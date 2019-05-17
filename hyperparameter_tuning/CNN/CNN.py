import os
import sys
import shutil
import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import RMSprop
from azureml.core import Run

# define the boundaries of validation and test sets
valid_start_dt = "2014-09-01 00:00:00"
test_start_dt = "2014-11-01 00:00:00"

# fixed parameters
EPOCHS = 50  # max number of epochs when training FNN
HORIZON = 24  # forecasting horizon (in hours)

# Get the run object
run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])

# create training, validation and test sets given the length of the history
def create_input(energy, T):

    from utils import TimeSeriesTensor

    # get training data
    train = energy.copy()[energy.index < valid_start_dt][["load", "temp"]]

    # normalize training data
    y_scaler = MinMaxScaler()
    y_scaler.fit(train[["load"]])
    X_scaler = MinMaxScaler()
    train[["load", "temp"]] = X_scaler.fit_transform(train)

    tensor_structure = {"X": (range(-T + 1, 1), ["load", "temp"])}
    train_inputs = TimeSeriesTensor(train, "load", HORIZON, tensor_structure)

    look_back_dt = dt.datetime.strptime(
        valid_start_dt, "%Y-%m-%d %H:%M:%S"
    ) - dt.timedelta(hours=T - 1)
    valid = energy.copy()[
        (energy.index >= look_back_dt) & (energy.index < test_start_dt)
    ][["load", "temp"]]
    valid[["load", "temp"]] = X_scaler.transform(valid)
    valid_inputs = TimeSeriesTensor(valid, "load", HORIZON, tensor_structure)

    look_back_dt = dt.datetime.strptime(
        test_start_dt, "%Y-%m-%d %H:%M:%S"
    ) - dt.timedelta(hours=T - 1)
    test = energy.copy()[test_start_dt:][["load", "temp"]]
    test[["load", "temp"]] = X_scaler.transform(test)
    test_inputs = TimeSeriesTensor(test, "load", HORIZON, tensor_structure)

    return train_inputs, valid_inputs, test_inputs, y_scaler


# create the model with the given values of hyperparameters
def get_model(LEARNING_RATE, T, ALPHA, LATENT_DIM_1, LATENT_DIM_2, KERNEL_SIZE):
    model = Sequential()
    model.add(
        Conv1D(
            LATENT_DIM_1,
            kernel_size=KERNEL_SIZE,
            padding='causal',
            activation='relu',
            dilation_rate=1,
            input_shape=(T, 2),
            kernel_regularizer=regularizers.l2(ALPHA),
            bias_regularizer=regularizers.l2(ALPHA),
        )
    )
    if LATENT_DIM_2:
        model.add(
            Conv1D(
                LATENT_DIM_2,
                kernel_size=KERNEL_SIZE,
                padding='causal',
                activation='relu',
                dilation_rate=2,
                kernel_regularizer=regularizers.l2(ALPHA),
                bias_regularizer=regularizers.l2(ALPHA),
            )
        )

    model.add(Flatten())
    model.add(
        Dense(
            HORIZON,
            activation='linear',
            kernel_regularizer=regularizers.l2(ALPHA),
            bias_regularizer=regularizers.l2(ALPHA),
        )
    )

    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse")

    return model


def run_training(
    energy,
    T_val,
    LATENT_DIM_1,
    LATENT_DIM_2,
    KERNEL_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    ALPHA,
):

    from utils import create_evaluation_df, mape
    
    train_inputs, valid_inputs, test_inputs, y_scaler = create_input(energy, T_val)

    # Initialize the model
    model = get_model(LEARNING_RATE, T_val, ALPHA, LATENT_DIM_1, LATENT_DIM_2, KERNEL_SIZE)
    earlystop = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    best_val = ModelCheckpoint(
        "model_{epoch:02d}.h5",
        save_best_only=True,
        mode="min",
        period=1,
        save_weights_only=True,
    )

    # Train the model
    history = model.fit(
        train_inputs["X"],
        train_inputs["target"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(valid_inputs["X"], valid_inputs["target"]),
        callbacks=[earlystop, best_val, LogRunMetrics()],
        verbose=0,
    )

    # load the model with the smallest validation MAPE
    best_epoch = np.argmin(np.array(history.history["val_loss"])) + 1
    validationLoss = np.min(np.array(history.history["val_loss"]))
    model.load_weights("model_{:02d}.h5".format(best_epoch))

    # Save best model for this experiment
    model_name = "bestmodel"
    # serialize NN architecture to JSON
    model_json = model.to_json()
    # save model JSON
    with open("{}.json".format(model_name), "w") as f:
        f.write(model_json)
    # save model weights
    model.save_weights("{}.h5".format(model_name))
    
    # Compute test MAPE
    predictions = model.predict(test_inputs["X"])
    eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)
    testMAPE = mape(eval_df["prediction"], eval_df["actual"])

    # clean up model files
    for m in glob("model_*.h5"):
        os.remove(m)
    
    # Log a list of validation and test MAPEs from all experiments
    run.log("validationLoss", validationLoss)
    run.log("testMAPE", testMAPE)
    
    # create a ./outputs/model folder in the compute target
    # files saved in the "./outputs" folder are automatically uploaded into run history
    os.makedirs("./outputs/model", exist_ok=True)
    model_files = glob("bestmodel*")
    for f in model_files:
        shutil.move(f, "./outputs/model")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--datadir",
        type=str,
        dest="datadir",
        help="Directory where the dataset is located",
        required=True,
    )
    parser.add_argument(
        "--scriptdir",
        type=str,
        dest="scriptdir",
        help="Directory where scripts are located",
        required=True,
    )
    parser.add_argument(
        "--latent-dim-1",
        type=int,
        dest="LATENT_DIM_1",
        help="number of neurons in the first hidden layer",
        required=True,
    )
    parser.add_argument(
        "--latent-dim-2",
        type=int,
        dest="LATENT_DIM_2",
        help="number of neurons in the second hidden layer",
        default=0,
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        dest="KERNEL_SIZE",
        help="width of the convolution kernel",
        default=0,
    )
    parser.add_argument(
        "--batch-size", type=int, dest="BATCH_SIZE", help="batch size", required=True
    )
    parser.add_argument("--T", dest="T", type=int, help="history length", required=True)
    parser.add_argument(
        "--learning-rate",
        type=float,
        dest="LEARNING_RATE",
        help="learning rate",
        required=True,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        dest="ALPHA",
        help="regularization coefficient",
        required=True,
    )
    
    args = parser.parse_args()

    commondir = args.scriptdir
    
    sys.path.append(commondir)
    from utils import load_data
    from extract_data import extract_data
    
    # load data into Pandas dataframe
    data_dir = args.datadir
    if not os.path.exists(os.path.join(data_dir, "energy.csv")):
        extract_data(data_dir)
        
    energy = load_data(data_dir)

    # parse values of hyperparameters
    T = int(args.T)
    LATENT_DIM_1 = int(args.LATENT_DIM_1)
    LATENT_DIM_2 = int(args.LATENT_DIM_2)
    KERNEL_SIZE = int(args.KERNEL_SIZE)
    BATCH_SIZE = int(args.BATCH_SIZE)
    LEARNING_RATE = float(args.LEARNING_RATE)
    ALPHA = float(args.ALPHA)
    
    # train and evaluate RNN multi-step network with given values of hyperaparameters
    run_training(energy, T, LATENT_DIM_1, LATENT_DIM_2, KERNEL_SIZE, BATCH_SIZE, LEARNING_RATE, ALPHA)
