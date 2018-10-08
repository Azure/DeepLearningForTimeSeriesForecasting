import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, CuDNNGRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop

sys.stdout=sys.stderr
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)

# define the boundaries of validation and test sets
valid_start_dt = '2014-09-01 00:00:00'
test_start_dt = '2014-11-01 00:00:00'

# fixed parameters
EPOCHS = 100          # max number of epochs when training FNN
HORIZON = 24          # forecasting horizon (in hours)
N_EXPERIMENTS = 1     # number of experiments for each combination of hyperparameter values

# create training, validation and test sets given the length of the history
def create_input(energy, T):

    # get training data
    train = energy.copy()[energy.index < valid_start_dt][['load', 'temp']]

    # normalize training data
    y_scaler = MinMaxScaler()
    y_scaler.fit(train[['load']])
    X_scaler = MinMaxScaler()
    train[['load', 'temp']] = X_scaler.fit_transform(train)

    tensor_structure = {'X':(range(-T+1, 1), ['load', 'temp'])}
    train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)

    look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load', 'temp']]
    valid[['load', 'temp']] = X_scaler.transform(valid)
    valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)

    look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    test = energy.copy()[test_start_dt:][['load', 'temp']]
    test[['load', 'temp']] = X_scaler.transform(test)
    test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)

    return train_inputs, valid_inputs, test_inputs, y_scaler

# create the model with the given values of hyperparameters
def get_model(LEARNING_RATE, T, ALPHA, LATENT_DIM_1, LATENT_DIM_2):
    model = Sequential()
    if LATENT_DIM_2:
        model.add(CuDNNGRU(LATENT_DIM_1, input_shape=(T, 2), return_sequences=True, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
        model.add(CuDNNGRU(LATENT_DIM_2, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    else:
        model.add(CuDNNGRU(LATENT_DIM_1, input_shape=(T, 2), kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))

    model.add(Dense(HORIZON, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def run(energy, T_val, LATENT_DIM_1, LATENT_DIM_2, BATCH_SIZE, LEARNING_RATE, ALPHA, out_file):

    train_inputs, valid_inputs, test_inputs, y_scaler = create_input(energy, T_val)
    validation_mapes_param = np.empty(N_EXPERIMENTS)
    test_mapes_param = np.empty(N_EXPERIMENTS)

    f = open(out_file, 'w')
    for ii in range(N_EXPERIMENTS):

        # Initialize the model
        model = get_model(LEARNING_RATE, T_val, ALPHA, LATENT_DIM_1, LATENT_DIM_2)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
        best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1, save_weights_only=True)

        # Train the model
        history = model.fit(train_inputs['X'],
                            train_inputs['target'],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(valid_inputs['X'], valid_inputs['target']),
                            callbacks=[earlystop, best_val],
                            verbose=0)

        # load the model with the smallest validation MAPE
        best_epoch = np.argmin(np.array(history.history['val_loss']))+1
        validation_mapes_param[ii] = np.min(np.array(history.history['val_loss']))
        model.load_weights("model_{:02d}.h5".format(best_epoch))

        # Compute test MAPE
        predictions = model.predict(test_inputs['X'])
        eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)
        test_mapes_param[ii] = mape(eval_df['prediction'], eval_df['actual'])
        result = 'Run results {0:.4f} {1:.4f}\n'.format(validation_mapes_param[ii], test_mapes_param[ii])
        f.write(result)

        # clean up model files
        for m in glob('model_*.h5'):
            os.remove(m)

    # output results
    params = 'T={0}, LATENT_DIM_1={1}, LATENT_DIM_2={2}, BATCH_SIZE={3}, LR={4}, ALPHA={5}\n'.format(T_val, LATENT_DIM_1, LATENT_DIM_2, BATCH_SIZE, LEARNING_RATE, ALPHA)
    f.write(params)
    result1 = 'Mean validation MAPE = {0:.4f} +/- {1:.4f}\n'.format(np.mean(validation_mapes_param), np.std(validation_mapes_param)/np.sqrt(N_EXPERIMENTS))
    f.write(result1)
    result2 = 'Mean test MAPE = {0:.4f} +/- {1:.4f}\n'.format(np.mean(test_mapes_param), np.std(test_mapes_param)/np.sqrt(N_EXPERIMENTS))
    f.write(result2)
    f.close()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Directory where the dataset is located', required=True, type=str)
    parser.add_argument('-scriptdir', '--scriptdir', help='Directory where scripts are located', required=True, type=str)
    parser.add_argument('-outdir', '--outdir', help='Output directory', required=True, type=str)
    parser.add_argument('-l', '--LATENT_DIM_1', help='number of neurons in the first hidden layer', required=True, type=int)
    parser.add_argument('-s', '--LATENT_DIM_2', help='number of neurons in the second hidden layer', type=int, default=0)
    parser.add_argument('-b', '--BATCH_SIZE', help='batch size', required=True, type=int)
    parser.add_argument('-T', '--T', help='history length', required=True, type=int)
    parser.add_argument('-r', '--LEARNING_RATE', help='learning rate', required=True, type=float)
    parser.add_argument('-a', '--ALPHA', help='regularization coefficient', required=True, type=float)

    args = parser.parse_args()

    commondir = args.scriptdir
    sys.path.append(commondir)
    from common.utils import *
    from common.extract_data import *

    # load data into Pandas dataframe
    data_dir = args.datadir
    if not os.path.exists(os.path.join(data_dir, 'energy.csv')):
        extract_data(data_dir)
    energy = load_data(data_dir)

    out_file = os.path.join(args.outdir, 'output.txt')

    run(energy, args.T, args.LATENT_DIM_1, args.LATENT_DIM_2, args.BATCH_SIZE, args.LEARNING_RATE, args.ALPHA, out_file)
