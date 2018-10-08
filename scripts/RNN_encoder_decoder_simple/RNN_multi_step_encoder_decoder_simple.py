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
from keras.layers import Dense, CuDNNGRU, RepeatVector, Flatten, TimeDistributed
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
def get_model(LEARNING_RATE, T, ALPHA, ENCODER_DIM_1, ENCODER_DIM_2, DECODER_DIM_1, DECODER_DIM_2):
    model = Sequential()
    if ENCODER_DIM_2:
        model.add(CuDNNGRU(ENCODER_DIM_1, input_shape=(T, 2), return_sequences=True, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
        model.add(CuDNNGRU(ENCODER_DIM_2, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    else:
        model.add(CuDNNGRU(ENCODER_DIM_1, input_shape=(T, 2), kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))

    model.add(RepeatVector(HORIZON))

    model.add(CuDNNGRU(DECODER_DIM_1, return_sequences=True, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    if DECODER_DIM_2:
        model.add(CuDNNGRU(DECODER_DIM_2, return_sequences=True, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def run(energy, T_val, ENCODER_DIM_1, ENCODER_DIM_2, DECODER_DIM_1, DECODER_DIM_2, BATCH_SIZE, LEARNING_RATE, ALPHA, out_file):

    train_inputs, valid_inputs, test_inputs, y_scaler = create_input(energy, T_val)
    validation_mapes_param = np.empty(N_EXPERIMENTS)
    test_mapes_param = np.empty(N_EXPERIMENTS)

    f = open(out_file, 'w')
    for ii in range(N_EXPERIMENTS):

        # Initialize the model
        model = get_model(LEARNING_RATE, T_val, ALPHA, ENCODER_DIM_1, ENCODER_DIM_2, DECODER_DIM_1, DECODER_DIM_2)
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
    params = 'T={0}, ENCODER_DIM_1={1}, ENCODER_DIM_2={2}, DECODER_DIM_1={3}, DECODER_DIM_2={4}, BATCH_SIZE={5}, LR={6}, ALPHA={7}\n'.format(T_val, ENCODER_DIM_1, ENCODER_DIM_2, 
                                                                                                                                             DECODER_DIM_1, DECODER_DIM_2,
                                                                                                                                             BATCH_SIZE, LEARNING_RATE, ALPHA)
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
    parser.add_argument('-e1', '--ENCODER_DIM_1', help='number of neurons in the first layer of encoder', required=True, type=int)
    parser.add_argument('-e2', '--ENCODER_DIM_2', help='number of neurons in the second layer of encoder', type=int, default=0)
    parser.add_argument('-d1', '--DECODER_DIM_1', help='number of neurons in the first layer of decoder', required=True, type=int)
    parser.add_argument('-d2', '--DECODER_DIM_2', help='number of neurons in the second layer of decoder', type=int, default=0)
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

    run(energy, args.T, args.ENCODER_DIM_1, args.ENCODER_DIM_2, args.DECODER_DIM_1, args.DECODER_DIM_2, 
        args.BATCH_SIZE, args.LEARNING_RATE, args.ALPHA, out_file)
