
# multi-step forecasting with multivariate input using feed-forward neural network
# train and evaluate feed-forward neural network with a given values of hyperparameters
# this code can be run both inside Batch AI node and as a standalone script

# The code uses evergy.csv file derived from the data in GEFCom2014 forecasting competition. It consists of 3 years of hourly electricity load and temperature values 
# between 2012 and 2014. The task is to forecast future values of electricity load.
# Reference: T. Hong, P. Pinson, S. Fan, H. Zareipour, A. Troccoli and . Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", 
# International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

import sys
import os
import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
from itertools import product
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras import losses, regularizers

sys.stdout = sys.stderr
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)

# define the boundaries of validation and test sets
valid_start_dt = '2014-09-01 00:00:00'
test_start_dt = '2014-11-01 00:00:00'

# fixed parameters
EPOCHS = 50           # max number of epochs when training FNN
HORIZON = 24          # forecasting horizon (in hours)
N_EXPERIMENTS = 5     # number of experiments for each combination of hyperparameter values

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
    X_train = train_inputs.dataframe.as_matrix()[:,HORIZON:]
    Y_train = train_inputs['target']

    # Construct validation set (keeping T hours from the training set in order to construct initial features)
    look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load', 'temp']]
    valid[['load', 'temp']] = X_scaler.transform(valid)
    valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)
    X_valid = valid_inputs.dataframe.as_matrix()[:,HORIZON:]
    Y_valid = valid_inputs['target']

    # Construct test set (keeping T hours from the validation set in order to construct initial features)
    look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    test = energy.copy()[test_start_dt:][['load', 'temp']]
    test[['load', 'temp']] = X_scaler.transform(test)
    test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)
    X_test = test_inputs.dataframe.as_matrix()[:,HORIZON:]

    return X_train, Y_train, X_valid, Y_valid, X_test, test_inputs, y_scaler

# create the model with the given values of hyperparameters
def get_model(LATENT_DIM, LEARNING_RATE, T, ALPHA, HIDDEN_LAYERS):
    model = Sequential()
    model.add(Dense(LATENT_DIM, activation="relu", input_shape=(2*T,), kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    for i in range(HIDDEN_LAYERS-1):
        model.add(Dense(LATENT_DIM, activation="relu", kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    model.add(Dense(HORIZON, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def run(energy, T_val, LATENT_DIM, BATCH_SIZE, LEARNING_RATE, ALPHA, HIDDEN_LAYERS, out_file):

    X_train, Y_train, X_valid, Y_valid, X_test, test_inputs, y_scaler = create_input(energy, T_val)
    validation_mapes_param = np.empty(N_EXPERIMENTS)
    test_mapes_param = np.empty(N_EXPERIMENTS)

    f = open(out_file, 'w')
    for ii in range(N_EXPERIMENTS):
        # Initialize the model
        model = get_model(BATCH_SIZE, LEARNING_RATE, T_val, ALPHA, HIDDEN_LAYERS)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
        best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1, save_weights_only=True)

        # Train the model
        history = model.fit(X_train, Y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(X_valid, Y_valid),
                            callbacks=[earlystop, best_val],
                            verbose=0)

        # load the model with the smallest validation MAPE
        best_epoch = np.argmin(np.array(history.history['val_loss']))+1
        validation_mapes_param[ii] = np.min(np.array(history.history['val_loss']))
        model.load_weights("model_{:02d}.h5".format(best_epoch))

        # Compute test MAPE 
        predictions = model.predict(X_test)
        eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)
        test_mapes_param[ii] = mape(eval_df['prediction'], eval_df['actual'])
        result = 'Run results {0:.4f} {1:.4f}\n'.format(validation_mapes_param[ii], test_mapes_param[ii])
        f.write(result)

        # clean up model files
        for m in glob('model_*.h5'):
            os.remove(m)

    # output results
    params = 'T={0}, LATENT_DIM={1}, HIDDEN_LAYERS={2}, BATCH_SIZE={3}, LR={4}, ALPHA={5}\n'.format(T_val, LATENT_DIM, HIDDEN_LAYERS, BATCH_SIZE, LEARNING_RATE, ALPHA)
    f.write(params)
    result1 = 'Mean validation MAPE = {0:.4f} +/- {1:.4f}\n'.format(np.mean(validation_mapes_param), np.std(validation_mapes_param)/np.sqrt(N_EXPERIMENTS))
    f.write(result1)
    result2 = 'Mean test MAPE = {0:.4f} +/- {1:.4f}\n'.format(np.mean(test_mapes_param), np.std(test_mapes_param)/np.sqrt(N_EXPERIMENTS))
    f.write(result2)
    f.close()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Directory where the dataset is located', required=True)
    parser.add_argument('-scriptdir', '--scriptdir', help='Directory where scripts are located', required=True)
    parser.add_argument('-outdir', '--outdir', help='Output directory', required=True)    
    parser.add_argument('-l', '--LATENT_DIM', help='number of neurons in each hidden layer', required=True)
    parser.add_argument('-n', '--HIDDEN_LAYERS', help='number of hidden layers', required=True)
    parser.add_argument('-b', '--BATCH_SIZE', help='batch size', required=True)
    parser.add_argument('-T', '--T', help='history length', required=True)
    parser.add_argument('-r', '--LEARNING_RATE', help='learning rate', required=True)
    parser.add_argument('-a', '--ALPHA', help='regularization coefficient', required=True)

    args = vars(parser.parse_args())

    commondir = args['scriptdir']
    sys.path.append(commondir)
    from common.utils import *
    from common.extract_data import *
    
    # load data into Pandas dataframe
    data_dir = args['datadir']
    if not os.path.exists(os.path.join(data_dir, 'energy.csv')):
        extract_data(data_dir)
    energy = load_data(data_dir)

    # parse values of hyperparameters
    T = int(args['T'])
    HIDDEN_LAYERS = int(args['HIDDEN_LAYERS'])
    LATENT_DIM = int(args['LATENT_DIM'])
    BATCH_SIZE = int(args['BATCH_SIZE'])
    LEARNING_RATE = float(args['LEARNING_RATE'])
    ALPHA = float(args['ALPHA'])
    out_file = os.path.join(args['outdir'], 'output.txt')

    # train and evaluate feed-forward NN with given values of hyperaparameters
    run(energy, T, LATENT_DIM, BATCH_SIZE, LEARNING_RATE, ALPHA, HIDDEN_LAYERS, out_file)


