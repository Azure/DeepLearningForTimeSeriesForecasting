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
from keras.layers import Dense, CuDNNGRU, RepeatVector, TimeDistributed, Flatten, Input
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
    train = energy.copy()[energy.index < valid_start_dt][['load']]

    # normalize training data
    y_scaler = MinMaxScaler()
    y_scaler.fit(train[['load']])
    X_scaler = MinMaxScaler()
    train[['load']] = X_scaler.fit_transform(train)

    tensor_structure = {'encoder_input':(range(-T+1, 1), ['load']), 'decoder_input':(range(0, HORIZON), ['load'])}
    train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)

    look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load']]
    valid[['load']] = X_scaler.transform(valid)
    valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)

    look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    test = energy.copy()[test_start_dt:][['load']]
    test[['load']] = X_scaler.transform(test)
    test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)

    return train_inputs, valid_inputs, test_inputs, y_scaler

# create the model with the given values of hyperparameters
def get_model(LEARNING_RATE, T, ALPHA, LATENT_DIM_1):

    # define training encoder
    encoder_input = Input(shape=(None, 1))
    encoder = CuDNNGRU(LATENT_DIM_1, return_state=True, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA))
    encoder_output, state_h = encoder(encoder_input)
    encoder_states = [state_h]

    # define training decoder
    decoder_input = Input(shape=(None, 1))
    decoder_GRU = CuDNNGRU(LATENT_DIM_1, return_state=True, return_sequences=True, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA))
    decoder_output, _ = decoder_GRU(decoder_input, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(1))
    decoder_output = decoder_dense(decoder_output)

    train_model = Model([encoder_input, decoder_input], decoder_output)

    optimizer = RMSprop(lr=LEARNING_RATE)
    train_model.compile(optimizer=optimizer, loss='mse')


    # build inference encoder model
    encoder_model = Model(encoder_input, encoder_states)

    # build inference decoder model
    decoder_state_input_h = Input(shape=(LATENT_DIM_1,))
    decoder_states_input = [decoder_state_input_h]

    decoder_output, state_h = decoder_GRU(decoder_input, initial_state=decoder_states_input)
    decoder_states = [state_h]
    decoder_output = decoder_dense(decoder_output)
    decoder_model = Model([decoder_input] + decoder_states_input, [decoder_output] + decoder_states)

    return train_model, encoder_model, decoder_model


def predict_single_sequence(encoder_model, decoder_model, single_input_seq, horizon, n_features):
    # apply encoder model to the input_seq to get state
    states_value = encoder_model.predict(single_input_seq)
    
    # get input for decoder's first time step (which is encoder input at time t)
    dec_input = np.zeros((1, 1, n_features))
    dec_input[0, 0, 0] = single_input_seq[0, -1, :]
    
    # create final output placeholder
    output = list()
    # collect predictions
    for t in range(horizon):
        # predict next value
        yhat, h = decoder_model.predict([dec_input] + [states_value])
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        #state = [h]
        states_value = h
        # update decoder input to be used as input for next prediction
        dec_input[0, 0, 0] = yhat
        
    return np.array(output)


def predict_multi_sequence(encoder_model, decoder_model, input_seq_multi, horizon, n_features):
    # create output placeholder
    predictions_all = list()
    for seq_index in range(input_seq_multi.shape[0]):       
        # Take one sequence for decoding
        input_seq = input_seq_multi[seq_index: seq_index + 1]
        # Generate prediction for the single sequence
        predictions = predict_single_sequence(encoder_model, decoder_model, input_seq, horizon, n_features)
        # store all the sequence prediction
        predictions_all.append(predictions)
        
    return np.array(predictions_all)


def run(energy, T_val, LATENT_DIM_1, BATCH_SIZE, LEARNING_RATE, ALPHA, out_file):

    train_inputs, valid_inputs, test_inputs, y_scaler = create_input(energy, T_val)
    validation_mapes_param = np.empty(N_EXPERIMENTS)
    test_mapes_param = np.empty(N_EXPERIMENTS)

    f = open(out_file, 'w')
    for ii in range(N_EXPERIMENTS):

        # Initialize the model
        train_model, encoder_model, decoder_model = get_model(LEARNING_RATE, T_val, ALPHA, LATENT_DIM_1)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
        best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1, save_weights_only=True)

        train_target = train_inputs['target'].reshape(train_inputs['target'].shape[0], train_inputs['target'].shape[1], 1)
        valid_target = valid_inputs['target'].reshape(valid_inputs['target'].shape[0], valid_inputs['target'].shape[1], 1)

        # Train the model
        history = train_model.fit([train_inputs['encoder_input'], train_inputs['decoder_input']],
                            train_target,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=([valid_inputs['encoder_input'], valid_inputs['decoder_input']], valid_target),
                            callbacks=[earlystop, best_val],
                            verbose=1)

        # load the model with the smallest validation MAPE
        best_epoch = np.argmin(np.array(history.history['val_loss']))+1
        validation_mapes_param[ii] = np.min(np.array(history.history['val_loss']))
        train_model.load_weights("model_{:02d}.h5".format(best_epoch))

        # Compute test MAPE
        #predictions = model.predict(test_inputs['X'])
        predictions = predict_multi_sequence(encoder_model, decoder_model, test_inputs['encoder_input'], HORIZON, 1)
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
        eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)
        test_mapes_param[ii] = mape(eval_df['prediction'], eval_df['actual'])
        result = 'Run results {0:.4f} {1:.4f}\n'.format(validation_mapes_param[ii], test_mapes_param[ii])
        f.write(result)

        # clean up model files
        for m in glob('model_*.h5'):
            os.remove(m)

    # output results
    params = 'T={0}, LATENT_DIM_1={1}, BATCH_SIZE={2}, LR={3}, ALPHA={4}\n'.format(T_val, LATENT_DIM_1, BATCH_SIZE, LEARNING_RATE, ALPHA)
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

    run(energy, args.T, args.LATENT_DIM_1, args.BATCH_SIZE, args.LEARNING_RATE, args.ALPHA, out_file)
