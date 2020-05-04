#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Code samples adapted from https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
# Licensed under the MIT License (the "License");
# =============================================================================
#

from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import train_utils
import matplotlib.pyplot as plt

import numpy as np
import random


# set random seed for reproducibility
tensorflow.random.set_seed(2018)
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class Seq2SeqModel():

     def __init__(self, n_features, hidden_layers=2, latent_dim=2, hidden_dim=[15, 7],
                 output_activation='sigmoid', learning_rate=0.01, epochs=15, batch_size=128, model_path=None):
         """ Build seq2seq model.

        Arguments:
            - n_features (int): number of features in the data
            - encoder_dim (list): list with number of units per encoder layer
            - decoder_dim (list): list with number of units per decoder layer
            - dropout (float): dropout for LSTM units
            - learning_rate (float): learning rate used during training
            - loss (str): loss function used
            - output_activation (str): activation type for the dense output layer in the decoder
        """




    def create_model(self, n_features, encoder_dim=[20], decoder_dim=[20], dropout=0., learning_rate=.001,
                  loss='mean_squared_error', output_activation='sigmoid'):
       
        
        enc_dim = len(encoder_dim)
        dec_dim = len(decoder_dim)

        # seq2seq = encoder + decoder
        # encoder
        encoder_hidden = encoder_inputs = Input(
            shape=(None, n_features), name='encoder_input')

        # add encoder hidden layers
        encoder_lstm = []
        for i in range(enc_dim-1):
            encoder_lstm.append(Bidirectional(LSTM(encoder_dim[i], dropout=dropout,
                                                   return_sequences=True, name='encoder_lstm_' + str(i))))
            encoder_hidden = encoder_lstm[i](encoder_hidden)

        encoder_lstm.append(Bidirectional(LSTM(encoder_dim[-1], dropout=dropout, return_state=True,
                                               name='encoder_lstm_' + str(enc_dim-1))))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm[-1](
            encoder_hidden)

        # only need to keep encoder states
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # decoder
        decoder_hidden = decoder_inputs = Input(
            shape=(None, n_features), name='decoder_input')

        # add decoder hidden layers
        # check if dimensions are correct
        dim_check = [(idx, dim) for idx, dim in enumerate(
            decoder_dim) if dim != encoder_dim[-1]*2]
        if len(dim_check) > 0:
            raise ValueError('\nDecoder (layer,units) {0} is not compatible with encoder hidden '
                             'states. Units should be equal to {1}'.format(dim_check, encoder_dim[-1]*2))

        # initialise decoder states with encoder states
        decoder_lstm = []
        for i in range(dec_dim):
            decoder_lstm.append(LSTM(decoder_dim[i], dropout=dropout, return_sequences=True,
                                     return_state=True, name='decoder_lstm_' + str(i)))
            decoder_hidden, _, _ = decoder_lstm[i](
                decoder_hidden, initial_state=encoder_states)

        # add linear layer on top of LSTM
        decoder_dense = Dense(
            n_features, activation=output_activation, name='dense_output')
        decoder_outputs = decoder_dense(decoder_hidden)

        # define seq2seq model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss=loss)

        # define encoder model returning encoder states
        encoder_model = Model(encoder_inputs, encoder_states * dec_dim)

        # define decoder model
        # need state inputs for each LSTM layer
        decoder_states_inputs = []
        for i in range(dec_dim):
            decoder_state_input_h = Input(
                shape=(decoder_dim[i],), name='decoder_state_input_h_' + str(i))
            decoder_state_input_c = Input(
                shape=(decoder_dim[i],), name='decoder_state_input_c_' + str(i))
            decoder_states_inputs.append(
                [decoder_state_input_h, decoder_state_input_c])
        decoder_states_inputs = [
            state for states in decoder_states_inputs for state in states]

        decoder_inference = decoder_inputs
        decoder_states = []
        for i in range(dec_dim):
            decoder_inference, state_h, state_c = decoder_lstm[i](decoder_inference,
                                                                  initial_state=decoder_states_inputs[2*i:2*i+2])
            decoder_states.append([state_h, state_c])
        decoder_states = [
            state for states in decoder_states for state in states]

        decoder_outputs = decoder_dense(decoder_inference)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)

        # return model, encoder_model, decoder_model

    def compute_anomaly_score(self, df):
        print( "  computing mse for test values")
        mse_holder = []
        for i in range( len(df)):
            X = df[i,:].reshape(1,df.shape[1],1) 
            input_data, preds = predict(X,)
            mse = np.mean(np.power(input_data - preds, 2))
            mse_holder.append(mse) 
            drawProgressBar(i/len(df))
            # print(i,len(mse_holder))
   
        return mse_holder

    def decode_sequence(self, input_seq):
         # ..
         """ Feed output of encoder to decoder and make sequential predictions. """

        # use encoder the get state vectors
        states_value = enc.predict(input_seq)

        # generate initial target sequence
        target_seq = input_seq[0,0,:].reshape((1,1,n_features))

        # sequential prediction of time series
        decoded_seq = np.zeros((1, timesteps, n_features))
        decoded_seq[0,0,:] = target_seq[0,0,:]
        i = 1
        while i < timesteps:

            decoder_output = dec.predict([target_seq] + states_value)
            # update the target sequence
            target_seq = np.zeros((1, 1, n_features))
            target_seq[0, 0, :] = decoder_output[0]

            # update output
            decoded_seq[0, i, :] = decoder_output[0]
            # update states
            states_value = decoder_output[1:]
            i+=1

        return decoded_seq

    def predict(self, X): 
        n_obs = X.shape[0] 
        obs = 0 
        input_seq = X[obs:obs+1,:,:]
        decoded_seq = decode_sequence(input_seq)  
        return input_seq[0,:,:], decoded_seq[0,:,:]
