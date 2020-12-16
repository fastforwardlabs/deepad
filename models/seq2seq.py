# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2020
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products 
#  made up of hundreds of individual components, each of which was 
#  individually copyrighted.  Each Cloudera open source product is a 
#  collective work under U.S. Copyright Law. Your license to use the 
#  collective work is as provided in your written agreement with  
#  Cloudera.  Used apart from the collective work, this file is 
#  licensed for your use pursuant to the open source license 
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute 
#  this code. If you do not have a written agreement with Cloudera nor 
#  with an authorized and properly licensed third party, you do not 
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED 
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO 
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND 
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, 
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS 
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE 
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES 
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF 
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
import tensorflow
import logging
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import train_utils

import numpy as np
import random
from utils import train_utils


# set random seed for reproducibility
tensorflow.random.set_seed(2018)
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class Seq2SeqModel():

    def __init__(self, n_features, encoder_dim=[10], decoder_dim=[20], dropout=0.2,
                 loss='mean_squared_error', learning_rate=0.01, epochs=15, batch_size=128, output_activation='sigmoid'):
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
        self.name = "seq2seq"
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_features = n_features

        self.create_model(n_features, encoder_dim=encoder_dim, decoder_dim=decoder_dim, dropout=dropout, learning_rate=learning_rate,
                          loss=loss, output_activation=output_activation)

    def create_model(self, n_features, encoder_dim=[20], decoder_dim=[20], dropout=0.2, learning_rate=.001,
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
        self.model = Model([encoder_inputs, decoder_inputs],
                           decoder_outputs, name="seq2seq")
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)

        # define encoder model returning encoder states
        self.encoder_model = Model(
            encoder_inputs, encoder_states * dec_dim, name="encoder")

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
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                                   [decoder_outputs] + decoder_states, name="decoder")

        logging.info(self.encoder_model.summary())
        logging.info(self.decoder_model.summary())
        logging.info(self.model.summary())

        # return model, encoder_model, decoder_model

    def train(self, in_train, in_test):

        # define inputs
        encoder_input_data = in_train
        decoder_input_data = in_train
        # offset decoder_input_data by 1 across time axis
        decoder_target_data = np.roll(in_train, -1, axis=1)

        kwargs = {}
        kwargs['epochs'] = self.epochs
        kwargs['batch_size'] = self.batch_size
        kwargs['shuffle'] = True
        # kwargs['validation_split'] = args.validation_split
        kwargs['validation_data'] = (
            [in_test, in_test], np.roll(in_test, -1, axis=1))
        kwargs['verbose'] = 1
        kwargs['callbacks'] = [train_utils.TimeHistory()]

        history = self.model.fit(
            [encoder_input_data, decoder_input_data], decoder_target_data, **kwargs)

    def decode_sequence(self, input_seq):
        """ Feed output of encoder to decoder and make sequential predictions. """

        timesteps = input_seq.shape[1]
        # use encoder the get state vectors
        states_value = self.encoder_model.predict(input_seq)

        # generate initial target sequence
        target_seq = input_seq[0, 0, :].reshape((1, 1, self.n_features))

        # sequential prediction of time series
        decoded_seq = np.zeros((1, timesteps, self.n_features))
        decoded_seq[0, 0, :] = target_seq[0, 0, :]
        i = 1
        while i < timesteps:

            decoder_output = self.decoder_model.predict(
                [target_seq] + states_value)
            # update the target sequence
            target_seq = np.zeros((1, 1, self.n_features))
            target_seq[0, 0, :] = decoder_output[0]

            # update output
            decoded_seq[0, i, :] = decoder_output[0]
            # update states
            states_value = decoder_output[1:]
            i += 1

        return decoded_seq

    def predict(self, input_data):
        n_obs = input_data.shape[0]
        obs = 0
        input_seq = input_data[obs:obs+1, :, :]
        decoded_seq = self.decode_sequence(input_seq)
        return input_seq[0, :, :], decoded_seq[0, :, :]

    def compute_anomaly_score(self, input_data):
        logging.info(" >> Computing mse for test values")
        anomaly_score_holder = []
        for i in range(len(input_data)):
            X = input_data[i, :].reshape(1, input_data.shape[1], 1)
            orig_input, preds = self.predict(X,)
            mse = np.mean(np.power(orig_input - preds, 2))
            anomaly_score_holder.append(mse)

        return anomaly_score_holder

    def save_model(self, model_path="models/savedmodels/seq2seq/"):
        logging.info(">> Saving Bigan model to " + model_path)
        self.model.save_weights(model_path + "model")
        self.encoder_model.save_weights(model_path + "encoder")
        self.decoder_model.save_weights(model_path + "decoder")

    def load_model(self, model_path="models/savedmodels/seq2seq/"):
        if (os.path.exists(model_path)):
            logging.info(">> Loading saved model weights")
            self.model.load_weights(model_path + "model")
            self.encoder_model.load_weights(model_path + "encoder")
            self.decoder_model.load_weights(model_path + "decoder")
