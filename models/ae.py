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

import tensorflow
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import logging

import os
from utils import train_utils

import numpy as np
import random


# set random seed for reproducibility
tensorflow.random.set_seed(2018)
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class AutoencoderModel():

    def __init__(self, n_features, hidden_layers=2, latent_dim=2, hidden_dim=[15, 7],
                 output_activation='sigmoid', learning_rate=0.01, epochs=15, batch_size=128, model_path=None):
        """ Build AE model.
        Arguments:
            - n_features (int): number of features in the data
            - hidden_layers (int): number of hidden layers used in encoder/decoder
            - latent_dim (int): dimension of latent variable
            - hidden_dim (list): list with dimension of each hidden layer
            - output_activation (str): activation type for last dense layer in the decoder
            - learning_rate (float): learning rate used during training
        """

        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = "ae"

        self.create_model(n_features, hidden_layers=hidden_layers, latent_dim=latent_dim,
                          hidden_dim=hidden_dim, output_activation=output_activation,
                          learning_rate=learning_rate, model_path=model_path)

    def create_model(self, n_features, hidden_layers=1, latent_dim=2, hidden_dim=[],
                     output_activation='sigmoid', learning_rate=0.001, model_path=None):

        # set dimensions hidden layers
        if hidden_dim == []:
            i = 0
            dim = n_features
            while i < hidden_layers:
                hidden_dim.append(int(np.max([dim/2, 2])))
                dim /= 2
                i += 1

        # Optional: add regularization to minimize overfitting?
        # kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)
        # kernel_regularizer = regularizers.l1(0.01)
        kernel_regularizer = None

        # AE = encoder + decoder
        # encoder
        inputs = Input(shape=(n_features,), name='encoder_input')
        # define hidden layers
        enc_hidden = Dense(hidden_dim[0], activation='relu', name='encoder_hidden_0',
                           kernel_regularizer=kernel_regularizer)(inputs)
        i = 1
        while i < hidden_layers:
            enc_hidden = Dense(hidden_dim[i], activation='relu', name='encoder_hidden_'+str(
                i), kernel_regularizer=kernel_regularizer)(enc_hidden)
            i += 1

        z_ = Dense(latent_dim, name='z_')(enc_hidden)

        encoder = Model(inputs, z_, name='encoder')
        logging.info(encoder.summary())
        # plot_model(encoder, to_file='ae_mlp_encoder.png',
        #            show_shapes=True)

        # decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_')
        # define hidden layers
        dec_hidden = Dense(hidden_dim[-1], activation='relu', name='decoder_hidden_0',
                           kernel_regularizer=kernel_regularizer)(latent_inputs)

        i = 2
        while i < hidden_layers+1:
            dec_hidden = Dense(hidden_dim[-i], activation='relu', name='decoder_hidden_'+str(
                i-1), kernel_regularizer=kernel_regularizer)(dec_hidden)
            i += 1

        outputs = Dense(n_features, activation=output_activation,
                        name='decoder_output')(dec_hidden)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        logging.info(decoder.summary())
        # plot_model(decoder, to_file='ae_mlp_decoder.png',
        #            show_shapes=True)

        # instantiate AE model
        outputs = decoder(encoder(inputs))
        self.model = Model(inputs, outputs, name='ae', )

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss="mse")

    def train(self, in_train, in_val):
        # default args

        # training

        X_train, X_val = in_train, in_val
        logging.info("Training with data of shape " + str(X_train.shape))

        kwargs = {}
        kwargs['epochs'] = self.epochs
        kwargs['batch_size'] = self.batch_size
        kwargs['shuffle'] = True
        kwargs['validation_data'] = (X_val, X_val)
        kwargs['verbose'] = 1
        kwargs['callbacks'] = [train_utils.TimeHistory()]

        history = self.model.fit(X_train, X_train, **kwargs)

    def compute_anomaly_score(self, df):
        preds = self.model.predict(df)
        mse = np.mean(np.power(df - preds, 2), axis=1)
        return mse

    def save_model(self, model_path="models/savedmodels/ae/"):
        logging.info(">> Saving AE model to " + model_path)
        self.model.save_weights(model_path + "model")

    def load_model(self, model_path="models/savedmodels/ae/"):
        if (os.path.exists(model_path)):
            logging.info(">> Loading saved model weights")
            self.model.load_weights(model_path + "model")
