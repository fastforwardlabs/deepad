#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Code samples adapted from https://keras.io/examples/variational_autoencoder/
# Licensed under the MIT License (the "License");
# =============================================================================
#

import tensorflow
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import logging

from utils import train_utils
import matplotlib.pyplot as plt

import numpy as np
import random


# set random seed for reproducibility
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class VAEModel():

    def __init__(self, n_features, hidden_layers=2, latent_dim=2, hidden_dim=[15, 7],
                 output_activation='sigmoid', learning_rate=0.01, epochs=15, batch_size=128, model_path=None):
        """ Build VAE model.
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

        self.create_model(n_features, hidden_layers=hidden_layers, latent_dim=latent_dim,
                          hidden_dim=hidden_dim, output_activation=output_activation,
                          learning_rate=learning_rate, model_path=model_path)

    def create_model(self, n_features, hidden_layers=1, latent_dim=2, hidden_dim=[],
                     output_activation='sigmoid', learning_rate=0.001, model_path=None):

        def sampling(args):
            """ Reparameterization trick by sampling from an isotropic unit Gaussian.
            Arguments:
                - args (tensor): mean and log of variance of Q(z|X)
            Returns:
                - z (tensor): sampled latent vector
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=(batch, dim))
            # mean + stdev * eps
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # set dimensions hidden layers
        if hidden_dim == []:
            i = 0
            dim = n_features
            while i < hidden_layers:
                hidden_dim.append(int(np.max([dim/2, 2])))
                dim /= 2
                i += 1

        # VAE = encoder + decoder
        # encoder
        inputs = Input(shape=(n_features,), name='encoder_input')
        # define hidden layers
        enc_hidden = Dense(hidden_dim[0], activation='relu',
                           name='encoder_hidden_0')(inputs)
        i = 1
        while i < hidden_layers:
            enc_hidden = Dense(
                hidden_dim[i], activation='relu', name='encoder_hidden_'+str(i))(enc_hidden)
            i += 1

        z_mean = Dense(latent_dim, name='z_mean')(enc_hidden)
        z_log_var = Dense(latent_dim, name='z_log_var')(enc_hidden)
        # reparametrization trick to sample z
        z = Lambda(sampling, output_shape=(latent_dim,),
                   name='z')([z_mean, z_log_var])
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        logging.debug(encoder.summary())

        # decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        # define hidden layers
        dec_hidden = Dense(hidden_dim[-1], activation='relu',
                           name='decoder_hidden_0')(latent_inputs)

        i = 2
        while i < hidden_layers+1:
            dec_hidden = Dense(
                hidden_dim[-i], activation='relu', name='decoder_hidden_'+str(i-1))(dec_hidden)
            i += 1

        outputs = Dense(n_features, activation=output_activation,
                        name='decoder_output')(dec_hidden)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        logging.debug(decoder.summary())

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        self.model = Model(inputs, outputs, name='vae')

        # define VAE loss, optimizer and compile model
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= n_features
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.model.add_loss(vae_loss)

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer)

        if model_path:
            logging.debug(">> Loading saved model weights")
            self.model.load_weights(model_path)

    def train(self, in_train, in_val):
        # default args

        # training

        X_train, X_val = in_train, in_val
        logging.debug("Training with data of shape " + str(X_train.shape))

        kwargs = {}
        kwargs['epochs'] = self.epochs
        kwargs['batch_size'] = self.batch_size
        kwargs['shuffle'] = True
        kwargs['validation_data'] = (X_val, None)
        kwargs['verbose'] = 1
        kwargs['callbacks'] = [train_utils.TimeHistory()]

        history = self.model.fit(X_train, **kwargs)
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def compute_anomaly_score(self, df):
        preds = self.model.predict(df)
        mse = np.mean(np.power(df - preds, 2), axis=1)
        return mse

    def save_model(self, save_path):
        logging.debug(">> Saving VAE model to " + save_path)
        self.model.save_weights(save_path)
