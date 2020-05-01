#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Code samples adapted from https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
# Licensed under the MIT License (the "License");
# =============================================================================
#

import tensorflow
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, Flatten, concatenate, Reshape, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import logging
from tensorflow.keras.layers.advanced_activations import LeakyReLU, ReLU

from utils import train_utils
import matplotlib.pyplot as plt

import numpy as np
import random


# set random seed for reproducibility
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


train_time_holder = []
LEARNING_RATE = 0.00001


class BiGANModel():
    def __init__(self, n_features, hidden_layers=2, latent_dim=2, hidden_dim=[15, 7],
                 output_activation='sigmoid', learning_rate=0.01, epochs=15, batch_size=128, model_path=None):

        self.max_accuracy = 0
        self.is_training = False
        self.img_rows = in_train.shape[1]
        self.anomaly_score_weight = 0.2
        self.img_cols = 1
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 32

        # optimizer = Adam(0.00001, 0.5)
        optimizer = Adam(lr=LEARNING_RATE)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        print("discriminator =========")
        self.discriminator.summary()
        self.discriminator.name = "discriminator"
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        print("generator =========")
        self.generator.summary()
        self.generator.name = "generator"

        # Build the encoder
        self.encoder = self.build_encoder()
        print("encoder =========")
        self.encoder.summary()

        self.encoder.name = "encoder"

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ), name="inputnoise")
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape,  name="inputimage")
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        #  self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
        self.bigan_generator.compile(loss=['mse', 'mse'],
                                     optimizer=optimizer)
        self.bigan_generator.name = "bigan"

        print("Full Model =========")
        self.bigan_generator.summary()

    def build_encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.latent_dim))

        # Specify the input to the encoder model
        img = Input(shape=self.img_shape)

        z = model(img)
        return Model(img, z)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(np.prod(self.img_shape), activation='sigmoid'))
        model.add(Reshape(self.img_shape))

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)

    def build_discriminator(self):

        x = Input(shape=self.img_shape, name="inputimage")
        x_ = LeakyReLU(alpha=0.2)(x)
        x_ = Dropout(0.2)(x)

        z = Input(shape=(self.latent_dim, ), name="latentz")
        z_ = LeakyReLU(alpha=0.2)(z)
        z_ = Dropout(0.2)(z)

        d_in = concatenate([z_, Flatten()(x_)])

        model = Dense(128)(d_in)
        model = LeakyReLU(alpha=0.1)(model)
        model = Dropout(0.2, name="discriminator_features")(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, x], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        self.is_training = True
        # Load the dataset
        X_train, _ = in_train, in_val

        print(">> Training on data, ", X_train.shape)

        self.train_ecg = X_train

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Store training metrics
        self.train_metrics = []

        for epoch in range(epochs):

            start_time = time.time()
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(batch_size, self.latent_dim))
            imgs_ = self.generator.predict(z)

            # Select a random batch of images and encode
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            z_ = self.encoder.predict(imgs)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch(
                [z, imgs], [valid, fake])

            # save time used to train gene and disceriminator
            train_time_holder.append(time.time() - start_time)

            d_accuracy = 100*d_loss[1]
            # Save training progresss statistics
            self.train_metrics.append({
                "d_loss": d_loss[0],
                "d_accuracy": d_accuracy,
                "g_loss": g_loss[0]
            })

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" %
                      (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
                self.sample_interval(epoch)
                # if ( d_accuracy > self.max_accuracy):
                #   print("saving best model")
                #   self.max_accuracy = d_accuracy
                #   self.generator.save("models/generator.h5")
                #   self.discriminator.save("models/discriminator.h5")
                #   self.encoder.save("models/encoder.h5")

    def sample_interval(self, epoch):
        # load test data, use to evaluate the model.
        # normal ecg is ecg belonging to class1, anomalous belongs to other 4 classes
        normal_scores = compute_anomaly_score(in_test, bigan)
        abnormal_scores = compute_anomaly_score(out_test, bigan)
        best_metrics = plot_roc_histogram(
            abnormal_scores, normal_scores, epoch)
        print("best metrics", best_metrics, len(
            normal_scores) + len(abnormal_scores))
        # compute_loss_and_plot(in_val[:5000], out_val[:5000], epoch)
