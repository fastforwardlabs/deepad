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
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, Flatten, concatenate, Reshape, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import logging
from tensorflow.keras.layers import LeakyReLU, ReLU
import os


import numpy as np
import random


# set random seed for reproducibility
tensorflow.random.set_seed(2018)
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class BiGANModel():
    def __init__(self,  input_shape, dense_dim=64, latent_dim=32,
                 output_activation='sigmoid', learning_rate=0.01, epochs=15, batch_size=128, model_path=None):
        print("bigan")
        self.name = "bigan"
        self.epochs = epochs
        self.dense_dim = dense_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.create_model(input_shape=input_shape,
                          learning_rate=learning_rate)

    def create_model(self, input_shape=(18, 1, 1),   learning_rate=0.01):
        self.input_shape = input_shape

        # optimizer = Adam(0.00001, 0.5)
        optimizer = Adam(lr=learning_rate)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        logging.info(self.discriminator.summary())
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        logging.info(self.generator.summary())

        # Build the encoder
        self.encoder = self.build_encoder()
        logging.info(self.encoder.summary())

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate output from sampled noise
        z = Input(shape=(self.latent_dim, ), name="inputnoise")
        input_data_ = self.generator(z)

        # Encode image
        input_data = Input(shape=self.input_shape,  name="inputimage")
        z_ = self.encoder(input_data)

        # Latent -> input_data is fake, and input_data -> latent is valid
        fake = self.discriminator([z, input_data_])
        valid = self.discriminator([z_, input_data])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model(
            [z, input_data], [fake, valid], name="bigan")
        self.bigan_generator.compile(
            loss=['binary_crossentropy', 'binary_crossentropy'],   optimizer=optimizer)  # ['mse', 'mse']

        logging.info(self.bigan_generator.summary())

    def build_encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(self.dense_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.latent_dim))

        # Specify the input to the encoder model
        input_data = Input(shape=self.input_shape)

        z = model(input_data)
        return Model(input_data, z, name="encoder")

    def build_generator(self):
        model = Sequential()

        model.add(Dense(self.dense_dim, input_dim=self.latent_dim))
        model.add(Activation('relu'))
        # model.add(Dense(32))
        # model.add(Activation('relu'))
        model.add(Dense(np.prod(self.input_shape), activation='sigmoid'))
        model.add(Reshape(self.input_shape))

        z = Input(shape=(self.latent_dim,))
        input_data_ = model(z)

        return Model(z, input_data_, name="generator")

    def build_discriminator(self):

        x = Input(shape=self.input_shape, name="inputdata")
        x_ = LeakyReLU(alpha=0.2)(x)
        x_ = Dropout(0.2)(x)

        z = Input(shape=(self.latent_dim, ), name="latentz")
        z_ = LeakyReLU(alpha=0.2)(z)
        z_ = Dropout(0.2)(z)

        d_in = concatenate([z_, Flatten()(x_)])

        model = Dense(self.dense_dim)(d_in)
        model = LeakyReLU(alpha=0.1)(model)
        model = Dropout(0.2, name="discriminator_features")(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, x], validity, name="discriminator")

    def train(self, in_train, in_val):

        print(">> Training on data, ", in_train.shape)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        # Store training metrics
        self.train_metrics = []

        for epoch in range(self.epochs):
            self.train_metrics = []
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(self.batch_size, self.latent_dim))
            input_data_ = self.generator.predict(z)

            # Select a random batch of images and encode
            idx = np.random.randint(0, in_train.shape[0], self.batch_size)
            input_data = in_train[idx]
            z_ = self.encoder.predict(input_data)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch(
                [z_, input_data], valid)
            d_loss_fake = self.discriminator.train_on_batch(
                [z, input_data_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch(
                [z, input_data], [valid, fake])

            d_accuracy = 100*d_loss[1]
            # Save training progresss statistics
            self.train_metrics.append({
                "d_loss": d_loss[0],
                "d_accuracy": d_accuracy,
                "g_loss": g_loss[0]
            })

            # If at save interval => save generated image samples

            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

    def compute_anomaly_score(self, input_data, weight_parameter=0.1, lnorm_degree=1):
        # Compute an intermediate model with the last layer of the trained discriminator
        # This model is used to compute feature loss
        feat_model = Model(inputs=self.discriminator.get_input_at(
            0), outputs=self.discriminator.get_layer("discriminator_features").output)

        # Compute z values from the input data using our encoder
        z_ = self.encoder.predict(input_data)

        # Now generate a reconstruction of input data using our generator and z_ from our encoder
        input_data_ = self.generator.predict(z_)

        # Generate loss as difference (L1 norm) between reconstructed input and actual input
        delta = input_data - input_data_
        delta = delta.reshape(delta.shape[0], -1)
        gen_loss = np.linalg.norm(delta,  axis=1, ord=lnorm_degree)

        # Compute discriminator loss based on cross entropy.
        valid = np.ones((input_data.shape[0], 1))
        # cross entropy loss.
        disc_loss = self.discriminator.test_on_batch([z_, input_data_], valid)

        # Compute discriminator loss based on feature matching differences (L1 norm)
        f1 = feat_model.predict([z_, input_data])
        f2 = feat_model.predict([z_, input_data_])
        f_delta = f1 - f2
        disc_loss_fm = np.linalg.norm(f_delta, axis=1, ord=lnorm_degree)

        # Now compute final loss as convex combination of  both generator aad discriminator loss
        final_loss_fm = (1 - weight_parameter) * gen_loss + \
            weight_parameter*disc_loss_fm
        final_loss = (1 - weight_parameter) * gen_loss + \
            weight_parameter*disc_loss[0]

        # print("Final loss :",  "min", np.min(final_loss), "max", np.max(final_loss), len(final_loss) )
        # print("Final loss fm :",  "min", np.min(final_loss_fm), "max", np.max(final_loss_fm), len(final_loss_fm)  )
        return final_loss

    def save_model(self, model_path="models/savedmodels/bigan/"):
        logging.info(">> Saving Bigan model to " + model_path)
        self.bigan_generator.save_weights(model_path + "bigan_generator")
        self.encoder.save_weights(model_path + "encoder")
        self.discriminator.save_weights(model_path + "discriminator")
        self.generator.save_weights(model_path + "generator")

    def load_model(self, model_path="models/savedmodels/bigan/"):
        if (os.path.exists(model_path)):
            logging.info(">> Loading saved model weights")
            self.bigan_generator.load_weights(model_path + "bigan_generator")
            self.encoder.load_weights(model_path + "encoder")
            self.discriminator.load_weights(model_path + "discriminator")
            self.generator.load_weights(model_path + "generator")
