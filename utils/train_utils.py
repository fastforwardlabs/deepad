#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Code samples adapted from https://keras.io/examples/variational_autoencoder/
# Licensed under the MIT License (the "License");
# =============================================================================
#


import time
import tensorflow as tf


class TimeHistory(tf.keras.callbacks.Callback):
    """[Keras callback for keras model ]

    Arguments:
        keras {[type]} -- [description]
    """

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# time_callback = TimeHistory()
# time_callback.times
