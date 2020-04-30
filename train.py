#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments for Deep Learning for Anomaly Detection https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

from mlflow import log_metric, log_param, log_artifact
import mlflow
import argparse
import logging

from models.ae import Autoencoder
from utils import data_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


parser = argparse.ArgumentParser(description='Process train parameters')
parser.add_argument('--model', dest='model', type=str,
                    choices=["ae", "vae", "seq2seq", "gan", "all"],
                    help='model type to train')
# parser.add_argument('--epochs', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')


args = parser.parse_args()


data_partition = "all"
in_train, out_train, scaler = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="train", partition=data_partition)
in_test, out_test, _ = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="test", partition=data_partition, scaler=scaler)

# Instantiate and Train Autoencoder
# ae = Autoencoder(in_train.shape[1])
# ae.train(in_train, in_test)

# print(args)
