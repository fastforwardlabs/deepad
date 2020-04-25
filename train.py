#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments for Deep Learning for Anomaly Detection https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#


import argparse
import mlflow
from mlflow import log_metric, log_param, log_artifact

parser = argparse.ArgumentParser(description='Process train parameters')
parser.add_argument('--model', dest='model', type=str,
                    choices=["ae", "vae", "seq2seq", "gan", "all"],
                    help='model type to train')
# parser.add_argument('--epochs', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args)
