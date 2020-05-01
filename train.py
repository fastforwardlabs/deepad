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
from models.ae import Autoencoder
from utils import data_utils, eval_utils


import logging
logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser(description='Process train parameters')
parser.add_argument('--model', dest='model', type=str,
                    choices=["ae", "vae", "seq2seq", "gan", "all"],
                    help='model type to train')
# parser.add_argument('--epochs', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')


args = parser.parse_args()


test_data_partition = "8020"
in_train, out_train, scaler = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="train", partition=test_data_partition)
in_test, out_test, _ = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="test", partition=test_data_partition, scaler=scaler)

# Instantiate and Train Autoencoder
ae_model_path = "models/savedmodels/ae/ae"
ae_kwargs = {}
ae_kwargs["latent_dim"] = 2
ae_kwargs["hidden_dim"] = [15, 7]
ae_kwargs["epochs"] = 14
ae_kwargs["batch_size"] = 128
ae_kwargs["model_path"] = ae_model_path
ae = Autoencoder(in_train.shape[1], **ae_kwargs)
# ae.train(in_train, in_test)
# ae.save_model(ae_model_path)

inlier_scores = ae.compute_anomaly_score(in_test)
outlier_scores = ae.compute_anomaly_score(out_test)
print(inlier_scores)
print(outlier_scores)
metrics = eval_utils.evaluate_model(
    inlier_scores, outlier_scores, model_name="ae")
print(metrics)
