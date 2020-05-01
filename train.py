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
from models.ae import AutoencoderModel
from models.pca import PCAModel
from models.ocsvm import SVMModel
from models.vae import VAEModel
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


def train_autoencoder():
    # Instantiate and Train Autoencoder
    ae_model_path = "models/savedmodels/ae/ae"
    ae_kwargs = {}
    ae_kwargs["latent_dim"] = 2
    ae_kwargs["hidden_dim"] = [15, 7]
    ae_kwargs["epochs"] = 14
    ae_kwargs["batch_size"] = 128
    # ae_kwargs["model_path"] = ae_model_path
    ae = AutoencoderModel(in_train.shape[1], **ae_kwargs)
    ae.train(in_train, in_test)
    ae.save_model(ae_model_path)

    inlier_scores = ae.compute_anomaly_score(in_test)
    outlier_scores = ae.compute_anomaly_score(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="ae")
    print(metrics)


def train_pca():
    num_features = 2
    pca = PCAModel()
    pca.train(in_train, in_test, num_features=num_features)

    inlier_scores = pca.compute_anomaly_score_unsupervised(in_test)
    outlier_scores = pca.compute_anomaly_score_unsupervised(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="pca")
    print(metrics)


def train_svm():
    svm_kwargs = {}
    svm_kwargs["kernel"] = "rbf"
    svm_kwargs["gamma"] = 0.5
    svm_kwargs["outlier_frac"] = 0.0001
    svm = SVMModel(**svm_kwargs)
    svm.train(in_train, in_train)

    inlier_scores = svm.compute_anomaly_score(in_test)
    outlier_scores = svm.compute_anomaly_score(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="ocsvm")
    print(metrics)


def train_vae():
    # Instantiate and Train Autoencoder
    vae_model_path = "models/savedmodels/vae/vae"
    vae_kwargs = {}
    vae_kwargs["latent_dim"] = 2
    vae_kwargs["hidden_dim"] = [15, 7]
    vae_kwargs["epochs"] = 8
    vae_kwargs["batch_size"] = 128
    # vae_kwargs["model_path"] = ae_model_path
    vae = VAEModel(in_train.shape[1], **vae_kwargs)
    vae.train(in_train, in_test)
    # # vae.save_model(ae_model_path)

    inlier_scores = vae.compute_anomaly_score(in_test)
    outlier_scores = vae.compute_anomaly_score(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="vae")
    # print(metrics)


# train_autoencoder()
# train_pca()
train_vae()


def train_all():
    train_autoencoder()
    train_pca()
    train_vae()
    train_svm()
