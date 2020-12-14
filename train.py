# ****************************************************************************
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
#  this code. If you do not have a written agreement with Cloudera nor *  with an authorized and properly licensed third party, you do not 
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY *  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED 
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO 
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND 
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, 
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS 
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE *  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES 
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF 
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ***************************************************************************

import argparse
from models.ae import AutoencoderModel
from models.pca import PCAModel
from models.ocsvm import SVMModel
from models.vae import VAEModel
from models.bigan import BiGANModel
from models.seq2seq import Seq2SeqModel
from utils import data_utils, eval_utils
import numpy as np


import logging
logging.basicConfig(level=logging.INFO)


def train_pca():
    num_features = 2
    pca = PCAModel()
    pca.train(in_train, in_test, num_features=num_features)

    inlier_scores = pca.compute_anomaly_score_unsupervised(in_test)
    outlier_scores = pca.compute_anomaly_score_unsupervised(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="pca", show_plot=False)
    print(metrics)
    return metrics


def train_svm():
    svm_kwargs = {}
    svm_kwargs["kernel"] = "rbf"
    svm_kwargs["gamma"] = 0.5
    svm_kwargs["outlier_frac"] = 0.0001
    svm = SVMModel(**svm_kwargs)
    svm.train(in_train, in_test)

    inlier_scores = svm.compute_anomaly_score(in_test)
    outlier_scores = svm.compute_anomaly_score(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="ocsvm", show_plot=False)
    print(metrics)
    return metrics


def train_autoencoder():
    # Instantiate and Train Autoencoder
    ae_kwargs = {}
    ae_kwargs["latent_dim"] = 2
    ae_kwargs["hidden_dim"] = [15, 7]
    ae_kwargs["epochs"] = 14
    ae_kwargs["batch_size"] = 128
    # ae_kwargs["model_path"] = ae_model_path
    ae = AutoencoderModel(in_train.shape[1], **ae_kwargs)
    ae.train(in_train, in_test)
    ae.save_model()

    inlier_scores = ae.compute_anomaly_score(in_test)
    outlier_scores = ae.compute_anomaly_score(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="ae", show_plot=False)
    print(metrics)
    return metrics


def train_vae():
    # Instantiate and Train Autoencoder
    vae_kwargs = {}
    vae_kwargs["latent_dim"] = 2
    vae_kwargs["hidden_dim"] = [15, 7]
    vae_kwargs["epochs"] = 8
    vae_kwargs["batch_size"] = 128
    # vae_kwargs["model_path"] = ae_model_path
    vae = VAEModel(in_train.shape[1], **vae_kwargs)
    vae.train(in_train, in_test)
    vae.save_model()

    inlier_scores = vae.compute_anomaly_score(in_test)
    outlier_scores = vae.compute_anomaly_score(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="vae", show_plot=False)
    print(metrics)
    return metrics


def train_bigan():
    bigan_kwargs = {}
    bigan_kwargs["latent_dim"] = 2
    bigan_kwargs["dense_dim"] = 128
    bigan_kwargs["epochs"] = 15
    bigan_kwargs["batch_size"] = 256
    bigan_kwargs["learning_rate"] = 0.01
    input_shape = (in_train.shape[1], )
    bigan = BiGANModel(input_shape, **bigan_kwargs)
    bigan.train(in_train, in_test)
    bigan.save_model()
    inlier_scores = bigan.compute_anomaly_score(in_test)
    outlier_scores = bigan.compute_anomaly_score(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="bigan", show_plot=False)
    print(metrics)
    return metrics


def train_seq2seq():
    # seq2seq models require a dim 3 input matrix (rows, timesteps, num_features )
    in_train_x, in_test_x, out_test_x = np.expand_dims(
        in_train, axis=2), np.expand_dims(in_test, axis=2),  np.expand_dims(out_test, axis=2)

    seq2seq_kwargs = {}
    seq2seq_kwargs["encoder_dim"] = [10]
    seq2seq_kwargs["decoder_dim"] = [20]
    seq2seq_kwargs["epochs"] = 40
    seq2seq_kwargs["batch_size"] = 256
    seq2seq_kwargs["learning_rate"] = 0.01
    n_features = 1  # single value per feature
    seq2seq = Seq2SeqModel(n_features, **seq2seq_kwargs)
    seq2seq.train(in_train_x, in_test_x)
    seq2seq.save_model()

    # seq2seq.load_model()
    inlier_scores = seq2seq.compute_anomaly_score(
        in_test_x[np.random.randint(100, size=400), :])
    outlier_scores = seq2seq.compute_anomaly_score(
        out_test_x[np.random.randint(100, size=80), :])

    print(inlier_scores[:5])
    print(outlier_scores[:5])
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="seq2seq", show_plot=False)
    print(metrics)
    return metrics


def train_all():
    train_autoencoder()
    train_pca()
    train_vae()
    train_svm()
    train_bigan()
    train_seq2seq()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process train parameters')
    parser.add_argument('-m', '--model', dest='model', type=str,
                        choices=["ae", "vae", "seq2seq", "gan", "all"],
                        help='model type to train', default="ae")
    # parser.add_argument('--epochs', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    args, unknown = parser.parse_known_args()

    test_data_partition = "8020"
    in_train, out_train, scaler, _ = data_utils.load_kdd(
        data_path="data/kdd/", dataset_type="train", partition=test_data_partition)
    in_test, out_test, _, _ = data_utils.load_kdd(
        data_path="data/kdd/", dataset_type="test", partition=test_data_partition, scaler=scaler)

    train_autoencoder()
