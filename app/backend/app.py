
#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

from utils import data_utils
import numpy as np
from flask import Flask, jsonify
from models.ae import AutoencoderModel
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

test_data_partition = "8020"
in_train, out_train, scaler, col_names = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="train", partition=test_data_partition)
in_test, out_test, _, _ = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="test", partition=test_data_partition, scaler=scaler)


def load_autoencoder():
    ae_kwargs = {}
    ae_kwargs["latent_dim"] = 2
    ae_kwargs["hidden_dim"] = [15, 7]
    ae_kwargs["epochs"] = 14
    ae_kwargs["batch_size"] = 128
    ae = AutoencoderModel(in_train.shape[1], **ae_kwargs)
    ae.load_model()
    return ae


ae = load_autoencoder()


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/data')
def data():
    inlier_size = 8
    outlier_size = 2
    response = {"inliers": in_test[np.random.randint(5, size=inlier_size), :].tolist(),
                "outliers":  out_test[np.random.randint(5, size=outlier_size), :].tolist(),
                "colnames": list(col_names)
                }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
