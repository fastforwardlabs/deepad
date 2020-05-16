
#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

from utils import data_utils
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from models.ae import AutoencoderModel
import logging

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

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


def data_to_json(data, label):
    data = pd.DataFrame(data, columns=list(col_names))
    data["label"] = label
    return data


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/data')
def data():
    inlier_size = 10
    outlier_size = 5
    in_liers = data_to_json(
        in_test[np.random.randint(5, size=inlier_size), :], 0)
    out_liers = data_to_json(
        out_test[np.random.randint(5, size=outlier_size), :], 1)

    response = pd.concat([in_liers, out_liers], axis=0)
    response = response.sample(frac=1)

    return response.to_json(orient="records")


@app.route('/colnames')
@cross_origin()
def colnames():
    coldesc = [
        "Server Count",
        "Server Error Rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "Same Server Error Rate",
        "Different Serer Error Rate",
        "srv_diff_host_rate",
        "Destination Host Count",
        "Destination Host Server Count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate"
    ]
    response = {"colnames":  list(
        col_names), "coldesc": coldesc, "label": "label"}
    return jsonify(response)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = []
    if request.method == 'POST':
        data = request.form["data"]

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
