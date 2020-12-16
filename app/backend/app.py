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

import argparse
from utils import data_utils
from utils.eval_utils import load_metrics
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
# from flask_cors import CORS, cross_origin
from models.ae import AutoencoderModel
import logging
import os


logging.basicConfig(level=logging.INFO)


def load_autoencoder():
    ae_kwargs = {}
    ae_kwargs["latent_dim"] = 2
    ae_kwargs["hidden_dim"] = [15, 7]
    ae_kwargs["epochs"] = 14
    ae_kwargs["batch_size"] = 128
    ae = AutoencoderModel(in_train.shape[1], **ae_kwargs)
    ae.load_model()
    metrics = load_metrics("metrics/" + ae.model_name + "/metrics.json")
    return ae, metrics


def data_to_json(data, label):
    data = pd.DataFrame(data, columns=list(col_names))
    data["label"] = label
    return data


# Point Flask to the front end directory

root_file_path = os.getcwd() + "/app/frontend"
print(root_file_path, os.getcwd())
# root_file_path = root_file_path.replace("backend", "frontend")
static_folder_root = os.path.join(root_file_path, "build")
print(static_folder_root)

app = Flask(__name__, static_url_path='',
            static_folder=static_folder_root, template_folder=static_folder_root)

# cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

test_data_partition = "8020"
in_train, out_train, scaler, col_names = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="train", partition=test_data_partition)
in_test, out_test, _, _ = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="test", partition=test_data_partition, scaler=scaler)

ae, metrics = load_autoencoder()


@app.route('/')
def hello():
    return render_template('index.html')

# @app.route('/build')
# def build():
#     return app.send_static_file('build/index.html')


@app.route('/data')
def data():

    data_size = request.args.get("n")
    data_size = 10 if data_size == None else int(data_size)
    inlier_size = int(0.8 * data_size)
    outlier_size = int(0.3 * data_size)
    in_liers = data_to_json(
        in_test[np.random.randint(5, size=inlier_size), :], 0)
    out_liers = data_to_json(
        out_test[np.random.randint(5, size=outlier_size), :], 1)

    response = pd.concat([in_liers, out_liers], axis=0)
    response = response.sample(frac=1)

    return response.to_json(orient="records")


@app.route('/colnames')
# @cross_origin()
def colnames():
    coldesc = [
        "Server Count",
        "Server Error Rate",
        "Server S Error Rate",
        "R Error Rate",
        "Server R Error Rate",
        "Same Server Error Rate",
        "Different Server Error Rate",
        "Server Different Host Rate",
        "Destination Host Count",
        "Destination Host Server Count",
        "Destination Host Same Server Rate",
        "Destination Host Different Server Rate",
        "Destination Host Same Source Port Rate",
        "Destination Host Server Different Host Rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate"
    ]
    response = {"colnames":  list(
        col_names), "coldesc": coldesc, "label": "label"}
    return jsonify(response)


def drop_cols(df, to_drop):
    for col in to_drop:
        if col in df.columns:
            df = df.drop(columns=col)
    return df


@app.route('/predict', methods=['GET', 'POST'])
# @cross_origin()
def predict():
    response = {}
    data, scores, predictions, ids = [], [], [], []
    if request.method == 'POST':
        data = request.get_json()["data"]
        data = pd.DataFrame(data)
        ids = data["id"].tolist()
        data = drop_cols(data, ["label", "id"])

        scores = ae.compute_anomaly_score(data)
        predictions = (scores > metrics["threshold"]) * 1

    response = {"scores": scores.tolist(), "predictions": predictions.tolist(),
                "threshold": metrics["threshold"], "ids": ids}
    return jsonify(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Application parameters')
    parser.add_argument('-p', '--port', dest='port', type=int,
                        help='port to run model', default=os.environ.get("CDSW_READONLY_PORT"))

    args, unknown = parser.parse_known_args()
    port = args.port
    app.run(port=port)
