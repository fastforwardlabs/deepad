#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments for Deep Learning for Anomaly Detection https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

from flask import Flask, send_from_directory, request, jsonify
import logging
import os

from flask import Flask
from flask_cors import CORS


app = Flask(__name__, static_url_path='')
CORS(app)


@app.route('/')
def home():
    return "bingo"


if __name__ == "__main__":
    port = 3002  # int(os.environ['CDSW_APP_PORT'])
    app.run(host='127.0.0.1', port=port, debug=True)
