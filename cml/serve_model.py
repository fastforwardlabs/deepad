#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments for Deep Learning for Anomaly Detection https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

from models.ae import AutoencoderModel
import pandas as pd
import numpy as np
from utils.eval_utils import load_metrics

ae_kwargs = {}
in_shape = 18
ae_kwargs["latent_dim"] = 2
ae_kwargs["hidden_dim"] = [15, 7]
ae_kwargs["epochs"] = 14
ae_kwargs["batch_size"] = 128
ae = AutoencoderModel(in_shape, **ae_kwargs)
ae.load_model()

metrics = load_metrics("metrics/" + ae.model_name + "/metrics.json")


def predict(data):
    scores = ae.compute_anomaly_score(data)
    preds = (scores > metrics["threshold"])
    return list(scores), list(preds)


data = np.random.rand(3, in_shape)
print(predict(data))
