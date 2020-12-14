#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments for Deep Learning for Anomaly Detection https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

from models.ae import AutoencoderModel
import numpy as np
from utils.eval_utils import load_metrics
import json

ae_kwargs = {}
in_shape = 18
ae_kwargs["latent_dim"] = 2
ae_kwargs["hidden_dim"] = [15, 7]
ae_kwargs["epochs"] = 14
ae_kwargs["batch_size"] = 128
ae = AutoencoderModel(in_shape, **ae_kwargs)
ae.load_model()

metrics = load_metrics("metrics/" + ae.model_name + "/metrics.json")


def predict(args):
    data = np.array(args)
    if data.shape[1] != 18:
        return {"status": "input data should have 18 features"}
    scores = ae.compute_anomaly_score(data)
    predictions = (scores > metrics["threshold"])
    result = {"scores": scores.tolist(),
              "predictions": list(predictions.tostring())
              }

    return json.dumps(result)
