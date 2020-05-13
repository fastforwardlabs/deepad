
#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

import cdsw
import os
import train


metrics = train.train_autoencoder()
cdsw.track_metric("test_accuracy", round(metrics["acc"], 2))
cdsw.track_metric("test_roc", round(metrics["roc"], 2))
cdsw.track_metric("test_precision", round(metrics["precision"], 2))
cdsw.track_metric("test_recall", round(metrics["recall"], 2))
cdsw.track_metric("threshold", round(metrics["threshold"], 2))
