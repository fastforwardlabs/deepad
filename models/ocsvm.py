#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#


import random
import numpy as np
from sklearn import svm


np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class SVMModel():

    def __init__(self, kernel="rbf", outlier_frac=0.0001, gamma=0.5):
        self.model = svm.OneClassSVM(
            nu=outlier_frac, kernel=kernel, gamma=gamma)

    def train(self, in_train, in_val):
        self.model.fit(in_train)

    def compute_anomaly_score(self, df):
        preds = self.model.decision_function(df)
        preds = preds * -1

        return preds
