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

# set random seed for reproducibility
from sklearn.decomposition import PCA
import numpy as np
import random
from scipy.spatial.distance import cdist

np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class PCAModel():
    def __init__(self):
        self.name = "pca"

    def train(self, in_train, in_val, num_features=2):
        num_features = min(num_features, in_train.shape[1])
        self.model = PCA(n_components=num_features)
        self.model .fit(in_train)
        print("Explained variation per principal component: ",
              np.sum(self.model.explained_variance_ratio_))

    def compute_anomaly_score(self, df):
        low_dim = self.model.transform(df)
        preds = self.model.inverse_transform(low_dim)
        mse = np.mean(np.power(df - preds, 2), axis=1)
        return mse

    def compute_anomaly_score_unsupervised(self, df):
        """Compute anomaly score as distance from learned PCA components

        Arguments:
            df {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        anomaly_scores = np.sum(
            cdist(df, self.model.components_),
            axis=1).ravel()
        return anomaly_scores
