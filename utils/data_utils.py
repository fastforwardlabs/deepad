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

from data import kdd_data_gen as kdd
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging


def scale_data(df, scaler=None, drop_col="target", dim_size=2):
    df = df.drop(columns=drop_col)
    col_names = df.columns
    if (not scaler):
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(df)
    else:
        df = scaler.transform(df)

    # df = np.expand_dims(df, axis=dim_size)
    # df = np.expand_dims(df, axis=3)
    return df, scaler, col_names


def load_kdd(data_path="data/kdd", dataset_type="all", partition="all", scaler=None):

    inlier_data_path = os.path.join(
        data_path, partition, dataset_type + "_inliers.csv")
    outlier_data_path = os.path.join(
        data_path, partition, dataset_type + "_outliers.csv")

    if not os.path.exists(os.path.join(inlier_data_path)):
        logging.info(" >> Generating KDD dataset")
        kdd.generate_dataset()

    inliers = pd.read_csv(inlier_data_path)
    outliers = pd.read_csv(outlier_data_path)

    logging.info(" >> KDD dataset loaded")
    inliers, scaler, col_names = scale_data(inliers, scaler=scaler, dim_size=2)
    outliers, _, _ = scale_data(outliers, scaler=scaler, dim_size=2)
    return inliers, outliers, scaler, col_names
