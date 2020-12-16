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

"""[Download KDD Dataset, clean and parse into train/val/test sets]

Returns:
    [type] -- [description]
"""


import pandas as pd
import os
import json
import logging
import urllib.request
import gzip
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

kdd_train_url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
kdd_test_url = 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
attack_types_url = "http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types"


def download_gzip(url, dest_path):
    filehandle, _ = urllib.request.urlretrieve(url)
    f = gzip.open(filehandle, 'rb')
    file_content = f.read()
    file_content = file_content.decode('utf-8')
    f_out = open(dest_path, 'w+')
    f_out.write(file_content)
    f.close()
    f_out.close()
    logging.debug(" >> KDD Dataset download completed.")


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_file(url, dest_path):
    f, _ = urllib.request.urlretrieve(url)
    f = open(f, "rb")
    file_content = f.read()
    file_content = file_content.decode('utf-8')
    f_out = open(dest_path, 'w+')
    f_out.write(file_content)
    f.close()
    f_out.close()


keep_cols_orig = ['srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                  'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                  'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                  'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'target']

keep_cols_no_corr = ['srv_count'
                     'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                     'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                     'dst_host_rerror_rate', 'target']


def get_kdd_data(data_dir="data/kdd/", filename="data/kdd/train.csv", keep_cols=keep_cols_orig):

    data_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                    'num_failed_logins', 'logged_in', 'num_compromised',
                    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate',
                    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                    'dst_host_srv_count', 'dst_host_same_srv_rate',
                    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                    'dst_host_srv_rerror_rate', 'labels']
    kdd99 = pd.read_csv(filename,
                        header=None, names=data_columns)
    dtypes = {
        'duration': 'int',
        'protocol_type': 'str',
        'service': 'str',
        'flag': 'str',
        'src_bytes': 'int',
        'dst_bytes': 'int',
        'land': 'int',
        'wrong_fragment': 'int',
        'urgent': 'int',
        'hot': 'int',
        'num_failed_logins': 'int',
        'logged_in': 'int',
        'num_compromised': 'int',
        'root_shell': 'int',
        'su_attempted': 'int',
        'num_root': 'int',
        'num_file_creations': 'int',
        'num_shells': 'int',
        'num_access_files': 'int',
        'num_outbound_cmds': 'int',
        'is_host_login': 'int',
        'is_guest_login': 'int',
        'count': 'int',
        'srv_count': 'int',
        'serror_rate': 'float',
        'srv_serror_rate': 'float',
        'rerror_rate': 'float',
        'srv_rerror_rate': 'float',
        'same_srv_rate': 'float',
        'diff_srv_rate': 'float',
        'srv_diff_host_rate': 'float',
        'dst_host_count': 'int',
        'dst_host_srv_count': 'int',
        'dst_host_same_srv_rate': 'float',
        'dst_host_diff_srv_rate': 'float',
        'dst_host_same_src_port_rate': 'float',
        'dst_host_srv_diff_host_rate': 'float',
        'dst_host_serror_rate': 'float',
        'dst_host_srv_serror_rate': 'float',
        'dst_host_rerror_rate': 'float',
        'dst_host_srv_rerror_rate': 'float',
        'labels': 'str'
    }

    for c in kdd99.columns:
        kdd99[c] = kdd99[c].astype(dtypes[c])

    kdd99['labels'] = kdd99['labels'].map(lambda x: str(x)[:-1])

    attack_types = pd.read_csv(os.path.join(data_dir, 'training_attack_types.txt'), sep="\s+", header=None,
                               names=['labels', 'attack'], dtype={'labels': str, 'attack': str})

    kdd99 = pd.merge(kdd99, attack_types, on=['labels'], how='left')
    kdd99['attack'].fillna('normal', inplace=True)
    kdd99['target'] = 0
    for t in ['dos', 'probe', 'r2l', 'u2r']:
        kdd99['target'][kdd99['attack'] == t] = 1

    # define columns to be dropped
    drop_cols = []
    for col in kdd99.columns.values:
        if col not in keep_cols:
            drop_cols.append(col)

    if drop_cols != []:
        kdd99.drop(columns=drop_cols, inplace=True)
    return kdd99


def subset_kdd_data(dataset_path, dataset_type, keep_cols_arr, conf):

    mkdir(dataset_path + conf)
    if (dataset_type == "train"):
        kdddata = get_kdd_data(filename=dataset_path +
                               "train.csv", keep_cols=keep_cols_arr)
    else:
        kdddata = get_kdd_data(filename=dataset_path +
                               "test.csv", keep_cols=keep_cols_arr)

    # Shuffle data
    kdddata = kdddata.sample(frac=1, random_state=200)
    inliers = kdddata[kdddata["target"] == 0]
    outliers = kdddata[kdddata["target"] == 1]

    if (dataset_type == "test"):
        # remove 20% of test data and use for validation
        val_inliers = inliers.sample(frac=0.2, random_state=200)
        inliers.drop(val_inliers.index, inplace=True)

        val_outliers = outliers.sample(frac=0.2, random_state=200)
        outliers.drop(val_outliers.index, inplace=True)

        # save validation set
        val_inliers.to_csv(dataset_path + conf + "/" +
                           "val" + "_inliers.csv", index=False)
        val_outliers.to_csv(dataset_path + conf + "/" +
                            "val" + "_outliers.csv", index=False)
        logging.info("saving validation data")

        # if test, set max inliers
        if (conf == "8020"):
            max_normal_samples = 8000
            max_abnormal_samples = 2000
        elif (conf == "9010"):
            max_normal_samples = 9000
            max_abnormal_samples = 1000
        elif (conf == "5050"):
            max_normal_samples = 5000
            max_abnormal_samples = 5000
        elif(conf == "all"):
            max_normal_samples = inliers.shape[0]
            max_abnormal_samples = outliers.shape[0]

        inliers = inliers.sample(max_normal_samples, random_state=250)
        outliers = outliers.sample(max_abnormal_samples, random_state=250)

    inliers.to_csv(dataset_path + conf + "/" +
                   dataset_type + "_inliers.csv", index=False)
    outliers.to_csv(dataset_path + conf + "/" +
                    dataset_type + "_outliers.csv", index=False)
    logging.info(">> saving " + dataset_type + " data")

    return (inliers), (outliers)


def load_json_file(json_file_path):
    with open(json_file_path) as f:
        json_data = json.load(f)
        return json_data


def save_json_file(json_file_path, json_data):
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f)


def generate_dataset():

    logging.debug(" >> Downloading KDD 99 dataset")

    mkdir("data/kdd")
    download_gzip(kdd_train_url, "data/kdd/train.csv")
    download_gzip(kdd_test_url, "data/kdd/test.csv")
    download_file(attack_types_url, "data/kdd/training_attack_types.txt")

    configs = ["all", "8020", "9010"]

    for conf in configs:
        in_train, out_train = subset_kdd_data(
            "data/kdd/", "train", keep_cols_orig, conf)
        in_test, out_test = subset_kdd_data(
            "data/kdd/", "test", keep_cols_orig, conf)

#   print("Train", conf, in_train.shape, out_train.shape, "Test", in_test.shape, out_test.shape )
