
# * @license
# * Copyright 2019
# * Conttributors: Victor Dibia
# * Deep Ad
# * Licensed under the MIT License(the "License")


import six.moves.urllib as urllib
import pandas as pd
import os
import json
import zipfile
import gzip
import shutil

from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


train_data_url_10k = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
test_data_url = "http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz"
data_file_path_10k = "data/kdd10k.gz"

data_base_path = "data/kdd/"


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def extract_file(file_path):
    if not os.path.exists(data_base_path + file_path):
        print("> Extracting files ", file_path)
        with gzip.open(file_path, 'r') as f_in, open(file_path.split(".")[0], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            print("> Extraction complete ", file_path)
            os.remove(file_path)


def download_file(file_url, file_type):
    opener = urllib.request.URLopener()
    if not os.path.exists(file_url):
        print("> Downloading kdd data ", file_type)
        opener.retrieve(file_url, data_base_path + file_type)
        print("> download complete", file_type)
    extract_file(data_base_path + file_type)


def download_files():
    download_file(train_data_url_10k, "train.gz")
    download_file(test_data_url, "test.gz")


keep_cols_orig = ['srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                  'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                  'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                  'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'target']

keep_cols_no_corr = ['srv_count'
                     'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                     'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                     'dst_host_rerror_rate', 'target']


def get_kdd_data(filename='kddcup.data_10_percent', keep_cols=keep_cols_orig):

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
    kdd99 = pd.read_csv(os.path.join(data_dir, filename),
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
    #print("data processing ..")
    # print(kdd99.info())
    # print(kdd99.shape)
    # print(kdd99.head())

#     print(kdd99['labels'].value_counts())
    # print(kdd99['protocol_type'].value_counts())
    # print(kdd99['service'].value_counts())
    # print(kdd99['flag'].value_counts())

    attack_types = pd.read_csv(os.path.join(data_dir, 'training_attack_types'), sep="\s+", header=None,
                               names=['labels', 'attack'], dtype={'labels': str, 'attack': str})
    # print(attack_types.shape)
    # print(attack_types.head())

    kdd99 = pd.merge(kdd99, attack_types, on=['labels'], how='left')
    kdd99['attack'].fillna('normal', inplace=True)
    # print(kdd99.shape)
    # print(kdd99.head())
#     print(kdd99['attack'].value_counts())

    # Only 4 types of attacks are seen in the training set -> ['dos', 'probe', 'r2l', 'u2r']

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

    #print("kdd99 obs, columns: ", kdd99.shape)
    # print(kdd99.info())
    #print("kdd['target'] distribution ")
    # print(kdd99['target'].value_counts())

    return kdd99


def scale_and_expand(df):
    # min max (0,1) scaling input to fit sigmoid function range predicted by model
    df = df.drop(columns='target')
#       df = np.asarray(df)
    minmaxscaler = MinMaxScaler(feature_range=(0, 1))
    df = minmaxscaler.fit_transform(df)
    df = np.expand_dims(df, axis=2)
    df = np.expand_dims(df, axis=3)

    return df


def subset_kdd_data(dataset_type, keep_cols_arr, conf):

    if (dataset_type == "train"):
        kdddata = get_kdd_data(keep_cols=keep_cols_arr)
    else:
        kdddata = get_kdd_data(filename="corrected", keep_cols=keep_cols_arr)

    # Shuffle data
    kdddata = kdddata.sample(frac=1, random_state=200)
    inliers = kdddata[kdddata["target"] == 0]
    outliers = kdddata[kdddata["target"] == 1]

    #   if test, we split test into validation 20% and test 80%
    if (dataset_type == "test"):
        # remove 20% of test data and use for validation
        val_inliers = inliers.sample(frac=0.2, random_state=200)
        inliers.drop(val_inliers.index, inplace=True)

        val_outliers = outliers.sample(frac=0.2, random_state=200)
        outliers.drop(val_outliers.index, inplace=True)

        val_inliers.to_csv(data_base_path + conf + "/" + "val" +
                           "_inliers.csv", index=False)
        val_outliers.to_csv(data_base_path + conf + "/" + "val" +
                            "_outliers.csv", index=False)
        print("saving validation data")

        # if test, set max inliers
        if (conf == "1080"):
            max_normal_samples = 8000
            max_abnormal_samples = 2000
        elif (conf == "1090"):
            max_normal_samples = 9000
            max_abnormal_samples = 1000

        elif(conf == "all"):
            max_normal_samples = inliers.shape[0]
            max_abnormal_samples = outliers.shape[0]

        inliers = inliers.sample(max_normal_samples, random_state=250)
        outliers = outliers.sample(max_abnormal_samples, random_state=250)

    inliers.to_csv(data_base_path + conf + "/" + dataset_type +
                   "_inliers.csv", index=False)
    outliers.to_csv(data_base_path + conf + "/" + dataset_type +
                    "_outliers.csv", index=False)
    print("saving ", dataset_type, " data")

    return scale_and_expand(inliers), scale_and_expand(outliers)


def load_json_file(json_file_path):
    with open(json_file_path) as f:
        json_data = json.load(f)
        return json_data


def save_json_file(json_file_path, json_data):
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f)
