
import numpy as np
import pandas as pd
from utils import data_utils

test_data_partition = "8020"
in_train, out_train, scaler, col_names = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="train", partition=test_data_partition)
in_test, out_test, _, _ = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="test", partition=test_data_partition, scaler=scaler)


inlier_size = 2
outlier_size = 2


def data_to_json(data, label):
    data = pd.DataFrame(data, columns=list(col_names))
    data["label"] = label
    return data


in_liers = data_to_json(
    in_test[np.random.randint(5, size=inlier_size), :], 0)
out_liers = data_to_json(
    out_test[np.random.randint(5, size=outlier_size), :], 1)

response = pd.concat([in_liers, out_liers], axis=0)
print(response.to_json(orient="records"))
