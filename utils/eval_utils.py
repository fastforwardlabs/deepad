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

import matplotlib.pyplot as plt
from multiprocessing import Queue, Pool
import json
import os
import logging
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, fbeta_score, roc_curve, auc, roc_auc_score
import pandas as pd
import matplotlib
matplotlib.use('Agg')


fig_size = (9, 6)
fig_font = 15
plot_params = {'legend.fontsize': 'large',
               'figure.figsize': fig_size,
               'axes.labelsize': fig_font,
               'axes.titlesize': fig_font,
               'xtick.labelsize': fig_font*0.75,
               'ytick.labelsize': fig_font*0.75,
               'axes.titlepad': fig_font}
plt.rcParams.update(plot_params)


def plot_anomaly_histogram(inlier_score, outlier_score, title="Anomaly Score Histogram", threshold=0.5, model_name="_", show_plot=True):
    plt.figure()
    ndf = pd.DataFrame(data=inlier_score, columns=["score"])
    adf = pd.DataFrame(data=outlier_score, columns=["score"])

    plt.hist(ndf["score"])
    plt.hist(adf["score"])
    plt.legend(["Inlier Data", "Outlier Data"])
    plt.title(model_name.upper() + " | " + title +
              " | Threshold: " + str(threshold))
    plt.axvline(threshold, color="r", linestyle="dashed")
    plt.savefig("metrics/" + model_name + "/histogram.png")

    plt.rcParams.update(plot_params)
    if (show_plot):
        plt.show()
    else:
        plt.close()


def compute_accuracy(threshold, loss, y, dataset_name, show_plot=False, model_name="_"):
    y_pred = np.array([1 if e > threshold else 0 for e in loss]).astype(int)
    acc_tot = accuracy_score(y, y_pred)
    prec_tot = precision_score(y, y_pred)
    rec_tot = recall_score(y, y_pred)
    f1_tot = f1_score(y, y_pred)
    f2_tot = fbeta_score(y, y_pred, beta=2)

    fpr, tpr, thresholds = roc_curve(y, loss)
    roc_auc = roc_auc_score(y, loss)

    plt.figure()

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title(model_name.upper() + " | " +
              'Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.rcParams.update(plot_params)
    plt.savefig("metrics/" + model_name + "/roc.png")
    if (show_plot):
        plt.show()
    else:
        plt.close()

    metrics = {"acc": acc_tot,
               "precision": prec_tot,
               "recall": rec_tot,
               "f1": f1_tot,
               "f2": f2_tot,
               "roc": roc_auc,
               "threshold": round(threshold, 3)
               }
    return metrics


def test_threshold(threshold, loss, y):
    y_pred = np.array([1 if e > threshold else 0 for e in loss]).astype(int)
    acc_tot = accuracy_score(y, y_pred)
    metrics = {"acc": acc_tot,
               "threshold": round(threshold, 3)}
    return metrics


def get_scores_and_labels(outlier_score, inlier_score):
    zero_vec = np.zeros(len(inlier_score))
    one_vec = np.ones(len(outlier_score))

    all_scores = list(inlier_score) + list(outlier_score)
    all_labels = list(zero_vec) + list(one_vec)

    return all_scores, all_labels


def plot_metrics(best_metrics, model_name="_", show_plot=False):
    fig, ax = plt.subplots()
    metrics = best_metrics.copy()
    del metrics["threshold"]
    ax.barh(list(metrics.keys()), list(metrics.values()), color="blue")
    plt.title(model_name.upper() + " | " + ' Model Performance Metrics')
    plt.xlabel('', fontsize=14)
    plt.ylabel('Model Metrics', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.box(False)
    for i, v in enumerate(list(metrics.values())):
        ax.text(v + 0.01, i, str(round(v, 3)), color='blue', fontsize=15)

    plt.savefig("metrics/" + model_name + "/metrics.png")
    if (show_plot):
        plt.show()
    else:
        plt.close()

    # save metrics to json file
    with open("metrics/" + model_name + "/metrics.json", 'w') as outfile:
        json.dump(best_metrics, outfile)


def evaluate_model(inlier_score, outlier_score, model_name="_", show_plot=True):
    image_directory = "metrics/" + model_name
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    all_scores, all_labels = get_scores_and_labels(
        outlier_score, inlier_score)
    all_thresholds = list(set(all_scores))
    all_thresholds.sort()

    logging.info(str(len(all_thresholds)) + "unique thresholds")
    logging.info("Testing all thresholds to find best accuracy ...")
    metric_holder = []

    for threshold in all_thresholds:
        metrics = test_threshold(threshold, all_scores, all_labels)
        metric_holder.append(metrics)

    logging.info("Threshold testing complete ...")

    metric_df = pd.DataFrame(metric_holder)
    max_acc = metric_df.sort_values(
        by='acc', ascending=False, na_position='first').iloc[0]
    logging.info("Best accuracy is .. " + str(dict(max_acc)))
    # visualize_tested_metrics(metric_df, epoch)

    # show ROC for best accuracy model
    best_metrics = compute_accuracy(
        dict(max_acc)["threshold"], all_scores, all_labels, "test data", model_name=model_name, show_plot=show_plot)

    plot_anomaly_histogram(inlier_score, outlier_score,
                           threshold=best_metrics["threshold"], model_name=model_name, show_plot=show_plot)

    plot_metrics(best_metrics, show_plot=show_plot, model_name=model_name)

    return best_metrics


def save_metrics(loss, threshold, save_path):
    y_pred = [1 if e > threshold else 0 for e in loss]
    scores = loss
    class_vals = y_pred
    result = pd.DataFrame(
        {"scores": scores, "class": class_vals, "threshold": threshold})
    result = result.to_json(save_path, orient='records', lines=True)


def load_metrics(metric_path):
    with open(metric_path) as json_file:
        data = json.load(json_file)
        return data
