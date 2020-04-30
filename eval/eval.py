
#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments detecting Anomalies with Deep Neural Networks https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, fbeta_score, roc_curve, auc, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import logging

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


def plot_anomaly_histogram(loss_normal, loss_anomalous, title="anomaly", epoch=2, threshold=0.5):
    plt.figure()
    ndf = pd.DataFrame(data=loss_normal, columns=["score"])
    adf = pd.DataFrame(data=loss_anomalous, columns=["score"])

    plt.hist(ndf["score"])
    plt.hist(adf["score"])
    plt.legend(["Normal Data", "Anomalous Data"])
    plt.title(title + " Epoch " + str(epoch) +
              " | Threshold: " + str(threshold))
    plt.axvline(threshold, color="r", linestyle="dashed")
    plt.savefig("images/histogram/%d.png" % epoch)

    plt.rcParams.update(plot_params)
    if(not is_training):
        plt.show()
    plt.close()

    # save anomaly scores
    anomalyscore_holder.append({"epoch": epoch, "scores": {
                               "inlier": list(loss_normal), "outlier": list(loss_anomalous)}})


def compute_accuracy(threshold, loss, y, dataset_name, show_roc=False):
    y_pred = np.array([1 if e > threshold else 0 for e in loss]).astype(int)
    acc_tot = accuracy_score(y, y_pred)
    prec_tot = precision_score(y, y_pred)
    rec_tot = recall_score(y, y_pred)
    f1_tot = f1_score(y, y_pred)
    f2_tot = fbeta_score(y, y_pred, beta=2)

    fpr, tpr, thresholds = roc_curve(y, loss)
    roc_auc = roc_auc_score(y, loss)

    if (show_roc):
        plt.figure()

        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        plt.rcParams.update(plot_params)

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


def get_scores_and_labels(outlier_score, inlier_score, epoch=2):
    zero_vec = np.zeros(len(inlier_score))
    one_vec = np.ones(len(outlier_score))

    all_scores = list(inlier_score) + list(outlier_score)
    all_labels = list(zero_vec) + list(one_vec)

    return all_scores, all_labels


def visualize_tested_metrics(metric_df, epoch):

    axis_font = 20
    labelpad = 20

    ax = metric_df[['acc', 'threshold']].plot(
        kind='line',  stacked=False, title=" Performance values at epoch  " + str(epoch), x="threshold", figsize=fig_size, legend=True)
    ax.set_xlabel("Thresholds",   labelpad=labelpad)
    ax.set_ylabel("Performance score",  labelpad=labelpad)

    plt.rcParams.update(plot_params)

    plt.savefig("images/accuracy/%d.png" % epoch)
    if(not is_training):
        plt.show()
    plt.close()


def plot_accuracy(outlier_score, inlier_score, epoch=2):
    all_scores, all_labels = get_scores_and_labels(
        outlier_score, inlier_score, epoch)
    all_thresholds = list(set(all_scores))
    all_thresholds.sort()

    logging.debug(str(len(all_thresholds)) + "unique thresholds")
    logging.debug("Testing all thresholds to find best accuracy ...")
    metric_holder = []

    for threshold in all_thresholds:
        metrics = test_threshold(threshold, all_scores, all_labels)
        metric_holder.append(metrics)

    logging.debug("Threshold testing complete ...")

    metric_df = pd.DataFrame(metric_holder)
    max_acc = metric_df.sort_values(
        by='acc', ascending=False, na_position='first').iloc[0]
    logging.debug("Best accuracy is .. " + str(dict(max_acc)))
    visualize_tested_metrics(metric_df, epoch)

    # show ROC for best accuracy model
    best_metrics = compute_accuracy(
        dict(max_acc)["threshold"], all_scores, all_labels, "test data", show_roc=True)
    # save maximum accuracy for this iteration
    accuracyscore_holder.append({"epoch": epoch, "accuracy": dict(max_acc)})

    return best_metrics


def save_classification(loss, threshold, save_path):
    y_pred = [1 if e > threshold else 0 for e in loss]
    scores = loss
    class_vals = y_pred
    result = pd.DataFrame(
        {"scores": scores, "class": class_vals, "threshold": threshold})
    result = result.to_json(save_path, orient='records', lines=True)


def get_classification(loss, threshold):
    y_pred = [1 if e > threshold else 0 for e in loss]
    return list(y_pred)
