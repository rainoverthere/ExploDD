# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
This module contains functions to compute and display classification KPIs
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from torchmetrics.functional import confusion_matrix, precision_recall_curve, roc, auroc
from metavision_ml.data import HDF5Iterator
from metavision_ml.preprocessing.viz import viz_histo_filtered
import os
import cv2
import random
from statistics import mean, median, stdev


def unpack_metrics_dict(metrics_per_category, label):
    """
    Unpack the dense metrics

    Args:
        metrics_per_category (dict): the torchmetrics result calculated per category
        label (list): list of class labels

    """
    new_metrics = {}
    for k, v in metrics_per_category.items():
        if isinstance(v, torch.Tensor):
            if v.numel() > 1:
                for i, t in enumerate(v):
                    new_k = k + "_" + label[i]
                    new_v = t
                    new_metrics[new_k] = new_v
            else:
                new_metrics[k] = v
        else:
            raise TypeError("The value in the metrics should be a tensor array.")
    return new_metrics


def plot_cm(preds_all, labels_all, labels):
    """
    Plot confusion metrics & error map by masking the diagonal values

    Args:
       preds_all (torch.Tensor): predictions
       labels_all (torch.Tensor): GT
       labels (list): list of all class labels

    """
    n_classes = len(labels)
    cm = confusion_matrix(preds_all, labels_all, n_classes, normalize='true', threshold=0.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # plot confusion matrix
    sns.heatmap(cm.numpy(), annot=True, fmt='.2f', ax=ax1, cmap='Blues')
    ax1.set(
        xticks=np.arange(n_classes) + 0.5,
        yticks=np.arange(n_classes) + 0.5,
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True labels",
        xlabel="Predicted labels",
        title='Normalised Confusion Matrix')

    # plot error matrix
    norm_cm_filled = np.copy(cm)
    np.fill_diagonal(norm_cm_filled, 0)
    sns.heatmap(norm_cm_filled, annot=True, fmt='.2f', ax=ax2, cmap=plt.get_cmap('gray'))
    ax2.set(
        xticks=np.arange(n_classes) + 0.5,
        yticks=np.arange(n_classes) + 0.5,
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True labels",
        xlabel="Predicted labels",
        title='Error Matrix')
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(preds_all, labels_all, labels):
    """
    plot the PR-curve

    Args:
        preds_all (torch.Tensor): predictions
        labels_all (torch.Tensor): GT
        labels (list): list of all class labels
    """
    n_classes = len(labels)
    precision, recall, _ = precision_recall_curve(preds_all, labels_all, n_classes)
    n_rows = 2 ** ((n_classes.bit_length() - 1) // 2)
    n_cols = int(np.ceil(n_classes / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols)
    axes = axes.flatten()
    for i, (p, r) in enumerate(zip(precision, recall)):
        r, p = r.tolist(), p.tolist()
        axes[i].step(r, p, color='r', alpha=0.99, where='post')
        axes[i].fill_between(r, p, alpha=0.2, color='b', step='post')
        axes[i].set(
            xlabel='Recall',
            ylabel='Precision',
            xlim=[0.0, 1],
            ylim=[0.0, 1.05],
            title=labels[i],
        )
    plt.tight_layout()
    plt.show()


def plot_roc(preds_all, labels_all, labels):
    """
    plot roc curve with auc_roc score

    Args:
        preds_all (torch.Tensor): predictions
        labels_all (torch.Tensor): GT
        labels (list): list of all class labels

    """
    n_classes = len(labels)
    auc_roc = auroc(preds_all, labels_all, num_classes=n_classes, average=None)
    fpr, tpr, _ = roc(preds_all, labels_all, n_classes)
    assert n_classes == len(auc_roc) == len(fpr) == len(tpr), "the number of classes is not the same among the inputs"
    n_rows = 2 ** ((n_classes.bit_length() - 1) // 2)
    n_cols = int(np.ceil(n_classes / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols)
    axes = axes.flatten()
    for i, (auc, fp, tp, label) in enumerate(zip(auc_roc, fpr, tpr, labels)):
        fp, tp = fp.tolist(), tp.tolist()
        axes[i].step(fp, tp, color='r', alpha=0.99, where='post', label="AUC:{:.2f}".format(auc))
        axes[i].plot([0, 1], [0, 1], 'k--')
        axes[i].set(
            xlabel='False Positive Rate',
            ylabel='True Positive Rate (Recall)',
            xlim=[0.0, 1],
            ylim=[0.0, 1.05],
            title=label,
        )
        axes[i].legend()
        axes[i].grid()
    plt.tight_layout()
    plt.legend()
    plt.show()


def evaluate_preds_seq(preds_seq, labels_seq, res_per_recording, hparams, latency_seq):
    """
    Inspect the test result by plotting the recording image together with prediction sequence

    Args:
        preds_seq (list): nested list of prediction sequences
        labels_seq (list): nested list of labeling sequences
        res_per_recording (defaultdict): defaultdict of time stamp, prediction and label vectors per HDF5 file
        hparams (dict): hyperparameters
        latency_seq : list of time to prediction for each data sample

    """
    recording_ids = list(res_per_recording.keys())
    latency_arr = np.array(latency_seq)

    # Get the worst 10 prediction samples
    worst_10 = np.argsort(latency_arr)[::-1][:10]

    label_map = ['background'] + hparams.classes

    for rank, i in enumerate(worst_10):  # pick up the worst 10 samples to analyze
        T = range(len(preds_seq[i]))
        preprocessor = HDF5Iterator(os.path.join(hparams.dataset_path, 'test', recording_ids[i]),
                                    num_tbins=hparams['num_tbins'],
                                    device=torch.device('cpu'), height=hparams['height'], width=hparams['width'])
        preds = np.argmax(preds_seq[i].numpy(), -1)
        ts_step = 0

        # Visualize the worst 10 data samples
        WIN_NAME = "worst {}: {}".format(rank+1, recording_ids[i])
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        for tensor_th in preprocessor:
            for nr, tensor in enumerate(tensor_th.detach().cpu().numpy()):
                img = preprocessor.get_vis_func()(
                    tensor) if hparams["preprocess"] != 'histo' else viz_histo_filtered(tensor)

                label = label_map[labels_seq[i][ts_step]]
                cv2.putText(
                    img, "GT: {}".format(label),
                    (10, img.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                cv2.putText(
                    img, "pred: {}".format(label_map[preds[ts_step]]),
                    (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 204))
                cv2.imshow(WIN_NAME, img[..., ::-1])

                key = cv2.waitKey(50)
                if key == 27 or key == ord("q"):
                    break
                if ts_step < len(preds)-1:
                    ts_step += 1

        cv2.destroyWindow(WIN_NAME)


def get_1st_nonzeros(tensor):
    """
    Get the 1st nonzero item along the last axis of the tensor
    If tensor only contains zeros, get the last item index
    Args:
        tensor (torch.Tensor): input tensor
    """
    reversed_order = torch.arange(tensor.shape[-1], 0, -1, device=tensor.device)
    ord_index = tensor * reversed_order
    res = torch.argmax(ord_index, -1)
    return res


def calculate_time_to_prediction(pred_seq, label_seq, delta_t=10000):
    """
    calculate KPI: time to 1st correct prediction
    & draw histogram of the time to prediction statistics

    Args:
        pred_seq (list): list of prediction tensors for each data sample
        labels_all (list): list of label tensors for each data sample
        delta_t (int): time interval of data sample

    """
    assert len(pred_seq) == len(label_seq)
    tot_latency = []
    for i in range(len(pred_seq)):
        preds = torch.argmax(pred_seq[i], -1)
        ts_gt_1st_nonzero = get_1st_nonzeros(label_seq[i])
        mask = (preds > 0) & (label_seq[i] == preds)
        ts_pred_1st_nonzero = get_1st_nonzeros(mask)
        no_frames_late = ts_pred_1st_nonzero - ts_gt_1st_nonzero
        tot_latency.append(no_frames_late.item())

    median = mean(tot_latency)
    std = stdev(tot_latency)

    plt.hist(tot_latency, density=True)
    plt.title("Histogram of Time to Prediction")
    plt.xlabel("Time to prediction (x{} ms)".format(delta_t//1000))
    plt.ylabel("Probability")
    plt.axvline(median, color='k', linestyle='dashed', linewidth=1)
    plt.text(0.5*max(tot_latency), 0.3, 'median={:.2f}, std={:.2f}'.format(median, std))
    plt.show()

    t2p = {'t2p_median': median*delta_t//1000, 't2p_std': std*delta_t//1000}
    return tot_latency, t2p


PLOTS_DICT = {'cm': plot_cm,
              'pr': plot_precision_recall_curve,
              'roc': plot_roc
              }
