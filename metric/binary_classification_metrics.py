import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision
from torchmetrics.classification import BinaryF1Score
import numpy as np
from sklearn import metrics as sklearn_metrics
from metric.es import es_score


def reverse_los(y, los_info):
    return y * los_info["los_std"] + los_info["los_mean"]


def minpse(preds, labels):
    precisions, recalls, thresholds = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score


def get_binary_metrics(preds, labels, y_true_los=None, threshold=None):
    accuracy = Accuracy(task="binary", threshold=0.5)
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")
    f1 = BinaryF1Score()

    # convert labels type to int
    labels = labels.type(torch.int)
    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    f1(preds, labels)

    minpse_score = minpse(preds, labels)

    if threshold is None:
        # return a dictionary
        return {
            "accuracy": accuracy.compute().item(),
            "auroc": auroc.compute().item(),
            "auprc": auprc.compute().item(),
            "f1": f1.compute().item(),
            "minpse": minpse_score,
        }
    else:
        es = es_score(labels, y_true_los, preds, threshold)
        return {
            "accuracy": accuracy.compute().item(),
            "auroc": auroc.compute().item(),
            "auprc": auprc.compute().item(),
            "f1": f1.compute().item(),
            "minpse": minpse_score,
            "es": es['es'],
        }
