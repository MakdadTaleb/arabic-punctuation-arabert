from sklearn.metrics import confusion_matrix
import numpy as np


def compute_confusion_matrix(true_labels, pred_labels):
    return confusion_matrix(true_labels, pred_labels)


def normalize_confusion_matrix(cm):
    return cm.astype("float") / cm.sum(axis=1, keepdims=True)