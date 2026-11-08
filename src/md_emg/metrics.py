from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    specificity = tn/(tn+fp) if (tn+fp) else np.nan
    sensitivity = tp/(tp+fn) if (tp+fn) else np.nan
    out = {
        "n": int(len(y_true)), "n0": int((y_true==0).sum()), "n1": int((y_true==1).sum()),
        "accuracy": accuracy_score(y_true,y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true,y_pred) if len(np.unique(y_true))==2 else np.nan,
        "f1_error": f1_score(y_true,y_pred,pos_label=1,zero_division=0),
        "precision_error": precision_score(y_true,y_pred,pos_label=1,zero_division=0),
        "sensitivity": sensitivity, "specificity": specificity,
        "mcc": matthews_corrcoef(y_true,y_pred) if len(np.unique(y_true))==2 else np.nan,
        "fpr": fp/(fp+tn) if (fp+tn) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_error_rate": float((y_pred==1).mean())
    }
    if score is not None and len(np.unique(y_true))==2:
        out["roc_auc"] = roc_auc_score(y_true, score)
        out["average_precision"] = average_precision_score(y_true, score)
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan
    return out
