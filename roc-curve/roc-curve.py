import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    desc_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_indices]
    y_true = y_true[desc_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]

    return fpr, tpr, thresholds