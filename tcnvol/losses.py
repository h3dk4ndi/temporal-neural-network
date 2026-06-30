import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score


# ─────────────────────────────────────────────────────────────────────
# Loss and evaluation (no state — kept as functions)
# ─────────────────────────────────────────────────────────────────────

def bce_loss(
    p_hat:      np.ndarray,
    y:          np.ndarray,
    pos_weight: float = 1.0,
) -> tuple[float, np.ndarray]:
    """
    Weighted binary cross-entropy loss and gradient w.r.t. logits.

    L = -1/N Σ [ w+ y log(p̂) + (1-y) log(1-p̂) ]

    Parameters
    ----------
    p_hat      : np.ndarray  predicted probabilities, shape (N, 1)
    y          : np.ndarray  binary labels, shape (N,) or (N, 1)
    pos_weight : float       weight on positive class (handles imbalance)

    Returns
    -------
    loss    : float
    dlogits : np.ndarray  gradient w.r.t. pre-sigmoid logits, shape (N, 1)
    """
    y       = y.reshape(-1, 1).astype(np.float32)
    eps     = 1e-8
    loss    = -float(np.mean(
        pos_weight * y * np.log(p_hat + eps)
        + (1.0 - y) * np.log(1.0 - p_hat + eps)
    ))
    dlogits = ((1.0 - y) * p_hat - pos_weight * y * (1.0 - p_hat)) / len(y)
    return loss, dlogits


def evaluate(
    y_true:    np.ndarray,
    p_hat:     np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Computes classification metrics at a given threshold.

    Returns
    -------
    dict with keys: acc, bacc, precision, recall, f1, auc, pr_auc
    """
    y   = y_true.astype(int).reshape(-1)
    p   = p_hat.reshape(-1)
    pred = (p >= threshold).astype(int)
    return {
        "acc"      : accuracy_score(y, pred),
        "bacc"     : balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall"   : recall_score(y, pred, zero_division=0),
        "f1"       : f1_score(y, pred, zero_division=0),
        "auc"      : roc_auc_score(y, p)           if len(np.unique(y)) > 1 else np.nan,
        "pr_auc"   : average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
    }
