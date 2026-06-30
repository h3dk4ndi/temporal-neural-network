import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

# ──────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────

def build_har_features(df_raw, target_col):
    """
    HAR-RV features:
        RV_d = yesterday's realised variance
        RV_w = previous 5-day average realised variance
        RV_m = previous 22-day average realised variance

    All features are lagged by one day to avoid lookahead.
    """
    r = np.log(df_raw[target_col] / df_raw[target_col].shift(1))
    rv = r ** 2

    har = pd.DataFrame({
        "RV_d": rv.shift(1),
        "RV_w": rv.rolling(5).mean().shift(1),
        "RV_m": rv.rolling(22).mean().shift(1),
    })

    return har.dropna()


def tune_threshold(y_val, p_val):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = []

    for t in thresholds:
        y_pred = (p_val >= t).astype(int)
        scores.append(f1_score(y_val, y_pred, zero_division=0))

    return float(thresholds[np.argmax(scores)])


def evaluate_binary(y_true, p, threshold):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).reshape(-1)
    y_pred = (p >= threshold).astype(int)

    return {
        "AUC": roc_auc_score(y_true, p),
        "PR-AUC": average_precision_score(y_true, p),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "BAcc": balanced_accuracy_score(y_true, y_pred),
        "Threshold": threshold,
    }


def align_xy(X, y):
    idx = X.index.intersection(y.index)
    return X.loc[idx], y.loc[idx]
