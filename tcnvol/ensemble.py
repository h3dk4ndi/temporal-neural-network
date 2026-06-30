from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

from tcnvol.trainer import Trainer
from tcnvol.config import TCNConfig


def evaluate(y_true, p, threshold: float = 0.5) -> dict:
    """
    Evaluate probabilistic binary-classification predictions.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    p : array-like
        Predicted probabilities for class 1.
    threshold : float
        Probability threshold used to convert probabilities into hard labels.

    Returns
    -------
    dict
        Dictionary containing AUC, PR-AUC, F1, accuracy, balanced accuracy,
        precision, and recall.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(p).reshape(-1)

    y_hat = (p >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        auc = np.nan
        pr_auc = np.nan
    else:
        auc = roc_auc_score(y_true, p)
        pr_auc = average_precision_score(y_true, p)

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "accuracy": accuracy_score(y_true, y_hat),
        "bacc": balanced_accuracy_score(y_true, y_hat),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
    }

# ─────────────────────────────────────────────────────────────────────
# EnsembleTrainer
# ─────────────────────────────────────────────────────────────────────

class EnsembleTrainer:
    """
    Multi-seed ensemble: trains one TCN per seed, averages predictions,
    and tunes the classification threshold on validation data.

    Parameters
    ----------
    F       : int          number of input features
    C_clean : np.ndarray   RMT-cleaned correlation matrix
    cfg     : TCNConfig    hyperparameter config
    """

    def __init__(
        self,
        F:       int,
        C_clean: np.ndarray,
        cfg:     TCNConfig,
    ) -> None:
        self.F            = F
        self.C_clean      = C_clean
        self.cfg          = cfg
        self._trainers:   list[Trainer] = []
        self._seed_metrics: list[dict]  = []
        self._threshold:  float         = 0.5

    def __repr__(self) -> str:
        return (f"EnsembleTrainer(seeds={self.cfg.seeds}, "
                f"F={self.F}, epochs={self.cfg.epochs})")

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
    ) -> pd.DataFrame:
        """
        Trains one model per seed and prints a full results summary.

        Returns
        -------
        pd.DataFrame   per-seed test metrics
        """
        pos_weight = float((1 - y_train.mean()) / max(y_train.mean(), 1e-6))
        print(f"pos_weight={pos_weight:.2f}  |  positive rate={y_train.mean():.3f}\n")

        val_preds, test_preds = [], []

        for seed in self.cfg.seeds:
            print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
            np.random.seed(seed)

            trainer = Trainer(self.F, self.C_clean, self.cfg, pos_weight)
            trainer.fit(X_train, y_train, X_val, y_val)

            val_preds.append(trainer.predict(X_val))
            test_preds.append(trainer.predict(X_test))

            m       = evaluate(y_test, test_preds[-1])
            m["seed"] = seed
            self._seed_metrics.append(m)
            self._trainers.append(trainer)
            print(f"  → Seed {seed}: AUC={m['auc']:.4f}  "
                  f"PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")

        # tune threshold on ensembled val predictions
        p_val_ens       = np.mean(val_preds, axis=0)
        ths             = np.linspace(0.05, 0.95, 181)
        self._threshold = ths[np.argmax([
            f1_score(y_val.astype(int),
                     (p_val_ens.reshape(-1) >= t).astype(int),
                     zero_division=0)
            for t in ths
        ])]

        # print summary
        results   = pd.DataFrame(self._seed_metrics)
        p_test_ens = np.mean(test_preds, axis=0)
        ens_m     = evaluate(y_test, p_test_ens, threshold=self._threshold)

        print(f"\n{'='*60}")
        print("PER-SEED TEST METRICS")
        print("=" * 60)
        for col in ["auc", "pr_auc", "f1", "precision", "recall", "bacc"]:
            mu, sd = results[col].mean(), results[col].std()
            print(f"  {col:<10}: {mu:.4f} ± {sd:.4f}")

        print(f"\n{'='*60}")
        print(f"ENSEMBLE  (threshold = {self._threshold:.3f})")
        print("=" * 60)
        for col in ["auc", "pr_auc", "f1", "bacc"]:
            print(f"  {col:<10}: {ens_m[col]:.4f}")
        print(f"\nBaseline PR-AUC (no-skill): {y_test.mean():.3f}")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Averages probabilities across all trained seeds."""
        if not self._trainers:
            raise RuntimeError("Call fit() before predict().")
        return np.mean([t.predict(X) for t in self._trainers], axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluates ensemble using tuned threshold."""
        return evaluate(y, self.predict(X), threshold=self._threshold)