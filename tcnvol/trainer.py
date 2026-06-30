import numpy as np
from tcnvol.tcn import TCN
from tcnvol.optim import AdamW
from tcnvol.losses import bce_loss, evaluate


# ─────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Single-seed training loop with early stopping on val PR-AUC.

    Parameters
    ----------
    F          : int           number of input features
    C_clean    : np.ndarray    RMT-cleaned correlation matrix
    cfg        : TCNConfig     hyperparameter config
    pos_weight : float         BCE positive class weight
    """

    def __init__(
        self,
        F:          int,
        C_clean:    np.ndarray,
        cfg:        TCNConfig,
        pos_weight: float = 1.0,
    ) -> None:
        self.F          = F
        self.C_clean    = C_clean
        self.cfg        = cfg
        self.pos_weight = pos_weight
        self._model: TCN | None = None

    def __repr__(self) -> str:
        return (f"Trainer(F={self.F}, epochs={self.cfg.epochs}, "
                f"batch={self.cfg.batch_size}, patience={self.cfg.patience})")

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> dict:
        """
        Trains the TCN and restores the best checkpoint.

        Parameters
        ----------
        X_train, X_val : np.ndarray  shape (N, W, F)
        y_train, y_val : np.ndarray  shape (N,)

        Returns
        -------
        dict   training history with keys train_loss, val_pr_auc
        """
        model = TCN(self.F, self.cfg.kernel_size, self.cfg.dilations,
                    self.cfg.dropout, self.C_clean)
        opt   = AdamW(self.cfg.lr, weight_decay=self.cfg.weight_decay,
                      clip_norm=self.cfg.clip_norm)

        best_pr_auc, best_state = -np.inf, None
        patience = 0
        history  = {"train_loss": [], "val_pr_auc": []}

        for epoch in range(1, self.cfg.epochs + 1):
            idx    = np.random.permutation(len(X_train))
            losses = []

            for start in range(0, len(X_train), self.cfg.batch_size):
                b             = idx[start:start + self.cfg.batch_size]
                p_hat         = model.forward(X_train[b], training=True)
                loss, dlogits = bce_loss(p_hat, y_train[b], self.pos_weight)
                model.backward(dlogits)
                opt.step(model.parameters(), model.gradients())
                losses.append(loss)

            train_loss  = float(np.mean(losses))
            p_val       = self._predict(X_val, model)
            val_loss, _ = bce_loss(p_val, y_val, self.pos_weight)
            val_m       = evaluate(y_val, p_val)

            history["train_loss"].append(train_loss)
            history["val_pr_auc"].append(val_m["pr_auc"])

            print(f"  Ep {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"AUC={val_m['auc']:.4f} | PR-AUC={val_m['pr_auc']:.4f} | "
                  f"F1={val_m['f1']:.4f} | P={val_m['precision']:.4f} | R={val_m['recall']:.4f}")

            if val_m["pr_auc"] > best_pr_auc + 1e-4:
                best_pr_auc = val_m["pr_auc"]
                best_state  = model.state_dict()
                patience    = 0
            else:
                patience   += 1

            if patience >= self.cfg.patience:
                print(f"  → Early stop. Best PR-AUC: {best_pr_auc:.4f}")
                break

        if best_state is not None:
            model.load_state(best_state)

        self._model = model
        return history

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Returns predicted probabilities, shape (N, 1)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._predict(X, self._model, batch_size)

    def _predict(self, X: np.ndarray, model: TCN, batch_size: int = 256) -> np.ndarray:
        return np.vstack([
            model.forward(X[s:s + batch_size], training=False)
            for s in range(0, len(X), batch_size)
        ])