import pandas as pd
import numpy as np 
from statsmodels.tsa.stattools import adfuller 


class FracDiff:
    def __init__(
        self,
        d:     float | None = None,
        tau:   float        = 1e-4,
        alpha: float        = 0.05,
    ) -> None:                            
        self.d       = d
        self.tau     = tau
        self.alpha   = alpha
        self._fitted = False             

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"FracDiff(d={self.d}, tau={self.tau}, alpha={self.alpha}, {status})"

    def fit(self, X: pd.DataFrame) -> "FracDiff":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}.")
        if self.d is None:
            X_pre     = self._pre_transform(X)
            self.d, _ = self._grid_search(X_pre)
            if self.d is None:
                raise ValueError(
                    f"No d in [0, 1] achieved stationarity at alpha={self.alpha}."
                )
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}.")
        X_pre   = self._pre_transform(X)
        weights = self._weights(self.d)
        X_tilde = pd.DataFrame(0.0, index=X_pre.index, columns=X_pre.columns)
        for k, w_k in enumerate(weights):
            X_tilde += w_k * X_pre.shift(k)
        return X_tilde.dropna()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on X then transform X."""
        return self.fit(X).transform(X)

    def _pre_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.arcsinh(X).astype(np.float64)

    def _weights(self, d: float) -> np.ndarray:
        w, k = [1.0], 1
        while True:
            w_new = -w[-1] * (d - k + 1) / k
            if np.abs(w_new) < self.tau:
                break
            w.append(w_new)
            k += 1
        return np.array(w)

    def _grid_search(
        self,
        X:      pd.DataFrame,
        d_grid: np.ndarray | None = None,
    ) -> tuple[float | None, pd.DataFrame]:
        if d_grid is None:
            d_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
        rows = []
        for d in d_grid:
            weights = self._weights(d)
            X_tilde = pd.DataFrame(0.0, index=X.index, columns=X.columns)
            for k, w_k in enumerate(weights):
                X_tilde += w_k * X.shift(k)
            X_tilde = X_tilde.dropna()
            pvals, corrs = [], []
            for col in X_tilde.columns:
                try:
                    pval = adfuller(X_tilde[col].dropna(), autolag="AIC")[1]
                    corr = X.loc[X_tilde.index, col].corr(X_tilde[col])
                    pvals.append(pval)
                    corrs.append(corr)
                except Exception:
                    pass
            if not pvals:
                continue
            rows.append({
                "d":              d,
                "width":          len(weights) - 1,
                "max_pvalue":     np.round(np.max(pvals), 6),
                "stationary_all": bool(np.max(pvals) < self.alpha),
                "mean_corr":      np.round(np.mean(corrs), 6),
            })
        results = pd.DataFrame(rows)
        passed  = results[results["stationary_all"]]
        best_d  = None if passed.empty else float(passed.iloc[0]["d"])
        return best_d, results