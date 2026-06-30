import numpy as np 
import pandas as pd


class MarchenkoPastur:
    def __init__(
        self,
        standardise: bool = True,
        grid_length: int  = 5000,
    ) -> None:
        self.standardise = standardise
        self.grid_length = grid_length
        self._fitted     = False

    def __repr__(self) -> str:
        if not self._fitted:
            return (
                f"MarchenkoPastur("
                f"standardise={self.standardise}, "
                f"grid_length={self.grid_length}, "
                f"not fitted)"
            )
        return (
            f"MarchenkoPastur("
            f"n={self.n_}, p={self.p_}, "
            f"n_signal={self.n_signal_}, "
            f"lambda_max={self.lambda_max_:.4f}, "
            f"standardise={self.standardise})"
        )

    # ── public ────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame | np.ndarray) -> "MarchenkoPastur":
        
        X_arr = (
            X.to_numpy(dtype=np.float64)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=np.float64)
        )

        n, p = X_arr.shape

        if self.standardise:
            X_arr  = X_arr - X_arr.mean(axis=0, keepdims=True)
            std    = X_arr.std(axis=0, ddof=1, keepdims=True)
            std[std == 0] = 1.0
            X_arr /= std

        S                    = (1 / n) * X_arr.T @ X_arr
        eigenvalues, eigenvectors = np.linalg.eigh(S)

        idx          = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        y          = p / n
        lambda_max = (1 + np.sqrt(y)) ** 2
        lambda_min = (1 - np.sqrt(y)) ** 2

        signal_mask  = eigenvalues > lambda_max
        noise_mask   = ~signal_mask

        W_s          = eigenvectors[:, signal_mask]
        L_s          = eigenvalues[signal_mask]
        lambda_noise = eigenvalues[noise_mask].mean()

        C_clean = (
            W_s @ np.diag(L_s) @ W_s.T
            + lambda_noise * (np.eye(p) - W_s @ W_s.T)
        )

        scale         = np.sqrt(np.diag(C_clean))
        scale[scale < 1e-10] = 1.0
        C_clean       = C_clean / np.outer(scale, scale)

        # store results as attributes — accessible after fit()
        self.C_clean_   = C_clean
        self.n_signal_  = int(signal_mask.sum())
        self.lambda_max_ = float(lambda_max)
        self.lambda_min_ = float(lambda_min)
        self.n_         = n
        self.p_         = p
        self._fitted    = True

        return self

    def to_dataframe(self, columns: list[str]) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before to_dataframe().")
        return pd.DataFrame(self.C_clean_, index=columns, columns=columns)

# -----------------------------------------------------------------------------

def custom_padding(arr, pad_width, iaxis, kwargs):
    limit = kwargs.get("limit", 1.0)
    if pad_width[0] > 0:
        arr[:pad_width[0]]  = np.random.uniform(-limit, limit, size=pad_width[0])
    if pad_width[1] > 0:
        arr[-pad_width[1]:] = np.random.uniform(-limit, limit, size=pad_width[1])

def matrix_expand(C_clean: np.ndarray, F: int, k: int) -> np.ndarray:
    """
    Expands the RMT-cleaned correlation matrix C_clean (shape C×C)
    to a full weight direction matrix of shape (k*F, F) for the
    first conv layer, seeding known correlations into V1.
    """
    C_in, C_out = C_clean.shape
    assert C_in == C_out, "C_clean must be square."
    limit = np.sqrt(6.0 / (k * C_in + k * F))
    K = np.pad(C_clean, pad_width=(0, F - C_in),
               mode=custom_padding, limit=limit)
    return np.tile(K, (k, 1))