import numpy as np 
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# train / validation / test split

def train_val_test_split(
        X: pd.DataFrame,
        train_end: float,
        val_end: float
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
    if not (0 < train_end < val_end < 1):
        raise ValueError(

            "Required: 0 < train_end < val_end < 1, "
            f"got train_end = {train_end}, val_end = {val_end}."
        )
    T, _ = X.shape
    train_index = int(T * train_end)
    val_index = int(T * val_end)

    X_train = X.iloc[:train_index, :]
    X_val = X.iloc[train_index:val_index, :]
    X_test = X.iloc[val_index:, :]
    
    return X_train, X_val, X_test


def build_features(
    X_raw:       pd.DataFrame,
    X_frac_diff: pd.DataFrame,
    window:      int = 20,
) -> pd.DataFrame:

    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(X_raw).__name__}.")

    features = {}
    for col in X_raw.columns:
        r = np.arcsinh(X_raw[col]) - np.arcsinh(X_raw[col].shift(1))

        features[f"{col}_fd"]   = X_frac_diff[col]
        features[f"{col}_r"]    = r
        features[f"{col}_vol"]  = r.rolling(window).std()
        features[f"{col}_skew"] = r.rolling(window).skew()
        features[f"{col}_kurt"] = r.rolling(window).kurt()

    return pd.DataFrame(features).dropna()


def build_target(
    df_raw:     pd.DataFrame,
    target_col: str,
    horizon:    int   = 5,
    lookback:   int   = 252,
    q:          float = 0.75,
) -> pd.Series:
   
    r   = np.log(df_raw[target_col] / df_raw[target_col].shift(1))
    r2  = r ** 2

    # vectorised forward RV — no Python loop
    fwd_rv = np.sqrt(252 / horizon * r2.rolling(horizon).sum().shift(-horizon))

    rolling_thr = (
        fwd_rv.shift(horizon)
              .rolling(lookback, min_periods=lookback // 2)
              .quantile(q)
    )

    tmp = pd.DataFrame({"fwd_rv": fwd_rv, "thr": rolling_thr}).dropna()
    return (tmp["fwd_rv"] > tmp["thr"]).astype(int)


def align(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:

    common = X.index.intersection(y.index)
    return X.loc[common], y.loc[common]


def standardisation(
    X:       pd.DataFrame,
    X_train: pd.DataFrame,
) -> pd.DataFrame:

    mu    = X_train.mean()
    sigma = X_train.std(ddof=1)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma

def sliding_windows(
    X:             pd.DataFrame | np.ndarray,
    y:             pd.Series    | np.ndarray,
    window_length: int,
) -> tuple[np.ndarray, np.ndarray]:

    X_arr = X.to_numpy(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)
    y_arr = y.to_numpy(np.float32) if isinstance(y, pd.Series)    else np.asarray(y, np.float32)

    if len(X_arr) != len(y_arr):
        raise ValueError(f"X/y length mismatch: {len(X_arr)} vs {len(y_arr)}")
    T, C = X_arr.shape
    if T < window_length:
        raise ValueError(f"Window {window_length} > sequence length {T}.")

    V     = np.lib.stride_tricks.sliding_window_view(X_arr, window_length, axis=0)
    V     = V.transpose(0, 2, 1)
    y_win = y_arr[window_length - 1:]
    return V.astype(np.float32), y_win.astype(np.float32)