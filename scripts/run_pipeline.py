"""
End-to-end pipeline: Bloomberg data -> fractional differencing -> Marchenko-Pastur
denoising -> features/target -> sliding windows -> TCN ensemble -> HAR-RV baseline.

Run from the REPO ROOT (so the `tcnvol` package resolves):

    python -m scripts.run_pipeline

Data: by default loads a local parquet you do NOT commit (see .gitignore). Set
USE_BLOOMBERG = True to pull live from a Bloomberg Terminal instead.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless: save figures to disk instead of a window
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from tcnvol.bloomberg import Bloomberg
from tcnvol.fracdiff import FracDiff
from tcnvol.rmt import MarchenkoPastur
from tcnvol.features import (train_val_test_split, build_features, build_target,
                             align, standardisation, sliding_windows)
from tcnvol.config import TCNConfig
from tcnvol.ensemble import EnsembleTrainer
from tcnvol.har_rv import (build_har_features, tune_threshold,
                           evaluate_binary, align_xy)

# ─────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────
USE_BLOOMBERG = False                              # True only on a Terminal machine
DATA_PATH     = "data/bloomberg_raw_data.parquet"  # gitignored; place your parquet here
TARGET_COL    = "PX_LAST XAU Curncy"
WINDOW        = 20
RESULTS_DIR   = "results"


def load_data() -> pd.DataFrame:
    """Either pull from Bloomberg (Terminal only) or load the cached parquet."""
    if USE_BLOOMBERG:
        bbg = Bloomberg(
            host="localhost",
            port=8194,
            securities=[
                "XAU Curncy",      # target — spot gold
                "DXY Curncy",      # USD strength
                "LF98TRUU Index",  # HY credit
                "XAG Curncy",      # silver
                "NZD Curncy",
                "USGG10YR Index",
                "CL1 Comdty",
            ],
            fields=["PX_LAST", "RSI_14D", "VOLATILITY_30D"],
            start="20060101",
            end="20260101",
        )
        print(bbg)
        df = bbg.fetch()
    else:
        df = pd.read_parquet(DATA_PATH, engine="pyarrow")

    df = df.ffill(axis=0)
    df = df.dropna(axis=0)
    return df


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────
    df = load_data()
    print(df.head())
    print("nulls:\n", df.isnull().sum())

    # ── 2. Train / val / test split (chronological, no shuffle) ──────
    X_train, X_val, X_test = train_val_test_split(df, 0.7, 0.85)

    # ── 3. Fractional differencing — fit d on TRAIN only ─────────────
    fd = FracDiff(tau=1e-4, alpha=0.05)
    fd.fit(X_train)                       # finds d once, on train only
    X_all_fd   = fd.transform(df)         # full series, for feature alignment
    X_train_fd = fd.transform(X_train)    # train slice, for the MP fit

    # ── 4. Marchenko-Pastur denoising of the train covariance ────────
    mp = MarchenkoPastur()
    mp.fit(X_train_fd)
    C_clean = mp.C_clean_                  # cleaned covariance -> conv weight init
    print(mp)

    # ── 5. Features (raw + fracdiff) and binary regime target ────────
    X_all_feat = build_features(X_raw=df, X_frac_diff=X_all_fd, window=WINDOW)
    X_train_feat = X_all_feat.loc[X_train.index.intersection(X_all_feat.index)]
    X_val_feat   = X_all_feat.loc[X_val.index.intersection(X_all_feat.index)]
    X_test_feat  = X_all_feat.loc[X_test.index.intersection(X_all_feat.index)]

    y_full  = build_target(df, TARGET_COL, horizon=5, lookback=252, q=0.75)
    y_train = y_full.loc[y_full.index.intersection(X_train.index)]
    y_val   = y_full.loc[y_full.index.intersection(X_val.index)]
    y_test  = y_full.loc[y_full.index.intersection(X_test.index)]

    X_train_feat, y_train = align(X_train_feat, y_train)
    X_val_feat,   y_val   = align(X_val_feat,   y_val)
    X_test_feat,  y_test  = align(X_test_feat,  y_test)

    print(f"y_train rate: {y_train.mean():.3f} | "
          f"y_val rate: {y_val.mean():.3f} | "
          f"y_test rate: {y_test.mean():.3f}")

    # ── 6. Standardise (fit on TRAIN, apply to all) + sliding windows ─
    X_train_z = standardisation(X_train_feat, X_train_feat)
    X_val_z   = standardisation(X_val_feat,   X_train_feat)
    X_test_z  = standardisation(X_test_feat,  X_train_feat)

    X_train_w, y_train_w = sliding_windows(X_train_z, y_train, WINDOW)
    X_val_w,   y_val_w   = sliding_windows(X_val_z,   y_val,   WINDOW)
    X_test_w,  y_test_w  = sliding_windows(X_test_z,  y_test,  WINDOW)
    print(f"X_train_w: {X_train_w.shape} | X_val_w: {X_val_w.shape} | "
          f"X_test_w: {X_test_w.shape}")

    # ── 7. Train the TCN ensemble ────────────────────────────────────
    cfg = TCNConfig(epochs=50, batch_size=64, patience=10,
                    seeds=[1, 2, 3, 4, 5], dropout=0.25)
    ensemble = EnsembleTrainer(F=X_train_w.shape[2], C_clean=C_clean, cfg=cfg)
    ensemble.fit(X_train_w, y_train_w, X_val_w, y_val_w, X_test_w, y_test_w)

    tcn_metrics_raw = ensemble.evaluate(X_test_w, y_test_w)
    print("TCN ensemble (full test window):", tcn_metrics_raw)
    print(ensemble)

    # ── 8. HAR-RV logistic-regression baseline ───────────────────────
    har_all = build_har_features(df, TARGET_COL)
    har_train, y_har_train = align_xy(har_all, y_train)
    har_val,   y_har_val   = align_xy(har_all, y_val)
    har_test,  y_har_test  = align_xy(har_all, y_test)

    scaler = StandardScaler()
    har_train_z = scaler.fit_transform(har_train)
    har_val_z   = scaler.transform(har_val)

    har_model = LogisticRegression(class_weight="balanced", C=1,
                                   max_iter=1000, random_state=42)
    har_model.fit(har_train_z, y_har_train)

    p_har_val = har_model.predict_proba(har_val_z)[:, 1]
    thr_har   = tune_threshold(y_har_val.values.astype(int), p_har_val)

    # ── 9. Align HAR and TCN to the SAME test dates, then compare ────
    # The TCN loses the first WINDOW-1 rows to sliding windows; align HAR to match.
    tcn_test_index = y_test.index[-len(y_test_w):]
    idx_test       = har_test.index.intersection(tcn_test_index)

    har_test_aligned   = har_test.loc[idx_test]
    y_test_aligned     = y_test.loc[idx_test].values.astype(int)
    har_test_aligned_z = scaler.transform(har_test_aligned)
    p_har_test         = har_model.predict_proba(har_test_aligned_z)[:, 1]

    p_tcn_test   = ensemble.predict(X_test_w).reshape(-1)
    p_tcn_series = pd.Series(p_tcn_test, index=tcn_test_index)
    y_tcn_series = pd.Series(np.asarray(y_test_w).astype(int), index=tcn_test_index)
    p_tcn_aligned = p_tcn_series.loc[idx_test].values
    y_tcn_aligned = y_tcn_series.loc[idx_test].values
    thr_tcn       = ensemble._threshold

    har_metrics = evaluate_binary(y_test_aligned, p_har_test, threshold=thr_har)
    tcn_metrics = evaluate_binary(y_tcn_aligned, p_tcn_aligned, threshold=thr_tcn)

    comparison = pd.DataFrame({
        "HAR-RV LogReg": har_metrics,
        "TCN Ensemble":  tcn_metrics,
    }).T
    print("\nHAR-RV vs TCN")
    print("=" * 80)
    print(comparison.round(4))
    print("=" * 80)
    print("Aligned test observations:", len(idx_test))

    # ── 10. Plots (saved for the README) + difference table ──────────
    ranking_cols   = ["AUC", "PR-AUC"]
    threshold_cols = ["Precision", "Recall", "F1", "BAcc"]

    ax = comparison[ranking_cols].plot(kind="bar", figsize=(7, 4))
    ax.set_title("HAR-RV vs TCN: Ranking Metrics")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tcn_vs_har_ranking.png"), dpi=150)
    plt.close()

    ax = comparison[threshold_cols].plot(kind="bar", figsize=(8, 4))
    ax.set_title("HAR-RV vs TCN: Threshold-Dependent Metrics")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tcn_vs_har_threshold.png"), dpi=150)
    plt.close()

    diff = (comparison.loc["TCN Ensemble", ranking_cols + threshold_cols]
            - comparison.loc["HAR-RV LogReg", ranking_cols + threshold_cols])
    print("\nTCN minus HAR-RV")
    print("=" * 80)
    print(diff.round(4))
    print("=" * 80)
    print(f"\nFigures written to ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
