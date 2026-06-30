"""
Sanity checks + ablation study for the TCN volatility-regime classifier.

Reproduces four checks:
  TEST 1  random-label   — AUC must collapse to ~0.50 (proves no leakage)
  TEST 2  shifted-target — AUC must drop when features/labels are misaligned in time
  TEST 3  no-RMT ablation — Glorot init instead of Marchenko-Pastur (tests H3)
  TEST 4  logistic on 105 features, last timestep (tests H2)
…then prints an ablation table and the H1/H2/H3 verdicts.

Run from the REPO ROOT:

    python -m scripts.sanity_checks

NOTE: this trains several ensembles, so it is compute-heavy. Set QUICK = True for a
fast smoke test (fewer seeds/epochs) — the relative verdicts still hold, but the
absolute numbers won't match the headline run.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from tcnvol.fracdiff import FracDiff
from tcnvol.rmt import MarchenkoPastur
from tcnvol.features import (train_val_test_split, build_features, build_target,
                             align, standardisation, sliding_windows)
from tcnvol.config import TCNConfig
from tcnvol.ensemble import EnsembleTrainer
from tcnvol.har_rv import build_har_features, align_xy

# reuse the exact same data loader as the main pipeline (no duplication)
from scripts.run_pipeline import load_data, TARGET_COL, WINDOW

# ─────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────
QUICK = False                                  # True = fast, non-headline numbers
SHIFT = 60                                     # days, for the shifted-target test

CFG_FULL = (TCNConfig(epochs=15, batch_size=64, patience=5, seeds=[1, 2], dropout=0.25)
            if QUICK else
            TCNConfig(epochs=50, batch_size=64, patience=10, seeds=[1, 2, 3, 4, 5], dropout=0.25))
CFG_FAST = TCNConfig(epochs=20, batch_size=64, patience=5, seeds=[1, 2])


# ─────────────────────────────────────────────────────────────────────
# Shared prep — mirrors run_pipeline steps 2–6 (could be factored into a
# shared module later; kept here so this script runs on its own).
# ─────────────────────────────────────────────────────────────────────
def prepare_windows(df: pd.DataFrame) -> dict:
    X_train, X_val, X_test = train_val_test_split(df, 0.7, 0.85)

    fd = FracDiff(tau=1e-4, alpha=0.05)
    fd.fit(X_train)
    X_all_fd   = fd.transform(df)
    X_train_fd = fd.transform(X_train)

    mp = MarchenkoPastur()
    mp.fit(X_train_fd)

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

    X_train_z = standardisation(X_train_feat, X_train_feat)
    X_val_z   = standardisation(X_val_feat,   X_train_feat)
    X_test_z  = standardisation(X_test_feat,  X_train_feat)

    X_train_w, y_train_w = sliding_windows(X_train_z, y_train, WINDOW)
    X_val_w,   y_val_w   = sliding_windows(X_val_z,   y_val,   WINDOW)
    X_test_w,  y_test_w  = sliding_windows(X_test_z,  y_test,  WINDOW)

    return dict(
        X_train_w=X_train_w, y_train_w=y_train_w,
        X_val_w=X_val_w,     y_val_w=y_val_w,
        X_test_w=X_test_w,   y_test_w=y_test_w,
        C_clean=mp.C_clean_,
        y_train=y_train, y_val=y_val, y_test=y_test,
        tcn_test_index=y_test.index[-len(y_test_w):],
    )


def fit_har(df, y_train, y_val, y_test, tcn_test_index):
    """Compact HAR-RV logistic baseline, aligned to the TCN test dates."""
    har_all = build_har_features(df, TARGET_COL)
    har_train, y_har_train = align_xy(har_all, y_train)
    har_test,  _           = align_xy(har_all, y_test)

    scaler = StandardScaler()
    model  = LogisticRegression(class_weight="balanced", C=1,
                                max_iter=1000, random_state=42)
    model.fit(scaler.fit_transform(har_train), y_har_train)

    idx = har_test.index.intersection(tcn_test_index)
    y_aligned = y_test.loc[idx].values.astype(int)
    p_aligned = model.predict_proba(scaler.transform(har_test.loc[idx]))[:, 1]
    return y_aligned, p_aligned


def main() -> None:
    if QUICK:
        print("*** QUICK mode: numbers are illustrative, not headline ***\n")

    d = prepare_windows(load_data())
    X_train_w, y_train_w = d["X_train_w"], d["y_train_w"]
    X_val_w,   y_val_w   = d["X_val_w"],   d["y_val_w"]
    X_test_w,  y_test_w  = d["X_test_w"],  d["y_test_w"]
    C_clean,   F         = d["C_clean"],   d["X_train_w"].shape[2]

    # ── reference: the real, full ensemble ───────────────────────────
    ensemble = EnsembleTrainer(F=F, C_clean=C_clean, cfg=CFG_FULL)
    ensemble.fit(X_train_w, y_train_w, X_val_w, y_val_w, X_test_w, y_test_w)
    p_real   = ensemble.predict(X_test_w).reshape(-1)
    auc_real = roc_auc_score(y_test_w, p_real)
    ap_real  = average_precision_score(y_test_w, p_real)

    # ── HAR-RV baseline (for the H1 row) ─────────────────────────────
    y_har_aligned, p_har_test = fit_har(load_data(), d["y_train"], d["y_val"],
                                        d["y_test"], d["tcn_test_index"])
    auc_har = roc_auc_score(y_har_aligned, p_har_test)
    ap_har  = average_precision_score(y_har_aligned, p_har_test)

    # ── TEST 1: random labels (expect AUC ≈ 0.50) ────────────────────
    rng = np.random.default_rng(42)
    y_rand_train = rng.permutation(y_train_w)
    y_rand_val   = rng.permutation(y_val_w)
    y_rand_test  = rng.permutation(y_test_w)

    ens_rand = EnsembleTrainer(F=F, C_clean=C_clean, cfg=CFG_FAST)
    ens_rand.fit(X_train_w, y_rand_train, X_val_w, y_rand_val, X_test_w, y_rand_test)
    p_rand   = ens_rand.predict(X_test_w).reshape(-1)
    auc_rand = roc_auc_score(y_rand_test, p_rand)
    ap_rand  = average_precision_score(y_rand_test, p_rand)

    print("=" * 52)
    print("TEST 1 — RANDOM LABEL")
    print("=" * 52)
    print(f"  Real-label AUC   : {auc_real:.4f}")
    print(f"  Random-label AUC : {auc_rand:.4f}  (expect ~0.50)")
    print(f"  Verdict          : "
          f"{'PASS — no leakage' if abs(auc_rand - 0.5) < 0.05 else 'SUSPICIOUS'}")

    # ── TEST 2: shifted target (expect AUC drop) ─────────────────────
    ens_shift = EnsembleTrainer(F=F, C_clean=C_clean, cfg=CFG_FAST)
    ens_shift.fit(X_train_w[SHIFT:], y_train_w[:-SHIFT],
                  X_val_w[SHIFT:],   y_val_w[:-SHIFT],
                  X_test_w[SHIFT:],  y_test_w[:-SHIFT])
    p_shift   = ens_shift.predict(X_test_w[SHIFT:]).reshape(-1)
    auc_shift = roc_auc_score(y_test_w[:-SHIFT], p_shift)
    drop      = auc_real - auc_shift

    print(f"\nTEST 2 — SHIFTED TARGET (shift = {SHIFT} days)")
    print(f"  Real AUC    : {auc_real:.4f}")
    print(f"  Shifted AUC : {auc_shift:.4f}  (expect ~0.50)")
    print(f"  AUC drop    : {drop:+.4f}  {'PASS' if drop > 0.05 else 'SUSPICIOUS'}")

    # ── TEST 3: no-RMT ablation (Glorot only) ────────────────────────
    ens_no_rmt = EnsembleTrainer(F=F, C_clean=None, cfg=CFG_FULL)
    ens_no_rmt.fit(X_train_w, y_train_w, X_val_w, y_val_w, X_test_w, y_test_w)
    p_no_rmt   = ens_no_rmt.predict(X_test_w).reshape(-1)
    auc_no_rmt = roc_auc_score(y_test_w, p_no_rmt)
    ap_no_rmt  = average_precision_score(y_test_w, p_no_rmt)

    # ── TEST 4: logistic regression on 105 features, last timestep ───
    sc_lr  = StandardScaler()
    lr_105 = LogisticRegression(class_weight="balanced", C=1.0,
                                max_iter=1000, random_state=42)
    lr_105.fit(sc_lr.fit_transform(X_train_w[:, -1, :]), y_train_w.astype(int))
    p_lr   = lr_105.predict_proba(sc_lr.transform(X_test_w[:, -1, :]))[:, 1]
    auc_lr = roc_auc_score(y_test_w, p_lr)
    ap_lr  = average_precision_score(y_test_w, p_lr)

    # ── Ablation table ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("ABLATION TABLE")
    print("=" * 62)
    print(f"  {'Model':<35} {'AUC':>8} {'PR-AUC':>8}")
    print("-" * 62)
    rows = [
        ("No-skill baseline",      0.500,      float(y_test_w.mean())),
        ("HAR-RV + LogReg",        auc_har,    ap_har),
        ("LogReg on 105 features", auc_lr,     ap_lr),
        ("TCN, no RMT init",       auc_no_rmt, ap_no_rmt),
        ("TCN, 5-seed ensemble",   auc_real,   ap_real),
        ("TCN, random labels",     auc_rand,   ap_rand),
    ]
    for name, auc_v, ap_v in rows:
        print(f"  {name:<35} {auc_v:>8.4f} {ap_v:>8.4f}")
    print("=" * 62)

    # ── Hypotheses ───────────────────────────────────────────────────
    print(f"\nH1: TCN ensemble vs HAR-RV     ΔPR-AUC = {ap_real - ap_har:+.4f}   "
          f"{'CONFIRMED' if ap_real > ap_har else 'REJECTED'}")
    print(f"H2: TCN no-RMT vs LogReg(105)  ΔPR-AUC = {ap_no_rmt - ap_lr:+.4f}   "
          f"{'CONFIRMED' if ap_no_rmt > ap_lr else 'REJECTED'}")
    print(f"H3: TCN full vs TCN no-RMT     ΔPR-AUC = {ap_real - ap_no_rmt:+.4f}   "
          f"{'CONFIRMED' if ap_real > ap_no_rmt else 'REJECTED'}")


if __name__ == "__main__":
    main()
