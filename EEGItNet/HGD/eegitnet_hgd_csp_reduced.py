# eegitnet_hgd_csp_reduced.py
#
# CSP-guided channel reduction for EEG-ITNet on HGD — the "CSP" row.
#
# Fixes applied relative to the original EEGITNet_HGD_CSP.py:
#
#   1. CLASS-MASK BUG FIX: the original script masked on
#         mask = np.isin(y, [0, 1])   # feet vs left_hand
#      instead of left-hand-vs-right-hand. It was trained on a different
#      binary task than "Ours", CSP's own 2-class table row was invalid.
#      Now uses the same apply_class_mode() as every other script.
#
#   2. LOSO MONTAGE FIX: the reduced channel montage for subject S now
#      comes only from the other 13 subjects' CSP scores (never S's own),
#      addressing the reviewer's "global montage must generalize to a new
#      user" objection.
#
#   3. 4-CLASS CSP EXTENSION: standard two-class CSP (Sigma1 w = lambda
#      Sigma2 w) is undefined for 4 classes. This script extends it with
#      the standard one-vs-rest (OVR) construction: for each class c, fit
#      binary CSP for "trials of class c" vs "all other trials", and sum
#      the absolute spatial-pattern weights across all four OVR filters
#      to get one channel-importance score. This is a substantive,
#      reviewer-relevant methodological addition (comment #28: "give
#      complete, reproducible ... including the 4-class extension") and
#      should be described in the paper's Methods section once you've
#      reviewed it.

import os
import gc
import numpy as np

from mne.decoding import CSP

from eegitnet_hgd_common import (
    SUBJECT_IDS, FULL_CHANNELS, REDUCED_N_CHANNELS,
    load_subject_windows, normalize_importance, top_k_channels,
    loso_montage, run_reduced_training_loso, write_comparison_block,
)

CSP_N_COMPONENTS = 4
CSP_LOG = False
CSP_NORM_TRACE = False

OUTPUT_BASE = "EEGITNet_HGD_CSP_Output"
CLASS_MODES = ["2class", "4class"]


def compute_csp_scores_binary(X, y, n_components=CSP_N_COMPONENTS):
    """Standard two-class CSP channel importance from spatial patterns."""
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1], dtype=np.float32)

    n_components = min(n_components, X.shape[1])
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.int64)

    csp = CSP(n_components=n_components, reg=None, log=CSP_LOG, norm_trace=CSP_NORM_TRACE)
    csp.fit(X64, y64)

    patterns = np.asarray(csp.patterns_, dtype=np.float64)
    if patterns.ndim != 2:
        raise RuntimeError(f"Unexpected CSP patterns_ shape: {patterns.shape}")

    used_patterns = patterns[:n_components, :]
    scores = np.sum(np.abs(used_patterns), axis=0)
    return normalize_importance(scores)


def compute_csp_scores(X, y, n_components=CSP_N_COMPONENTS):
    """
    Dispatches to binary CSP for 2-class data, or to a one-vs-rest (OVR)
    extension for >2 classes: fit one binary CSP per class (that class vs.
    everything else) and average the resulting per-class importance
    vectors. Each OVR sub-problem is itself standard CSP, so this reduces
    exactly to the binary case when there are only two classes.
    """
    classes = np.unique(y)
    if len(classes) <= 2:
        return compute_csp_scores_binary(X, y, n_components)

    per_class_scores = []
    for c in classes:
        y_ovr = (y == c).astype(np.int64)
        if len(np.unique(y_ovr)) < 2:
            continue
        per_class_scores.append(compute_csp_scores_binary(X, y_ovr, n_components))

    if not per_class_scores:
        return np.zeros(X.shape[1], dtype=np.float32)

    return normalize_importance(np.mean(per_class_scores, axis=0))


def compute_per_subject_scores(class_mode):
    print("\n" + "=" * 80)
    print(f"PHASE 1: PER-SUBJECT CSP CHANNEL SCORES FROM FULL 22-CHANNEL DATA — {class_mode}")
    print("=" * 80)

    subject_scores = {}
    for sid in SUBJECT_IDS:
        X, y = load_subject_windows(sid, FULL_CHANNELS, class_mode)
        print(f"Subject {sid:02d}: X={X.shape}, y={y.shape}, classes={np.unique(y)}")
        scores = compute_csp_scores(X, y, n_components=CSP_N_COMPONENTS)
        subject_scores[sid] = scores
        top12 = top_k_channels(scores, FULL_CHANNELS, REDUCED_N_CHANNELS)
        print(f"  Subject {sid:02d} CSP Top-{REDUCED_N_CHANNELS}: "
              + ", ".join(f"{ch} ({v:.4f})" for ch, v in top12))
        del X, y
        gc.collect()

    return subject_scores


def run_one_class_mode(class_mode, output_base):
    subject_scores = compute_per_subject_scores(class_mode)

    # *** LOSO FIX: montage for subject S excludes S's own CSP score ***
    subject_montages = loso_montage(subject_scores, FULL_CHANNELS, REDUCED_N_CHANNELS)

    print(f"\n[{class_mode}] === PHASE 2: TRAINING ON CSP LOSO {REDUCED_N_CHANNELS}-CHANNEL MONTAGES ===")
    for sid in SUBJECT_IDS:
        print(f"  S{sid:02d} montage: {', '.join(subject_montages[sid])}")

    fold_results, means, stds, grand_mean, grand_std = run_reduced_training_loso(
        subject_montages, class_mode, tag="CSP-LOSO"
    )

    report_path = os.path.join(output_base, f"csp_reduced_{class_mode}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EEG-ITNet — HGD — CSP-GUIDED CHANNEL REDUCTION — {class_mode}\n")
        f.write("LOSO montage: each subject's channels come only from the other 13 subjects.\n")
        if class_mode == "4class":
            f.write("4-class CSP: one-vs-rest extension (see script header for definition).\n")
        f.write("=" * 80 + "\n\n")
        for sid, mean_acc, std_acc in zip(SUBJECT_IDS, means, stds):
            f.write(f"Subject {sid:02d} — montage: {', '.join(subject_montages[sid])}\n")
            for fold_i, acc in enumerate(fold_results[sid], start=1):
                f.write(f"  Fold {fold_i}: {acc*100:.2f}%\n")
            f.write(f"  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%\n\n")
        f.write(f"GRAND MEAN: {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%\n")

    print(f"[{class_mode}] GRAND MEAN (CSP, LOSO): {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
    print(f"Report saved: {report_path}")
    return grand_mean, grand_std


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    results = {}
    for class_mode in CLASS_MODES:
        results[class_mode] = run_one_class_mode(class_mode, OUTPUT_BASE)

    print("\n" + "=" * 80)
    print("SUMMARY — CSP-GUIDED REDUCTION")
    print("=" * 80)
    for class_mode, (grand_mean, grand_std) in results.items():
        print(f"{class_mode}: {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
