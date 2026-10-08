# eegitnet_hgd_relieff_reduced.py
#
# ReliefF-guided channel reduction for EEG-ITNet on HGD — the "RLF" row.
#
# Fixes applied relative to the original EEGITNet_HGD_RelieF.py:
#
#   1. CLASS-MASK BUG FIX: the original script masked on
#         mask = np.isin(y, [0, 1])   # feet vs left_hand
#      instead of left-hand-vs-right-hand — same bug as CSP/MI, now fixed
#      via the shared apply_class_mode().
#
#   2. LOSO MONTAGE FIX: subject S's reduced montage now comes only from
#      the other 13 subjects' ReliefF scores.
#
#   3. 4-CLASS: the manual ReliefF implementation here already handles any
#      number of classes correctly as written (hits = same label, misses =
#      any different label), so no algorithmic change was needed for
#      4-class beyond removing the 2-class-only mask.

import os
import gc
import numpy as np

from eegitnet_hgd_common import (
    SUBJECT_IDS, FULL_CHANNELS, REDUCED_N_CHANNELS,
    load_subject_windows, normalize_importance, top_k_channels,
    loso_montage, run_reduced_training_loso,
)

RELIEFF_N_NEIGHBORS = 10
OUTPUT_BASE = "EEGITNet_HGD_ReliefF_Output"
CLASS_MODES = ["2class", "4class"]


def compute_relieff_importance(X, y, n_neighbors=RELIEFF_N_NEIGHBORS):
    """Manual ReliefF using one channel-level feature per channel (temporal
    variance over the trial window). Hits = same-label trials, misses =
    any different-label trial — this generalizes correctly to >2 classes
    without modification."""
    n_trials, n_chans, _ = X.shape
    features = np.var(X, axis=2).astype(np.float64)
    features = (features - features.min(axis=0, keepdims=True)) / (
        features.max(axis=0, keepdims=True) - features.min(axis=0, keepdims=True) + 1e-8
    )

    weights = np.zeros(n_chans, dtype=np.float64)
    n_neighbors = min(n_neighbors, max(1, n_trials - 1))

    for i in range(n_trials):
        dists = np.linalg.norm(features - features[i], axis=1)
        dists[i] = np.inf

        hits_idx = np.where(np.asarray(y) == y[i])[0]
        hits_idx = hits_idx[hits_idx != i]
        misses_idx = np.where(np.asarray(y) != y[i])[0]

        near_hits = hits_idx[np.argsort(dists[hits_idx])[:n_neighbors]] if len(hits_idx) > 0 else np.array([], dtype=int)
        near_misses = misses_idx[np.argsort(dists[misses_idx])[:n_neighbors]] if len(misses_idx) > 0 else np.array([], dtype=int)

        diff_hits = (np.mean(np.abs(features[i] - features[near_hits]), axis=0)
                     if len(near_hits) > 0 else np.zeros(n_chans, dtype=np.float64))
        diff_misses = (np.mean(np.abs(features[i] - features[near_misses]), axis=0)
                       if len(near_misses) > 0 else np.zeros(n_chans, dtype=np.float64))

        weights += diff_misses - diff_hits

    return normalize_importance(weights / n_trials)


def compute_per_subject_scores(class_mode):
    print("\n" + "=" * 80)
    print(f"PHASE 1: PER-SUBJECT RELIEFF CHANNEL SCORES FROM FULL 22-CHANNEL DATA — {class_mode}")
    print("=" * 80)

    subject_scores = {}
    for sid in SUBJECT_IDS:
        X, y = load_subject_windows(sid, FULL_CHANNELS, class_mode)
        print(f"Subject {sid:02d}: X={X.shape}, y={y.shape}, classes={np.unique(y)}")
        scores = compute_relieff_importance(X, y, n_neighbors=RELIEFF_N_NEIGHBORS)
        subject_scores[sid] = scores
        top12 = top_k_channels(scores, FULL_CHANNELS, REDUCED_N_CHANNELS)
        print(f"  Subject {sid:02d} ReliefF Top-{REDUCED_N_CHANNELS}: "
              + ", ".join(f"{ch} ({v:.4f})" for ch, v in top12))
        del X, y
        gc.collect()

    return subject_scores


def run_one_class_mode(class_mode, output_base):
    subject_scores = compute_per_subject_scores(class_mode)

    # *** LOSO FIX ***
    subject_montages = loso_montage(subject_scores, FULL_CHANNELS, REDUCED_N_CHANNELS)

    print(f"\n[{class_mode}] === PHASE 2: TRAINING ON RELIEFF LOSO {REDUCED_N_CHANNELS}-CHANNEL MONTAGES ===")
    for sid in SUBJECT_IDS:
        print(f"  S{sid:02d} montage: {', '.join(subject_montages[sid])}")

    fold_results, means, stds, grand_mean, grand_std = run_reduced_training_loso(
        subject_montages, class_mode, tag="RLF-LOSO"
    )

    report_path = os.path.join(output_base, f"relieff_reduced_{class_mode}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EEG-ITNet — HGD — RELIEFF CHANNEL REDUCTION — {class_mode}\n")
        f.write("LOSO montage: each subject's channels come only from the other 13 subjects.\n")
        f.write("=" * 80 + "\n\n")
        for sid, mean_acc, std_acc in zip(SUBJECT_IDS, means, stds):
            f.write(f"Subject {sid:02d} — montage: {', '.join(subject_montages[sid])}\n")
            for fold_i, acc in enumerate(fold_results[sid], start=1):
                f.write(f"  Fold {fold_i}: {acc*100:.2f}%\n")
            f.write(f"  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%\n\n")
        f.write(f"GRAND MEAN: {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%\n")

    print(f"[{class_mode}] GRAND MEAN (ReliefF, LOSO): {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
    print(f"Report saved: {report_path}")
    return grand_mean, grand_std


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    results = {}
    for class_mode in CLASS_MODES:
        results[class_mode] = run_one_class_mode(class_mode, OUTPUT_BASE)

    print("\n" + "=" * 80)
    print("SUMMARY — RELIEFF REDUCTION")
    print("=" * 80)
    for class_mode, (grand_mean, grand_std) in results.items():
        print(f"{class_mode}: {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
