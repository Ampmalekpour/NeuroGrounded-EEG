# eegitnet_hgd_mi_reduced.py
#
# Mutual-Information-guided channel reduction for EEG-ITNet on HGD — the
# "MI" row.
#
# Fixes applied relative to the original EEGITNet_HGD_MI.py:
#
#   1. CLASS-MASK BUG FIX: the original script masked on
#         mask = np.isin(y, [0, 1])   # feet vs left_hand
#      instead of left-hand-vs-right-hand — same bug as the CSP script, now
#      using the shared apply_class_mode().
#
#   2. LOSO MONTAGE FIX: subject S's reduced montage now comes only from
#      the other 13 subjects' MI scores.
#
#   3. 4-CLASS: mutual_info_classif natively supports a multi-class target,
#      so no OVR extension is needed here (unlike CSP) — the same
#      per-channel-variance-feature -> MI pipeline is used for both
#      2-class and 4-class, only the label vector differs.

import os
import gc
import numpy as np
from sklearn.feature_selection import mutual_info_classif

from eegitnet_hgd_common import (
    SEED, SUBJECT_IDS, FULL_CHANNELS, REDUCED_N_CHANNELS,
    load_subject_windows, normalize_importance, top_k_channels,
    loso_montage, run_reduced_training_loso,
)

OUTPUT_BASE = "EEGITNet_HGD_MI_Output"
CLASS_MODES = ["2class", "4class"]


def compute_mi_importance(X, y):
    features = np.var(X, axis=2).astype(np.float64)  # trials x channels
    scores = mutual_info_classif(features, y, discrete_features=False, random_state=SEED)
    return normalize_importance(scores)


def compute_per_subject_scores(class_mode):
    print("\n" + "=" * 80)
    print(f"PHASE 1: PER-SUBJECT MI CHANNEL SCORES FROM FULL 22-CHANNEL DATA — {class_mode}")
    print("=" * 80)

    subject_scores = {}
    for sid in SUBJECT_IDS:
        X, y = load_subject_windows(sid, FULL_CHANNELS, class_mode)
        print(f"Subject {sid:02d}: X={X.shape}, y={y.shape}, classes={np.unique(y)}")
        scores = compute_mi_importance(X, y)
        subject_scores[sid] = scores
        top12 = top_k_channels(scores, FULL_CHANNELS, REDUCED_N_CHANNELS)
        print(f"  Subject {sid:02d} MI Top-{REDUCED_N_CHANNELS}: "
              + ", ".join(f"{ch} ({v:.4f})" for ch, v in top12))
        del X, y
        gc.collect()

    return subject_scores


def run_one_class_mode(class_mode, output_base):
    subject_scores = compute_per_subject_scores(class_mode)

    # *** LOSO FIX ***
    subject_montages = loso_montage(subject_scores, FULL_CHANNELS, REDUCED_N_CHANNELS)

    print(f"\n[{class_mode}] === PHASE 2: TRAINING ON MI LOSO {REDUCED_N_CHANNELS}-CHANNEL MONTAGES ===")
    for sid in SUBJECT_IDS:
        print(f"  S{sid:02d} montage: {', '.join(subject_montages[sid])}")

    fold_results, means, stds, grand_mean, grand_std = run_reduced_training_loso(
        subject_montages, class_mode, tag="MI-LOSO"
    )

    report_path = os.path.join(output_base, f"mi_reduced_{class_mode}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EEG-ITNet — HGD — MUTUAL-INFORMATION CHANNEL REDUCTION — {class_mode}\n")
        f.write("LOSO montage: each subject's channels come only from the other 13 subjects.\n")
        f.write("=" * 80 + "\n\n")
        for sid, mean_acc, std_acc in zip(SUBJECT_IDS, means, stds):
            f.write(f"Subject {sid:02d} — montage: {', '.join(subject_montages[sid])}\n")
            for fold_i, acc in enumerate(fold_results[sid], start=1):
                f.write(f"  Fold {fold_i}: {acc*100:.2f}%\n")
            f.write(f"  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%\n\n")
        f.write(f"GRAND MEAN: {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%\n")

    print(f"[{class_mode}] GRAND MEAN (MI, LOSO): {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
    print(f"Report saved: {report_path}")
    return grand_mean, grand_std


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    results = {}
    for class_mode in CLASS_MODES:
        results[class_mode] = run_one_class_mode(class_mode, OUTPUT_BASE)

    print("\n" + "=" * 80)
    print("SUMMARY — MUTUAL-INFORMATION REDUCTION")
    print("=" * 80)
    for class_mode, (grand_mean, grand_std) in results.items():
        print(f"{class_mode}: {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
