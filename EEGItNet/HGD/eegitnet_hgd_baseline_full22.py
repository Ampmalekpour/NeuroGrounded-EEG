# eegitnet_hgd_baseline_full22.py
#
# Full 22-channel baseline for EEG-ITNet on HGD (Schirrmeister2017).
# Produces the "22ch" row of Tables 1 and 4 for both 2-class and 4-class.
#
# Run this first (or alongside the other four _reduced.py scripts) — the
# other scripts do NOT depend on this one's output; each is self-contained
# and re-trains its own full-channel models where needed (saliency does,
# CSP/MI/ReliefF don't). This script exists so you have one clean, direct
# "22ch" number per class-mode to put in the tables' 22ch rows and to sanity
# check against the 22ch numbers embedded in the saliency script's own run.

import os
import gc
import numpy as np

from eegitnet_hgd_common import (
    SUBJECT_IDS, FULL_CHANNELS, TARGET_SFREQ,
    load_subject_windows, make_blockwise_folds, train_one_fold,
    compute_subject_erd_maps, make_topo_info, plot_erd_ers_topomaps,
    save_array,
)

OUTPUT_BASE = "EEGITNet_HGD_Output"
CLASS_MODES = ["2class", "4class"]


def run_baseline(class_mode):
    output_subdir = os.path.join(OUTPUT_BASE, f"baseline_full22_{class_mode}")
    os.makedirs(output_subdir, exist_ok=True)

    topo_info = make_topo_info(FULL_CHANNELS)

    subject_means, subject_stds = [], []
    subject_fold_results = {}

    for sid in SUBJECT_IDS:
        print("\n" + "=" * 80)
        print(f"SUBJECT {sid:02d} — BASELINE 22ch — {class_mode}")
        print("=" * 80)

        X, y = load_subject_windows(sid, FULL_CHANNELS, class_mode)
        print(f"Trials: {X.shape[0]}, shape={X.shape}, classes={np.unique(y)}")

        folds = make_blockwise_folds(len(y), n_blocks=4)
        fold_test_accs = []
        for fold_i, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
            test_acc, best_val_acc, _, _, _, _ = train_one_fold(
                X, y, tr_idx, val_idx, te_idx, tag=f"S{sid:02d} F{fold_i}"
            )
            fold_test_accs.append(test_acc)
            print(f"S{sid:02d} Fold {fold_i}: test {test_acc*100:.2f}% (best val {best_val_acc*100:.2f}%)")

        subj_mean = float(np.mean(fold_test_accs))
        subj_std = float(np.std(fold_test_accs))
        subject_means.append(subj_mean)
        subject_stds.append(subj_std)
        subject_fold_results[sid] = fold_test_accs
        print(f"Subject {sid:02d} mean: {subj_mean*100:.2f}% +/- {subj_std*100:.2f}%")

        # ERD/ERS reference maps (always left-vs-right, regardless of class_mode)
        erd_maps = compute_subject_erd_maps(sid, FULL_CHANNELS)
        for k, v in erd_maps.items():
            save_array(os.path.join(output_subdir, f"{k}_S{sid:02d}.npy"), v)
        erd_png = os.path.join(output_subdir, f"erd_S{sid:02d}.png")
        plot_erd_ers_topomaps(erd_maps["ERD_L"], erd_maps["ERD_R"],
                               erd_maps["ERD_(L-R)"], erd_maps["ERD_comb"],
                               topo_info, erd_png)

        del X, y
        gc.collect()

    grand_mean = float(np.mean(subject_means))
    grand_std = float(np.std(subject_means))

    report_path = os.path.join(output_subdir, f"baseline_full22_{class_mode}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EEG-ITNet — HGD — FULL 22-CHANNEL BASELINE — {class_mode}\n")
        f.write("=" * 80 + "\n\n")
        for sid, mean_acc, std_acc in zip(SUBJECT_IDS, subject_means, subject_stds):
            f.write(f"Subject {sid:02d}\n")
            for i, acc in enumerate(subject_fold_results[sid], start=1):
                f.write(f"  Fold {i}: {acc*100:.2f}%\n")
            f.write(f"  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%\n\n")
        f.write(f"GRAND MEAN (22ch, {class_mode}): {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%\n")

    print(f"\n[{class_mode}] GRAND MEAN (22ch): {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
    print(f"Report saved: {report_path}")
    return subject_means, grand_mean, grand_std


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    results = {}
    for class_mode in CLASS_MODES:
        results[class_mode] = run_baseline(class_mode)

    print("\n" + "=" * 80)
    print("SUMMARY — FULL 22-CHANNEL BASELINE")
    print("=" * 80)
    for class_mode, (means, grand_mean, grand_std) in results.items():
        print(f"{class_mode}: grand mean {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
