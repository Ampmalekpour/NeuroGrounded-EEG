# eegitnet_hgd_saliency_reduced.py
#
# Saliency-guided channel reduction for EEG-ITNet on HGD — the "Ours" row.
#
# Fixes applied relative to the original
# EEGITNet_BD_HGD_BD_Final_2_class.py / _4c.py:
#
#   1. LEAKAGE FIX: channel-importance (saliency) scores are now computed
#      from the VALIDATION set of each fold, never the test set. The
#      original code computed them from X_te/y_te — the same trials the
#      reduced model is later "tested" on — which lets test-set information
#      leak into channel selection. Matches paper Section 3.5's claim.
#
#   2. LOSO MONTAGE FIX: the reduced channel montage used for subject S is
#      now computed from the OTHER 13 subjects' saliency only (never
#      including S's own data). The original script averaged saliency over
#      ALL 14 subjects (including the one it then evaluated on), which is
#      the reviewer's "global montage must generalize to a new user"
#      objection (their comment #6/#22).
#
#   3. Extra ERD correlations (ERD_L, ERD_R, ERD_comb) are computed and
#      written to the report for every subject, in addition to the
#      existing ERD_(L-R) correlation — but, per instruction, they are NOT
#      used anywhere in the decision logic. Attribution-method ranking
#      (which methods count as "best") still uses ERD_(L-R) only, exactly
#      as before.
#
#   4. CLASS_MODE ("2class" / "4class") is a single shared parameter instead
#      of two near-duplicate files, so the mask/remap logic can't drift.

import os
import gc
import numpy as np

from eegitnet_hgd_common import (
    SUBJECT_IDS, FULL_CHANNELS, REDUCED_N_CHANNELS, ATTR_METHODS,
    load_subject_windows, make_blockwise_folds, train_one_fold, build_model,
    compute_fold_channel_importances, normalize_importance, corrcoef_safe,
    top_k_channels, loso_montage, run_reduced_training_loso,
    compute_subject_erd_maps, correlations_report,
    make_topo_info, plot_saliency_topomap, plot_erd_ers_topomaps, save_array,
    write_comparison_block,
)

import torch

BEST_METHODS_COUNT = 3   # how many top attribution methods to average, as before
OUTPUT_BASE = "EEGITNet_HGD_Saliency_Output"
CLASS_MODES = ["2class", "4class"]


def run_full22_and_collect_saliency(class_mode, topo_info):
    """Phase 1: train the full-22ch model per subject/fold, and collect
    per-subject saliency vectors from the VALIDATION set of each fold."""
    output_subdir = os.path.join(OUTPUT_BASE, f"full_22ch_{class_mode}")
    os.makedirs(output_subdir, exist_ok=True)

    subject_summaries = []
    full_means = []

    for sid in SUBJECT_IDS:
        print("\n" + "=" * 80)
        print(f"SUBJECT {sid:02d} — FULL 22ch (saliency source) — {class_mode}")
        print("=" * 80)

        X, y = load_subject_windows(sid, FULL_CHANNELS, class_mode)
        print(f"Trials: {X.shape[0]}, shape={X.shape}, classes={np.unique(y)}")

        folds = make_blockwise_folds(len(y), n_blocks=4)
        fold_test_accs = []
        importance_accum = {m: np.zeros(len(FULL_CHANNELS), dtype=np.float64) for m in ATTR_METHODS}

        for fold_i, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
            test_acc, best_val_acc, best_state, n_ch, n_t, n_cl = train_one_fold(
                X, y, tr_idx, val_idx, te_idx, tag=f"S{sid:02d} F{fold_i}"
            )
            fold_test_accs.append(test_acc)

            model = build_model(n_ch, n_t, n_cl)
            model.load_state_dict(best_state)
            model.eval()

            # *** FIX: validation set, not test set ***
            X_val, y_val = X[val_idx], y[val_idx]
            fold_imps = compute_fold_channel_importances(model, X_val, y_val, ATTR_METHODS, batch_size=32)
            for m in ATTR_METHODS:
                importance_accum[m] += fold_imps[m]

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        subj_mean = float(np.mean(fold_test_accs))
        subj_std = float(np.std(fold_test_accs))
        full_means.append(subj_mean)
        print(f"Subject {sid:02d} 22ch mean: {subj_mean*100:.2f}% +/- {subj_std*100:.2f}%")

        subject_importances = {m: normalize_importance(importance_accum[m] / len(folds)) for m in ATTR_METHODS}
        erd_maps = compute_subject_erd_maps(sid, FULL_CHANNELS)

        subject_corrs = {m: correlations_report(vec, erd_maps) for m, vec in subject_importances.items()}

        subject_summaries.append({
            "subject": sid,
            "fold_test_accs": fold_test_accs,
            "subject_mean_acc": subj_mean,
            "subject_std_acc": subj_std,
            "importance": subject_importances,
            "erd_maps": erd_maps,
            "correlations": subject_corrs,
        })

        for m, scores in subject_importances.items():
            plot_saliency_topomap(scores, topo_info, f"S{sid:02d} {m}",
                                   os.path.join(output_subdir, f"saliency_{m}_S{sid:02d}.png"))
            save_array(os.path.join(output_subdir, f"saliency_{m}_S{sid:02d}.npy"), scores)
        for k, v in erd_maps.items():
            save_array(os.path.join(output_subdir, f"{k}_S{sid:02d}.npy"), v)
        plot_erd_ers_topomaps(erd_maps["ERD_L"], erd_maps["ERD_R"], erd_maps["ERD_(L-R)"], erd_maps["ERD_comb"],
                               topo_info, os.path.join(output_subdir, f"erd_S{sid:02d}.png"))

        del X, y
        gc.collect()

    grand_mean = float(np.mean(full_means))
    grand_std = float(np.std(full_means))
    print(f"\n[{class_mode}] FULL 22ch grand mean: {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")

    return subject_summaries, full_means, grand_mean, grand_std, output_subdir


def run_one_class_mode(class_mode, report_lines):
    topo_info = make_topo_info(FULL_CHANNELS)

    full_summaries, full_means, grand_mean, grand_std, full_subdir = \
        run_full22_and_collect_saliency(class_mode, topo_info)

    # ───── Rank attribution methods by avg ERD_(L-R) correlation only
    #        (decision logic unchanged; ERD_L/ERD_R/ERD_comb are report-only) ─────
    method_avg_corrs = {
        m: float(np.nanmean([s["correlations"][m]["ERD_(L-R)"] for s in full_summaries]))
        for m in ATTR_METHODS
    }
    ranked_methods = sorted(method_avg_corrs, key=method_avg_corrs.get, reverse=True)
    best_methods = ranked_methods[:BEST_METHODS_COUNT]
    print(f"[{class_mode}] Best attribution methods (top {BEST_METHODS_COUNT} by avg ERD_(L-R) corr): {best_methods}")

    # ───── Per-subject "avg best methods" saliency vector, used for LOSO montage ─────
    subject_avg_best = {
        s["subject"]: normalize_importance(np.mean([s["importance"][m] for m in best_methods], axis=0))
        for s in full_summaries
    }

    # *** LOSO FIX: montage for subject S excludes S's own saliency ***
    subject_montages = loso_montage(subject_avg_best, FULL_CHANNELS, REDUCED_N_CHANNELS)

    # ───── Phase 2: reduced-channel training, one LOSO montage per subject ─────
    print(f"\n[{class_mode}] === REDUCED TRAINING: {REDUCED_N_CHANNELS} channels, LOSO montage per subject ===")
    for sid in SUBJECT_IDS:
        print(f"  S{sid:02d} montage: {', '.join(subject_montages[sid])}")

    fold_results, reduced_means, reduced_stds, red_grand_mean, red_grand_std = \
        run_reduced_training_loso(subject_montages, class_mode, tag="Saliency-LOSO")

    # ───── Report ─────
    report_lines.append(f"\n{'='*80}\nCLASS MODE: {class_mode}\n{'='*80}\n")
    for subj in full_summaries:
        sid = subj["subject"]
        report_lines.append(f"SUBJECT {sid:02d}")
        report_lines.append("-" * 60)
        for i, acc in enumerate(subj["fold_test_accs"], 1):
            report_lines.append(f"  Fold {i}: {acc*100:.2f}%")
        report_lines.append(f"  22ch Mean: {subj['subject_mean_acc']*100:.2f}% +/- {subj['subject_std_acc']*100:.2f}%\n")
        report_lines.append("  Correlations (all four report-only + decision metric):")
        for m in ATTR_METHODS:
            c = subj["correlations"][m]
            report_lines.append(
                f"    {m:>20} -> ERD_(L-R)={c['ERD_(L-R)']:+.3f} (used for ranking)  "
                f"ERD_L={c['ERD_L']:+.3f}  ERD_R={c['ERD_R']:+.3f}  ERD_comb={c['ERD_comb']:+.3f}"
            )
        report_lines.append(f"  Reduced ({REDUCED_N_CHANNELS}ch, LOSO montage): "
                             + ", ".join(subject_montages[sid]))
        reduced_mean = reduced_means[SUBJECT_IDS.index(sid)]
        report_lines.append(f"  Reduced Mean: {reduced_mean*100:.2f}%\n")

    report_lines.append(f"\nBest attribution methods used for channel selection "
                         f"(ranked by ERD_(L-R) only): {best_methods}")
    report_lines.append(f"GRAND 22ch:    {grand_mean*100:.2f}% +/- {grand_std*100:.2f}%")
    report_lines.append(f"GRAND reduced: {red_grand_mean*100:.2f}% +/- {red_grand_std*100:.2f}%")

    return full_means, grand_mean, reduced_means, red_grand_mean


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_lines = ["EEG-ITNet — HGD — SALIENCY-GUIDED CHANNEL REDUCTION ('Ours')",
                 "LOSO montage: each subject's reduced channels come only from the other 13 subjects.",
                 "Saliency computed from the validation set (not the test set).\n"]

    summary = {}
    for class_mode in CLASS_MODES:
        full_means, grand_mean, reduced_means, red_grand_mean = run_one_class_mode(class_mode, all_lines)
        summary[class_mode] = (grand_mean, red_grand_mean)

    report_path = os.path.join(OUTPUT_BASE, "saliency_reduced_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

    print("\n" + "=" * 80)
    print("FINAL SUMMARY — SALIENCY-GUIDED ('Ours')")
    print("=" * 80)
    for class_mode, (g, r) in summary.items():
        print(f"{class_mode}: 22ch {g*100:.2f}%  ->  reduced {r*100:.2f}%")
    print(f"\nReport saved: {report_path}")
