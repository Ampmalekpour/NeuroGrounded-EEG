# deep4net_hgd_saliency_reduced.py
#
# Saliency-guided 12-channel montage for Deep4Net on HGD (the "Ours" row), plus the
# saliency <-> ERD/ERS alignment analysis.
#
# Pipeline (per class mode):
#   1. Train the full 22-channel model for every subject / seed / fold.
#   2. Compute gradient-based attributions (grad, grad x input, integrated gradients,
#      SmoothGrad) on each fold's VALIDATION block only, overall and per true class.
#      The test block is never used for channel ranking.
#   3. Alignment of each subject's saliency with that subject's ERD/ERS maps using
#      sign-invariant, class-specific metrics (see alignment_metrics in the common module);
#      participant-level values and CIs are written to alignment_metrics.json.
#   4. Leave-one-subject-out montage: subject S's 12 channels come only from the other
#      subjects' saliency.
#   5. Retrain from scratch on those montages.
#
# Attribution-method choice (EEG_ATTR_SELECTION):
#   all  (default) average all four methods - the choice does NOT depend on ERD, so
#                  nothing is selected and then "validated" against the same reference
#   best           EXPLORATORY: keep the 3 methods with the highest mean 'magnitude' alignment
#                  computed from the OTHER subjects only, per held-out subject.

import os
import json
import numpy as np

import deep4net_hgd_common as C

ATTR_SELECTION = os.environ.get("EEG_ATTR_SELECTION", "all")
BEST_METHODS_COUNT = 3
SELECTION_METRIC = "magnitude_pearson"
METRIC_KEYS = ["magnitude_pearson", "magnitude_spearman", "class_specific_pearson",
               "class_specific_spearman", "lateralisation_pearson", "lateralisation_spearman",
               "legacy_signed_LR_pearson", "legacy_signed_LR_spearman"]
assert ATTR_SELECTION in ("all", "best"), "EEG_ATTR_SELECTION must be 'all' or 'best'"


def compute_alignment(subject_sal, class_mode):
    metrics = {}
    for sid, by_method in subject_sal.items():
        erd = C.get_erd_maps(sid, C.FULL_CHANNELS)
        metrics[sid] = {m: C.alignment_metrics(v["all"], v["by_class"], erd, class_mode)
                        for m, v in by_method.items()}
    return metrics


def saliency_montages(subject_sal, metrics):
    ids = list(subject_sal.keys())
    montages, chosen_by_subject = {}, {}
    for held_out in ids:
        others = [s for s in ids if s != held_out]
        if ATTR_SELECTION == "best":
            score = {m: float(np.nanmean([metrics[o][m][SELECTION_METRIC] for o in others]))
                     for m in C.ATTR_METHODS}
            chosen = sorted(score, key=score.get, reverse=True)[:BEST_METHODS_COUNT]
        else:
            chosen = list(C.ATTR_METHODS)
        chosen_by_subject[held_out] = chosen
        per_subject = [C.normalize_importance(np.mean([subject_sal[o][m]["all"] for m in chosen], axis=0))
                       for o in others]
        avg = C.normalize_importance(np.mean(per_subject, axis=0))
        montages[held_out] = [ch for ch, _ in C.top_k_channels(avg, C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)]
    return montages, chosen_by_subject


def save_alignment(out_dir, metrics, class_mode):
    summary = {}
    for m in C.ATTR_METHODS:
        summary[m] = {k: C.summarize_metric([metrics[s][m][k] for s in metrics]) for k in METRIC_KEYS}
    payload = {"class_mode": class_mode, "attribution_selection": ATTR_SELECTION,
               "note": ("magnitude/class_specific/lateralisation compare UNSIGNED saliency with "
                        "desynchronisation strength (-ERD); legacy_signed_LR is the old "
                        "corr(saliency, ERD_L - ERD_R) and its sign is NOT interpretable."),
               "per_subject": {str(s): metrics[s] for s in metrics}, "summary": summary}
    with open(os.path.join(out_dir, "alignment_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return summary


def save_saliency_files(out_dir, subject_sal):
    topo = C.make_topo_info(C.FULL_CHANNELS)
    for sid, by_method in subject_sal.items():
        for m, v in by_method.items():
            C.save_array(os.path.join(out_dir, f"saliency_{m}_S{sid:02d}.npy"), v["all"])
            C.plot_saliency_topomap(v["all"], topo, f"S{sid:02d} {m}",
                                    os.path.join(out_dir, f"saliency_{m}_S{sid:02d}.png"))
            for c, vec in v["by_class"].items():
                C.save_array(os.path.join(out_dir, f"saliency_{m}_class{c}_S{sid:02d}.npy"), vec)
    for m in C.ATTR_METHODS:
        avg = C.normalize_importance(np.mean([subject_sal[s][m]["all"] for s in subject_sal], axis=0))
        C.plot_saliency_topomap(avg, topo, f"Group mean {m}", os.path.join(out_dir, f"group_{m}.png"))


def run(class_mode):
    # Phase 1: full 22-channel models + validation-set saliency
    full_records, subject_sal = C.run_experiment(
        "saliency_full22", class_mode, C.full_montages(), collect_saliency=True)
    full_dir = C.experiment_dir("saliency_full22", class_mode)

    metrics = compute_alignment(subject_sal, class_mode)
    align_summary = save_alignment(full_dir, metrics, class_mode)
    save_saliency_files(full_dir, subject_sal)

    # Phase 2: LOSO montages and retraining
    montages, chosen = saliency_montages(subject_sal, metrics)
    red_records, _ = C.run_experiment(
        "saliency_reduced", class_mode, montages,
        extra_meta={"selection": "gradient saliency (validation blocks), LOSO montage",
                    "attribution_selection": ATTR_SELECTION,
                    "methods_per_heldout_subject": {str(k): v for k, v in chosen.items()}})
    red_dir = C.experiment_dir("saliency_reduced", class_mode)

    lines = ["", f"Attribution methods used: {ATTR_SELECTION}", "Alignment (participant-level mean [95% CI]):"]
    for m in C.ATTR_METHODS:
        for k in ("magnitude_pearson", "class_specific_pearson", "lateralisation_pearson"):
            s = align_summary[m][k]
            lines.append(f"  {m:>20} {k:<24} {s['mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] n={s['n']}")
    C.write_text_report(os.path.join(red_dir, "saliency_reduced_report.txt"),
                        f"Deep4Net - HGD - SALIENCY 12-channel (LOSO) - {class_mode}",
                        red_records, montages, lines)
    return C.summarize(full_records), C.summarize(red_records)


if __name__ == "__main__":
    results = {cm: run(cm) for cm in C.CLASS_MODES}
    print("\n" + "=" * 80 + "\nFINAL SUMMARY - SALIENCY-GUIDED ('Ours')\n" + "=" * 80)
    for cm, (full, red) in results.items():
        print(f"{cm}: 22ch {full['grand_mean']*100:.2f}%  ->  12ch {red['grand_mean']*100:.2f}%")
