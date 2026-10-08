# eegitnet_hgd_2class_saliency.py
#
# MAIN SCRIPT - EEG-ITNet on HGD, 2class. Produces the "22ch" and "Ours" rows.
#
# Pipeline:
#   1. Train the full 22-channel model for every subject / seed / fold      -> results/full22/
#   2. Gradient attributions (grad, grad x input, integrated gradients, SmoothGrad) on each fold's
#      VALIDATION block only, overall and per true class. The test block is never used for ranking.
#   3. Saliency <-> ERD/ERS alignment with sign-invariant, class-specific metrics
#      (participant-level values and 95% CIs in results/full22/alignment_metrics.json)
#      and the ERD/ERS reference maps per subject.
#   4. Leave-one-subject-out montage: subject S's 12 channels come only from the other subjects' saliency.
#   5. Retrain from scratch on those montages                                -> results/saliency_reduced/
#
# Attribution-method choice (EEG_ATTR_SELECTION):
#   all  (default) average all four methods - the choice does NOT depend on ERD
#   best           EXPLORATORY: top-3 methods by mean 'magnitude' alignment of the OTHER subjects
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # folder holding eegitnet_hgd_common.py
os.environ["EEG_CLASS_MODE"] = "2class"           # this folder is the 2class experiment
os.environ.setdefault("EEG_RESULTS_DIR", os.path.join(HERE, "results"))

import json
import numpy as np

import eegitnet_hgd_common as C

ATTR_SELECTION = os.environ.get("EEG_ATTR_SELECTION", "all")
BEST_METHODS_COUNT = 3
SELECTION_METRIC = "magnitude_pearson"
METRIC_KEYS = ["magnitude_pearson", "magnitude_spearman", "class_specific_pearson",
               "class_specific_spearman", "lateralisation_pearson", "lateralisation_spearman",
               "legacy_signed_LR_pearson", "legacy_signed_LR_spearman"]
assert ATTR_SELECTION in ("all", "best"), "EEG_ATTR_SELECTION must be 'all' or 'best'"


def compute_alignment(subject_sal):
    metrics = {}
    for sid, by_method in subject_sal.items():
        erd = C.get_erd_maps(sid, C.FULL_CHANNELS)
        metrics[sid] = {m: C.alignment_metrics(v["all"], v["by_class"], erd, C.CLASS_MODE)
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


def save_alignment(out_dir, metrics):
    summary = {m: {k: C.summarize_metric([metrics[s][m][k] for s in metrics]) for k in METRIC_KEYS}
               for m in C.ATTR_METHODS}
    payload = {"class_mode": C.CLASS_MODE, "attribution_selection": ATTR_SELECTION,
               "note": ("magnitude/class_specific/lateralisation compare UNSIGNED saliency with "
                        "desynchronisation strength (-ERD); legacy_signed_LR is the old "
                        "corr(saliency, ERD_L - ERD_R) and its sign is NOT interpretable."),
               "per_subject": {str(s): metrics[s] for s in metrics}, "summary": summary}
    with open(os.path.join(out_dir, "alignment_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return summary


def save_figures(out_dir, subject_sal):
    topo = C.make_topo_info(C.FULL_CHANNELS)
    for sid, by_method in subject_sal.items():
        for m, v in by_method.items():
            C.save_array(os.path.join(out_dir, f"saliency_{m}_S{sid:02d}.npy"), v["all"])
            C.plot_saliency_topomap(v["all"], topo, f"S{sid:02d} {m}",
                                    os.path.join(out_dir, f"saliency_{m}_S{sid:02d}.png"))
            for c, vec in v["by_class"].items():
                C.save_array(os.path.join(out_dir, f"saliency_{m}_class{c}_S{sid:02d}.npy"), vec)
        erd = C.get_erd_maps(sid, C.FULL_CHANNELS)
        for k, vec in erd.items():
            C.save_array(os.path.join(out_dir, f"{k}_S{sid:02d}.npy"), vec)
        C.plot_erd_ers_topomaps(erd["ERD_L"], erd["ERD_R"], erd["ERD_(L-R)"], erd["ERD_comb"],
                                topo, os.path.join(out_dir, f"erd_S{sid:02d}.png"))
    for m in C.ATTR_METHODS:
        avg = C.normalize_importance(np.mean([subject_sal[s][m]["all"] for s in subject_sal], axis=0))
        C.plot_saliency_topomap(avg, topo, f"Group mean {m}", os.path.join(out_dir, f"group_{m}.png"))


def main():
    # Phase 1: full 22-channel models + validation-set saliency
    full_records, subject_sal = C.run_experiment(
        "full22", C.CLASS_MODE, C.full_montages(), collect_saliency=True)
    full_dir = C.experiment_dir("full22")
    C.write_text_report(os.path.join(full_dir, "full22_report.txt"),
                        "EEG-ITNet - HGD - FULL 22-CHANNEL BASELINE - 2class", full_records)

    metrics = compute_alignment(subject_sal)
    align_summary = save_alignment(full_dir, metrics)
    save_figures(full_dir, subject_sal)

    # Phase 2: LOSO montages and retraining
    montages, chosen = saliency_montages(subject_sal, metrics)
    red_records, _ = C.run_experiment(
        "saliency_reduced", C.CLASS_MODE, montages,
        extra_meta={"selection": "gradient saliency (validation blocks), LOSO montage",
                    "attribution_selection": ATTR_SELECTION,
                    "methods_per_heldout_subject": {str(k): v for k, v in chosen.items()}})
    red_dir = C.experiment_dir("saliency_reduced")

    lines = ["", f"Attribution methods used: {ATTR_SELECTION}",
             "Alignment (participant-level mean [95% CI]):"]
    for m in C.ATTR_METHODS:
        for k in ("magnitude_pearson", "class_specific_pearson", "lateralisation_pearson"):
            s = align_summary[m][k]
            lines.append(f"  {m:>20} {k:<24} {s['mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] n={s['n']}")
    C.write_text_report(os.path.join(red_dir, "saliency_reduced_report.txt"),
                        "EEG-ITNet - HGD - SALIENCY 12-channel (LOSO) - 2class", red_records, montages, lines)

    full, red = C.summarize(full_records), C.summarize(red_records)
    print("\n" + "=" * 80 + "\nFINAL SUMMARY - EEG-ITNet HGD 2class\n" + "=" * 80)
    print(f"22ch {full['grand_mean']*100:.2f}%  ->  saliency 12ch {red['grand_mean']*100:.2f}%  "
          f"(chance {full['chance_acc']*100:.1f}%, majority {full['majority_class_acc_mean']*100:.1f}%)")


if __name__ == "__main__":
    main()
