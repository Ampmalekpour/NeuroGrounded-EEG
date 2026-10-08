# deep4net_hgd_controls_reduced.py
#
# Control montages of the same size (12 channels) that the saliency-ranked montage must
# be compared with (reviewer: "does the network explanation add anything?"):
#
#   sensorimotor  a fixed, symmetric montage over the sensorimotor strip chosen a priori
#   random        N_RANDOM random 12-channel subsets per subject (seeded)
#   erd           ERD-only ranking: channels with the largest mean |ERD| (left & right hand)
#                 of the OTHER subjects (LOSO) - tests whether saliency adds value beyond the
#                 physiological reference itself
#
# Usage:  python deep4net_hgd_controls_reduced.py sensorimotor|random|erd [...]

import os
import sys
import numpy as np

import deep4net_hgd_common as C

SENSORIMOTOR_12 = ['FC3', 'FC1', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CPz', 'CP4']
N_RANDOM = 5
RANDOM_SEED = 2024
assert len(SENSORIMOTOR_12) == C.REDUCED_N_CHANNELS and set(SENSORIMOTOR_12) <= set(C.FULL_CHANNELS)


def sensorimotor_montages():
    return {sid: list(SENSORIMOTOR_12) for sid in C.SUBJECT_IDS}


def random_montages(draw):
    rng = np.random.default_rng(RANDOM_SEED + draw)
    return {sid: [C.FULL_CHANNELS[i] for i in sorted(
        rng.choice(len(C.FULL_CHANNELS), C.REDUCED_N_CHANNELS, replace=False))]
        for sid in C.SUBJECT_IDS}


def erd_montages():
    scores = {}
    for sid in C.SUBJECT_IDS:
        erd = C.get_erd_maps(sid, C.FULL_CHANNELS)
        scores[sid] = C.normalize_importance(0.5 * (np.abs(erd["ERD_L"]) + np.abs(erd["ERD_R"])))
    return C.loso_montage(scores, C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)


def run_one(name, montages, class_mode, meta):
    records, _ = C.run_experiment(name, class_mode, montages, extra_meta=meta)
    out_dir = C.experiment_dir(name, class_mode)
    C.write_text_report(os.path.join(out_dir, f"{name}_report.txt"),
                        f"Deep4Net - HGD - control '{name}' - {class_mode}", records, montages)
    return C.summarize(records)


def run(control, class_mode):
    if control == "sensorimotor":
        return {"control_sensorimotor": run_one("control_sensorimotor", sensorimotor_montages(), class_mode,
                                               {"selection": "fixed sensorimotor montage"})}
    if control == "erd":
        return {"control_erd": run_one("control_erd", erd_montages(), class_mode,
                                      {"selection": "ERD-only ranking, LOSO"})}
    if control == "random":
        return {f"control_random_draw{d}": run_one(f"control_random_draw{d}", random_montages(d), class_mode,
                                                  {"selection": "random montage", "draw": d})
                for d in range(N_RANDOM)}
    raise SystemExit("control must be one of: sensorimotor, random, erd")


if __name__ == "__main__":
    controls = sys.argv[1:] or ["sensorimotor", "random", "erd"]
    for control in controls:
        for cm in C.CLASS_MODES:
            for name, s in run(control, cm).items():
                print(f"{name} [{cm}]: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}%")
