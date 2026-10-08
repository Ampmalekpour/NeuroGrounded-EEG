# deep4net_hgd_2class_controls.py
#
# Deep4Net on HGD, 2class: control montages of the same size (12 channels) that the saliency-ranked
# montage must be compared with ("does the network explanation add anything?"):
#   sensorimotor  a fixed, symmetric montage over the sensorimotor strip chosen a priori
#   random        N_RANDOM random 12-channel subsets per subject (seeded)
#   erd           ERD-only ranking: largest mean |ERD| (left & right hand) of the OTHER subjects (LOSO)
#
# Usage:  python deep4net_hgd_2class_controls.py sensorimotor random erd
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # folder holding deep4net_hgd_common.py
os.environ["EEG_CLASS_MODE"] = "2class"           # this folder is the 2class experiment
os.environ.setdefault("EEG_RESULTS_DIR", os.path.join(HERE, "results"))

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


def run_one(name, montages, meta):
    records, _ = C.run_experiment(name, C.CLASS_MODE, montages, extra_meta=meta)
    C.write_text_report(os.path.join(C.experiment_dir(name), f"{name}_report.txt"),
                        f"Deep4Net - HGD - control '{name}' - 2class", records, montages)
    s = C.summarize(records)
    print(f"{name} [2class]: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}%")


def main(controls):
    for control in controls:
        if control == "sensorimotor":
            run_one("control_sensorimotor", sensorimotor_montages(), {"selection": "fixed sensorimotor montage"})
        elif control == "erd":
            run_one("control_erd", erd_montages(), {"selection": "ERD-only ranking, LOSO"})
        elif control == "random":
            for d in range(N_RANDOM):
                run_one(f"control_random_draw{d}", random_montages(d), {"selection": "random montage", "draw": d})
        else:
            raise SystemExit("control must be one of: sensorimotor, random, erd")


if __name__ == "__main__":
    main(sys.argv[1:] or ["sensorimotor", "random", "erd"])
