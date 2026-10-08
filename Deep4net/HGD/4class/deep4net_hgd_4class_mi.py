# deep4net_hgd_4class_mi.py
#
# Deep4Net on HGD, 4class: mutual-information-ranked 12-channel montage (the "MI" row).
#
# Method (fully specified for reproducibility):
#   * feature : log of the temporal variance of each channel, per trial (trials x channels)
#   * score   : sklearn mutual_info_classif (k-NN estimator, n_neighbors=3), each channel scored
#               independently against the class label (multi-class supported natively)
#   * montage : leave-one-subject-out
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # folder holding deep4net_hgd_common.py
os.environ["EEG_CLASS_MODE"] = "4class"           # this folder is the 4class experiment
os.environ.setdefault("EEG_RESULTS_DIR", os.path.join(HERE, "results"))

import numpy as np
from sklearn.feature_selection import mutual_info_classif

import deep4net_hgd_common as C

MI_N_NEIGHBORS = 3
MI_SEED = 42


def band_power_features(X):
    return np.log(np.var(X.astype(np.float64), axis=2) + 1e-12)


def mi_scores(X, y):
    scores = mutual_info_classif(band_power_features(X), y, discrete_features=False,
                                 n_neighbors=MI_N_NEIGHBORS, random_state=MI_SEED)
    return C.normalize_importance(scores)


def main():
    print("\n" + "=" * 80 + "\nPHASE 1: PER-SUBJECT MI SCORES (22 ch) - 4class\n" + "=" * 80)
    subject_scores = {}
    for sid in C.SUBJECT_IDS:
        X, y, _ = C.load_subject_windows(sid, C.FULL_CHANNELS, C.CLASS_MODE)
        subject_scores[sid] = mi_scores(X, y)
        top = C.top_k_channels(subject_scores[sid], C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
        print(f"  S{sid:02d} MI top-{C.REDUCED_N_CHANNELS}: " + ", ".join(f"{c} ({v:.3f})" for c, v in top))

    montages = C.loso_montage(subject_scores, C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
    records, _ = C.run_experiment("mi_reduced", C.CLASS_MODE, montages,
                                  extra_meta={"selection": "mutual information, LOSO montage",
                                              "feature": "log temporal variance",
                                              "mi_n_neighbors": MI_N_NEIGHBORS})
    C.write_text_report(os.path.join(C.experiment_dir("mi_reduced"), "mi_reduced_report.txt"),
                        "Deep4Net - HGD - MI 12-channel (LOSO) - 4class", records, montages)
    s = C.summarize(records)
    print(f"\nMI 4class: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}%")


if __name__ == "__main__":
    main()
