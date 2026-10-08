# deep4net_hgd_mi_reduced.py
#
# Mutual-information-ranked 12-channel montage for Deep4Net on HGD (the "MI" row).
#
# Method (fully specified for reproducibility):
#   * feature : log of the temporal variance of each channel, per trial (trials x channels)
#   * score   : sklearn mutual_info_classif (k-NN estimator, n_neighbors=3), each channel
#               scored independently against the class label (multi-class supported natively)
#   * montage : leave-one-subject-out

import os
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


def run(class_mode):
    print("\n" + "=" * 80 + f"\nPHASE 1: PER-SUBJECT MI SCORES (22 ch) - {class_mode}\n" + "=" * 80)
    subject_scores = {}
    for sid in C.SUBJECT_IDS:
        X, y, _ = C.load_subject_windows(sid, C.FULL_CHANNELS, class_mode)
        subject_scores[sid] = mi_scores(X, y)
        top = C.top_k_channels(subject_scores[sid], C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
        print(f"  S{sid:02d} MI top-{C.REDUCED_N_CHANNELS}: " + ", ".join(f"{c} ({v:.3f})" for c, v in top))

    montages = C.loso_montage(subject_scores, C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
    records, _ = C.run_experiment("mi_reduced", class_mode, montages,
                                  extra_meta={"selection": "mutual information, LOSO montage",
                                              "feature": "log temporal variance",
                                              "mi_n_neighbors": MI_N_NEIGHBORS})
    out_dir = C.experiment_dir("mi_reduced", class_mode)
    C.write_text_report(os.path.join(out_dir, "mi_reduced_report.txt"),
                        f"Deep4Net - HGD - MI 12-channel (LOSO) - {class_mode}", records, montages)
    return C.summarize(records)


if __name__ == "__main__":
    results = {cm: run(cm) for cm in C.CLASS_MODES}
    print("\n" + "=" * 80 + "\nSUMMARY - MUTUAL-INFORMATION REDUCTION\n" + "=" * 80)
    for cm, s in results.items():
        print(f"{cm}: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}%")
