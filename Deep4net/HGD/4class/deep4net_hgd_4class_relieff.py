# deep4net_hgd_4class_relieff.py
#
# Deep4Net on HGD, 4class: ReliefF-ranked 12-channel montage (the "RLF" row).
#
# ReliefF (Kononenko, 1994) as in manuscript Eq. 8, for any number of classes:
#   W_j <- W_j - (1/(m k)) sum_k diff(j, R, H_k)
#              + (1/(m k)) sum_{C != class(R)} [P(C) / (1 - P(class(R)))] sum_k diff(j, R, M_k(C))
# i.e. k nearest HITS of the same class and k nearest MISSES from EACH other class, prior-weighted.
#   * feature : log temporal variance per channel, min-max scaled to [0, 1] (diff() in [0, 1])
#   * distance: Euclidean over all 22 channel features;  k = 10;  every trial is used as R
#   * montage : leave-one-subject-out
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # folder holding deep4net_hgd_common.py
os.environ["EEG_CLASS_MODE"] = "4class"           # this folder is the 4class experiment
os.environ.setdefault("EEG_RESULTS_DIR", os.path.join(HERE, "results"))

import numpy as np

import deep4net_hgd_common as C

RELIEFF_K = 10


def relieff_scores(X, y, k=RELIEFF_K):
    feats = np.log(np.var(X.astype(np.float64), axis=2) + 1e-12)
    feats = (feats - feats.min(0, keepdims=True)) / (feats.max(0, keepdims=True)
                                                     - feats.min(0, keepdims=True) + 1e-12)
    n, n_ch = feats.shape
    y = np.asarray(y)
    classes = np.unique(y)
    prior = {c: float(np.mean(y == c)) for c in classes}
    class_idx = {c: np.where(y == c)[0] for c in classes}
    dist = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)

    w = np.zeros(n_ch)
    for i in range(n):
        ci = y[i]
        hits = class_idx[ci][class_idx[ci] != i]
        if len(hits):
            near = hits[np.argsort(dist[i, hits])[:k]]
            w -= np.abs(feats[i] - feats[near]).mean(axis=0) / n
        for c in classes:
            if c == ci or len(class_idx[c]) == 0:
                continue
            near = class_idx[c][np.argsort(dist[i, class_idx[c]])[:k]]
            w += prior[c] / (1.0 - prior[ci]) * np.abs(feats[i] - feats[near]).mean(axis=0) / n
    return C.normalize_importance(w)


def main():
    print("\n" + "=" * 80 + "\nPHASE 1: PER-SUBJECT RELIEFF SCORES (22 ch) - 4class\n" + "=" * 80)
    subject_scores = {}
    for sid in C.SUBJECT_IDS:
        X, y, _ = C.load_subject_windows(sid, C.FULL_CHANNELS, C.CLASS_MODE)
        subject_scores[sid] = relieff_scores(X, y)
        top = C.top_k_channels(subject_scores[sid], C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
        print(f"  S{sid:02d} ReliefF top-{C.REDUCED_N_CHANNELS}: " + ", ".join(f"{c} ({v:.3f})" for c, v in top))

    montages = C.loso_montage(subject_scores, C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
    records, _ = C.run_experiment("relieff_reduced", C.CLASS_MODE, montages,
                                  extra_meta={"selection": "ReliefF, LOSO montage",
                                              "feature": "log temporal variance, min-max scaled",
                                              "relieff_k": RELIEFF_K,
                                              "relieff_misses": "k nearest per other class, prior-weighted"})
    C.write_text_report(os.path.join(C.experiment_dir("relieff_reduced"), "relieff_reduced_report.txt"),
                        "Deep4Net - HGD - ReliefF 12-channel (LOSO) - 4class", records, montages)
    s = C.summarize(records)
    print(f"\nReliefF 4class: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}%")


if __name__ == "__main__":
    main()
