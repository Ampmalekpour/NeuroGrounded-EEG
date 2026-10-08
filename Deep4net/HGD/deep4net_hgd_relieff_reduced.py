# deep4net_hgd_relieff_reduced.py
#
# ReliefF-ranked 12-channel montage for Deep4Net on HGD (the "RLF" row).
#
# Implements ReliefF (Kononenko, 1994) as in manuscript Eq. 8, for any number of classes:
#   W_j <- W_j - (1/(m k)) sum_k diff(j, R, H_k)
#              + (1/(m k)) sum_{C != class(R)} [P(C) / (1 - P(class(R)))] sum_k diff(j, R, M_k(C))
# i.e. k nearest HITS of the same class, and k nearest MISSES from EACH other class,
# weighted by the class priors. (The earlier script pooled misses over all other classes.)
#   * feature : log temporal variance per channel, min-max scaled to [0, 1] (so diff() is in [0, 1])
#   * distance: Euclidean over all 22 channel features
#   * k = 10, every trial is used as R (m = n_trials)
#   * montage : leave-one-subject-out

import os
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
            weight = prior[c] / (1.0 - prior[ci])
            w += weight * np.abs(feats[i] - feats[near]).mean(axis=0) / n
    return C.normalize_importance(w)


def run(class_mode):
    print("\n" + "=" * 80 + f"\nPHASE 1: PER-SUBJECT RELIEFF SCORES (22 ch) - {class_mode}\n" + "=" * 80)
    subject_scores = {}
    for sid in C.SUBJECT_IDS:
        X, y, _ = C.load_subject_windows(sid, C.FULL_CHANNELS, class_mode)
        subject_scores[sid] = relieff_scores(X, y)
        top = C.top_k_channels(subject_scores[sid], C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
        print(f"  S{sid:02d} ReliefF top-{C.REDUCED_N_CHANNELS}: " + ", ".join(f"{c} ({v:.3f})" for c, v in top))

    montages = C.loso_montage(subject_scores, C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
    records, _ = C.run_experiment("relieff_reduced", class_mode, montages,
                                  extra_meta={"selection": "ReliefF, LOSO montage",
                                              "feature": "log temporal variance, min-max scaled",
                                              "relieff_k": RELIEFF_K,
                                              "relieff_misses": "k nearest per other class, prior-weighted"})
    out_dir = C.experiment_dir("relieff_reduced", class_mode)
    C.write_text_report(os.path.join(out_dir, "relieff_reduced_report.txt"),
                        f"Deep4Net - HGD - ReliefF 12-channel (LOSO) - {class_mode}", records, montages)
    return C.summarize(records)


if __name__ == "__main__":
    results = {cm: run(cm) for cm in C.CLASS_MODES}
    print("\n" + "=" * 80 + "\nSUMMARY - RELIEFF REDUCTION\n" + "=" * 80)
    for cm, s in results.items():
        print(f"{cm}: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}%")
