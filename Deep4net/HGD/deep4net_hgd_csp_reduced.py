# deep4net_hgd_csp_reduced.py
#
# CSP-ranked 12-channel montage for Deep4Net on HGD (the "CSP" row).
#
# Method (fully specified for reproducibility):
#   * input   : the preprocessed 22-channel trials (4-38 Hz, EMS), whole window
#   * filters : mne.decoding.CSP(n_components=4, reg=None, log=False, norm_trace=False)
#   * score   : sum over the 4 retained spatial patterns of |pattern weight|, per channel
#   * 2-class : standard binary CSP
#   * 4-class : one-vs-rest - one binary CSP per class (class c vs all others), the four
#               channel-score vectors are averaged
#   * montage : leave-one-subject-out - subject S's 12 channels come only from the
#               other subjects' scores

import os
import numpy as np
from mne.decoding import CSP

import deep4net_hgd_common as C

CSP_N_COMPONENTS = 4


def csp_scores_binary(X, y):
    n_comp = min(CSP_N_COMPONENTS, X.shape[1])
    csp = CSP(n_components=n_comp, reg=None, log=False, norm_trace=False)
    csp.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64))
    patterns = np.asarray(csp.patterns_, dtype=np.float64)       # rows = patterns
    return C.normalize_importance(np.sum(np.abs(patterns[:n_comp, :]), axis=0))


def csp_scores(X, y):
    classes = np.unique(y)
    if len(classes) <= 2:
        return csp_scores_binary(X, y)
    per_class = [csp_scores_binary(X, (y == c).astype(np.int64)) for c in classes]
    return C.normalize_importance(np.mean(per_class, axis=0))


def run(class_mode):
    print("\n" + "=" * 80 + f"\nPHASE 1: PER-SUBJECT CSP SCORES (22 ch) - {class_mode}\n" + "=" * 80)
    subject_scores = {}
    for sid in C.SUBJECT_IDS:
        X, y, _ = C.load_subject_windows(sid, C.FULL_CHANNELS, class_mode)
        subject_scores[sid] = csp_scores(X, y)
        top = C.top_k_channels(subject_scores[sid], C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
        print(f"  S{sid:02d} CSP top-{C.REDUCED_N_CHANNELS}: " + ", ".join(f"{c} ({v:.3f})" for c, v in top))

    montages = C.loso_montage(subject_scores, C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
    records, _ = C.run_experiment("csp_reduced", class_mode, montages,
                                  extra_meta={"selection": "CSP, LOSO montage",
                                              "csp_n_components": CSP_N_COMPONENTS,
                                              "csp_4class": "one-vs-rest, averaged"})
    out_dir = C.experiment_dir("csp_reduced", class_mode)
    C.write_text_report(os.path.join(out_dir, "csp_reduced_report.txt"),
                        f"Deep4Net - HGD - CSP 12-channel (LOSO) - {class_mode}", records, montages)
    return C.summarize(records)


if __name__ == "__main__":
    results = {cm: run(cm) for cm in C.CLASS_MODES}
    print("\n" + "=" * 80 + "\nSUMMARY - CSP-GUIDED REDUCTION\n" + "=" * 80)
    for cm, s in results.items():
        print(f"{cm}: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}%")
