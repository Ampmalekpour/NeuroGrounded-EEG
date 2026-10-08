# eegitnet_hgd_2class_csp.py
#
# EEG-ITNet on HGD, 2class: CSP-ranked 12-channel montage (the "CSP" row).
#
# Method (fully specified for reproducibility):
#   * input   : the preprocessed 22-channel trials, whole window
#   * filters : mne.decoding.CSP(n_components=4, reg=None, log=False, norm_trace=False)
#   * score   : per channel, sum over the 4 retained spatial patterns of |pattern weight|
#   * 2-class : standard binary CSP;  4-class: one-vs-rest (one binary CSP per class, scores averaged)
#   * montage : leave-one-subject-out - subject S's 12 channels come only from the other subjects
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # folder holding eegitnet_hgd_common.py
os.environ["EEG_CLASS_MODE"] = "2class"           # this folder is the 2class experiment
os.environ.setdefault("EEG_RESULTS_DIR", os.path.join(HERE, "results"))

import numpy as np
from mne.decoding import CSP

import eegitnet_hgd_common as C

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


def main():
    print("\n" + "=" * 80 + "\nPHASE 1: PER-SUBJECT CSP SCORES (22 ch) - 2class\n" + "=" * 80)
    subject_scores = {}
    for sid in C.SUBJECT_IDS:
        X, y, _ = C.load_subject_windows(sid, C.FULL_CHANNELS, C.CLASS_MODE)
        subject_scores[sid] = csp_scores(X, y)
        top = C.top_k_channels(subject_scores[sid], C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
        print(f"  S{sid:02d} CSP top-{C.REDUCED_N_CHANNELS}: " + ", ".join(f"{c} ({v:.3f})" for c, v in top))

    montages = C.loso_montage(subject_scores, C.FULL_CHANNELS, C.REDUCED_N_CHANNELS)
    records, _ = C.run_experiment("csp_reduced", C.CLASS_MODE, montages,
                                  extra_meta={"selection": "CSP, LOSO montage",
                                              "csp_n_components": CSP_N_COMPONENTS,
                                              "csp_4class": "one-vs-rest, averaged"})
    C.write_text_report(os.path.join(C.experiment_dir("csp_reduced"), "csp_reduced_report.txt"),
                        "EEG-ITNet - HGD - CSP 12-channel (LOSO) - 2class", records, montages)
    s = C.summarize(records)
    print(f"\nCSP 2class: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}%")


if __name__ == "__main__":
    main()
