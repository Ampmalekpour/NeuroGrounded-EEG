# eegnetv4_bnci_4class_tune.py
#
# EEGNetv4 on BNCI2014-001, 4class: FAIR, validation-only choice of the training/preprocessing profile.
#
# Every candidate profile in TUNE_CANDIDATES (eegnetv4_bnci_common.py) is trained on the 22-channel data for
# every subject and fold; only the VALIDATION accuracy is recorded (the test block is never scored).
# For each subject S the profile with the best mean validation accuracy over the OTHER subjects is
# chosen (leave-one-subject-out), so no data of S influence S's settings. The choice is written to
# results/selected_profile.json and used automatically by every other script in this folder, for
# every method, so the comparison between methods stays fair.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # folder holding eegnetv4_bnci_common.py
os.environ["EEG_CLASS_MODE"] = "4class"           # this folder is the 4class experiment
os.environ.setdefault("EEG_RESULTS_DIR", os.path.join(
    HERE, "results" if os.environ.get("EEG_PROTOCOL", "session") == "blocks" else "results_session"))

import csv
import json
import numpy as np

import eegnetv4_bnci_common as C

CANDIDATES = [p for p in os.environ.get("EEG_TUNE_CANDIDATES", ",".join(C.TUNE_CANDIDATES)).split(",") if p]


def main():
    val_acc = {}                                  # profile -> subject -> mean validation accuracy
    seed = C.SEEDS[0]
    for name in CANDIDATES:
        cfg = C.get_cfg(name=name)
        val_acc[name] = {}
        for sid in C.SUBJECT_IDS:
            X, y, info = C.load_subject_windows(sid, C.FULL_CHANNELS, C.CLASS_MODE, cfg)
            accs = []
            for fold_i, (tr, va, te) in enumerate(C.make_folds(len(y), info), start=1):
                res = C.train_one_fold(X, y, tr, va, te, C.fold_seed(seed, sid, fold_i),
                                       tag=f"[tune {name}] S{sid:02d} F{fold_i}", cfg=cfg,
                                       evaluate_test=False)
                accs.append(res["val_acc"])
            val_acc[name][sid] = float(np.mean(accs))
            print(f"[tune] {name:>14} S{sid:02d}: mean validation accuracy {val_acc[name][sid]*100:.2f}%")

    ids = list(C.SUBJECT_IDS)
    per_subject = {}
    for sid in ids:
        others = [s for s in ids if s != sid] or ids
        score = {n: float(np.mean([val_acc[n][o] for o in others])) for n in CANDIDATES}
        per_subject[str(sid)] = max(CANDIDATES, key=lambda n: score[n])      # ties -> listed first
    overall_score = {n: float(np.mean([val_acc[n][s] for s in ids])) for n in CANDIDATES}
    overall = max(CANDIDATES, key=lambda n: overall_score[n])

    os.makedirs(C.RESULTS_ROOT, exist_ok=True)
    selection = {"class_mode": C.CLASS_MODE, "rule": "leave-one-subject-out mean validation accuracy",
                 "per_subject": per_subject, "overall": overall,
                 "unanimous": len(set(per_subject.values())) == 1,
                 "validation_accuracy": {n: {str(s): a for s, a in v.items()} for n, v in val_acc.items()},
                 "mean_validation_accuracy": overall_score, "seed": seed, "candidates": CANDIDATES}
    with open(C.SELECTED_PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=1)
    with open(os.path.join(C.RESULTS_ROOT, "tuning_validation_accuracy.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["profile"] + [f"S{s:02d}" for s in ids] + ["mean"])
        for n in CANDIDATES:
            w.writerow([n] + [f"{val_acc[n][s]*100:.2f}" for s in ids] + [f"{overall_score[n]*100:.2f}"])

    print("\n" + "=" * 80 + "\nPROFILE SELECTION - 4class (validation accuracy only)\n" + "=" * 80)
    for n in CANDIDATES:
        print(f"  {n:>14}: {overall_score[n]*100:.2f}%")
    print(f"chosen per subject (LOSO): {per_subject}")
    print(f"unanimous: {selection['unanimous']} | written to {C.SELECTED_PROFILE_FILE}")


if __name__ == "__main__":
    main()
