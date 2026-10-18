# check_hgd_data.py
#
# Diagnostic for the High-Gamma Dataset (Schirrmeister2017) as delivered by MOABB / braindecode.
# For every subject it prints, per recording (the "0train" and "1test" runs): duration, number of
# annotated trials per class, and a fingerprint of the signal. It then flags
#   * a test recording whose signal or event times are IDENTICAL to the training recording
#     (that would put the test trials into the training data),
#   * a test recording larger than the training recording,
#   * trials counts that differ from the usual ~880 training / ~160 test trials.
#
#   python analysis/check_hgd_data.py                 # subjects 1-14
#   python analysis/check_hgd_data.py 1 6 8           # some subjects
#
# Writes analysis/hgd_data_check.csv next to this file. Nothing is modified; the data are only read.

import os
import sys
import csv
import hashlib
import collections

import numpy as np

CLASSES = ("feet", "left_hand", "rest", "right_hand")


def fingerprint(raw, n=20000):
    data = raw.get_data(picks=[0, 1, 2], start=1000, stop=1000 + n)
    return hashlib.md5(np.round(data * 1e9).astype(np.int64).tobytes()).hexdigest()[:10]


def describe_run(base_ds):
    raw = base_ds.raw
    ann = raw.annotations
    counts = collections.Counter(d for d in ann.description if d in CLASSES)
    onsets = np.round(np.asarray(ann.onset)[[d in CLASSES for d in ann.description]], 3)
    return {
        "run": str(base_ds.description.get("run", "?")),
        "session": str(base_ds.description.get("session", "?")),
        "minutes": round(raw.n_times / raw.info["sfreq"] / 60, 1),
        "sfreq": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
        "n_trials": int(sum(counts.values())),
        "per_class": {c: int(counts.get(c, 0)) for c in CLASSES},
        "fingerprint": fingerprint(raw),
        "onsets": onsets,
    }


def inspect(subject, dataset):
    """Returns (rows, warnings) for one subject's dataset (anything with .datasets[i].raw/.description)."""
    runs = [describe_run(d) for d in dataset.datasets]
    warns = []
    by_name = {r["run"]: r for r in runs}
    tr = next((r for k, r in by_name.items() if "train" in k), None)
    te = next((r for k, r in by_name.items() if "test" in k), None)
    if tr is None or te is None:
        warns.append(f"S{subject:02d}: could not find both a 'train' and a 'test' run (found {list(by_name)})")
    else:
        if tr["fingerprint"] == te["fingerprint"]:
            warns.append(f"S{subject:02d}: TEST and TRAIN signals are IDENTICAL -> test trials are in the training data")
        if len(tr["onsets"]) and len(te["onsets"]) and np.array_equal(tr["onsets"], te["onsets"]):
            warns.append(f"S{subject:02d}: TEST and TRAIN event times are identical")
        if te["n_trials"] > tr["n_trials"]:
            warns.append(f"S{subject:02d}: the test run ({te['n_trials']} trials) is larger than the training run "
                         f"({tr['n_trials']})")
        if abs(te["n_trials"] - 160) > 40:
            warns.append(f"S{subject:02d}: {te['n_trials']} test trials (the usual number is about 160)")
        if abs(tr["n_trials"] - 880) > 120:
            warns.append(f"S{subject:02d}: {tr['n_trials']} training trials (the usual number is about 880)")
    return runs, warns


def main():
    subjects = [int(a) for a in sys.argv[1:]] or list(range(1, 15))
    from braindecode.datasets import MOABBDataset
    rows, all_warns = [], []
    print(f"{'subj':>4} {'run':>7} {'min':>6} {'Hz':>5} {'chan':>5} {'trials':>7}  per class (feet/left/rest/right)   fingerprint")
    for s in subjects:
        try:
            ds = MOABBDataset("Schirrmeister2017", subject_ids=[s])
        except Exception as exc:                                    # keep going: report what is readable
            print(f"S{s:02d}: could not load ({type(exc).__name__}: {exc})")
            continue
        runs, warns = inspect(s, ds)
        all_warns += warns
        for r in runs:
            pc = r["per_class"]
            print(f"{s:>4} {r['run']:>7} {r['minutes']:>6} {r['sfreq']:>5.0f} {r['n_channels']:>5} {r['n_trials']:>7}  "
                  f"{pc['feet']}/{pc['left_hand']}/{pc['rest']}/{pc['right_hand']}   {r['fingerprint']}")
            rows.append([s, r["run"], r["session"], r["minutes"], r["sfreq"], r["n_channels"], r["n_trials"],
                         *[pc[c] for c in CLASSES], r["fingerprint"]])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hgd_data_check.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "run", "session", "minutes", "sfreq", "n_channels", "n_trials", *CLASSES, "fingerprint"])
        w.writerows(rows)
    print("\n" + ("=" * 70))
    if all_warns:
        print("WARNINGS:")
        for w_ in all_warns:
            print("  - " + w_)
    else:
        print("No warnings: train and test recordings are distinct and have the usual sizes.")
    print(f"Table written to {out}")


if __name__ == "__main__":
    main()
