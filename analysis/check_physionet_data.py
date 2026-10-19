# check_physionet_data.py
#
# Diagnostic for the PhysioNet EEG Motor Movement/Imagery dataset (MOABB "PhysionetMI", imagery runs)
# as delivered by MOABB / braindecode. Run it ONCE before the PhysioNet suite: it also downloads every
# subject (about 1.5 GB), so the long run does not stop to download or to ask for a data folder.
#
# For every subject it reports, per imagery run: MOABB run name, the run type (left/right fist or
# fists/feet), sampling rate, number of channels, duration and the number of cues per class. It flags
#   * a sampling rate other than 160 Hz (the pipeline resamples, but it is reported),
#   * a run with a missing class or with no cues, fewer than 6 runs, or very few trials,
#   * irregular cue spacing (imagery period + rest normally ~ 8.3 s apart),
# and prints, for the record, what the ORIGINAL PhysioNet code kept: it built windows WITHOUT a
# mapping (braindecode then numbers the event names alphabetically: feet=0, hands=1, left_hand=2,
# rest=3, right_hand=4) and removed label 0, i.e. it dropped FEET and kept REST.
#
#   python analysis/check_physionet_data.py                 # subjects 1-109
#   python analysis/check_physionet_data.py 1 2 88 104      # some subjects
#
# Writes analysis/physionet_data_check.csv. Nothing is modified; the data are only read.

import os
import sys
import csv
import collections

import numpy as np

CLASSES = ("left_hand", "right_hand", "hands", "feet")
ALPHABETICAL = sorted(CLASSES + ("rest",))          # braindecode's default label order


def ensure_eegbci_path():
    import mne
    key = "MNE_DATASETS_EEGBCI_PATH"
    cur = mne.get_config(key)
    if cur is None:
        base = mne.get_config("MNE_DATA") or os.path.join(os.path.expanduser("~"), "mne_data")
        mne.set_config(key, os.path.abspath(base))
    elif cur != os.path.abspath(cur):
        mne.set_config(key, os.path.abspath(cur))


def describe_run(base_ds):
    raw = base_ds.raw
    ann = raw.annotations
    desc = np.asarray(ann.description)
    counts = collections.Counter(d for d in desc if d in CLASSES + ("rest",))
    onsets = np.sort(np.asarray(ann.onset)[np.isin(desc, CLASSES)])
    gaps = np.diff(onsets) if len(onsets) > 1 else np.array([np.nan])
    kind = "left/right" if counts.get("left_hand", 0) + counts.get("right_hand", 0) else "fists/feet"
    return {
        "run": str(base_ds.description.get("run", "?")), "type": kind,
        "sfreq": float(raw.info["sfreq"]), "n_channels": len(raw.ch_names),
        "minutes": round(raw.n_times / raw.info["sfreq"] / 60, 2),
        "per_class": {c: int(counts.get(c, 0)) for c in CLASSES}, "rest": int(counts.get("rest", 0)),
        "cue_gap_median_s": float(np.nanmedian(gaps)), "cue_gap_min_s": float(np.nanmin(gaps)),
        "cue_duration_s": float(np.median(np.asarray(ann.duration)[np.isin(desc, CLASSES)]))
        if np.isin(desc, CLASSES).any() else float("nan"),
    }


def inspect(subject, dataset):
    """Returns (rows, warnings) for one subject (anything with .datasets[i].raw / .description)."""
    runs = [describe_run(d) for d in dataset.datasets]
    warns = []
    if len(runs) != 6:
        warns.append(f"S{subject:03d}: {len(runs)} imagery runs (expected 6)")
    for r in runs:
        pc = r["per_class"]
        need = ("left_hand", "right_hand") if r["type"] == "left/right" else ("hands", "feet")
        if any(pc[c] == 0 for c in need):
            warns.append(f"S{subject:03d} run {r['run']}: a class has no cues ({pc})")
        if abs(r["sfreq"] - 160) > 1e-6:
            warns.append(f"S{subject:03d} run {r['run']}: sampling rate {r['sfreq']:g} Hz (resampled to 160 Hz)")
        if np.isfinite(r["cue_gap_min_s"]) and r["cue_gap_min_s"] < 6.0:
            warns.append(f"S{subject:03d} run {r['run']}: cues only {r['cue_gap_min_s']:.2f} s apart "
                         f"(irregular timing)")
    totals = {c: sum(r["per_class"][c] for r in runs) for c in CLASSES}
    if min(totals.values()) < 15:
        warns.append(f"S{subject:03d}: few trials per class {totals}")
    return runs, totals, warns


def main():
    subjects = [int(a) for a in sys.argv[1:]] or list(range(1, 110))
    ensure_eegbci_path()
    from braindecode.datasets import MOABBDataset
    rows, all_warns, old_rule = [], [], []
    print(f"{'subj':>4} {'run':>4} {'type':>10} {'Hz':>5} {'chan':>4} {'min':>5}  "
          f"left/right/hands/feet  rest  cue gap (median/min s)")
    for s in subjects:
        try:
            try:
                ds = MOABBDataset("PhysionetMI", subject_ids=[s], dataset_kwargs={"imagined": True, "executed": False})
            except TypeError:
                ds = MOABBDataset("PhysionetMI", subject_ids=[s])
        except Exception as exc:                                    # keep going: report what is readable
            print(f"S{s:03d}: could not load ({type(exc).__name__}: {exc})")
            all_warns.append(f"S{s:03d}: could not load ({type(exc).__name__})")
            continue
        runs, totals, warns = inspect(s, ds)
        all_warns += warns
        for r in runs:
            pc = r["per_class"]
            print(f"{s:>4} {r['run']:>4} {r['type']:>10} {r['sfreq']:>5.0f} {r['n_channels']:>4} {r['minutes']:>5}  "
                  f"{pc['left_hand']:>4}/{pc['right_hand']}/{pc['hands']}/{pc['feet']}        {r['rest']:>3}   "
                  f"{r['cue_gap_median_s']:.2f} / {r['cue_gap_min_s']:.2f}")
            rows.append([s, r["run"], r["type"], r["sfreq"], r["n_channels"], r["minutes"],
                         *[pc[c] for c in CLASSES], r["rest"], round(r["cue_gap_median_s"], 3),
                         round(r["cue_gap_min_s"], 3), round(r["cue_duration_s"], 3)])
        rest = sum(r["rest"] for r in runs)
        kept = {c: totals.get(c, rest if c == "rest" else 0) for c in ALPHABETICAL[1:]}   # old: label 0 removed
        old_rule.append(rest / max(1, sum(kept.values())))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physionet_data_check.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "run", "type", "sfreq", "n_channels", "minutes", *CLASSES, "rest",
                    "cue_gap_median_s", "cue_gap_min_s", "cue_duration_s"])
        w.writerows(rows)
    print("\n" + "=" * 70)
    print(f"Original code ('label != 0' on alphabetical labels {ALPHABETICAL}): it removed "
          f"'{ALPHABETICAL[0]}' and kept 'rest'; rest = {100 * np.mean(old_rule):.1f}% of the kept windows "
          f"(mean over subjects), i.e. the accuracy of always predicting rest.")
    if all_warns:
        print("WARNINGS:")
        for w_ in all_warns:
            print("  - " + w_)
    else:
        print("No warnings.")
    print("Subjects excluded by default in the pipelines: 88, 92, 100, 104 (EEG_PHYSIONET_EXCLUDE to change).")
    print(f"Table written to {out}")


if __name__ == "__main__":
    main()
