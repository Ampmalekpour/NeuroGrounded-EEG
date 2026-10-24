# master_table.py
#
# One master table for every result we have, next to the paper's tables.
#
#   python analysis/master_table.py                    # writes reports/master_table/ (xlsx + html + csv)
#   python analysis/master_table.py --out D:\somewhere
#
# What it reads (nothing is modified, nothing is recomputed):
#   <Model>/<Dataset>/report*/<class>/T1_accuracy_per_subject.csv       written by run_all.py / aggregate_results.py
# for Model in EEGNetV4, EEGItNet, Deep4net, EEGConformer; Dataset in BNCI_001, HGD, PhysioNet; and every report
# folder that exists:
#   report_session           session protocol (the main results)
#   report                   blocks protocol (first revision)
#   report_session_holdout   HGD run with the hold-out caused by the broken MOABB cache
#   report_session_wide      HGD full-bandwidth sensitivity run (250 Hz, 4-100 Hz)
#   report_session_invalid_v1  (only with --include-invalid)
# plus the paper's per-subject tables (Tables 1, 2, 4, 5), typed in below, as "Paper" rows.
#
# Layout of the "Master" sheet: one row per subject, one column per method
#   Full montage (22ch BNCI / HGD, 64ch PhysioNet) | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random
# with a Mean and an SD row at the end of each block. In every row the WINNER is written in BLUE
# (white bold text on a blue fill; ties are all blue). By default the winner is chosen among the five
# columns of the paper's tables (Full, Ours, CSP, RLF, MI); --winner-over all includes the three controls.
#
# Other sheets: "Means" (one row per block, plus 'ours - paper' rows), "Wins" (how many subjects each method
# wins per block), "Montages" (recorded / baseline / reduced channel counts and the selected electrodes of every
# method and subject: reviewer comment 5), "Paper" (the printed means and their recomputation), "Missing".
# Needs openpyxl for the xlsx (pip install openpyxl); the html and csv need nothing.

import os
import csv
import sys
import html
import argparse
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

MODELS = ["EEGNetV4", "EEGItNet", "Deep4net", "EEGConformer"]
DATASETS = [("BNCI_001", "BNCI2014-001"), ("HGD", "HGD"), ("PhysioNet", "PhysioNet")]
CLASSES = ["2class", "4class"]
VARIANTS = [                       # report folder -> label shown in the table
    ("report_session", "ours: session protocol"),
    ("report", "ours: blocks (1st revision)"),
    ("report_session_holdout", "ours: HGD hold-out (cache bug)"),
    ("report_session_wide", "ours: HGD 4-100 Hz"),
    ("report_session_invalid_v1", "ours: INVALID v1"),
]
COLUMNS = ["Full montage", "Ours", "CSP", "RLF", "MI", "Fixed sensorimotor", "ERD-only", "Random"]
PAPER5 = COLUMNS[:5]
ALIAS = {"22ch": "Full montage", "64ch": "Full montage", "Ours": "Ours", "CSP": "CSP", "RLF": "RLF", "MI": "MI",
         "Fixed sensorimotor": "Fixed sensorimotor", "ERD-only": "ERD-only",
         "Random (mean of draws)": "Random"}

BLUE_FILL, BLUE_TEXT = "2F75B5", "FFFFFF"

# ──────────────────────────── the paper's tables, per subject ────────────────────────────
# Each row: 9 (BNCI) or 14 (HGD) participant values, then the mean printed in the paper.
# Row order in every block: Full, CSP, RLF, MI, Ours. PhysioNet (Tables 3, 6) shows only 10 of its subjects and is
# not transcribed.
PAPER_TXT = {
 ("BNCI_001", "4class", "Conformer"): """
78.82 30.9 86.46 55.9 32.64 33.33 81.6 80.9 77.08 61.96
69.79 37.15 85.07 48.96 33.68 34.72 73.61 75.69 71.53 58.91
71.53 35.07 82.64 51.39 31.25 36.81 78.12 71.88 67.01 58.41
69.79 34.03 86.46 50.69 31.25 37.15 79.86 77.08 76.74 60.34
70.83 32.29 85.76 50.69 34.03 35.76 72.92 79.17 76.04 59.72""",
 ("BNCI_001", "4class", "Deep4Net"): """
76.04 29.34 82.47 55.73 32.99 42.71 77.08 82.29 78.99 61.96
71.18 32.47 77.43 45.31 32.12 42.53 62.15 77.26 78.12 57.62
68.06 31.42 71.01 38.19 36.98 35.42 43.23 69.97 75 52.14
67.71 32.47 72.05 44.97 36.46 35.76 42.01 72.92 74.31 53.18
72.57 28.65 81.42 51.22 32.64 43.58 67.88 80.38 77.78 59.57""",
 ("BNCI_001", "4class", "EEGItNet"): """
58.68 37.15 71.18 53.99 37.5 44.44 59.72 65.45 66.49 54.96
58.68 38.37 60.42 53.99 39.58 45.31 56.6 59.2 63.89 52.89
56.25 36.46 64.58 47.05 38.72 42.53 56.08 61.81 67.01 52.28
63.37 40.1 64.06 56.08 37.85 44.27 61.46 63.19 66.15 55.17
59.55 41.49 57.64 57.12 41.32 41.67 62.33 59.03 65.45 53.95""",
 ("BNCI_001", "4class", "EEGNetv4"): """
69.44 40.8 86.28 48.96 40.1 45.83 79.51 76.91 85.94 63.75
65.45 34.2 81.42 48.26 28.99 44.44 73.96 75.17 84.9 59.65
69.97 41.67 83.33 48.61 45.49 45.83 71.53 76.56 81.77 62.75
67.19 39.58 83.85 55.21 30.38 46.53 70.66 76.56 84.03 61.55
69.27 36.46 84.9 48.09 36.28 44.1 71.01 74.65 85.07 61.54""",
 ("BNCI_001", "2class", "Conformer"): """
90.97 60.42 97.22 75 72.92 65.97 81.94 97.22 93.06 81.64
89.58 60.42 93.75 74.31 65.97 67.36 78.47 97.92 93.75 80.17
89.58 55.56 95.83 72.92 64.58 65.28 78.47 92.36 93.75 78.7
88.89 55.56 93.06 75 65.97 66.67 79.86 97.92 94.44 79.71
89.58 59.72 97.22 74.31 70.14 66.67 80.56 97.22 95.14 81.17""",
 ("BNCI_001", "2class", "Deep4Net"): """
80.56 52.08 94.1 59.38 56.94 64.58 56.94 96.18 92.71 72.61
76.39 47.22 89.93 60.42 55.9 61.11 57.99 94.44 84.37 69.75
68.75 57.99 91.67 63.19 61.46 59.72 59.72 93.4 94.1 72.22
70.83 54.86 91.32 63.19 60.07 61.46 56.6 96.88 93.4 72.07
84.38 48.26 94.44 62.15 58.68 62.5 67.71 96.18 92.36 74.07""",
 ("BNCI_001", "2class", "EEGItNet"): """
67.71 52.78 81.6 78.12 65.97 63.54 67.36 91.67 76.39 71.68
57.29 52.08 68.06 67.01 61.11 57.99 61.46 83.33 73.26 64.62
72.22 54.17 81.25 76.74 64.58 62.15 68.75 89.58 79.51 72.11
70.83 54.86 83.33 75 62.85 63.89 68.06 91.67 80.21 72.3
64.58 56.25 83.68 75.35 66.32 60.07 71.88 89.58 81.94 72.18""",
 ("BNCI_001", "2class", "EEGNetv4"): """
79.51 54.86 79.17 65.28 61.46 61.81 85.07 84.72 82.64 72.72
67.71 57.64 51.04 51.74 55.9 57.99 72.57 67.71 80.9 62.58
70.14 61.46 64.58 56.25 55.56 50.69 82.29 66.67 70.83 64.27
69.79 65.97 78.12 56.25 60.42 59.72 92.71 83.68 81.25 71.99
50 56.25 90.28 61.81 54.51 61.46 61.11 88.89 81.94 67.36""",
 ("HGD", "4class", "Conformer"): """
67.71 80.98 91.73 92.06 78.86 99.37 99.94 100 99.94 99.66 100 99.89 100 99.94 93.58
62.5 77.08 79.33 89.31 68.41 99.94 99.94 99.85 100 99.72 100 99.77 100 99.94 91.13
72.71 80.37 92.79 94.99 79.2 100 100 99.77 99.89 99.83 100 99.77 100 99.83 94.22
76.04 80.16 90.87 93.29 77.61 99.94 99.89 99.77 100 99.66 100 99.77 100 99.89 94.06
68.12 79.34 90.96 77.16 77.16 99.77 99.94 99.92 99.94 99.77 100 99.83 100 99.83 93.45""",
 ("HGD", "4class", "Deep4Net"): """
79.38 87.56 95.1 97.16 98.98 100 100 100 100 100 98.44 100 100 100 96.9
47.71 84.89 81.63 93.37 98.07 100 100 100 100 100 100 100 100 100 93.26
73.75 85.61 95.58 95.46 98.98 100 100 100 100 100 93.12 100 100 100 95.89
76.88 88.7 95.29 93.75 98.75 100 100 100 100 100 95.62 100 100 100 96.36
78.75 87.35 95.77 95.46 98.86 100 100 100 100 100 98.75 100 100 100 96.78""",
 ("HGD", "4class", "EEGItNet"): """
82.5 87.68 95.38 97.73 82.73 100 100 100 100 100 100 100 100 100 96.14
82.08 88.08 98.08 95.08 82.5 100 99.89 100 99.89 100 100 99.89 100 100 96.11
88.33 90.35 98.08 89.59 76.82 99.66 99.09 100 100 100 100 98.07 100 100 95.71
92.5 90.56 97.31 96.98 84.09 100 100 100 100 100 100 68.41 100 100 94.99
78.75 88.5 98.65 97.92 94.55 100 100 99.85 100 100 100 99.2 100 100 96.96""",
 ("HGD", "4class", "EEGNetv4"): """
82.5 83.76 80.67 96.12 98.98 100 100 99.92 100 99.94 100 100 99.94 100 95.77
81.46 81.5 78.46 90.82 97.16 100 99.77 99.92 99.94 99.66 100 100 99.81 100 94.89
89.17 84.48 93.27 95.74 99.09 100 99.94 100 100 99.77 99.69 100 99.88 100 97.22
85 83.35 96.63 94.51 98.64 100 99.94 100 100 99.77 100 100 99.94 100 96.98
86.25 75.64 96.15 95.18 98.64 100 99.94 100 99.83 99.66 100 100 99.94 100 96.23""",
 ("HGD", "2class", "Conformer"): """
54.58 90.95 95.96 98.86 74.32 100 100 100 100 100 100 100 100 100 93.91
60.42 88.68 88.46 98.3 62.95 100 100 100 100 99.77 100 100 100 100 92.76
63.75 91.15 97.88 97.73 70.68 100 100 100 100 100 100 100 100 100 94.37
68.33 89.3 94.23 99.24 66.59 100 100 100 100 99.89 100 100 100 100 94.11
58.33 89.09 96.35 97.35 79.32 100 100 100 100 99.89 100 100 100 100 94.31""",
 ("HGD", "2class", "Deep4Net"): """
85.83 97.74 98.08 99.43 99.77 100 100 100 100 99.89 100 100 100 100 98.62
92.08 97.73 97.31 99.05 99.09 100 100 99.85 100 100 100 100 100 100 98.94
93.33 96.3 99.42 99.05 100 100 100 100 100 100 100 100 100 100 99.15
92.92 97.74 99.62 99.24 98.86 100 100 100 100 100 100 100 100 100 99.17
92.08 96.71 100 99.62 99.77 100 100 100 100 100 100 100 100 100 99.16""",
 # the paper's EEGItNet 2-class HGD table is identical to its 4-class table (reviewer comment 1)
 ("HGD", "2class", "EEGItNet"): None,
 ("HGD", "2class", "EEGNetv4"): """
98.75 98.15 99.42 99.43 100 100 100 99.85 100 100 100 100 100 100 99.69
95 97.73 99.81 99.24 100 100 100 100 100 100 100 100 100 100 99.41
97.92 97.53 99.23 99.43 99.77 100 100 100 100 100 100 100 100 100 99.56
97.92 97.73 99.62 99.43 99.77 100 100 100 100 100 100 100 100 100 99.6
96.25 98.35 99.81 99.43 100 100 100 100 100 100 100 100 100 100 99.56""",
}
PAPER_ORDER = ["Full montage", "CSP", "RLF", "MI", "Ours"]            # row order inside each paper block
PAPER_TABLE = {("BNCI_001", "4class"): "Table 2", ("BNCI_001", "2class"): "Table 5",
               ("HGD", "4class"): "Table 1", ("HGD", "2class"): "Table 4"}
MODEL_OF = {"Conformer": "EEGConformer", "Deep4Net": "Deep4net", "EEGItNet": "EEGItNet", "EEGNetv4": "EEGNetV4"}


def paper_blocks():
    """[(dataset, classes, model, {method: [per-subject values]}, {method: printed mean})]"""
    out = []
    for (ds, cls, name), txt in PAPER_TXT.items():
        if txt is None:                                              # duplicated table: copy the 4-class block
            txt = PAPER_TXT[(ds, "4class", name)]
        rows = [[float(x) for x in line.split()] for line in txt.strip().splitlines()]
        n_subj = 9 if ds == "BNCI_001" else 14
        assert len(rows) == 5 and all(len(r) == n_subj + 1 for r in rows), (ds, cls, name)
        vals = {m: r[:-1] for m, r in zip(PAPER_ORDER, rows)}
        printed = {m: r[-1] for m, r in zip(PAPER_ORDER, rows)}
        out.append((ds, cls, MODEL_OF[name], vals, printed, name == "EEGItNet" and ds == "HGD" and cls == "2class"))
    return out


# ──────────────────────────── reading our results ────────────────────────────
def read_t1(path):
    """(subject labels, {method column: [values]}) from a T1_accuracy_per_subject.csv."""
    rows = list(csv.reader(open(path, encoding="utf-8")))
    header = rows[0]
    subj = header[1:-2]
    data = {}
    for r in rows[1:]:
        col = ALIAS.get(r[0])
        if col is None:
            continue
        data[col] = [float(x) if x not in ("", "—", "nan") else float("nan") for x in r[1:-2]]
    return subj, data


def collect(include_invalid, only_main):
    blocks = []                                  # dicts: dataset, classes, model, source, subjects, values
    for ds_dir, ds_label in DATASETS:
        for model in MODELS:
            for rep, label in VARIANTS:
                if rep == "report_session_invalid_v1" and not include_invalid:
                    continue
                if only_main and rep != "report_session":
                    continue
                for cls in CLASSES:
                    p = os.path.join(REPO, model, ds_dir, rep, cls, "T1_accuracy_per_subject.csv")
                    if os.path.exists(p):
                        subj, data = read_t1(p)
                        blocks.append(dict(dataset=ds_label, ds_dir=ds_dir, classes=cls, model=model,
                                           source=label, subjects=subj, values=data, path=os.path.relpath(p, REPO)))
    return blocks


RECORDED = {"BNCI_001": 22, "HGD": 128, "PhysioNet": 64}


def collect_montages(include_invalid, only_main):
    """Rows of every T2_montages.csv: where each electrode set came from, per method and subject."""
    rows = []
    for ds_dir, ds_label in DATASETS:
        for model in MODELS:
            for rep, label in VARIANTS:
                if rep == "report_session_invalid_v1" and not include_invalid:
                    continue
                if only_main and rep != "report_session":
                    continue
                for cls in CLASSES:
                    p = os.path.join(REPO, model, ds_dir, rep, cls, "T2_montages.csv")
                    if not os.path.exists(p):
                        continue
                    for r in list(csv.reader(open(p, encoding="utf-8")))[1:]:
                        method, subject, profile, n_ch, chans = r[0], r[1], r[2], int(r[3]), r[4]
                        rows.append([ds_label, model, cls, label, method, subject, profile, RECORDED[ds_dir], n_ch, chans])
    return rows


def paper_as_blocks():
    out = []
    for ds, cls, model, vals, printed, dup in paper_blocks():
        n = len(next(iter(vals.values())))
        label = f"PAPER {PAPER_TABLE[(ds, cls)]}" + (" (identical to its 4-class table)" if dup else "")
        out.append(dict(dataset=dict(DATASETS)[ds], ds_dir=ds, classes=cls, model=model, source=label,
                        subjects=[f"S{i + 1:02d}" for i in range(n)], values=vals, printed=printed, path="paper"))
    return out


# ──────────────────────────── table assembly ────────────────────────────
def sort_key(b):
    ds_order = [d for d, _ in DATASETS].index(b["ds_dir"])
    src = 0 if b["source"].startswith("PAPER") else 1 + [l for _, l in VARIANTS].index(b["source"]) \
        if b["source"] in [l for _, l in VARIANTS] else 9
    return (ds_order, CLASSES.index(b["classes"]), MODELS.index(b["model"]), src)


def winners(row_vals, contenders):
    """Indices (into COLUMNS) of the winning columns among `contenders`; ties are all winners."""
    cand = [(c, row_vals.get(c, float("nan"))) for c in contenders]
    cand = [(c, v) for c, v in cand if np.isfinite(v)]
    if not cand:
        return set()
    best = max(v for _, v in cand)
    return {c for c, v in cand if abs(v - best) < 1e-9}


def block_rows(b):
    """[(label, {col: value}, kind)] with kind in subject | mean | sd"""
    rows = []
    n = len(b["subjects"])
    for i, s in enumerate(b["subjects"]):
        rows.append((s, {c: b["values"][c][i] for c in COLUMNS if c in b["values"]}, "subject"))
    mean = {c: float(np.nanmean(b["values"][c])) for c in COLUMNS if c in b["values"]}
    sd = {c: float(np.nanstd(b["values"][c], ddof=1)) if n > 1 else float("nan") for c in COLUMNS if c in b["values"]}
    rows.append(("Mean", mean, "mean"))
    rows.append(("SD", sd, "sd"))
    return rows


def build(blocks, contenders):
    """Flat list of table rows for the Master sheet / html / csv."""
    master, means, wins = [], [], []
    paper_idx = {(b["ds_dir"], b["classes"], b["model"]): b for b in blocks if b["source"].startswith("PAPER")}
    for b in sorted(blocks, key=sort_key):
        rows = block_rows(b)
        count = {c: 0 for c in COLUMNS}
        for label, vals, kind in rows:
            win = winners(vals, contenders) if kind in ("subject", "mean") else set()
            if kind == "subject":
                for c in win:
                    count[c] += 1
            master.append(dict(dataset=b["dataset"], model=b["model"], classes=b["classes"], source=b["source"],
                               subject=label, kind=kind, values=vals, winners=win))
        mean_row = next(r for r in rows if r[2] == "mean")
        sd_row = next(r for r in rows if r[2] == "sd")
        means.append(dict(dataset=b["dataset"], model=b["model"], classes=b["classes"], source=b["source"],
                          n=len(b["subjects"]), values=mean_row[1], sd=sd_row[1], winners=winners(mean_row[1], contenders),
                          delta=False))
        p = paper_idx.get((b["ds_dir"], b["classes"], b["model"]))
        if p is not None and not b["source"].startswith("PAPER"):
            pm = {c: float(np.nanmean(p["values"][c])) for c in p["values"]}
            means.append(dict(dataset=b["dataset"], model=b["model"], classes=b["classes"],
                              source=f"   {b['source']} MINUS paper", n=len(b["subjects"]),
                              values={c: mean_row[1][c] - pm[c] for c in mean_row[1] if c in pm}, sd={}, winners=set(),
                              delta=True))
        wins.append(dict(dataset=b["dataset"], model=b["model"], classes=b["classes"], source=b["source"],
                         n=len(b["subjects"]), counts=count))
    return master, means, wins


def missing_report(blocks):
    have = {(b["ds_dir"], b["model"], b["classes"]) for b in blocks if b["source"] == "ours: session protocol"}
    out = []
    for ds_dir, ds_label in DATASETS:
        for model in MODELS:
            for cls in CLASSES:
                if (ds_dir, model, cls) not in have:
                    out.append((ds_label, model, cls))
    return out


def self_check_paper(pblocks):
    """Recompute the paper's means from its printed participant values (reviewer comment 1)."""
    lines = []
    for ds, cls, model, vals, printed, _dup in paper_blocks():
        for m in PAPER_ORDER:
            re_mean = float(np.mean(vals[m]))
            if abs(re_mean - printed[m]) > 0.35:
                lines.append(f"{dict(DATASETS)[ds]} {cls} {model} {m}: printed {printed[m]:.2f}, recomputed {re_mean:.2f}")
    return lines


# ──────────────────────────── writers ────────────────────────────
def fmt(v):
    return "" if v is None or not np.isfinite(v) else f"{v:.2f}"


def write_montages_csv(path, montages):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Model", "Classes", "Source", "Method", "Subject", "Training profile",
                    "Channels recorded", "Channels used by this model", "Selected / used electrodes"])
        w.writerows(montages)


def write_csv(path, master):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Model", "Classes", "Source", "Subject"] + COLUMNS + ["Winner(s)"])
        for r in master:
            w.writerow([r["dataset"], r["model"], r["classes"], r["source"], r["subject"]]
                       + [fmt(r["values"].get(c)) for c in COLUMNS] + [", ".join(c for c in COLUMNS if c in r["winners"])])


def write_html(path, master, means, wins, title, contenders_note, paper_issues, missing):
    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:16px;color:#1b1b1b}h1{font-size:20px}h2{font-size:16px;margin-top:28px}"
           "table{border-collapse:collapse;font-size:13px;margin-bottom:10px}th,td{border:1px solid #d0d0d0;padding:3px 8px;text-align:right}"
           "th{background:#f0f0f0;position:sticky;top:0}td.l,th.l{text-align:left}"
           f"td.win{{background:#{BLUE_FILL};color:#{BLUE_TEXT};font-weight:700}}tr.mean td{{font-weight:700;border-top:2px solid #888}}"
           "tr.sd td{color:#666;font-style:italic}tr.paper td.l{color:#7a4b00}.note{color:#555;font-size:12px}")
    h = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style>",
         f"<h1>{html.escape(title)}</h1><p class='note'>Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. {html.escape(contenders_note)} "
         "Blue = winner of the row (ties are all blue). Paper rows are the printed participant values.</p>"]

    def table(head, rows_html):
        return "<table><tr>" + "".join(f"<th class='{'l' if i < len(head) - len(COLUMNS) else ''}'>{html.escape(x)}</th>" for i, x in enumerate(head)) + "</tr>" + "".join(rows_html) + "</table>"

    h.append("<h2>Means per block</h2>")
    rows = []
    for m in means:
        cells = "".join(f"<td class='{'win' if c in m['winners'] else ''}'>{fmt(m['values'].get(c)) if not m['delta'] else (('%+.2f' % m['values'][c]) if c in m['values'] else '')}</td>" for c in COLUMNS)
        cls = "paper" if m["source"].startswith("PAPER") else ""
        rows.append(f"<tr class='{cls}'><td class='l'>{html.escape(m['dataset'])}</td><td class='l'>{m['model']}</td><td class='l'>{m['classes']}</td><td class='l'>{html.escape(m['source'])}</td><td>{m['n']}</td>{cells}</tr>")
    h.append(table(["Dataset", "Model", "Classes", "Source", "N subj"] + COLUMNS, rows))
    h.append("<h2>Wins per block (number of subjects in which a method is the winner; ties count for each)</h2>")
    rows = []
    for w in wins:
        top = max(w["counts"][c] for c in PAPER5) if w["counts"] else 0
        cells = "".join(f"<td>{w['counts'][c]}</td>" for c in COLUMNS)
        rows.append(f"<tr><td class='l'>{html.escape(w['dataset'])}</td><td class='l'>{w['model']}</td><td class='l'>{w['classes']}</td><td class='l'>{html.escape(w['source'])}</td><td>{w['n']}</td>{cells}</tr>")
    h.append(table(["Dataset", "Model", "Classes", "Source", "N subj"] + COLUMNS, rows))
    h.append("<h2>Master table: every subject</h2>")
    rows = []
    for r in master:
        cells = "".join(f"<td class='{'win' if c in r['winners'] else ''}'>{fmt(r['values'].get(c))}</td>" for c in COLUMNS)
        cls = r["kind"] + (" paper" if r["source"].startswith("PAPER") else "")
        rows.append(f"<tr class='{cls}'><td class='l'>{html.escape(r['dataset'])}</td><td class='l'>{r['model']}</td><td class='l'>{r['classes']}</td><td class='l'>{html.escape(r['source'])}</td><td class='l'>{r['subject']}</td>{cells}</tr>")
    h.append(table(["Dataset", "Model", "Classes", "Source", "Subject"] + COLUMNS, rows))
    if paper_issues:
        h.append("<h2>Paper: printed mean differs from the mean of the printed participant values</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in paper_issues) + "</ul>")
    if missing:
        h.append("<h2>Not run yet (session protocol)</h2><p>" + html.escape(", ".join(f"{a} {b} {c}" for a, b, c in missing)) + "</p>")
    open(path, "w", encoding="utf-8").write("\n".join(h))


def write_xlsx(path, master, means, wins, paper_pblocks, paper_issues, missing, contenders_note, montages=()):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    blue = PatternFill("solid", fgColor=BLUE_FILL)
    blue_font = Font(bold=True, color=BLUE_TEXT)
    head_fill = PatternFill("solid", fgColor="E7E6E6")
    paper_fill = PatternFill("solid", fgColor="FFF2CC")
    mean_border = Border(top=Side(style="medium", color="808080"))
    wb = Workbook()

    def header(ws, cols):
        ws.append(cols)
        for c in ws[ws.max_row]:
            c.font = Font(bold=True)
            c.fill = head_fill
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def widths(ws, ws_widths):
        for i, wd in enumerate(ws_widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = wd

    # Master
    ws = wb.active
    ws.title = "Master"
    header(ws, ["Dataset", "Model", "Classes", "Source", "Subject"] + COLUMNS)
    for r in master:
        ws.append([r["dataset"], r["model"], r["classes"], r["source"], r["subject"]] + [
            (r["values"].get(c) if r["values"].get(c) is not None and np.isfinite(r["values"].get(c)) else None) for c in COLUMNS])
        row = ws[ws.max_row]
        for j, c in enumerate(COLUMNS):
            cell = row[5 + j]
            cell.number_format = "0.00"
            if c in r["winners"]:
                cell.fill, cell.font = blue, blue_font
        if r["source"].startswith("PAPER"):
            for k in range(5):
                if row[k].fill.fgColor.rgb in (None, "00000000"):
                    row[k].fill = paper_fill
        if r["kind"] == "mean":
            for j, cell in enumerate(row):
                cell.border = mean_border
                is_win = j >= 5 and COLUMNS[j - 5] in r["winners"]
                if not is_win:
                    cell.font = Font(bold=True)
        if r["kind"] == "sd":
            for cell in row:
                cell.font = Font(italic=True, color="666666")
    ws.freeze_panes = "F2"
    ws.auto_filter.ref = ws.dimensions
    widths(ws, [14, 14, 9, 34, 9] + [13] * len(COLUMNS))

    # Means
    ws = wb.create_sheet("Means")
    header(ws, ["Dataset", "Model", "Classes", "Source", "N subjects"] + COLUMNS)
    for m in means:
        ws.append([m["dataset"], m["model"], m["classes"], m["source"], m["n"]] + [m["values"].get(c) for c in COLUMNS])
        row = ws[ws.max_row]
        for j, c in enumerate(COLUMNS):
            cell = row[5 + j]
            cell.number_format = "+0.00;-0.00;0.00" if m["delta"] else "0.00"
            if c in m["winners"]:
                cell.fill, cell.font = blue, blue_font
        if m["source"].startswith("PAPER"):
            for k in range(5):
                row[k].fill = paper_fill
        if m["delta"]:
            for cell in row:
                cell.font = Font(italic=True, color="444444")
    ws.freeze_panes = "F2"
    widths(ws, [14, 14, 9, 40, 10] + [13] * len(COLUMNS))

    # Wins
    ws = wb.create_sheet("Wins")
    header(ws, ["Dataset", "Model", "Classes", "Source", "N subjects"] + COLUMNS)
    for w in wins:
        ws.append([w["dataset"], w["model"], w["classes"], w["source"], w["n"]] + [w["counts"][c] for c in COLUMNS])
        row = ws[ws.max_row]
        top = max(w["counts"][c] for c in PAPER5)
        for j, c in enumerate(COLUMNS):
            if c in PAPER5 and w["counts"][c] == top and top > 0:
                row[5 + j].fill, row[5 + j].font = blue, blue_font
    ws.freeze_panes = "F2"
    widths(ws, [14, 14, 9, 40, 10] + [13] * len(COLUMNS))
    ws.append([])
    ws.append(["A subject counts for every method that ties for the best value in that row. Blue = most wins in the block "
               "(among the five paper columns)."])

    # Paper
    ws = wb.create_sheet("Paper")
    header(ws, ["Dataset", "Classes", "Model", "Method", "Printed mean", "Mean of printed participant values", "Difference"])
    for b in paper_pblocks:
        for m in PAPER_ORDER:
            re_mean = float(np.mean(b["values"][m]))
            ws.append([b["dataset"], b["classes"], b["model"], m, b["printed"][m], round(re_mean, 2), round(re_mean - b["printed"][m], 2)])
            if abs(re_mean - b["printed"][m]) > 0.35:
                for cell in ws[ws.max_row]:
                    cell.fill = paper_fill
    widths(ws, [14, 9, 14, 16, 13, 18, 11])
    ws.append([])
    ws.append(["Yellow rows: the printed mean differs from the mean of the printed participant values (reviewer comment 1). "
               "PhysioNet (Tables 3, 6) shows only 10 subjects and is not transcribed."])

    # Montages
    ws = wb.create_sheet("Montages")
    header(ws, ["Dataset", "Model", "Classes", "Source", "Method", "Subject", "Training profile",
                "Channels recorded", "Channels used by this model", "Selected / used electrodes"])
    for r in montages:
        ws.append(list(r))
    ws.freeze_panes = "G2"
    ws.auto_filter.ref = ws.dimensions
    widths(ws, [14, 14, 9, 34, 22, 9, 16, 11, 12, 90])

    # Missing
    ws = wb.create_sheet("Missing")
    header(ws, ["Dataset", "Model", "Classes (session protocol not run yet)"])
    for a, b, c in missing:
        ws.append([a, b, c])
    widths(ws, [16, 16, 38])

    ws = wb.create_sheet("Read me")
    for line in ["Master table of all results (analysis/master_table.py).", contenders_note,
                 "Blue cell = winner of the row; ties are all blue. Yellow = paper rows.",
                 "Sources: <Model>/<Dataset>/report*/<class>/T1_accuracy_per_subject.csv and the paper's Tables 1, 2, 4, 5.",
                 f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}."]:
        ws.append([line])
    ws.column_dimensions["A"].width = 120
    wb.move_sheet("Read me", offset=-(len(wb.sheetnames) - 1))
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description="Master table of all results with the winner of each row in blue")
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "master_table"))
    ap.add_argument("--winner-over", choices=["paper", "all"], default="paper",
                    help="paper (default): Full, Ours, CSP, RLF, MI; all: also the three control montages")
    ap.add_argument("--include-invalid", action="store_true", help="also list the invalid first HGD run")
    ap.add_argument("--only-main", action="store_true", help="only the session protocol results (+ the paper)")
    ap.add_argument("--no-paper", action="store_true", help="leave the paper rows out")
    a = ap.parse_args()

    contenders = PAPER5 if a.winner_over == "paper" else COLUMNS
    note = ("Winner = best of " + ", ".join(contenders) + " in each row (the 12-channel controls are shown but "
            "not in the contest)." if a.winner_over == "paper" else "Winner = best of all eight columns in each row.")
    ours = collect(a.include_invalid, a.only_main)
    pap = [] if a.no_paper else paper_as_blocks()
    blocks = ours + pap
    if not blocks:
        raise SystemExit("no results found: run from the repository, after run_all.py has produced report*/ folders")
    master, means, wins = build(blocks, contenders)
    issues = self_check_paper(pap)
    missing = missing_report(ours)
    os.makedirs(a.out, exist_ok=True)

    montages = collect_montages(a.include_invalid, a.only_main)
    write_csv(os.path.join(a.out, "master_table.csv"), master)
    write_montages_csv(os.path.join(a.out, "montages.csv"), montages)
    write_html(os.path.join(a.out, "master_table.html"), master, means, wins, "Master table - all models, datasets and subjects",
               note, issues, missing)
    wrote = ["master_table.csv", "montages.csv", "master_table.html"]
    try:
        write_xlsx(os.path.join(a.out, "master_table.xlsx"), master, means, wins, [b for b in pap], issues, missing, note,
                   montages)
        wrote.insert(0, "master_table.xlsx")
    except ImportError:
        print("openpyxl is not installed, so no .xlsx was written. Run:  pip install openpyxl   and start this again "
              "(the html and csv are complete).")
    n_ours = len([b for b in ours])
    print(f"{n_ours} result blocks from this repository + {len(pap)} paper blocks, {len(master)} rows.")
    for line in ("Not run yet (session protocol): " + ", ".join(f"{x[0]} {x[1]} {x[2]}" for x in missing),):
        print(line)
    if issues:
        print("Paper: printed means that differ from the mean of the printed participant values:")
        for x in issues:
            print("   " + x)
    print("Written to " + a.out + ": " + ", ".join(wrote))


if __name__ == "__main__":
    main()
