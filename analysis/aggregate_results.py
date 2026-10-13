# aggregate_results.py
#
# Build the tables the reviewer asked for, for ONE model/dataset folder, from the saved result files.
#
#   python analysis/aggregate_results.py EEGNetV4/BNCI_001 [--classes 2class 4class] [--margin 6]
#
# Reads <folder>/<class>/results/<experiment>/results.json (plus alignment_metrics.json and
# selected_profile.json) and writes <folder>/report/REPORT.md and, per class mode, CSV files:
#   T1 accuracy per subject and method, mean and SD                      (review 1, 2, 11)
#   T2 montage / configuration per subject and method                    (review 5, 12)
#   T3 equivalence (TOST) of every 12-channel montage vs 22 channels      (review 10)
#   T4 paired comparison of "Ours" vs every other montage (Holm)          (review 11, 12)
#   T5 confusion matrices, per-class recall, chance / majority, collapse   (review 15)
#   T6 seed stability                                                    (review 14)
#   T7 saliency-ERD alignment, participant level with 95% CI              (review 7, 9)
#   T8 faithfulness: deletion test and model-randomisation check          (review 8)
#   T9 configuration and selected training profile                       (review 13, 14)
# and copies the topomaps of the 22-channel run to report/<class>/figures/.
# Missing experiments are skipped and listed in the report.

import os
import sys
import csv
import json
import glob
import shutil
import argparse
import datetime

import numpy as np
from scipy import stats

METHODS = [
    ("full22", "22ch"),
    ("saliency_reduced", "Ours"),
    ("csp_reduced", "CSP"),
    ("relieff_reduced", "RLF"),
    ("mi_reduced", "MI"),
    ("control_sensorimotor", "Fixed sensorimotor"),
    ("control_erd", "ERD-only"),
    ("control_random", "Random (mean of draws)"),
]
ALIGN_KEYS = ["magnitude_pearson", "class_specific_pearson", "lateralisation_pearson",
              "magnitude_spearman", "class_specific_spearman", "lateralisation_spearman",
              "legacy_signed_LR_pearson"]


# ───────────────────────────── loading ─────────────────────────────
def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_methods(results_root):
    """{label: {"records": [...], "meta": {...}, "draws": [records, ...] (random only)}}."""
    out, missing = {}, []
    for exp, label in METHODS:
        if exp == "control_random":
            paths = sorted(glob.glob(os.path.join(results_root, "control_random_draw*", "results.json")))
            if not paths:
                missing.append(label)
                continue
            draws = [_load(p) for p in paths]
            out[label] = {"records": [r for d in draws for r in d["records"]],
                          "meta": draws[0]["meta"], "draws": [d["records"] for d in draws],
                          "draw_montages": [d["meta"]["montages"] for d in draws]}
            continue
        path = os.path.join(results_root, exp, "results.json")
        if os.path.exists(path):
            d = _load(path)
            out[label] = {"records": d["records"], "meta": d["meta"]}
        else:
            missing.append(label)
    return out, missing


def subject_acc(records):
    """{subject: accuracy} = mean over folds within a seed, then mean over seeds."""
    per = {}
    for r in records:
        per.setdefault((r["subject"], r["seed"]), []).append(r["test_acc"])
    out = {}
    for (sid, _), accs in per.items():
        out.setdefault(sid, []).append(np.mean(accs))
    return {sid: float(np.mean(v)) for sid, v in sorted(out.items())}


def method_subject_acc(entry):
    if "draws" in entry:                                   # random montages: mean over draws
        per_draw = [subject_acc(r) for r in entry["draws"]]
        sids = sorted(set.intersection(*[set(d) for d in per_draw]))
        return {s: float(np.mean([d[s] for d in per_draw])) for s in sids}
    return subject_acc(entry["records"])


def seed_grand_means(records):
    per = {}
    for r in records:
        per.setdefault(r["seed"], {}).setdefault(r["subject"], []).append(r["test_acc"])
    return {seed: float(np.mean([np.mean(v) for v in subj.values()])) for seed, subj in sorted(per.items())}


# ───────────────────────────── statistics ─────────────────────────────
def tost(diff, margin, alpha=0.05):
    diff = np.asarray(diff, dtype=float)
    n, d = len(diff), float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    if sd < 1e-12:
        p = 0.0 if abs(d) < margin else 1.0
        return {"n": n, "mean_diff": d, "sd_diff": sd, "ci_low": d, "ci_high": d,
                "p_tost": p, "equivalent": p < alpha}
    se = sd / np.sqrt(n)
    p_lower = 1 - stats.t.cdf((d + margin) / se, n - 1)
    p_upper = stats.t.cdf((d - margin) / se, n - 1)
    tcrit = stats.t.ppf(1 - alpha, n - 1)
    p = float(max(p_lower, p_upper))
    return {"n": n, "mean_diff": d, "sd_diff": sd, "ci_low": d - tcrit * se, "ci_high": d + tcrit * se,
            "p_tost": p, "equivalent": p < alpha}


def paired(diff):
    diff = np.asarray(diff, dtype=float)
    n, d = len(diff), float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 1e-12:
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        p_t = float(stats.ttest_1samp(diff, 0.0).pvalue)
    else:
        h, p_t = 0.0, float("nan")
    try:
        p_w = float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        p_w = float("nan")
    return {"n": n, "mean_diff": d, "ci_low": d - h, "ci_high": d + h, "p_t": p_t, "p_wilcoxon": p_w,
            "wins": int((diff > 1e-9).sum()), "ties": int((np.abs(diff) <= 1e-9).sum()),
            "losses": int((diff < -1e-9).sum())}


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ok) - rank) * p[i]))
        out[i] = running
    return out


def mean_ci(values):
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), float("nan"), len(v)
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h, len(v)


# ───────────────────────────── output helpers ─────────────────────────────
def pct(x, nd=2):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def num(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def pval(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ───────────────────────────── tables ─────────────────────────────
def build_class_report(folder, cls, out_dir, margin):
    results_root = os.path.join(folder, cls, "results")
    os.makedirs(out_dir, exist_ok=True)
    methods, missing = load_methods(results_root)
    md = [f"## {cls}"]
    if not methods:
        md.append(f"_No results found in `{results_root}`._")
        return "\n\n".join(md)
    if missing:
        md.append("Not available yet: " + ", ".join(missing))

    accs = {label: method_subject_acc(e) for label, e in methods.items()}
    subjects = sorted(set().union(*[set(a) for a in accs.values()]))
    any_rec = next(iter(methods.values()))["records"][0]
    chance = any_rec["chance_acc"]

    # T1 accuracy
    header = ["Method"] + [f"S{s:02d}" for s in subjects] + ["Mean", "SD"]
    rows = []
    for label, a in accs.items():
        vals = [a.get(s, float("nan")) for s in subjects]
        finite = [v for v in vals if np.isfinite(v)]
        sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
        rows.append([label] + [pct(v) for v in vals] + [pct(np.mean(finite)), pct(sd)])
    write_csv(os.path.join(out_dir, "T1_accuracy_per_subject.csv"), header, rows)
    n_seeds = len({r["seed"] for r in next(iter(methods.values()))["records"]})
    md += [f"### T1. Test accuracy (%) per subject — chance {pct(chance, 1)}%",
           f"Subject accuracy = mean over the 4 folds, then over seeds ({n_seeds} seed(s)). "
           "SD = across subjects. Random = mean of the random-montage draws.",
           md_table(header, rows)]

    # T2 montages
    header2 = ["Method", "Subject", "Profile", "N channels", "Channels"]
    rows2 = []
    for label, e in methods.items():
        prof = e["meta"].get("profiles_used", {})
        if "draw_montages" in e:
            for d_i, mont in enumerate(e["draw_montages"]):
                for s in subjects:
                    ch = mont.get(str(s), [])
                    rows2.append([f"{label} draw {d_i}", f"S{s:02d}", prof.get(str(s), ""), len(ch), " ".join(ch)])
            continue
        for s in subjects:
            ch = e["meta"]["montages"].get(str(s), [])
            rows2.append([label, f"S{s:02d}", prof.get(str(s), ""), len(ch), " ".join(ch)])
    write_csv(os.path.join(out_dir, "T2_montages.csv"), header2, rows2)
    md += ["### T2. Montages and profiles", "Full list per subject in `T2_montages.csv` "
           "(all 12-channel montages are leave-one-subject-out)."]
    summary_rows = []
    for label, e in methods.items():
        if label == "22ch" or "draw_montages" in e:
            continue
        monts = [tuple(e["meta"]["montages"].get(str(s), [])) for s in subjects]
        summary_rows.append([label, len(set(monts)), " ".join(max(set(monts), key=monts.count))])
    if summary_rows:
        md.append(md_table(["Method", "Distinct montages across subjects", "Most frequent montage"], summary_rows))

    # T3 equivalence vs 22ch
    if "22ch" in accs:
        header3 = ["Method", "n", "Mean diff (pp)", "SD", "90% CI", "TOST p", f"Equivalent (±{margin:g} pp)",
                   "Worst subject (pp)", f"Subjects losing >{margin:g} pp", "Equivalent at margins (pp)"]
        rows3 = []
        for label, a in accs.items():
            if label == "22ch":
                continue
            common = [s for s in subjects if s in a and s in accs["22ch"]]
            if len(common) < 3:
                continue
            diff = np.array([100 * (a[s] - accs["22ch"][s]) for s in common])
            t = tost(diff, margin)
            ok_margins = [m for m in (1, 2, 3, 4, 5, 6, 8, 10) if tost(diff, m)["equivalent"]]
            rows3.append([label, t["n"], f"{t['mean_diff']:+.2f}", f"{t['sd_diff']:.2f}",
                          f"[{t['ci_low']:+.2f}, {t['ci_high']:+.2f}]", pval(t["p_tost"]),
                          "yes" if t["equivalent"] else "no", f"{diff.min():+.2f}",
                          int((diff < -margin).sum()), ",".join(str(m) for m in ok_margins) or "none"])
        write_csv(os.path.join(out_dir, "T3_equivalence_vs_22ch.csv"), header3, rows3)
        md += ["### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)",
               "Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside "
               "the margin; it does not mean identical accuracy.", md_table(header3, rows3)]

    # T4 Ours vs others
    if "Ours" in accs:
        comps = []
        for label, a in accs.items():
            if label in ("Ours", "22ch"):
                continue
            common = [s for s in subjects if s in a and s in accs["Ours"]]
            if len(common) < 3:
                continue
            res = paired([100 * (accs["Ours"][s] - a[s]) for s in common])
            comps.append((label, res))
        adj_t = holm([c[1]["p_t"] for c in comps])
        adj_w = holm([c[1]["p_wilcoxon"] for c in comps])
        header4 = ["Ours vs", "n", "Mean diff (pp)", "95% CI", "paired t p", "Holm", "Wilcoxon p", "Holm",
                   "Ours better / tie / worse"]
        rows4 = [[label, r["n"], f"{r['mean_diff']:+.2f}", f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]",
                  pval(r["p_t"]), pval(adj_t[i]), pval(r["p_wilcoxon"]), pval(adj_w[i]),
                  f"{r['wins']} / {r['ties']} / {r['losses']}"] for i, (label, r) in enumerate(comps)]
        write_csv(os.path.join(out_dir, "T4_ours_vs_other_montages.csv"), header4, rows4)
        md += ["### T4. Saliency montage (Ours) vs every other 12-channel montage",
               "Positive = Ours more accurate. Holm correction across the comparisons in this table.",
               md_table(header4, rows4)]

    # T5 confusion / collapse
    header5 = ["Method", "Pooled acc", "Per-class recall", "Predicted share per class",
               "Mean majority-class acc", "Folds predicting one class ≥90%"]
    rows5, cm_rows = [], []
    for label, e in methods.items():
        recs = e["records"]
        names = list(recs[0]["test_class_counts"].keys())
        cm = np.sum([np.array(r["confusion_matrix"]) for r in recs], axis=0)
        recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
        share = cm.sum(axis=0) / max(cm.sum(), 1)
        collapse = sum(1 for r in recs
                       if max(r["pred_class_counts"].values()) >= 0.9 * max(r["n_test"], 1))
        rows5.append([label, pct(cm.diagonal().sum() / max(cm.sum(), 1)),
                      ", ".join(f"{n} {pct(v, 1)}" for n, v in zip(names, recall)),
                      ", ".join(f"{n} {pct(v, 1)}" for n, v in zip(names, share)),
                      pct(np.mean([r["majority_class_acc"] for r in recs])), f"{collapse} / {len(recs)}"])
        for i, n in enumerate(names):
            cm_rows.append([label, n] + [int(v) for v in cm[i]])
    write_csv(os.path.join(out_dir, "T5_confusion_summary.csv"), header5, rows5)
    write_csv(os.path.join(out_dir, "T5_confusion_matrices.csv"),
              ["Method", "True class"] + [f"pred {n}" for n in names], cm_rows)
    md += [f"### T5. Confusion, class balance and majority baseline (chance {pct(chance, 1)}%)",
           "Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. "
           "The last column flags models that collapsed to one class.", md_table(header5, rows5)]

    # T6 seeds
    header6 = ["Method", "Seeds", "Grand mean per seed (%)", "Mean across seeds", "SD across seeds"]
    rows6 = []
    for label, e in methods.items():
        g = seed_grand_means(e["records"])
        v = np.array(list(g.values()))
        rows6.append([label, len(g), ", ".join(f"{s}: {pct(x)}" for s, x in g.items()), pct(v.mean()),
                      pct(v.std(ddof=1)) if len(v) > 1 else "—"])
    write_csv(os.path.join(out_dir, "T6_seed_stability.csv"), header6, rows6)
    md += ["### T6. Seed stability", md_table(header6, rows6)]

    # T7 alignment
    align_path = os.path.join(results_root, "full22", "alignment_metrics.json")
    if os.path.exists(align_path):
        al = _load(align_path)
        header7 = ["Attribution", "Metric", "Mean", "95% CI", "SD", "n subjects"]
        rows7 = []
        for m, metrics in al["summary"].items():
            for k in ALIGN_KEYS:
                if k in metrics:
                    s = metrics[k]
                    rows7.append([m, k, num(s["mean"]), f"[{num(s['ci_low'])}, {num(s['ci_high'])}]",
                                  num(s["sd"]), s["n"]])
        write_csv(os.path.join(out_dir, "T7_alignment.csv"), header7, rows7)
        md += ["### T7. Saliency vs ERD/ERS alignment (participant level)",
               "magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation "
               "strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only "
               "for comparison (its sign is not interpretable). Per-subject values: "
               "`results/full22/alignment_metrics.json`.", md_table(header7, rows7)]

    # T8 faithfulness / randomisation
    recs22 = methods.get("22ch", {}).get("records", [])
    faith = [r for r in recs22 if "faithfulness" in r]
    if faith:
        ks = sorted({k for r in faith for k in r["faithfulness"] if k != "intact"}, key=int)
        header8 = ["Removed channels k", "Drop: top-ranked (pp)", "Drop: random (pp)", "Drop: bottom-ranked (pp)",
                   "top vs random p", "top vs bottom p"]
        rows8 = []
        for k in ks:
            drops = {"top": {}, "random": {}, "bottom": {}}
            for r in faith:
                f = r["faithfulness"]
                if k not in f:
                    continue
                for kind in drops:
                    drops[kind].setdefault(r["subject"], []).append(100 * (f["intact"] - f[k][kind]))
            subj = sorted(drops["top"])
            per = {kind: np.array([np.mean(drops[kind][s]) for s in subj]) for kind in drops}

            def fmt(v):
                return f"{v.mean():+.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():+.2f}"

            p_tr = paired(per["top"] - per["random"])["p_t"]
            p_tb = paired(per["top"] - per["bottom"])["p_t"]
            rows8.append([k, fmt(per["top"]), fmt(per["random"]), fmt(per["bottom"]), pval(p_tr), pval(p_tb)])
        write_csv(os.path.join(out_dir, "T8_deletion_test.csv"), header8, rows8)
        md += ["### T8a. Faithfulness: deletion test on the test block",
               "Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation "
               "saliency; removal = channel set to its standardised mean. A faithful ranking gives the "
               "largest drop for top-ranked channels.", md_table(header8, rows8)]
        rando = [r for r in recs22 if "randomisation_spearman" in r]
        if rando:
            header8b = ["Attribution", "Spearman(trained, untrained model)", "95% CI", "n subjects"]
            rows8b = []
            for m in rando[0]["randomisation_spearman"]:
                per_s = {}
                for r in rando:
                    per_s.setdefault(r["subject"], []).append(r["randomisation_spearman"][m])
                vals = [float(np.nanmean(v)) for v in per_s.values()]
                mean, lo, hi, n = mean_ci(vals)
                rows8b.append([m, num(mean), f"[{num(lo)}, {num(hi)}]", n])
            write_csv(os.path.join(out_dir, "T8_randomisation_check.csv"), header8b, rows8b)
            md += ["### T8b. Model-randomisation sanity check",
                   "Similarity between saliency of the trained model and of an untrained model. High values "
                   "mean the map reflects the input more than what the network learned.",
                   md_table(header8b, rows8b)]

    # T9 configuration
    meta22 = methods.get("22ch", next(iter(methods.values())))["meta"]
    with open(os.path.join(out_dir, "T9_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta22.get("config", {}), f, indent=1)
    md += ["### T9. Configuration", "Full snapshot: `T9_config.json`."]
    sel_path = os.path.join(results_root, "selected_profile.json")
    if os.path.exists(sel_path):
        sel = _load(sel_path)
        shutil.copy(sel_path, os.path.join(out_dir, "selected_profile.json"))
        rows9 = [[n, pct(v)] for n, v in sel["mean_validation_accuracy"].items()]
        md += [f"Training profile chosen on validation data only, leave-one-subject-out "
               f"(unanimous: {sel['unanimous']}): " +
               ", ".join(f"S{int(s):02d}={p}" for s, p in sel["per_subject"].items()),
               md_table(["Profile", "Mean validation accuracy (%)"], rows9)]
    cfg = meta22.get("config", {})
    keys = [k for k in ("model", "dataset", "sfreq_hz", "classification_band_hz", "ems", "profiles",
                        "checkpoint_rule", "folds", "seeds") if k in cfg]
    md.append("```\n" + "\n".join(f"{k}: {json.dumps(cfg[k])}" for k in keys) + "\n```")

    # figures
    fig_src = os.path.join(results_root, "full22")
    pngs = sorted(glob.glob(os.path.join(fig_src, "*.png")))
    if pngs:
        fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for p in pngs:
            shutil.copy(p, fig_dir)
        md.append(f"Figures ({len(pngs)} topomaps: saliency per method/subject, group maps, ERD/ERS) "
                  f"copied to `{os.path.relpath(fig_dir, os.path.dirname(out_dir))}`.")
    return "\n\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="model/dataset folder, e.g. EEGNetV4/BNCI_001")
    ap.add_argument("--classes", nargs="*", default=["2class", "4class"])
    ap.add_argument("--margin", type=float, default=6.0, help="TOST equivalence margin (percentage points)")
    ap.add_argument("--out", default=None, help="output folder (default <folder>/report)")
    a = ap.parse_args()

    folder = os.path.abspath(a.folder)
    out = os.path.abspath(a.out or os.path.join(folder, "report"))
    os.makedirs(out, exist_ok=True)
    parts = [f"# Results report — {os.path.relpath(folder, os.getcwd())}",
             f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from the saved `results.json` files. "
             f"TOST margin ±{a.margin:g} percentage points."]
    for cls in a.classes:
        parts.append(build_class_report(folder, cls, os.path.join(out, cls), a.margin))
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts) + "\n")
    print(f"Report written to {os.path.join(out, 'REPORT.md')}")


if __name__ == "__main__":
    sys.exit(main())
