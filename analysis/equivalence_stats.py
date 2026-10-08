# equivalence_stats.py
#
# Paired two-one-sided-tests (TOST) equivalence analysis between two experiments, computed
# from the machine-readable results.json files written by the HGD scripts.
#
#   python analysis/equivalence_stats.py FULL/results.json REDUCED/results.json --margin 6
#
# Reports, in percentage points (reduced - full, so a negative number is a loss):
#   * mean paired difference, SD, and the (1 - 2*alpha) = 90% confidence interval
#   * TOST p-value (equivalent iff the 90% CI lies inside +/- margin)
#   * sensitivity of the verdict to the margin (--sensitivity)
#   * the per-subject differences, the worst loss, and how many subjects lose more than the margin
#
# "Equivalent" only means the mean difference is inside the chosen margin; it does not mean
# identical performance, and it says nothing about individual subjects (see the per-subject list).

import sys
import json
import argparse
import numpy as np
from scipy import stats


def subject_means(path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)["records"]
    per = {}
    for r in records:
        per.setdefault((r["subject"], r["seed"]), []).append(r["test_acc"])
    by_subject = {}
    for (sid, _), accs in per.items():
        by_subject.setdefault(sid, []).append(np.mean(accs))
    return {sid: 100.0 * float(np.mean(v)) for sid, v in by_subject.items()}


def tost(diff, margin, alpha):
    n = len(diff)
    d, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n)
    p_lower = 1 - stats.t.cdf((d + margin) / se, n - 1)     # H0: d <= -margin
    p_upper = stats.t.cdf((d - margin) / se, n - 1)         # H0: d >=  margin
    tcrit = stats.t.ppf(1 - alpha, n - 1)
    return {"n": n, "mean_diff": d, "sd_diff": sd, "ci_low": d - tcrit * se, "ci_high": d + tcrit * se,
            "p_tost": float(max(p_lower, p_upper)), "equivalent": bool(max(p_lower, p_upper) < alpha)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full"), ap.add_argument("reduced")
    ap.add_argument("--margin", type=float, default=6.0, help="equivalence margin, percentage points")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--sensitivity", type=float, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--json", help="optionally write the result to this file")
    a = ap.parse_args()

    full, red = subject_means(a.full), subject_means(a.reduced)
    ids = sorted(set(full) & set(red))
    if len(ids) < 3:
        sys.exit(f"need >= 3 common subjects, found {len(ids)}")
    diff = np.array([red[s] - full[s] for s in ids])
    res = tost(diff, a.margin, a.alpha)
    res.update({"margin": a.margin, "alpha": a.alpha,
                "per_subject_diff": {int(s): float(d) for s, d in zip(ids, diff)},
                "worst_loss": float(diff.min()),
                "n_subjects_losing_more_than_margin": int((diff < -a.margin).sum()),
                "sensitivity": {str(m): tost(diff, m, a.alpha)["equivalent"] for m in a.sensitivity}})

    print(f"paired subjects: {res['n']}   margin: +/-{a.margin} pp   alpha: {a.alpha}")
    print(f"mean difference (reduced - full): {res['mean_diff']:+.2f} pp (SD {res['sd_diff']:.2f})")
    print(f"{100*(1-2*a.alpha):.0f}% CI: [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}] pp")
    print(f"TOST p = {res['p_tost']:.4f} -> {'equivalent within the margin' if res['equivalent'] else 'NOT shown to be equivalent'}")
    print(f"worst single-subject change: {res['worst_loss']:+.2f} pp; "
          f"subjects losing > {a.margin} pp: {res['n_subjects_losing_more_than_margin']}")
    print("verdict by margin:", ", ".join(f"{m}pp={'yes' if v else 'no'}" for m, v in res["sensitivity"].items()))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
