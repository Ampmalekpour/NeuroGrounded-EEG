# run_all.py  (repository root)
#
# Coordinator for BNCI2014-001: runs every experiment of EEGNetV4 first and then of EEG-ITNet,
# for both class modes, builds each model's reviewer tables and finally a cross-model summary.
#
#   EEGNetV4/BNCI_001  ->  EEGItNet/BNCI_001
#
# For each model and class mode (2class, then 4class) the steps run in this order:
#   tune      fair, validation-only choice of the training profile (leave-one-subject-out)
#   saliency  22-channel baseline + saliency montage ("22ch", "Ours"), ERD/ERS maps, saliency-ERD
#             correlations (sign-invariant + original), deletion and randomisation checks,
#             exploratory physiology correlations, topomaps
#   csp, relieff, mi   classical 12-channel montages
#   controls  fixed sensorimotor, random and ERD-only 12-channel montages
# and then
#   aggregate -> <model folder>/report/REPORT.md (+ T*.csv, figures/) for every model, and
#                reports/BNCI_001/SUMMARY.md (+ C*.csv) comparing the models
#
# Each step runs as its own process inside its class folder. Output is shown and also saved to
# <model folder>/<class>/logs/<step>.log. A step whose results already exist is skipped, so an
# interrupted run can simply be started again (use --force to redo steps).
#
# Examples
#   python run_all.py                                 # everything: EEGNetV4 then EEG-ITNet
#   python run_all.py --models eegitnet               # one model only
#   python run_all.py --classes 2class                # one class mode
#   python run_all.py --steps saliency csp            # some steps only
#   python run_all.py --steps aggregate               # rebuild all tables from existing results
#   python run_all.py --no-tune                       # skip tuning (profile 'original' unless tuned before)
# Environment variables (EEG_SEEDS, EEG_SUBJECTS, EEG_SYNTHETIC, EEG_SKIP_PHYSIOLOGY, ...) are passed on.

import os
import sys
import glob
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
MODELS = {                      # run in this order
    "eegnetv4": os.path.join(REPO, "EEGNetV4", "BNCI_001"),
    "eegitnet": os.path.join(REPO, "EEGItNet", "BNCI_001"),
}
COMBINED_OUT = os.path.join(REPO, "reports", "BNCI_001")
AGGREGATOR = os.path.join(REPO, "analysis", "aggregate_results.py")
STEPS = ["tune", "saliency", "csp", "relieff", "mi", "controls"]
DONE = {
    "tune": ["selected_profile.json"],
    "saliency": ["full22/results.json", "saliency_reduced/results.json"],
    "csp": ["csp_reduced/results.json"],
    "relieff": ["relieff_reduced/results.json"],
    "mi": ["mi_reduced/results.json"],
}
CONTROL_DONE = {"sensorimotor": "control_sensorimotor/results.json",
                "erd": "control_erd/results.json",
                "random": "control_random_draw4/results.json"}


def step_script(cls_dir, step):
    found = glob.glob(os.path.join(cls_dir, f"*_{step}.py"))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one *_{step}.py in {cls_dir}, found {found}")
    return found[0]


def is_done(results_dir, step, controls):
    targets = [CONTROL_DONE[c] for c in controls] if step == "controls" else DONE[step]
    return all(os.path.exists(os.path.join(results_dir, t)) for t in targets)


def run_step(cls_dir, step, controls, log_path):
    cmd = [sys.executable, "-u", step_script(cls_dir, step)] + (controls if step == "controls" else [])
    env = dict(os.environ, EEG_RESULTS_DIR=os.path.join(cls_dir, "results"), PYTHONUNBUFFERED="1")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=cls_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(MODELS), help=f"subset of {list(MODELS)}")
    ap.add_argument("--classes", nargs="*", default=["2class", "4class"])
    ap.add_argument("--steps", nargs="*", default=STEPS + ["aggregate"],
                    help=f"subset of {STEPS + ['aggregate']}")
    ap.add_argument("--controls", nargs="*", default=["sensorimotor", "random", "erd"])
    ap.add_argument("--no-tune", action="store_true", help="skip the tuning step")
    ap.add_argument("--force", action="store_true", help="re-run steps whose results already exist")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    ap.add_argument("--margin", type=float, default=6.0, help="TOST margin for the tables (pp)")
    a = ap.parse_args()

    steps = [s for s in a.steps if not (a.no_tune and s == "tune")]
    unknown = set(steps) - set(STEPS + ["aggregate"])
    if unknown:
        raise SystemExit(f"unknown steps: {sorted(unknown)}")

    bad = set(a.models) - set(MODELS)
    if bad:
        raise SystemExit(f"unknown models: {sorted(bad)}; choose from {list(MODELS)}")
    models = [m for m in MODELS if m in a.models]
    status_path = os.path.join(HERE, "run_all_status.json")
    status = json.load(open(status_path, encoding="utf-8")) if os.path.exists(status_path) else {}
    failed = []
    for model, cls in [(m, c) for m in models for c in a.classes]:      # EEGNetV4 first, then EEG-ITNet
        cls_dir = os.path.join(MODELS[model], cls)
        results_dir = os.path.join(cls_dir, "results")
        for step in [s for s in STEPS if s in steps]:
            key = f"{model}/{cls}/{step}"
            if not a.force and is_done(results_dir, step, a.controls):
                print(f"[run_all] {key}: results exist - skipped (use --force to redo)")
                status[key] = dict(status.get(key, {}), state="skipped (done)")
                continue
            print("\n" + "#" * 90 + f"\n[run_all] {key}  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n" + "#" * 90)
            log_path = os.path.join(cls_dir, "logs", f"{step}.log")
            t0 = time.time()
            rc = run_step(cls_dir, step, a.controls, log_path)
            status[key] = {"state": "ok" if rc == 0 else f"FAILED (exit {rc})",
                           "minutes": round((time.time() - t0) / 60, 1), "log": os.path.relpath(log_path, HERE),
                           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=1)
            if rc != 0:
                failed.append(key)
                print(f"[run_all] {key} FAILED - see {log_path}")
                if not a.keep_going:
                    raise SystemExit(1)

    if "aggregate" in steps:
        for model in models:
            cmd = [sys.executable, AGGREGATOR, MODELS[model], "--classes", *a.classes, "--margin", str(a.margin)]
            print("\n[run_all] building tables: " + " ".join(cmd))
            if subprocess.call(cmd, cwd=REPO) != 0:
                failed.append(f"aggregate {model}")
        cmd = [sys.executable, AGGREGATOR, "--combine", *[MODELS[m] for m in MODELS],
               "--classes", *a.classes, "--out", COMBINED_OUT]
        print("\n[run_all] cross-model summary: " + " ".join(cmd))
        if subprocess.call(cmd, cwd=REPO) != 0:
            failed.append("aggregate combined")

    print("\n[run_all] summary:")
    for k, v in status.items():
        print(f"  {k:<30} {v.get('state', '')}  {v.get('minutes', '')} min")
    if failed:
        print("[run_all] FAILED steps: " + ", ".join(failed))
        raise SystemExit(1)
    print("[run_all] done. Tables: " + ", ".join(os.path.join(MODELS[m], "report", "REPORT.md") for m in models)
          + f" and {os.path.join(COMBINED_OUT, 'SUMMARY.md')}")


if __name__ == "__main__":
    main()
