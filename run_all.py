# run_all.py  (repository root)
#
# Coordinator for all datasets (--dataset bnci | hgd | physionet, default bnci). Runs every experiment
# of each model in this order:
#
#   EEGNetV4 -> EEG-ITNet -> Deep4Net -> EEGConformer     (folders <Model>/BNCI_001, <Model>/HGD or
#                                                           <Model>/PhysioNet)
#
# for both class modes, then builds each model's reviewer tables and a cross-model summary.
#
# Protocols (--protocol):
#   session (default)  the dataset's OFFICIAL split. BNCI2014-001: train on session 1, test on
#                      session 2. HGD: train on the official training runs, test on the official test
#                      runs. Validation = last 25 % of the training data; refit on all of it; test
#                      once; seeds averaged (seed ensemble). Writes to <class>/results_session/,
#                      <class>/logs_session/, <model folder>/report_session/ and
#                      reports/<BNCI_001|HGD>_session/.
#   blocks             4 chronological blocks (first revision). Writes to <class>/results/,
#                      <class>/logs/, <model folder>/report/ and reports/<BNCI_001|HGD>/.
#   wide               HGD only: the session protocol with full-bandwidth profiles (250 Hz, 4-100 Hz;
#                      Schirrmeister et al. 2017 used a 4 Hz high-pass only). A sensitivity analysis next
#                      to the 4-38 Hz result; writes to <class>/results_session_wide/, logs_session_wide/,
#                      report_session_wide/ and reports/HGD_session_wide/ (never touches results_session/).
# PhysioNet has no official split: its session protocol trains on the first two imagery runs of each
# type and tests once on the last one (see <Model>/PhysioNet/README.md); blocks is not available.
# The two protocols never write into each other's folders, so earlier results are kept.
#
# For each model and class mode (2class, then 4class) the steps run in this order:
#   tune      fair, validation-only choice of the training profile (leave-one-subject-out)
#   saliency  22-channel baseline + saliency montage ("22ch", "Ours"), ERD/ERS maps, saliency-ERD
#             correlations, deletion and randomisation checks, exploratory physiology, topomaps
#   csp, relieff, mi   classical 12-channel montages
#   controls  fixed sensorimotor, random and ERD-only 12-channel montages
# and then
#   aggregate -> <model>/report*/REPORT.md (+ T*.csv, figures/) and reports/BNCI_001*/SUMMARY.md
#
# Each step runs as its own process inside its class folder; output is shown and saved to the logs
# folder. A step whose results already exist is skipped, so an interrupted run can simply be started
# again (--force redoes steps). A failed step stops the run (--keep-going continues).
#
# Examples
#   python run_all.py                                  # all 4 models, BNCI, session protocol
#   python run_all.py --dataset hgd --models eegnetv4  # EEGNetV4 on HGD
#   python run_all.py --dataset physionet --models eegnetv4   # the whole EEGNetV4 PhysioNet suite
#                                                     # (same as EEGNetV4/PhysioNet/run_suite.py)
#   python run_all.py --models eegitnet deep4net       # some models (still in the fixed order)
#   python run_all.py --classes 2class                 # one class mode
#   python run_all.py --steps aggregate                # rebuild all tables from existing results
#   python run_all.py --protocol blocks --models eegitnet --classes 4class   # finish the old protocol
# Environment variables (EEG_SEEDS, EEG_SUBJECTS, EEG_SYNTHETIC, EEG_TUNE_CANDIDATES,
# EEG_SKIP_PHYSIOLOGY, ...) are passed on. Default for every protocol except blocks: EEG_SEEDS=42,43,44.

import os
import sys
import glob
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
MODEL_DIRS = {                  # run in this order
    "eegnetv4": "EEGNetV4",
    "eegitnet": "EEGItNet",
    "deep4net": "Deep4net",
    "eegconformer": "EEGConformer",
}
DATASET_DIRS = {"bnci": "BNCI_001", "hgd": "HGD", "physionet": "PhysioNet"}
PROTOCOL_DIRS = {               # results, logs, per-model report, combined-report suffix, status suffix
    "session": ("results_session", "logs_session", "report_session", "_session", "_session"),
    "blocks": ("results", "logs", "report", "", ""),
    "wide": ("results_session_wide", "logs_session_wide", "report_session_wide", "_session_wide", "_session_wide"),
}
DATASET_PROTOCOLS = {"bnci": ["session", "blocks"], "hgd": ["session", "blocks", "wide"], "physionet": ["session"]}
BASELINE_DIR = {"bnci": "full22", "hgd": "full22", "physionet": "full64"}   # full-montage experiment


def protocol_paths(dataset, protocol):
    """(results, logs, report, combined report folder, status file) for a dataset and protocol.
    The BNCI names are unchanged from earlier runs, so their status files and reports are reused."""
    res_sub, log_sub, rep_sub, rep_suffix, status_suffix = PROTOCOL_DIRS[protocol]
    ds = DATASET_DIRS[dataset]
    status = (f"run_all_status{status_suffix}.json" if dataset == "bnci"
              else f"run_all_status_{dataset}{status_suffix}.json")
    return res_sub, log_sub, rep_sub, os.path.join(REPO, "reports", ds + rep_suffix), status
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


def is_done(results_dir, step, controls, dataset="bnci"):
    targets = [CONTROL_DONE[c] for c in controls] if step == "controls" else DONE[step]
    targets = [t.replace("full22", BASELINE_DIR[dataset]) for t in targets]
    return all(os.path.exists(os.path.join(results_dir, t)) for t in targets)


def run_step(cls_dir, step, controls, log_path, results_dir, protocol):
    cmd = [sys.executable, "-u", step_script(cls_dir, step)] + (controls if step == "controls" else [])
    wide = protocol == "wide"                         # session protocol, full-bandwidth profiles
    env = dict(os.environ, EEG_RESULTS_DIR=results_dir, EEG_PROTOCOL="session" if wide else protocol,
               EEG_BAND="wide" if wide else "narrow", PYTHONUNBUFFERED="1")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + f"   [EEG_PROTOCOL={protocol}]\n")
        proc = subprocess.Popen(cmd, cwd=cls_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def hgd_preflight(models_dirs, classes, res_sub):
    """HGD only. Returns a list of problems that would silently invalidate a run:
    1. cached train and test recordings with the same size (the MOABB cache bug: the test file is a copy of
       the train file, see analysis/fix_hgd_cache.py),
    2. existing results that were computed with such duplicates dropped (a hold-out instead of the official
       test run): the coordinator would skip those steps and report the old numbers."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fix_hgd_cache", os.path.join(REPO, "analysis", "fix_hgd_cache.py"))
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    problems = []
    subjects = [int(v) for v in os.environ.get("EEG_SUBJECTS", "").replace(" ", "").split(",") if v] or list(range(1, 15))
    root = fx.cache_root()
    bad = []
    for sid in subjects:
        size = {}
        for kind in ("train", "test"):
            canonical, legacy = fx.local_paths(root, kind, sid)
            path = legacy if os.path.isfile(legacy) else canonical          # MOABB reads the legacy copy first
            if os.path.isfile(path):
                size[kind] = os.path.getsize(path)
        if len(size) == 2 and size["train"] == size["test"]:
            bad.append(sid)
    if bad:
        problems.append("the cached HGD train and test recordings have the same size for subject(s) "
                        + ", ".join(f"S{b:02d}" for b in bad) + " (a copy of one file): run  python "
                        "analysis\\fix_hgd_cache.py  and  python analysis\\check_hgd_data.py  first")
    for d in models_dirs:
        for cls in classes:
            path = os.path.join(d, cls, res_sub, "full22", "results.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    dropped = json.load(f).get("meta", {}).get("config", {}).get("duplicated_recordings_dropped")
                if dropped:
                    problems.append(f"{os.path.relpath(path, REPO)} was computed with duplicated test recordings "
                                    f"dropped (hold-out for {len(dropped)} subjects); rename that results folder "
                                    f"(see docs/HGD_RERUN.md) or the steps would be skipped")
    return problems


def main():
    ap = argparse.ArgumentParser(description="BNCI2014-001 / HGD / PhysioNet coordinator (see the header of run_all.py)")
    ap.add_argument("--dataset", choices=list(DATASET_DIRS), default="bnci")
    ap.add_argument("--protocol", choices=list(PROTOCOL_DIRS), default=None,
                    help="default: session")
    ap.add_argument("--models", nargs="*", default=list(MODEL_DIRS), help=f"subset of {list(MODEL_DIRS)}")
    ap.add_argument("--classes", nargs="*", default=["2class", "4class"])
    ap.add_argument("--steps", nargs="*", default=STEPS + ["aggregate"],
                    help=f"subset of {STEPS + ['aggregate']}")
    ap.add_argument("--controls", nargs="*", default=["sensorimotor", "random", "erd"])
    ap.add_argument("--no-tune", action="store_true", help="skip the tuning step")
    ap.add_argument("--force", action="store_true", help="re-run steps whose results already exist")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    ap.add_argument("--allow-holdout", action="store_true",
                    help="HGD: do not stop for duplicated cached recordings / stale hold-out results")
    ap.add_argument("--margin", type=float, default=6.0, help="TOST margin for the tables (pp)")
    a = ap.parse_args()

    a.protocol = a.protocol or DATASET_PROTOCOLS[a.dataset][0]
    if a.protocol not in DATASET_PROTOCOLS[a.dataset]:
        raise SystemExit(f"--protocol {a.protocol} is not available for {a.dataset}; "
                         f"choose from {DATASET_PROTOCOLS[a.dataset]}")
    steps = [s for s in a.steps if not (a.no_tune and s == "tune")]
    unknown = set(steps) - set(STEPS + ["aggregate"])
    if unknown:
        raise SystemExit(f"unknown steps: {sorted(unknown)}")
    bad = set(a.models) - set(MODEL_DIRS)
    if bad:
        raise SystemExit(f"unknown models: {sorted(bad)}; choose from {list(MODEL_DIRS)}")
    MODELS = {m: os.path.join(REPO, d, DATASET_DIRS[a.dataset]) for m, d in MODEL_DIRS.items()}
    models = [m for m in MODELS if m in a.models]                    # always in the fixed order
    res_sub, log_sub, rep_sub, combined_out, status_name = protocol_paths(a.dataset, a.protocol)
    if a.protocol != "blocks" and not os.environ.get("EEG_SEEDS"):
        os.environ["EEG_SEEDS"] = "42,43,44"                          # needed for the seed ensemble
    print(f"[run_all] dataset={a.dataset} protocol={a.protocol} models={models} classes={a.classes} "
          f"seeds={os.environ.get('EEG_SEEDS', '42')} -> {res_sub}/")

    if a.dataset == "hgd" and not a.allow_holdout and os.environ.get("EEG_SYNTHETIC", "0") != "1":
        problems = hgd_preflight([MODELS[m] for m in models], a.classes, res_sub)
        if problems:
            print("[run_all] HGD pre-flight check failed:")
            for pr in problems:
                print("  - " + pr)
            raise SystemExit("[run_all] fix the above, or use --allow-holdout to run anyway")
        print("[run_all] HGD pre-flight check passed: train and test recordings differ, no stale hold-out results")

    status_path = os.path.join(HERE, status_name)
    status = json.load(open(status_path, encoding="utf-8")) if os.path.exists(status_path) else {}
    failed = []
    for model, cls in [(m, c) for m in models for c in a.classes]:
        cls_dir = os.path.join(MODELS[model], cls)
        results_dir = os.path.join(cls_dir, res_sub)
        for step in [s for s in STEPS if s in steps]:
            key = f"{model}/{cls}/{step}"
            if not a.force and is_done(results_dir, step, a.controls, a.dataset):
                print(f"[run_all] {key}: results exist - skipped (use --force to redo)")
                status[key] = dict(status.get(key, {}), state="skipped (done)")
                continue
            print("\n" + "#" * 90 + f"\n[run_all] {key}  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n" + "#" * 90)
            log_path = os.path.join(cls_dir, log_sub, f"{step}.log")
            t0 = time.time()
            rc = run_step(cls_dir, step, a.controls, log_path, results_dir, a.protocol)
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
            cmd = [sys.executable, AGGREGATOR, MODELS[model], "--classes", *a.classes, "--margin", str(a.margin),
                   "--results-subdir", res_sub, "--out", os.path.join(MODELS[model], rep_sub)]
            print("\n[run_all] building tables: " + " ".join(cmd))
            if subprocess.call(cmd, cwd=REPO) != 0:
                failed.append(f"aggregate {model}")
        cmd = [sys.executable, AGGREGATOR, "--combine", *MODELS.values(), "--classes", *a.classes,
               "--results-subdir", res_sub, "--out", combined_out]
        print("\n[run_all] cross-model summary: " + " ".join(cmd))
        if subprocess.call(cmd, cwd=REPO) != 0:
            failed.append("aggregate combined")

    print("\n[run_all] summary:")
    for k, v in status.items():
        print(f"  {k:<34} {v.get('state', '')}  {v.get('minutes', '')} min")
    if failed:
        print("[run_all] FAILED steps: " + ", ".join(failed))
        raise SystemExit(1)
    print("[run_all] done. Tables: " + ", ".join(os.path.join(MODELS[m], rep_sub, "REPORT.md") for m in models)
          + f" and {os.path.join(combined_out, 'SUMMARY.md')}")


if __name__ == "__main__":
    main()
