# eegconformer_hgd_common.py
#
# Shared code for all EEGConformer / High-Gamma Dataset (Schirrmeister2017) experiments, second revision.
# It is imported by the scripts in the 2class/ and 4class/ folders next to this file:
#   *_tune.py       validation-only, leave-one-subject-out choice of the training profile
#   *_saliency.py   full 22-channel baseline ("22ch") + saliency-ranked 12 channels ("Ours"),
#                   saliency-ERD alignment (sign-invariant + original correlations), deletion and
#                   model-randomisation checks, exploratory physiology correlations
#   *_csp.py / *_mi.py / *_relieff.py   CSP / mutual-information / ReliefF 12-channel montages
#   *_controls.py   fixed / random / ERD-only montages
# The root run_all.py (--dataset hgd) runs everything in order and builds the reviewer tables.
#
# Provided original scripts (kept in legacy_unrevised/original_scripts/): EEGConformer_BD_HGD_BD_Final.py, EEGConformer_BD_HGD_BD_Final_2_LR.py (both 4-class), EEGConformer_BD_HGD_BD_Final_2_LR_REST.py, EEGConformer_Channel_Reduction_{CSP,MutualInformation,ReliefF}.py
#
# Fixes relative to the original HGD scripts (docs/REVIEW_CROSSWALK.md, section I):
#   * labels: braindecode maps HGD events ALPHABETICALLY (feet=0, left_hand=1, rest=2, right_hand=3);
#     the EEGNetv4 / Deep4Net (and EEG-ITNet CSP/MI/ReliefF) originals kept labels [0, 1], i.e. trained
#     FEET vs LEFT HAND while calling it left vs right. 2class is now left vs right hand (asserted).
#   * windows: the originals' stop offset of +4 s produced 0-8 s windows (into the next trial);
#     windows are now relative to the cue (0-4 s unless a profile says otherwise)
#   * official HGD split: train on the training runs (last 25 % = validation), refit, test once on the
#     official test runs, seed ensemble; the test runs are never used for any choice
#   * saliency on VALIDATION trials (originals: test fold); attribution methods are NOT chosen by
#     their ERD correlation; leave-one-subject-out montages for every method (originals: one global
#     montage from all 14 subjects, test trials included)
#   * checkpoints / LR schedules on validation loss only (EEGConformer originals used the test fold)
#   * ERD maps from raw (non-standardised) data with a PRE-cue baseline (originals: 0-0.5 s after the cue)
#   * 2class AND 4class for every model (EEG-ITNet and Deep4Net originals had no 4-class version)
#   * per-trial predictions, confusion matrices, class counts, several seeds, TOST tables
#
# Environment variables (all optional): EEG_PROTOCOL (session | blocks), EEG_SEEDS, EEG_SUBJECTS,
# EEG_MAX_EPOCHS, EEG_PATIENCE, EEG_RESULTS_DIR, EEG_SYNTHETIC=1 (dry run), EEG_PROFILE,
# EEG_TUNE_CANDIDATES, EEG_REFIT=0, EEG_SKIP_PHYSIOLOGY=1.

import os
import gc
import csv
import copy
import json
import random
import hashlib
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from scipy.stats import spearmanr, t as student_t

import mne
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from braindecode.datasets import MOABBDataset
from braindecode.models import EEGConformer
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
)

mne.set_log_level("WARNING")


# ────────────────────────────────────────────────
# Environment helpers / configuration
# ────────────────────────────────────────────────
def _env_int_list(name, default):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [int(v) for v in raw.replace(" ", "").split(",") if v]


def _env_str_list(name, default):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [v for v in raw.replace(" ", "").split(",") if v]


SEEDS = _env_int_list("EEG_SEEDS", [42])
SUBJECT_IDS = _env_int_list("EEG_SUBJECTS", range(1, 10))
MODEL_NAME = "EEGConformer"
CLASS_MODE = os.environ.get("EEG_CLASS_MODE", "")
if CLASS_MODE not in ("2class", "4class"):
    raise SystemExit("Run the scripts inside the 2class/ or 4class/ folders "
                     "(they set EEG_CLASS_MODE before importing this module).")
SYNTHETIC = os.environ.get("EEG_SYNTHETIC", "0") == "1"

HERE = os.path.dirname(os.path.abspath(__file__))
# ---- Evaluation protocol --------------------------------------------------------------------
# session (default, folder name kept for all datasets): the OFFICIAL HGD split. Train on the
#   training runs ("0train"), test on the test runs ("1test"). The last VAL_FRACTION of the training
#   runs (chronological) is the validation set used for early stopping, LR scheduling, profile tuning
#   and saliency. The model is then refitted on ALL training runs with the chosen number of epochs (and
#   the same LR schedule), and the softmax outputs of all seeds are averaged (seed ensemble). The test
#   runs are scored once, at the end, and never used for any choice. Writes to results_session/.
# blocks: 4 chronological blocks within the pooled runs (train / validation / test). Writes to
#   results/, the folder that holds the FIRST-revision HGD results (different code: they used the
#   0-8 s windows), so do not mix the two.
PROTOCOL = os.environ.get("EEG_PROTOCOL", "session")
if PROTOCOL not in ("session", "blocks"):
    raise SystemExit("EEG_PROTOCOL must be 'session' or 'blocks'")
VAL_FRACTION = 0.25
REFIT = os.environ.get("EEG_REFIT", "1") == "1"
MIN_REFIT_EPOCHS = 20
RESULTS_ROOT = os.environ.get("EEG_RESULTS_DIR",
                              os.path.join(HERE, "results" if PROTOCOL == "blocks" else "results_session"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE, "| class mode:", CLASS_MODE, "| seeds:", SEEDS, "| subjects:", list(SUBJECT_IDS),
      "| synthetic:", SYNTHETIC, "| protocol:", PROTOCOL)

# Every preprocessing / training constant is written to results.json (config_snapshot), so the
# paper can quote exactly what was run.
DATASET_NAME = "Schirrmeister2017"

TARGET_SFREQ = 250               # default only; every profile sets its own sampling rate ("sfreq")
EMS_FACTOR_NEW = 1e-3
EMS_INIT_BLOCK = 1000
# braindecode gives every MOABB event the dataset interval as duration: Schirrmeister2017 interval is
# [0, 4] s, i.e. the event starts at the cue and lasts 4 s. Windows below are relative to the cue.
EVENT_DURATION_S = 4.0
# HGD keeps its official train/test split in the RUN name ("0train", "1test"), not in the session
SPLIT_KEY = "run"

# ---- Training / preprocessing profiles -----------------------------------------------------
# "original" = the provided HGD scripts for this model, with the bugs fixed (see header). The others
# are literature settings for this architecture. *_tune.py picks one per subject using ONLY the other
# subjects' VALIDATION accuracy (the test runs are never scored); whatever is chosen is used for every
# method (22ch, Ours, CSP, MI, ReliefF, controls), so the comparison stays fair.
_DROP = {"2class": 0.6, "4class": 0.5}[CLASS_MODE]
_BUTTER4 = dict(order=4, ftype="butter")
# the provided HGD EEGConformer scripts: 22 channels average-referenced, 500 Hz, 4-125 Hz IIR,
# 0-4 s, AdamW lr 1e-3 / wd 1e-4, batch 32 + segmentation-reconstruction, 100 epochs, no label
# smoothing, no clipping. Two fixes: the LR scheduler and the checkpoint now use VALIDATION loss
# (the originals used the TEST fold), and EMS replaces the per-fold z-score.
_BASE = dict(sfreq=500, band_hz=(4.0, 125.0), iir=_BUTTER4, window_s=(0.0, 4.0), norm="ems", car=True,
             lr=1e-3, weight_decay=1e-4, scheduler="plateau", plateau_factor=0.5, plateau_patience=10,
             max_epochs=100, patience=100, batch_size=32, label_smoothing=0.0, grad_clip=None,
             dropout=_DROP, augment="sr")
PROFILES = {
    "original": dict(_BASE),
    # EEGConformer paper setting (Song et al. 2022): 250 Hz, 4-40 Hz, Adam lr 2e-4, betas (0.5, 0.999),
    # no weight decay, constant learning rate, batch 72 with segmentation-reconstruction
    "song2022": dict(_BASE, sfreq=250, band_hz=(4.0, 40.0), lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0,
                     scheduler="none", max_epochs=300, patience=100, batch_size=72),
}
DEFAULT_PROFILE = "original"
TUNE_CANDIDATES = ["original", "song2022"]
CSP_REG = 1e-4        # average-referenced data are rank-deficient; plain CSP fails
AUGMENT = dict(mixup_alpha=0.2, scale=(0.8, 1.2), noise_rel_std=0.05, channel_drop_p=0.1, sr_segments=8)
SELECTED_PROFILE_FILE = os.path.join(RESULTS_ROOT, "selected_profile.json")

# deletion (faithfulness) test: channels removed from the TEST block of each fold
FAITH_K = [2, 4, 6]
FAITH_RANDOM_DRAWS = 10

# recipes without early stopping (fixed epochs + SWA) set this to True: their refit simply uses
# max_epochs, and models that are not needed for saliency skip the first (validation) fit
FIXED_EPOCHS = False
# profiles tried by *_tune.py unless EEG_TUNE_CANDIDATES is set


def profile_name_for(subject_id=None):
    """EEG_PROFILE (if set) > tuned choice in results/selected_profile.json > 'original'."""
    if os.environ.get("EEG_PROFILE"):
        return os.environ["EEG_PROFILE"]
    if os.path.exists(SELECTED_PROFILE_FILE):
        with open(SELECTED_PROFILE_FILE, encoding="utf-8") as f:
            sel = json.load(f)
        return sel["per_subject"].get(str(subject_id), sel["overall"])
    return DEFAULT_PROFILE


def get_cfg(subject_id=None, name=None):
    name = name or profile_name_for(subject_id)
    if name not in PROFILES:
        raise SystemExit(f"Unknown profile {name!r}; choose from {list(PROFILES)}")
    cfg = dict(PROFILES[name], name=name)
    if os.environ.get("EEG_MAX_EPOCHS"):
        cfg["max_epochs"] = int(os.environ["EEG_MAX_EPOCHS"])
    if os.environ.get("EEG_PATIENCE"):
        cfg["patience"] = int(os.environ["EEG_PATIENCE"])
    return cfg


IG_STEPS = 32
SMOOTHGRAD_SAMPLES = 12
SMOOTHGRAD_NOISE_STD = 0.10
ATTR_METHODS = ["grad", "grad_x_input", "integrated_gradients", "smoothgrad"]

FULL_CHANNELS = [                # the 22 sensorimotor channels of the provided HGD scripts
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Pz",
]
REDUCED_N_CHANNELS = 12

MONTAGE = mne.channels.make_standard_montage("standard_1005")

# ERD/ERS reference intervals, seconds relative to the cue.
ERD_BASELINE_S = (-0.5, 0.0)     # PRE-cue baseline (as described in the manuscript)
ERD_ACTIVE_S = (1.0, 3.0)
ERD_BAND_HZ = (8.0, 30.0)
ERD_SFREQ = 128                  # ERD/physiology maps: same sampling rate for every model


def config_snapshot():
    selection = None
    if os.path.exists(SELECTED_PROFILE_FILE):
        with open(SELECTED_PROFILE_FILE, encoding="utf-8") as f:
            selection = json.load(f)
    return {
        "model": MODEL_NAME, "class_mode": CLASS_MODE, "dataset": DATASET_NAME,
        "note_dataset": "High-Gamma Dataset (Schirrmeister2017): EXECUTED movements, 14 subjects, official "
                        "training runs (~880 trials) and test runs (~160 trials).",
        "sfreq_hz": "per profile (profiles[*].sfreq)",
        "ems_params": {"factor_new": EMS_FACTOR_NEW, "init_block_size": EMS_INIT_BLOCK,
                       "used_when": "profile norm == 'ems'"},
        "split_key": SPLIT_KEY,
        "event_duration_s": EVENT_DURATION_S,
        "profiles": PROFILES, "augmentation": AUGMENT, "profile_default": DEFAULT_PROFILE,
        "profile_selection": selection,
        "env_overrides": {k: os.environ.get(k) for k in ("EEG_PROFILE", "EEG_MAX_EPOCHS", "EEG_PATIENCE")},
        "model_fixed": {"n_filters_time": 40, "filter_time_length": 25, "pool": [75, 15], "size_per_class": CONFORMER_SIZE},
        "optimizer": "AdamW (betas per profile)",
        "scheduler": "per profile: plateau = ReduceLROnPlateau on VALIDATION loss; cosine; none. "
                     "Refit replays the plateau learning rates of the first fit",
        "checkpoint_rule": "lowest validation loss, early stopping with patience; refit keeps final weights",
        "folds": FOLDS_NOTE[PROTOCOL],
        "protocol": {"name": PROTOCOL, "val_fraction": VAL_FRACTION, "refit": REFIT,
                     "min_refit_epochs": MIN_REFIT_EPOCHS, "fixed_epochs": FIXED_EPOCHS,
                     "seed_ensemble": PROTOCOL == "session" and len(SEEDS) > 1},
        "attribution": {"methods": ATTR_METHODS, "ig_steps": IG_STEPS,
                        "smoothgrad_samples": SMOOTHGRAD_SAMPLES,
                        "smoothgrad_noise_std": SMOOTHGRAD_NOISE_STD,
                        "target": "pre-softmax logit of the true class",
                        "computed_on": "validation block of each fold"},
        "faithfulness": {"k": FAITH_K, "random_draws": FAITH_RANDOM_DRAWS,
                         "removal": "channel set to 0 (its standardised mean) on the TEST block; "
                                    "evaluation only, never used for any choice",
                         "ranking": "mean of the four attribution maps from the VALIDATION block"},
        "randomisation_check": "Spearman between trained-model and untrained-model saliency",
        "erd": {"baseline_s": ERD_BASELINE_S, "active_s": ERD_ACTIVE_S, "sfreq_hz": ERD_SFREQ,
                "band_hz": ERD_BAND_HZ, "standardisation": "none"},
        "physiology_exploratory": {"sfreq_hz": PHYS_SFREQ, "bands_hz": PHYS_BANDS,
                                   "epoch_s": PHYS_KEEP_S, "used_for_decisions": False},
        "full_channels": FULL_CHANNELS, "reduced_n_channels": REDUCED_N_CHANNELS,
        "seeds": SEEDS, "subject_ids": list(SUBJECT_IDS), "label_names": LABEL_NAMES,
        "versions": {"torch": torch.__version__, "mne": mne.__version__, "numpy": np.__version__},
    }


# ────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fold_seed(seed, subject_id, fold_i):
    """Unique, order-independent seed for every (seed, subject, fold): the model is
    re-initialised independently for each fold, and a result does not depend on
    which scripts were run before it."""
    return int(seed) * 1_000_003 + int(subject_id) * 1009 + int(fold_i)


# ────────────────────────────────────────────────
# Labels
#
# braindecode builds the event mapping with np.unique() over the annotation
# descriptions, i.e. ALPHABETICALLY:
#     0 = feet, 1 = left_hand, 2 = rest, 3 = right_hand
# load_subject_windows() asserts this against the annotations of every subject.
# ────────────────────────────────────────────────
LABEL_NAMES = ["feet", "left_hand", "rest", "right_hand"]
FEET_LABEL, LEFT_HAND_LABEL, REST_LABEL, RIGHT_HAND_LABEL = 0, 1, 2, 3


def class_ids(class_mode):
    """(left_id, right_id) in the label space of the given class mode."""
    if class_mode == "2class":
        return 0, 1
    if class_mode == "4class":
        return LEFT_HAND_LABEL, RIGHT_HAND_LABEL
    raise ValueError(f"Unknown class_mode: {class_mode!r}")


def class_mask(y, class_mode):
    """Trials kept by a class mode (same rule as apply_class_mode)."""
    if class_mode == "2class":
        return np.isin(y, [LEFT_HAND_LABEL, RIGHT_HAND_LABEL])
    return np.ones(len(y), dtype=bool)


def _window_sessions(windows):
    """Split index per window: 0 = official training runs, 1 = official test runs. HGD stores the
    split in the run name ('0train', '1test'), so sorting the names gives the order."""
    names = []
    for d in windows.datasets:
        desc = d.description
        names.extend([str(desc[SPLIT_KEY]) if SPLIT_KEY in desc else "0"] * len(d))
    order = sorted(set(names))
    return np.array([order.index(n) for n in names], dtype=np.int64)


def apply_class_mode(X, y, class_mode):
    if class_mode == "2class":      # left hand vs right hand only, remapped to 0/1
        mask = np.isin(y, [LEFT_HAND_LABEL, RIGHT_HAND_LABEL])
        return X[mask], np.where(y[mask] == LEFT_HAND_LABEL, 0, 1).astype(np.int64)
    if class_mode == "4class":
        return X, y.astype(np.int64)
    raise ValueError(f"Unknown class_mode: {class_mode!r} (use '2class' or '4class')")


def class_names(class_mode):
    return ["left_hand", "right_hand"] if class_mode == "2class" else list(LABEL_NAMES)


# ────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────
def scale_to_microvolts(x):
    return x * 1e6


def extract_xy(ds):
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i][:2]
        xs.append(np.asarray(x, dtype=np.float32))
        ys.append(int(y))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def _check_label_mapping(dataset, y_all):
    """Fail loudly if braindecode's integer labels are not the alphabetical mapping
    assumed everywhere in this code base."""
    names = []
    for d in dataset.datasets:
        names.extend([n for n in np.asarray(d.raw.annotations.description) if n in LABEL_NAMES])
    found = sorted(set(names))
    assert found == sorted(LABEL_NAMES), (
        f"Unexpected annotation names {found}; expected {sorted(LABEL_NAMES)}")
    if len(names) == len(y_all):
        expected = np.array([LABEL_NAMES.index(n) for n in names])
        assert np.array_equal(expected, y_all), (
            "Integer labels do not match annotation names - label mapping assumption broken")
    else:
        print(f"  [label check] {len(names)} annotated events vs {len(y_all)} windows "
              f"(some windows dropped); name check passed, order check skipped")


def _synthetic_windows(channels, n_times, seed, n_per_class=60, pre_samples=0, sfreq=None):
    """Synthetic 4-class data with lateralised mu-band-like variance changes, used only
    for pipeline dry-runs (EEG_SYNTHETIC=1). Returns X (trials, ch, n_times), y (0..3)."""
    rng = np.random.default_rng(seed)
    n = n_per_class * 4
    y = np.tile(np.arange(4), n_per_class).astype(np.int64)
    X = rng.standard_normal((n, len(channels), n_times)).astype(np.float32)
    ch = {c: i for i, c in enumerate(channels)}
    for k in range(n):
        gain = np.ones(len(channels), dtype=np.float32)
        if y[k] == LEFT_HAND_LABEL and "C4" in ch:      # contralateral drop in power
            gain[ch["C4"]] = 0.4
        if y[k] == RIGHT_HAND_LABEL and "C3" in ch:
            gain[ch["C3"]] = 0.4
        if y[k] == FEET_LABEL and "Cz" in ch:
            gain[ch["Cz"]] = 0.4
        X[k, :, pre_samples + int(1.0 * (sfreq or TARGET_SFREQ)):] *= gain[:, None]
    return X, y


def window_n_samples(cfg):
    start, stop = cfg["window_s"]
    return int(round((stop - start) * cfg["sfreq"]))


def common_average_reference(x):
    """Average reference over the picked channels (used by the EEGConformer profiles)."""
    return x - x.mean(axis=0, keepdims=True)


def load_subject_windows(subject_id, channels, class_mode, cfg=None):
    """Load one subject with the profile `cfg` (default: the subject's selected profile) and
    return (X, y, info). Every preprocessing choice (sampling rate, band, filter type, average
    reference, standardisation) comes from the profile. The window is relative to the cue;
    braindecode's stop offset is counted from the END of the 4-s event, hence the
    EVENT_DURATION_S correction (the original HGD scripts used +4 s and got 0-8 s windows)."""
    cfg = cfg or get_cfg(subject_id)
    sf = cfg["sfreq"]
    start_s, stop_s = cfg["window_s"]
    n_expected = window_n_samples(cfg)
    if SYNTHETIC:
        X, y = _synthetic_windows(list(channels), n_expected, seed=subject_id)
        sessions = (np.arange(len(y)) >= int(0.85 * len(y))).astype(np.int64)   # ~ HGD train/test sizes
    else:
        dataset = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
        preprocessors = [
            Preprocessor("pick_channels", ch_names=list(channels), ordered=True),
            Preprocessor(scale_to_microvolts, apply_on_array=True),
        ]
        if cfg.get("car"):
            preprocessors.append(Preprocessor(common_average_reference, apply_on_array=True))
        preprocessors += [
            Preprocessor("resample", sfreq=sf),
            Preprocessor("filter", l_freq=cfg["band_hz"][0], h_freq=cfg["band_hz"][1],
                         **({"method": "iir", "iir_params": cfg["iir"]} if cfg.get("iir") else {})),
        ]
        if cfg.get("norm") == "ems":
            preprocessors.append(Preprocessor(exponential_moving_standardize,
                                              factor_new=EMS_FACTOR_NEW,
                                              init_block_size=EMS_INIT_BLOCK))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocess(dataset, preprocessors, n_jobs=1)
        windows = create_windows_from_events(
            dataset,
            trial_start_offset_samples=int(round(start_s * sf)),
            trial_stop_offset_samples=int(round((stop_s - EVENT_DURATION_S) * sf)),
            preload=True,
        )
        X, y = extract_xy(windows)
        sessions = _window_sessions(windows)
        _check_label_mapping(dataset, y)
        del dataset, windows
        gc.collect()
        assert X.shape[2] == n_expected, (
            f"window has {X.shape[2]} samples, expected {n_expected} for {cfg['window_s']} s; "
            f"check EVENT_DURATION_S")
    sessions = sessions[class_mask(y, class_mode)]
    X, y = apply_class_mode(X, y, class_mode)

    names = class_names(class_mode)
    counts = {names[c]: int((y == c).sum()) for c in range(len(names))}
    info = {"n_trials": int(len(y)), "n_channels": int(X.shape[1]), "n_times": int(X.shape[2]),
            "class_counts": counts, "profile": cfg["name"], "window_s": list(cfg["window_s"]),
            "sessions": sessions, "n_per_session": [int((sessions == s).sum()) for s in (0, 1)]}
    print(f"  S{subject_id:02d} [{class_mode}] profile={cfg['name']} X={X.shape} class counts={counts}")
    assert len(np.unique(y)) == len(names), f"Missing classes: {counts}"
    return X.astype(np.float32), y.astype(np.int64), info


def make_blockwise_folds(n_samples, n_blocks=4):
    assert n_samples >= n_blocks
    blocks = np.array_split(np.arange(n_samples), n_blocks)
    folds = []
    for test_i in range(n_blocks):
        val_i = (test_i - 1) % n_blocks
        train_blocks = [b for b in range(n_blocks) if b not in (val_i, test_i)]
        train_idx = np.concatenate([blocks[b] for b in train_blocks])
        folds.append((train_idx, blocks[val_i], blocks[test_i]))
    return folds


FOLDS_NOTE = {
    "session": "train on the official HGD training runs (last 25% = validation, chronological), refit "
               "on all training runs, test once on the official test runs",
    "blocks": "4 chronological blocks; test block i, validation block i-1, rest train",
}


def make_folds(n_samples, info=None):
    """[(train, validation, test), ...] for the current PROTOCOL."""
    if PROTOCOL == "blocks":
        return make_blockwise_folds(n_samples)
    s = np.asarray(info["sessions"])
    first, second = np.where(s == 0)[0], np.where(s == 1)[0]
    if len(first) == 0 or len(second) == 0:
        raise RuntimeError("session protocol needs two sessions; found "
                           f"{len(first)} / {len(second)} trials. Use EEG_PROTOCOL=blocks.")
    n_val = max(1, int(round(VAL_FRACTION * len(first))))
    return [(first[:-n_val], first[-n_val:], second)]


def load_selection_trials(subject_id, channels, class_mode, cfg=None):
    """Trials that channel-SELECTION methods (CSP, MI, ReliefF) may use: under the session protocol
    only session 1, so session 2 never influences any choice (montages are also leave-one-subject-out)."""
    X, y, info = load_subject_windows(subject_id, channels, class_mode, cfg)
    if PROTOCOL == "session":
        keep = np.asarray(info["sessions"]) == 0
        X, y = X[keep], y[keep]
    return X, y, info


def batch_iter(X, y, batch_size, shuffle=True):
    idx = np.arange(len(y))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sl = idx[start:start + batch_size]
        yield (torch.tensor(X[sl], dtype=torch.float32, device=DEVICE),
               torch.tensor(y[sl], dtype=torch.long, device=DEVICE))


@torch.no_grad()
def predict(model, X, batch_size=256):
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=DEVICE)
        out.append(torch.argmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_proba(model, X, batch_size=256):
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=DEVICE)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate(model, X, y, criterion, batch_size=256):
    model.eval()
    losses, preds = [], []
    for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
        out = model(xb)
        losses.append(criterion(out, yb).item())
        preds.append(torch.argmax(out, dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    return float(np.mean(losses)), float((preds == y).mean())


# ────────────────────────────────────────────────
# Channel-importance helpers
# ────────────────────────────────────────────────
def normalize_importance(vec):
    vec = np.asarray(vec, dtype=np.float64)
    m = np.max(np.abs(vec))
    return (vec / m).astype(np.float32) if m > 1e-12 else vec.astype(np.float32)


def top_k_channels(scores, ch_names, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(ch_names[i], float(scores[i])) for i in idx]


def loso_montage(subject_scores, ch_names, k):
    """subject_scores: {subject_id: 1D score vector over ch_names}.
    Returns {subject_id: [k channel names]} where subject S's montage is built ONLY
    from the other subjects' scores (leave-one-subject-out)."""
    ids = list(subject_scores.keys())
    assert len(ids) >= 2, "LOSO montages need at least two subjects"
    montages = {}
    for held_out in ids:
        others = [subject_scores[s] for s in ids if s != held_out]
        avg = normalize_importance(np.mean(np.stack(others, axis=0), axis=0))
        montages[held_out] = [ch for ch, _ in top_k_channels(avg, ch_names, k)]
    return montages


def full_montages():
    return {sid: list(FULL_CHANNELS) for sid in SUBJECT_IDS}


# ────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────
CONFORMER_SIZE = {
    "2class": dict(att_depth=2, att_heads=4),
    "4class": dict(att_depth=6, att_heads=10),       # as in the provided 4-class HGD scripts
}


def build_model(n_chans, n_times, n_classes, cfg=None):
    """braindecode EEGConformer (40 temporal filters of 25 samples, pooling 75 / stride 15, as in the
    provided scripts); depth/heads per class mode, dropout from the profile."""
    cfg = cfg or get_cfg()
    size = CONFORMER_SIZE[CLASS_MODE]
    common = dict(n_outputs=n_classes, n_chans=n_chans, n_times=n_times, n_filters_time=40,
                  filter_time_length=25, pool_time_length=75, pool_time_stride=15,
                  drop_prob=cfg["dropout"], att_drop_prob=cfg["dropout"], final_fc_length="auto")
    try:                                   # current braindecode argument names
        model = EEGConformer(num_layers=size["att_depth"], num_heads=size["att_heads"], **common)
    except TypeError:                      # older names
        model = EEGConformer(att_depth=size["att_depth"], att_heads=size["att_heads"], **common)
    return model.to(DEVICE)


def augment_batch(xb):
    """Manuscript Sec. 3.4 augmentation (training batches only): random amplitude scaling,
    additive Gaussian noise and random channel dropout."""
    B, C_, _ = xb.shape
    lo, hi = AUGMENT["scale"]
    xb = xb * torch.empty(B, 1, 1, device=xb.device).uniform_(lo, hi)
    xb = xb + torch.randn_like(xb) * AUGMENT["noise_rel_std"] * xb.std(dim=(1, 2), keepdim=True)
    keep = (torch.rand(B, C_, 1, device=xb.device) >= AUGMENT["channel_drop_p"]).float()
    return xb * keep


def mixup(xb, yb, alpha):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(xb.shape[0], device=xb.device)
    return lam * xb + (1.0 - lam) * xb[perm], yb, yb[perm], lam


def sr_augment(xb, yb, n_seg):
    """Segmentation and reconstruction (Song et al., 2022, the EEGConformer augmentation): one new
    trial per trial in the batch, assembled from n_seg time segments taken from random trials of the
    same class. The new trials are appended, doubling the batch (as in the original HGD script)."""
    T = xb.shape[2]
    edges = np.linspace(0, T, n_seg + 1).astype(int)
    new = torch.empty_like(xb)
    for c in torch.unique(yb):
        idx = torch.where(yb == c)[0]
        for s in range(n_seg):
            pick = idx[torch.randint(len(idx), (len(idx),), device=xb.device)]
            new[idx, :, edges[s]:edges[s + 1]] = xb[pick, :, edges[s]:edges[s + 1]]
    return torch.cat([xb, new]), torch.cat([yb, yb])


def make_scheduler(optimizer, cfg, max_epochs):
    """(scheduler, steps_on_validation_loss). 'plateau' = the paper's HGD scheduler."""
    kind = cfg.get("scheduler", "plateau")
    if kind == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=cfg["plateau_factor"],
                                 patience=cfg["plateau_patience"]), True
    if kind == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6), False
    return None, False


def train_one_fold(X, y, train_idx, val_idx, test_idx, seed, tag="", cfg=None, evaluate_test=True,
                   final_epochs=None, lr_schedule=None):
    """HGD recipe with the profile `cfg`: AdamW, scheduler from the profile (paper: ReduceLROnPlateau
    on VALIDATION loss), label smoothing, gradient clipping, early stopping on validation loss,
    optional mixup/scaling/noise/channel-dropout ('mix') or segmentation-reconstruction ('sr')
    augmentation. The TEST trials are touched once, after training, never when evaluate_test=False
    (tuning). Refit mode (final_epochs): exactly that many epochs on train_idx, no validation, final
    weights kept; with a plateau scheduler the learning rates recorded in the first fit are replayed
    (lr_schedule), so the refit follows the same schedule without needing validation data."""
    seed_everything(seed)
    cfg = cfg or get_cfg()
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    n_chans, n_times = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))
    model = build_model(n_chans, n_times, n_classes, cfg)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                  betas=tuple(cfg.get("betas", (0.9, 0.999))))
    max_epochs = int(final_epochs or cfg["max_epochs"])
    replay = final_epochs is not None and lr_schedule is not None and len(lr_schedule) > 0
    scheduler, on_val = (None, False) if replay else make_scheduler(optimizer, cfg, max_epochs)
    aug = cfg.get("augment", False)

    best_state, best_val_loss, best_val_acc, best_epoch, no_improve = None, float("inf"), 0.0, 0, 0
    epochs_run, lr_history = 0, []
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        if replay:
            for g in optimizer.param_groups:
                g["lr"] = float(lr_schedule[min(epoch - 1, len(lr_schedule) - 1)])
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        model.train()
        losses = []
        for xb, yb in batch_iter(X_tr, y_tr, cfg["batch_size"], shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            if aug == "sr":
                xb, yb = sr_augment(xb, yb, AUGMENT["sr_segments"])
                loss = criterion(model(xb), yb)
            elif aug:
                xb, ya, yb2, lam = mixup(augment_batch(xb), yb, AUGMENT["mixup_alpha"])
                out = model(xb)
                loss = lam * criterion(out, ya) + (1.0 - lam) * criterion(out, yb2)
            else:
                loss = criterion(model(xb), yb)
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            optimizer.step()
            losses.append(loss.item())
        if final_epochs:
            if scheduler is not None and not on_val:
                scheduler.step()
            if epoch == 1 or epoch % 50 == 0:
                print(f"{tag} Ep {epoch:03d}/{max_epochs} | tr {np.mean(losses):.4f} (refit, no early stopping)")
            continue

        val_loss, val_acc = evaluate(model, X_val, y_val, criterion)
        if scheduler is not None:
            if on_val:
                scheduler.step(val_loss)
            else:
                scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch = val_loss, val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 50 == 0:
            print(f"{tag} Ep {epoch:03d} | tr {np.mean(losses):.4f} | val {val_acc:.1%} | "
                  f"best_val_loss={best_val_loss:.4f} (ep {best_epoch}) | lr {lr_history[-1]:.1e}")
        if no_improve >= cfg["patience"]:
            print(f"{tag} Early stop @ ep {epoch} (best epoch {best_epoch})")
            break

    if final_epochs:
        best_state, best_epoch, best_val_acc = copy.deepcopy(model.state_dict()), max_epochs, float("nan")
    if best_state is None:
        raise RuntimeError("No best state found")
    model.load_state_dict(best_state)
    test_proba = None
    if evaluate_test:
        y_te = y[test_idx]
        test_proba = predict_proba(model, X[test_idx])
        y_pred = test_proba.argmax(axis=1)
        test_acc = float((y_pred == y_te).mean())
    else:
        y_te = y_pred = np.zeros(0, dtype=np.int64)
        test_acc = float("nan")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"test_acc": test_acc, "val_acc": float(best_val_acc), "best_epoch": best_epoch,
            "epochs_run": epochs_run, "y_true": y_te, "y_pred": y_pred,
            "best_state": best_state, "n_chans": n_chans, "n_times": n_times,
            "n_classes": n_classes, "cfg": cfg, "test_proba": test_proba, "lr_history": lr_history}


def _confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def make_fold_record(sid, seed, fold_i, class_mode, channels, res, tr_idx, val_idx, te_idx):
    names = class_names(class_mode)
    n_classes = len(names)
    y_true, y_pred = res["y_true"], res["y_pred"]
    counts = np.bincount(y_true, minlength=n_classes)
    return {
        "subject": int(sid), "seed": int(seed), "fold": int(fold_i), "class_mode": class_mode,
        "channels": list(channels), "n_channels": len(channels),
        "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)), "n_test": int(len(te_idx)),
        "n_times": int(res["n_times"]),
        "test_acc": res["test_acc"], "val_acc": res["val_acc"],
        "best_epoch": int(res["best_epoch"]), "epochs_run": int(res["epochs_run"]),
        "chance_acc": 1.0 / n_classes,
        "majority_class_acc": float(counts.max() / counts.sum()),
        "test_class_counts": {names[c]: int(counts[c]) for c in range(n_classes)},
        "pred_class_counts": {names[c]: int((y_pred == c).sum()) for c in range(n_classes)},
        "confusion_matrix": _confusion(y_true, y_pred, n_classes).tolist(),
        "test_idx": [int(i) for i in te_idx],
        "y_true": [int(v) for v in y_true], "y_pred": [int(v) for v in y_pred],
    }


# ────────────────────────────────────────────────
# Gradient-based attribution (saliency script only)
# ────────────────────────────────────────────────
def _gather_true_class_score(logits, targets):
    return logits.gather(1, targets.view(-1, 1)).sum()


def compute_attr_batch(model, xb, yb, method, ig_steps=IG_STEPS,
                       sg_samples=SMOOTHGRAD_SAMPLES, sg_noise_std=SMOOTHGRAD_NOISE_STD):
    model.eval()
    if method == "grad":
        x = xb.clone().detach().requires_grad_(True)
        _gather_true_class_score(model(x), yb).backward()
        return x.grad.detach()
    if method == "grad_x_input":
        x = xb.clone().detach().requires_grad_(True)
        _gather_true_class_score(model(x), yb).backward()
        return x.grad.detach() * x.detach()
    if method == "integrated_gradients":
        baseline = torch.zeros_like(xb)
        total_grad = torch.zeros_like(xb)
        for alpha in torch.linspace(0., 1., ig_steps + 1, device=xb.device)[1:]:
            xi = baseline + alpha * (xb - baseline)
            xi.requires_grad_(True)
            _gather_true_class_score(model(xi), yb).backward()
            total_grad += xi.grad.detach()
        return ((xb - baseline) * (total_grad / ig_steps)).detach()
    if method == "smoothgrad":
        total = torch.zeros_like(xb)
        noise_std = sg_noise_std * xb.detach().std().clamp(min=1e-8)
        for _ in range(sg_samples):
            xn = xb + torch.randn_like(xb) * noise_std
            xn.requires_grad_(True)
            _gather_true_class_score(model(xn), yb).backward()
            total += xn.grad.detach().abs()
        return (total / sg_samples).detach()
    raise ValueError(method)


def compute_fold_channel_importances(model, X, y, methods, batch_size=32):
    """Channel importance = mean over trials and time of |attribution| wrt the true-class
    logit, overall AND separately for each true class (class-specific explanations).

    Call with the VALIDATION block of the fold, never the test block.
    Returns {method: {"all": vec, "by_class": {class_id: vec}}} (vectors max-normalised).
    """
    n_chans = X.shape[1]
    classes = sorted(int(c) for c in np.unique(y))
    sums = {m: np.zeros(n_chans) for m in methods}
    csums = {m: {c: np.zeros(n_chans) for c in classes} for m in methods}
    ccount = {c: 0 for c in classes}
    n_total = 0
    for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
        yb_np = yb.cpu().numpy()
        n_total += len(yb_np)
        for c in classes:
            ccount[c] += int((yb_np == c).sum())
        for m in methods:
            per_trial = compute_attr_batch(model, xb, yb, m).abs().mean(dim=2).cpu().numpy()
            sums[m] += per_trial.sum(axis=0)
            for c in classes:
                sel = yb_np == c
                if sel.any():
                    csums[m][c] += per_trial[sel].sum(axis=0)
    out = {}
    for m in methods:
        out[m] = {
            "all": normalize_importance(sums[m] / max(1, n_total)),
            "by_class": {c: normalize_importance(csums[m][c] / ccount[c])
                         for c in classes if ccount[c] > 0},
        }
    return out


# ────────────────────────────────────────────────
# Experiment runner (baseline / reduced / saliency collection)
# ────────────────────────────────────────────────
def experiment_dir(exp_name, class_mode=None):
    """One folder per experiment; the class mode is fixed by the 2class/4class folder."""
    path = os.path.join(RESULTS_ROOT, exp_name)
    os.makedirs(path, exist_ok=True)
    return path


@torch.no_grad()
def deletion_test(model, X, y, ranking, seed):
    """Faithfulness check (reviewer #8): remove (zero = standardised mean) the k highest-, k lowest-
    and k random-ranked channels of the TEST block and re-score the trained model. A faithful
    ranking hurts accuracy most when its top channels are removed. Evaluation only."""
    order = np.argsort(np.asarray(ranking))[::-1]
    rng = np.random.default_rng(seed)

    def acc_without(idx):
        Xd = X.copy()
        Xd[:, list(idx), :] = 0.0
        return float((predict(model, Xd) == y).mean())

    out = {"intact": float((predict(model, X) == y).mean())}
    for k in FAITH_K:
        if k >= X.shape[1]:
            continue
        rand = [acc_without(rng.choice(X.shape[1], k, replace=False)) for _ in range(FAITH_RANDOM_DRAWS)]
        out[str(k)] = {"top": acc_without(order[:k]), "bottom": acc_without(order[-k:]),
                       "random": float(np.mean(rand))}
    return out


def session_fit(X, y, tr, va, te, seed, tag, cfg, need_val_model):
    """Session protocol for one seed:
      1. fit on session-1 training part, validation part for early stopping (and saliency)
      2. refit on ALL of session 1 with the chosen number of epochs (REFIT=1)
    Returns the train_one_fold dict of the final model plus 'val_state' (the step-1 model, used for
    saliency on the validation trials) and the step-1 validation accuracy."""
    first = None
    if need_val_model or not FIXED_EPOCHS or not REFIT:
        first = train_one_fold(X, y, tr, va, te, seed, tag=tag + " [fit]", cfg=cfg,
                               evaluate_test=not REFIT)
    if not REFIT:
        first["val_state"] = first["best_state"]
        return first
    n_ep = int(cfg["max_epochs"]) if FIXED_EPOCHS else max(int(first["best_epoch"]), MIN_REFIT_EPOCHS)
    full = np.concatenate([tr, va])
    replay = first["lr_history"][:n_ep] if first and cfg.get("scheduler") == "plateau" else None
    res = train_one_fold(X, y, full, np.zeros(0, dtype=np.int64), te, seed, tag=tag + " [refit]",
                         cfg=cfg, final_epochs=n_ep, lr_schedule=replay)
    res["val_acc"] = first["val_acc"] if first else float("nan")
    res["val_state"] = first["best_state"] if first else None
    res["fit_best_epoch"] = int(first["best_epoch"]) if first else None
    return res


def _ensemble_record(sid, class_mode, channels, seed_recs, probas, y_te, tr, va, te, n_times):
    proba = np.mean(probas, axis=0)
    y_pred = proba.argmax(axis=1)
    res = {"y_true": y_te, "y_pred": y_pred, "test_acc": float((y_pred == y_te).mean()),
           "val_acc": float(np.nanmean([r["val_acc"] for r in seed_recs])) if seed_recs else float("nan"),
           "best_epoch": int(np.mean([r["best_epoch"] for r in seed_recs])),
           "epochs_run": int(np.mean([r["epochs_run"] for r in seed_recs])), "n_times": n_times}
    rec = make_fold_record(sid, -1, 1, class_mode, channels, res, tr, va, te)
    rec["ensemble_of_seeds"] = [r["seed"] for r in seed_recs]
    rec["seed_accs"] = [r["test_acc"] for r in seed_recs]
    for key in ("faithfulness", "randomisation_spearman"):            # mean over seeds
        vals = [r[key] for r in seed_recs if key in r]
        if vals:
            rec[key] = _mean_nested(vals)
    return rec


def _mean_nested(dicts):
    out = {}
    for k, v in dicts[0].items():
        if isinstance(v, dict):
            out[k] = _mean_nested([d[k] for d in dicts])
        else:
            out[k] = float(np.nanmean([d[k] for d in dicts]))
    return out


def run_experiment(exp_name, class_mode, montages, collect_saliency=False,
                   methods=ATTR_METHODS, extra_meta=None):
    """Train the model for every subject / seed (/ fold) on that subject's montage, with that
    subject's profile, under PROTOCOL. With collect_saliency: attributions on the VALIDATION trials,
    the deletion test on the TEST trials (evaluation only) and the model-randomisation check.
    Session protocol: `records` holds one seed-ensemble record per subject, `seed_records` the
    individual seeds. Returns (records, subject_saliency)."""
    out_dir = experiment_dir(exp_name, class_mode)
    records, seed_records, subject_saliency, profiles_used = [], [], {}, {}

    for sid in SUBJECT_IDS:
        channels = list(montages[sid])
        cfg = get_cfg(sid)
        profiles_used[sid] = cfg["name"]
        print("\n" + "=" * 80)
        print(f"SUBJECT {sid:02d} — {exp_name} — {class_mode} — {PROTOCOL} — profile {cfg['name']} — "
              f"{len(channels)} ch: {', '.join(channels)}")
        print("=" * 80)
        X, y, info = load_subject_windows(sid, channels, class_mode, cfg)
        folds = make_folds(len(y), info)

        sal_all = {m: [] for m in methods}
        sal_cls = {m: {} for m in methods}

        for fold_i, (tr, va, te) in enumerate(folds, start=1):
            fold_recs, probas = [], []
            for seed in SEEDS:
                tag = f"S{sid:02d} seed{seed} F{fold_i}"
                fseed = fold_seed(seed, sid, fold_i)
                if PROTOCOL == "session":
                    res = session_fit(X, y, tr, va, te, fseed, tag, cfg, need_val_model=collect_saliency)
                    rec = make_fold_record(sid, seed, fold_i, class_mode, channels, res,
                                           np.concatenate([tr, va]) if REFIT else tr, va, te)
                    rec["refit"] = REFIT
                    rec["fit_best_epoch"] = res.get("fit_best_epoch")
                    val_state = res["val_state"]
                else:
                    res = train_one_fold(X, y, tr, va, te, fseed, tag=tag, cfg=cfg)
                    rec = make_fold_record(sid, seed, fold_i, class_mode, channels, res, tr, va, te)
                    val_state = res["best_state"]
                rec["profile"] = cfg["name"]
                print(f"{tag}: test {res['test_acc']*100:.2f}% "
                      f"(val {res['val_acc']*100:.2f}%, epochs {res['best_epoch']})")
                if collect_saliency:
                    vmodel = build_model(res["n_chans"], res["n_times"], res["n_classes"], cfg)
                    vmodel.load_state_dict(val_state)
                    vmodel.eval()
                    imps = compute_fold_channel_importances(vmodel, X[va], y[va], methods)
                    for m in methods:
                        sal_all[m].append(imps[m]["all"])
                        for c, v in imps[m]["by_class"].items():
                            sal_cls[m].setdefault(c, []).append(v)
                    ranking = normalize_importance(np.mean([imps[m]["all"] for m in methods], axis=0))
                    fmodel = build_model(res["n_chans"], res["n_times"], res["n_classes"], cfg)
                    fmodel.load_state_dict(res["best_state"])
                    fmodel.eval()
                    rec["faithfulness"] = deletion_test(fmodel, X[te], y[te], ranking, seed=fseed)
                    seed_everything(fseed + 7)
                    rand_model = build_model(res["n_chans"], res["n_times"], res["n_classes"], cfg)
                    rand_model.eval()
                    rimps = compute_fold_channel_importances(rand_model, X[va], y[va], methods)
                    rec["randomisation_spearman"] = {
                        m: _corr(imps[m]["all"], rimps[m]["all"], "spearman") for m in methods}
                    del vmodel, fmodel, rand_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                fold_recs.append(rec)
                if res.get("test_proba") is not None:
                    probas.append(res["test_proba"])
                del res
            if PROTOCOL == "session":
                seed_records.extend(fold_recs)
                ens = _ensemble_record(sid, class_mode, channels, fold_recs, probas, y[te],
                                       np.concatenate([tr, va]) if REFIT else tr, va, te, X.shape[2])
                ens["profile"] = cfg["name"]
                records.append(ens)
                print(f"S{sid:02d} seed ensemble ({len(probas)} seeds): test {ens['test_acc']*100:.2f}% "
                      f"(single seeds {', '.join(f'{a*100:.1f}' for a in ens['seed_accs'])})")
            else:
                records.extend(fold_recs)

        if collect_saliency:
            subject_saliency[sid] = {
                m: {"all": normalize_importance(np.mean(sal_all[m], axis=0)),
                    "by_class": {c: normalize_importance(np.mean(v, axis=0))
                                 for c, v in sal_cls[m].items()}}
                for m in methods
            }
        del X, y
        gc.collect()

    meta = {"experiment": exp_name, "class_mode": class_mode, "protocol": PROTOCOL,
            "montages": {str(k): list(v) for k, v in montages.items()},
            "profiles_used": {str(k): v for k, v in profiles_used.items()},
            "config": config_snapshot()}
    if extra_meta:
        meta.update(extra_meta)
    write_results(out_dir, records, meta, seed_records=seed_records or None)
    return records, subject_saliency


# ────────────────────────────────────────────────
# Results files (machine readable) and summaries
# ────────────────────────────────────────────────
def subject_accuracies(records):
    """{subject: accuracy}: mean over folds within a seed, then mean over seeds."""
    per = {}
    for r in records:
        per.setdefault((r["subject"], r["seed"]), []).append(r["test_acc"])
    by_subject = {}
    for (sid, _seed), accs in per.items():
        by_subject.setdefault(sid, []).append(float(np.mean(accs)))
    return {sid: float(np.mean(v)) for sid, v in sorted(by_subject.items())}


def summarize(records):
    accs = subject_accuracies(records)
    vals = np.array(list(accs.values()))
    seed_means = {}
    for r in records:
        seed_means.setdefault(r["seed"], {}).setdefault(r["subject"], []).append(r["test_acc"])
    per_seed_grand = {s: float(np.mean([np.mean(v) for v in subj.values()]))
                      for s, subj in seed_means.items()}
    return {"subject_acc": accs, "grand_mean": float(vals.mean()),
            "grand_std_across_subjects": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_subjects": int(len(vals)), "grand_mean_per_seed": per_seed_grand,
            "majority_class_acc_mean": float(np.mean([r["majority_class_acc"] for r in records])),
            "chance_acc": records[0]["chance_acc"] if records else None}


def write_results(out_dir, records, meta, seed_records=None):
    """records: what the tables use (session protocol: one seed-ensemble record per subject);
    seed_records: the individual seeds (session protocol only)."""
    summary = summarize(records)
    payload = {"meta": meta, "summary": summary, "records": records}
    if seed_records:
        payload["seed_records"] = seed_records
        payload["seed_summary"] = summarize(seed_records)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    with open(os.path.join(out_dir, "fold_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "seed", "fold", "n_train", "n_val", "n_test", "n_channels",
                    "test_acc", "val_acc", "best_epoch", "epochs_run", "majority_class_acc",
                    "chance_acc", "channels"])
        for r in records:
            w.writerow([r["subject"], r["seed"], r["fold"], r["n_train"], r["n_val"],
                        r["n_test"], r["n_channels"], f"{r['test_acc']:.6f}",
                        f"{r['val_acc']:.6f}", r["best_epoch"], r["epochs_run"],
                        f"{r['majority_class_acc']:.6f}", f"{r['chance_acc']:.6f}",
                        " ".join(r["channels"])])
    with open(os.path.join(out_dir, "subject_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "mean_test_acc"])
        for sid, acc in summary["subject_acc"].items():
            w.writerow([sid, f"{acc:.6f}"])
    print(f"Results written to {out_dir} "
          f"(grand mean {summary['grand_mean']*100:.2f}% over {summary['n_subjects']} subjects; "
          f"chance {summary['chance_acc']*100:.1f}%, majority {summary['majority_class_acc_mean']*100:.1f}%)")
    return summary


def write_text_report(path, title, records, montages=None, lines=None):
    summary = summarize(records)
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n" + "=" * 80 + "\n\n")
        for sid, acc in summary["subject_acc"].items():
            f.write(f"Subject {sid:02d}: {acc*100:.2f}%")
            if montages:
                f.write(f"  | montage: {', '.join(montages[sid])}")
            f.write("\n")
            for r in sorted((r for r in records if r["subject"] == sid),
                            key=lambda r: (r["seed"], r["fold"])):
                f.write(f"    seed {r['seed']} fold {r['fold']}: {r['test_acc']*100:.2f}% "
                        f"(majority-class {r['majority_class_acc']*100:.1f}%)\n")
        f.write(f"\nGRAND MEAN: {summary['grand_mean']*100:.2f}% "
                f"+/- {summary['grand_std_across_subjects']*100:.2f}% (SD across subjects)\n")
        f.write(f"chance {summary['chance_acc']*100:.1f}% | mean majority-class "
                f"{summary['majority_class_acc_mean']*100:.1f}%\n")
        for line in lines or []:
            f.write(line + "\n")
    print(f"Report saved: {path}")


# ────────────────────────────────────────────────
# ERD/ERS reference maps
#
# Always left- vs right-hand trials, whatever the class mode. Pre-cue baseline,
# 8-30 Hz, NO exponential moving standardisation, trial-averaged power ratio.
# ────────────────────────────────────────────────
def load_subject_for_erd(subject_id, channels):
    """Left/right-hand trials for the ERD maps. Always at ERD_SFREQ, whatever the model's sampling
    rate, so every model in this dataset is compared with the SAME reference maps."""
    sf = ERD_SFREQ
    pre = int(round(-ERD_BASELINE_S[0] * sf))                 # samples before the cue
    if SYNTHETIC:
        X, y = _synthetic_windows(list(channels), pre + int(4.5 * sf),
                                  seed=subject_id + 500, pre_samples=pre, sfreq=sf)
    else:
        dataset = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
        preprocessors = [
            Preprocessor("pick_channels", ch_names=list(channels), ordered=True),
            Preprocessor(scale_to_microvolts, apply_on_array=True),
            Preprocessor("resample", sfreq=sf),
            Preprocessor("filter", l_freq=ERD_BAND_HZ[0], h_freq=ERD_BAND_HZ[1]),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocess(dataset, preprocessors, n_jobs=1)
        windows = create_windows_from_events(
            dataset, trial_start_offset_samples=-pre, trial_stop_offset_samples=0, preload=True)
        X, y = extract_xy(windows)
        _check_label_mapping(dataset, y)
        del dataset, windows
    need = pre + int(ERD_ACTIVE_S[1] * sf)
    assert X.shape[2] >= need, (
        f"ERD windows have {X.shape[2]} samples, need >= {need} "
        f"(cue-0.5 s .. cue+3 s). Check the annotation duration / stop offset.")
    mask = np.isin(y, [LEFT_HAND_LABEL, RIGHT_HAND_LABEL])
    return X[mask], y[mask], sf


def compute_class_erd_map(X_class, sfreq, n_channels):
    """ERD% = 100 * (P_active - P_baseline) / P_baseline with powers averaged over trials
    (Pfurtscheller & Lopes da Silva, 1999). Negative = desynchronisation."""
    if len(X_class) == 0:
        return np.full(n_channels, np.nan, dtype=np.float32)
    pre = int(round(-ERD_BASELINE_S[0] * sfreq))
    b0, b1 = pre + int(ERD_BASELINE_S[0] * sfreq), pre + int(ERD_BASELINE_S[1] * sfreq)
    a0, a1 = pre + int(ERD_ACTIVE_S[0] * sfreq), pre + int(ERD_ACTIVE_S[1] * sfreq)
    p_base = np.mean(X_class[:, :, b0:b1] ** 2, axis=(0, 2))
    p_act = np.mean(X_class[:, :, a0:a1] ** 2, axis=(0, 2))
    return (100 * (p_act - p_base) / (p_base + 1e-12)).astype(np.float32)


def compute_subject_erd_maps(subject_id, channels):
    X, y, sfreq = load_subject_for_erd(subject_id, channels)
    erd_l = compute_class_erd_map(X[y == LEFT_HAND_LABEL], sfreq, len(channels))
    erd_r = compute_class_erd_map(X[y == RIGHT_HAND_LABEL], sfreq, len(channels))
    return {"ERD_L": erd_l, "ERD_R": erd_r, "ERD_(L-R)": erd_l - erd_r,
            "ERD_comb": 0.5 * (erd_l + erd_r)}


def get_erd_maps(subject_id, channels):
    """Cached on disk (results/erd_cache) so every script reuses the same maps."""
    key = hashlib.md5(("|".join(channels) + f"|{SYNTHETIC}|{ERD_SFREQ}").encode()).hexdigest()[:10]
    cache_dir = os.path.join(RESULTS_ROOT, "erd_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"S{subject_id:02d}_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    maps = compute_subject_erd_maps(subject_id, channels)
    np.savez(path, **maps)
    return maps


# ────────────────────────────────────────────────
# Saliency <-> ERD alignment metrics
#
# Saliency is UNSIGNED (|attribution|) while ERD is signed, so correlating the two
# directly (and interpreting the sign of r) is not meaningful. Instead, saliency is
# compared with desynchronisation STRENGTH D = -ERD (positive = power drop):
#   magnitude      corr(saliency_all , (|ERD_L|+|ERD_R|)/2)
#   class_specific mean( corr(saliency_L, D_L), corr(saliency_R, D_R) )
#   lateralisation corr(saliency_L - saliency_R , D_L - D_R)   (invariant to swapping L/R)
# 'legacy_signed_LR' reproduces the old corr(saliency_all, ERD_(L-R)) for comparison only.
# ────────────────────────────────────────────────
def _corr(a, b, kind):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
        return float("nan")
    if kind == "pearson":
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(spearmanr(a[ok], b[ok])[0])


def alignment_metrics(sal_all, sal_by_class, erd, class_mode, grad_ref=None):
    """Sign-invariant metrics (used in the paper's argument) plus the ORIGINAL scripts'
    correlations, kept so the old numbers can be reproduced and compared:
      orig_ERD_L / orig_ERD_R / orig_ERD_LR / orig_ERD_comb   corr(saliency_all, ERD map)
      orig_vs_grad                                           corr(saliency_all, plain-gradient saliency)
    The orig_* values correlate an unsigned map with a signed one: report them, but do not read
    the sign as 'agreement'."""
    left_id, right_id = class_ids(class_mode)
    d_l, d_r = -erd["ERD_L"], -erd["ERD_R"]
    magnitude = 0.5 * (np.abs(erd["ERD_L"]) + np.abs(erd["ERD_R"]))
    out = {}
    for kind in ("pearson", "spearman"):
        out[f"magnitude_{kind}"] = _corr(sal_all, magnitude, kind)
        out[f"legacy_signed_LR_{kind}"] = _corr(sal_all, erd["ERD_(L-R)"], kind)
        if left_id in sal_by_class and right_id in sal_by_class:
            sl, sr = sal_by_class[left_id], sal_by_class[right_id]
            cl, cr = _corr(sl, d_l, kind), _corr(sr, d_r, kind)
            out[f"class_specific_{kind}"] = float(np.nanmean([cl, cr]))
            out[f"lateralisation_{kind}"] = _corr(np.asarray(sl) - np.asarray(sr), d_l - d_r, kind)
        else:
            out[f"class_specific_{kind}"] = float("nan")
            out[f"lateralisation_{kind}"] = float("nan")
        for name, key in (("ERD_L", "ERD_L"), ("ERD_R", "ERD_R"), ("ERD_LR", "ERD_(L-R)"),
                          ("ERD_comb", "ERD_comb")):
            out[f"orig_{name}_{kind}"] = _corr(sal_all, erd[key], kind)
        out[f"orig_vs_grad_{kind}"] = (_corr(sal_all, grad_ref, kind) if grad_ref is not None
                                       else float("nan"))
    return out


def fisher_mean_ci(values, confidence=0.95):
    """Mean of correlations and CI via Fisher z (stays inside [-1, 1])."""
    r = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if len(r) == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    m = float(np.tanh(z.mean()))
    if len(z) < 2:
        return {"n": 1, "mean": m, "ci_low": float("nan"), "ci_high": float("nan")}
    h = student_t.ppf(0.5 + confidence / 2, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
    return {"n": int(len(z)), "mean": m, "ci_low": float(np.tanh(z.mean() - h)),
            "ci_high": float(np.tanh(z.mean() + h))}


def summarize_metric(values, confidence=0.95):
    """Participant-level distribution: mean, SD, t-based CI, n."""
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan")}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = float(student_t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n))
    else:
        h = float("nan")
    return {"n": int(n), "mean": mean, "sd": sd, "ci_low": mean - h, "ci_high": mean + h}


# ────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────
def make_topo_info(channels):
    info = mne.create_info(ch_names=list(channels), sfreq=TARGET_SFREQ, ch_types="eeg")
    info.set_montage(MONTAGE)
    return info


def save_array(path, arr):
    np.save(path, np.asarray(arr, dtype=np.float32))


def plot_saliency_topomap(vec, info, title, out_png, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(vec, info, axes=ax, show=False, cmap=cmap)
    ax.set_title(title)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_erd_ers_topomaps(erd_l, erd_r, contrast, comb, info, out_png):
    vals = [erd_l, erd_r, contrast, comb]
    finite = [v for v in vals if np.all(np.isfinite(v))]
    limit = float(np.max(np.abs(np.concatenate([v.ravel() for v in finite])))) if finite else 1.0
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, data, title in zip(axes, vals, ["Left Hand ERD", "Right Hand ERD",
                                            "Contrast (L-R)", "Combined"]):
        mne.viz.plot_topomap(data, info, axes=ax, cmap="RdBu_r", vlim=(-limit, limit), show=False)
        ax.set_title(title)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    plt.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-limit, limit)), cax=cbar_ax)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────
# EXPLORATORY physiology block (curiosity only)
#
# More neurophysiological channel maps beyond the 8-30 Hz ERD, all computed from the
# same raw recordings (left/right-hand trials, no EMS, no model involved), and their
# correlation with the saliency maps. NOTHING here is used for any choice: montages,
# profiles and the paper's alignment statistics use only the ERD maps above.
#
#   ERD_<band>_L/R/mag   band-limited ERD% (theta 4-8, mu 8-13, beta 13-30, low-gamma 30-40,
#                        high-gamma 60-90 Hz),
#                        active 1-3 s vs pre-cue baseline; mag = (|L|+|R|)/2
#   ERS_beta_rebound_*   post-imagery beta (13-30 Hz) power 4-5 s after the cue vs baseline
#   logpow_8_30          log 8-30 Hz power during imagery (absolute, not relative)
#   r2_signed / r2_abs   left-vs-right discriminability of log 8-30 Hz power per channel
#                        (signed r^2 and r^2 of the point-biserial correlation)
#   TFR_ERD_dB_L/R/mag   Morlet time-frequency ERD in dB, 8-30 Hz, 1-3 s vs baseline
#   ITC_4_30             inter-trial phase coherence 4-30 Hz, 0-1 s after the cue
#   ERP_abs_0.2_0.8      mean |amplitude| of the cue-evoked potential (< 5 Hz), 0.2-0.8 s
#   SlowPot_0.5_3        mean slow-potential amplitude (< 5 Hz), 0.5-3 s (MRCP/CNV-like)
# For every map: corr(saliency, map) and corr(saliency, |map|), Pearson and Spearman, per
# attribution method and for the mean of the four methods ("mean4"); for maps with a left
# and a right version also the class-specific corr(saliency_L, |map_L|), corr(saliency_R, |map_R|).
# ────────────────────────────────────────────────
PHYS_SFREQ = 250                 # HGD: high enough for the high-gamma band
PHYS_LOWPASS = 100.0
PHYS_TMIN, PHYS_TMAX = -1.5, 5.5          # loaded epoch (0.5 s padding for filter edges)
PHYS_KEEP_S = (-1.0, 5.0)
PHYS_BANDS = {"theta": (4.0, 8.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0), "lowgamma": (30.0, 40.0),
              "highgamma": (60.0, 90.0)}       # HGD: Schirrmeister et al. 2017 found high-gamma informative


def load_subject_for_physiology(subject_id, channels):
    """Left/right-hand epochs, PHYS_TMIN..PHYS_TMAX around the cue, 0.1-PHYS_LOWPASS Hz, uV, no EMS.
    Built with MNE from the annotation names (not braindecode windows), so trials whose window
    would run past the end of a run are simply dropped instead of failing."""
    sf = PHYS_SFREQ
    n = int(round((PHYS_TMAX - PHYS_TMIN) * sf)) + 1
    if SYNTHETIC:
        X, y = _synthetic_windows(list(channels), n, seed=subject_id + 900,
                                  pre_samples=int(round(-PHYS_TMIN * sf)), sfreq=sf)
        mask = np.isin(y, [LEFT_HAND_LABEL, RIGHT_HAND_LABEL])
        return X[mask].astype(np.float64), y[mask], sf
    dataset = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
    Xs, ys = [], []
    want = {LABEL_NAMES[LEFT_HAND_LABEL]: LEFT_HAND_LABEL, LABEL_NAMES[RIGHT_HAND_LABEL]: RIGHT_HAND_LABEL}
    for d in dataset.datasets:
        raw = d.raw.copy().load_data().pick(list(channels))
        raw.reorder_channels(list(channels))
        raw.apply_function(scale_to_microvolts, channel_wise=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.resample(sf, npad="auto")
            raw.filter(0.1, PHYS_LOWPASS, verbose=False)
        present = {k: v + 1 for k, v in want.items() if k in set(raw.annotations.description)}
        if not present:
            continue
        events, _ = mne.events_from_annotations(raw, event_id=present, verbose=False)
        ep = mne.Epochs(raw, events, event_id=present, tmin=PHYS_TMIN, tmax=PHYS_TMAX,
                        baseline=None, preload=True, reject_by_annotation=False, verbose=False)
        Xs.append(ep.get_data()[:, :, :n])
        ys.append(ep.events[:, 2] - 1)
    del dataset
    gc.collect()
    return np.concatenate(Xs).astype(np.float64), np.concatenate(ys).astype(np.int64), sf


def _seg(sf, t0, t1):
    """Sample slice of the loaded epoch for times t0..t1 (s, relative to the cue)."""
    return slice(int(round((t0 - PHYS_TMIN) * sf)), int(round((t1 - PHYS_TMIN) * sf)))


def _bandpass(X, sf, lo, hi):
    return mne.filter.filter_data(X, sf, lo, hi, method="iir",
                                  iir_params=dict(order=4, ftype="butter", output="sos"), verbose=False)


def _erd_pct(Xb, sf, active):
    base = np.mean(Xb[:, :, _seg(sf, *ERD_BASELINE_S)] ** 2, axis=(0, 2))
    act = np.mean(Xb[:, :, _seg(sf, *active)] ** 2, axis=(0, 2))
    return 100.0 * (act - base) / (base + 1e-12)


def physiology_maps(X, y, sf):
    """{name: vector over channels}. X: (trials, channels, times) from load_subject_for_physiology."""
    L, R = y == LEFT_HAND_LABEL, y == RIGHT_HAND_LABEL
    maps = {}
    for band, (lo, hi) in PHYS_BANDS.items():
        Xb = _bandpass(X, sf, lo, hi)
        el, er = _erd_pct(Xb[L], sf, ERD_ACTIVE_S), _erd_pct(Xb[R], sf, ERD_ACTIVE_S)
        maps[f"ERD_{band}_L"], maps[f"ERD_{band}_R"] = el, er
        maps[f"ERD_{band}_mag"] = 0.5 * (np.abs(el) + np.abs(er))
        if band == "beta":
            rl, rr = _erd_pct(Xb[L], sf, (4.0, 5.0)), _erd_pct(Xb[R], sf, (4.0, 5.0))
            maps["ERS_beta_rebound_L"], maps["ERS_beta_rebound_R"] = rl, rr
            maps["ERS_beta_rebound_mag"] = 0.5 * (np.abs(rl) + np.abs(rr))

    X830 = _bandpass(X, sf, 8.0, 30.0)
    logp = np.log(np.mean(X830[:, :, _seg(sf, *ERD_ACTIVE_S)] ** 2, axis=2) + 1e-12)   # trials x ch
    maps["logpow_8_30"] = logp.mean(axis=0)
    lab = R.astype(float)[L | R]
    r = np.array([np.corrcoef(logp[L | R, c], lab)[0, 1] if logp[L | R, c].std() > 0 else 0.0
                  for c in range(X.shape[1])])
    maps["r2_signed"], maps["r2_abs"] = np.sign(r) * r ** 2, r ** 2

    freqs = np.arange(8.0, 31.0, 2.0)
    keep = _seg(sf, *PHYS_KEEP_S)
    for cls, m in (("L", L), ("R", R)):
        p = mne.time_frequency.tfr_array_morlet(X[m][:, :, keep], sf, freqs, n_cycles=freqs / 2.0,
                                                output="avg_power", verbose=False)       # ch x f x t
        t = np.arange(p.shape[-1]) / sf + PHYS_KEEP_S[0]
        base = p[:, :, (t >= ERD_BASELINE_S[0]) & (t < ERD_BASELINE_S[1])].mean(axis=2)
        act = p[:, :, (t >= ERD_ACTIVE_S[0]) & (t < ERD_ACTIVE_S[1])].mean(axis=2)
        maps[f"TFR_ERD_dB_{cls}"] = (10.0 * np.log10((act + 1e-20) / (base + 1e-20))).mean(axis=1)
    maps["TFR_ERD_dB_mag"] = 0.5 * (np.abs(maps["TFR_ERD_dB_L"]) + np.abs(maps["TFR_ERD_dB_R"]))

    fi = np.arange(4.0, 31.0, 2.0)
    itc = mne.time_frequency.tfr_array_morlet(X[L | R][:, :, keep], sf, fi, n_cycles=fi / 2.0,
                                              output="itc", verbose=False)                # ch x f x t
    t = np.arange(itc.shape[-1]) / sf + PHYS_KEEP_S[0]
    maps["ITC_4_30"] = np.abs(itc[:, :, (t >= 0.0) & (t < 1.0)]).mean(axis=(1, 2))

    slow = mne.filter.filter_data(X[L | R], sf, None, 5.0, method="iir",
                                  iir_params=dict(order=4, ftype="butter", output="sos"), verbose=False)
    slow = slow - slow[:, :, _seg(sf, *ERD_BASELINE_S)].mean(axis=2, keepdims=True)
    avg = slow.mean(axis=0)                                                                 # ch x t
    maps["ERP_abs_0.2_0.8"] = np.abs(avg[:, _seg(sf, 0.2, 0.8)]).mean(axis=1)
    maps["SlowPot_0.5_3"] = avg[:, _seg(sf, 0.5, 3.0)].mean(axis=1)
    return {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()}


def get_physiology_maps(subject_id, channels):
    key = hashlib.md5(("|".join(channels) + f"|{SYNTHETIC}|{PHYS_SFREQ}|v1").encode()).hexdigest()[:10]
    cache_dir = os.path.join(RESULTS_ROOT, "physiology_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"S{subject_id:02d}_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    X, y, sf = load_subject_for_physiology(subject_id, channels)
    maps = physiology_maps(X, y, sf)
    np.savez(path, **maps)
    return maps


def physiology_correlations(sal_all, sal_by_class, maps, class_mode):
    left_id, right_id = class_ids(class_mode)
    out = {}
    for name, vec in maps.items():
        for kind in ("pearson", "spearman"):
            out[f"{name}|{kind}"] = _corr(sal_all, vec, kind)
            out[f"{name}|abs|{kind}"] = _corr(sal_all, np.abs(vec), kind)
    pairs = sorted({k[:-2] for k in maps if k.endswith("_L") and k[:-2] + "_R" in maps})
    if left_id in sal_by_class and right_id in sal_by_class:
        for base in pairs:
            for kind in ("pearson", "spearman"):
                cl = _corr(sal_by_class[left_id], np.abs(maps[base + "_L"]), kind)
                cr = _corr(sal_by_class[right_id], np.abs(maps[base + "_R"]), kind)
                out[f"{base}_classspecific|abs|{kind}"] = float(np.nanmean([cl, cr]))
    return out


def plot_map_grid(maps, info, out_png, title):
    names = list(maps)
    ncol = 6
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, n in zip(np.ravel(axes), names):
        v = np.asarray(maps[n], dtype=float)
        if not np.all(np.isfinite(v)):
            continue
        signed = v.min() < 0 < v.max()
        lim = float(np.max(np.abs(v))) or 1.0
        mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r" if signed else "viridis",
                             vlim=(-lim, lim) if signed else (None, None))
        ax.set_title(n, fontsize=8)
    fig.suptitle(title)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_physiology(subject_sal, out_dir, channels=None):
    """EXPLORATORY. Writes physiology_correlations.json, physiology_maps_group.png and the
    per-subject maps (physiology_maps_SXX.npz). Never raises: a failure is recorded in the json
    so the main pipeline is not affected."""
    channels = list(channels or FULL_CHANNELS)
    payload = {"class_mode": CLASS_MODE, "exploratory": True,
               "note": ("Curiosity analysis. Not used for montage, profile or method choices; the "
                        "paper's alignment uses only the 8-30 Hz ERD maps (alignment_metrics.json)."),
               "maps": {}, "per_subject": {}, "summary": {}}
    try:
        all_maps = {}
        for sid, by_method in subject_sal.items():
            maps = get_physiology_maps(sid, channels)
            all_maps[sid] = maps
            np.savez(os.path.join(out_dir, f"physiology_maps_S{sid:02d}.npz"), **maps)
            sal = dict(by_method)
            sal["mean4"] = {
                "all": normalize_importance(np.mean([v["all"] for v in by_method.values()], axis=0)),
                "by_class": {c: normalize_importance(np.mean([v["by_class"][c] for v in by_method.values()
                                                              if c in v["by_class"]], axis=0))
                             for c in next(iter(by_method.values()))["by_class"]}}
            payload["per_subject"][str(sid)] = {
                m: physiology_correlations(v["all"], v["by_class"], maps, CLASS_MODE) for m, v in sal.items()}
        names = list(next(iter(all_maps.values())))
        payload["maps"] = {n: "see block comment in the common module" for n in names}
        first = next(iter(payload["per_subject"].values()))
        for m in first:
            payload["summary"][m] = {k: fisher_mean_ci([payload["per_subject"][s][m][k]
                                                        for s in payload["per_subject"]])
                                     for k in first[m]}
        group = {n: np.mean([all_maps[s][n] for s in all_maps], axis=0) for n in names}
        plot_map_grid(group, make_topo_info(channels), os.path.join(out_dir, "physiology_maps_group.png"),
                      f"Group physiology maps ({CLASS_MODE}, left/right-hand trials) - exploratory")
        print("\nEXPLORATORY physiology correlations (mean4 saliency, Pearson, Fisher mean over subjects):")
        for k, s in payload["summary"]["mean4"].items():
            if k.endswith("|pearson") and "|abs|" not in k:
                print(f"  {k[:-8]:<28} r = {s['mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]")
    except Exception as exc:                                        # exploratory: never fatal
        import traceback
        traceback.print_exc()
        payload["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[physiology] skipped: {payload['error']}")
    with open(os.path.join(out_dir, "physiology_correlations.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return payload
