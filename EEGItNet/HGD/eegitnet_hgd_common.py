# deep4net_hgd_common.py
#
# Shared code for all EEG-ITNet / HGD (Schirrmeister2017) experiments. It is imported by the
# scripts in the 2class/ and 4class/ folders next to this file:
#   *_saliency.py   full 22-channel baseline ("22ch") + saliency-ranked 12 channels ("Ours")
#   *_csp.py        CSP-ranked 12 channels            ("CSP")
#   *_mi.py         mutual-information-ranked         ("MI")
#   *_relieff.py    ReliefF-ranked                    ("RLF")
#   *_controls.py   fixed / random / ERD-only montages (controls)
#
# Everything that must not drift between methods lives here: label handling,
# preprocessing, folds, the training loop, leave-one-subject-out (LOSO) montages,
# ERD/ERS references, alignment metrics and machine-readable result files.
#
# Review items addressed in this module (see REVIEW_CROSSWALK.md):
#   * explicit, ASSERTED label mapping (feet/left_hand/rest/right_hand = 0/1/2/3).
#     The earlier Deep4Net scripts treated labels 0/1 as left/right hand, which are
#     feet/left hand under braindecode's alphabetical mapping.
#   * no test-set evaluation inside the training loop (checkpoint = best val loss)
#   * per-fold predictions, confusion matrices, class counts, majority baseline and
#     the exact channels are written to results.json / fold_results.csv
#   * several random seeds (EEG_SEEDS), re-initialised for every (seed, subject, fold)
#   * LOSO montages (subject S's montage never uses S's data)
#   * ERD baseline = PRE-cue interval (-0.5..0 s), trial-averaged power ratio
#   * alignment metrics that are sign-invariant and class-specific (unsigned
#     saliency is compared with desynchronisation STRENGTH, not signed ERD)
#
# Environment variables (all optional):
#   EEG_SEEDS=42,43,44       random seeds (default 42)
#   EEG_SUBJECTS=1,2,3       subject subset (default 1..14)
#   (the class mode is fixed by the folder: each script in 2class/ or 4class/ sets
#    EEG_CLASS_MODE itself before importing this module)
#   EEG_MAX_EPOCHS, EEG_PATIENCE
#   EEG_RESULTS_DIR          output root (scripts set it to <their folder>/results)
#   EEG_SYNTHETIC=1          use synthetic data (pipeline dry-run, no download)

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
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import spearmanr, t as student_t

import mne
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from braindecode.datasets import MOABBDataset
from braindecode.models import EEGITNet
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
SUBJECT_IDS = _env_int_list("EEG_SUBJECTS", range(1, 15))
MODEL_NAME = "EEG-ITNet"
USE_EMS = False           # the original EEG-ITNet scripts did not use exponential moving standardisation
CLASS_MODE = os.environ.get("EEG_CLASS_MODE", "")
if CLASS_MODE not in ("2class", "4class"):
    raise SystemExit("Run the scripts inside the 2class/ or 4class/ folders "
                     "(they set EEG_CLASS_MODE before importing this module).")
SYNTHETIC = os.environ.get("EEG_SYNTHETIC", "0") == "1"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.environ.get("EEG_RESULTS_DIR", os.path.join(HERE, "results"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE, "| class mode:", CLASS_MODE, "| seeds:", SEEDS, "| subjects:", list(SUBJECT_IDS),
      "| synthetic:", SYNTHETIC)

# NOTE: preprocessing / training constants are deliberately left exactly as in the
# original Deep4Net scripts. Differences from the manuscript (Section 3.2/3.4) are
# resolved by editing the manuscript, not the code. They are written to every
# results.json (see config_snapshot) so the paper can quote them exactly.
DATASET_NAME = "Schirrmeister2017"

LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0
TARGET_SFREQ = 100
EMS_FACTOR_NEW = 1e-3
EMS_INIT_BLOCK = 1000

TRIAL_START_OFFSET_S = 0.0
TRIAL_STOP_OFFSET_S = 4.0

MAX_EPOCHS = int(os.environ.get("EEG_MAX_EPOCHS", 600))
PATIENCE = int(os.environ.get("EEG_PATIENCE", 150))
BATCH_SIZE = 96
LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30
LABEL_SMOOTHING = 0.1

IG_STEPS = 32
SMOOTHGRAD_SAMPLES = 12
SMOOTHGRAD_NOISE_STD = 0.10
ATTR_METHODS = ["grad", "grad_x_input", "integrated_gradients", "smoothgrad"]

FULL_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Pz'
]
REDUCED_N_CHANNELS = 12

MONTAGE = mne.channels.make_standard_montage("standard_1005")

# ERD/ERS reference intervals, seconds relative to the cue.
ERD_BASELINE_S = (-0.5, 0.0)     # PRE-cue baseline (as described in the manuscript)
ERD_ACTIVE_S = (1.0, 3.0)
ERD_BAND_HZ = (8.0, 30.0)


def config_snapshot():
    return {
        "model": MODEL_NAME, "class_mode": CLASS_MODE,
        "dataset": DATASET_NAME,
        "note_dataset": "HGD contains EXECUTED movements (not imagined).",
        "sfreq_hz": TARGET_SFREQ,
        "classification_band_hz": [LOW_CUT_HZ, HIGH_CUT_HZ],
        "ems": ({"factor_new": EMS_FACTOR_NEW, "init_block_size": EMS_INIT_BLOCK}
                if USE_EMS else None),
        "trial_start_offset_s": TRIAL_START_OFFSET_S,
        "trial_stop_offset_s": TRIAL_STOP_OFFSET_S,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "checkpoint_rule": "lowest validation loss",
        "folds": "4 chronological blocks; test block i, validation block i-1, rest train",
        "attribution": {"methods": ATTR_METHODS, "ig_steps": IG_STEPS,
                        "smoothgrad_samples": SMOOTHGRAD_SAMPLES,
                        "smoothgrad_noise_std": SMOOTHGRAD_NOISE_STD,
                        "target": "pre-softmax logit of the true class",
                        "computed_on": "validation block of each fold"},
        "erd": {"baseline_s": ERD_BASELINE_S, "active_s": ERD_ACTIVE_S,
                "band_hz": ERD_BAND_HZ, "standardisation": "none"},
        "full_channels": FULL_CHANNELS, "reduced_n_channels": REDUCED_N_CHANNELS,
        "seeds": SEEDS, "subject_ids": list(SUBJECT_IDS),
        "label_names": LABEL_NAMES,
        "versions": {"torch": torch.__version__, "mne": mne.__version__,
                     "numpy": np.__version__},
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
        f"Unexpected HGD annotation names {found}; expected {sorted(LABEL_NAMES)}")
    if len(names) == len(y_all):
        expected = np.array([LABEL_NAMES.index(n) for n in names])
        assert np.array_equal(expected, y_all), (
            "Integer labels do not match annotation names - label mapping assumption broken")
    else:
        print(f"  [label check] {len(names)} annotated events vs {len(y_all)} windows "
              f"(some windows dropped); name check passed, order check skipped")


def _synthetic_windows(channels, n_times, seed, n_per_class=60, pre_samples=0):
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
        X[k, :, pre_samples + int(1.0 * TARGET_SFREQ):] *= gain[:, None]
    return X, y


def load_subject_windows(subject_id, channels, class_mode):
    """Load one subject with the shared pipeline and return (X, y, info)."""
    if SYNTHETIC:
        n_times = int((TRIAL_STOP_OFFSET_S - TRIAL_START_OFFSET_S) * TARGET_SFREQ)
        X, y = _synthetic_windows(channels, n_times, seed=subject_id)
        X, y = apply_class_mode(X, y, class_mode)
    else:
        dataset = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
        preprocessors = [
            Preprocessor("pick_channels", ch_names=list(channels), ordered=True),
            Preprocessor(scale_to_microvolts, apply_on_array=True),
            Preprocessor("resample", sfreq=TARGET_SFREQ),
            Preprocessor("filter", l_freq=LOW_CUT_HZ, h_freq=HIGH_CUT_HZ),
        ]
        if USE_EMS:
            preprocessors.append(Preprocessor(exponential_moving_standardize,
                                              factor_new=EMS_FACTOR_NEW,
                                              init_block_size=EMS_INIT_BLOCK))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocess(dataset, preprocessors, n_jobs=1)
        windows = create_windows_from_events(
            dataset,
            trial_start_offset_samples=int(TRIAL_START_OFFSET_S * TARGET_SFREQ),
            trial_stop_offset_samples=int(TRIAL_STOP_OFFSET_S * TARGET_SFREQ),
            preload=True,
        )
        X, y = extract_xy(windows)
        _check_label_mapping(dataset, y)
        X, y = apply_class_mode(X, y, class_mode)
        del dataset, windows
        gc.collect()

    names = class_names(class_mode)
    counts = {names[c]: int((y == c).sum()) for c in range(len(names))}
    info = {"n_trials": int(len(y)), "n_channels": int(X.shape[1]),
            "n_times": int(X.shape[2]), "class_counts": counts}
    print(f"  S{subject_id:02d} [{class_mode}] X={X.shape} class counts={counts}")
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
def build_model(n_chans, n_times, n_classes):
    return EEGITNet(
        n_outputs=n_classes, n_chans=n_chans, n_times=n_times, drop_prob=DROPOUT,
    ).to(DEVICE)  # MODEL-SPECIFIC LINE (the Deep4Net copy of this file differs only here)


def train_one_fold(X, y, train_idx, val_idx, test_idx, seed, tag=""):
    """Train one fold. The TEST block is touched exactly once, after training, with
    the checkpoint that had the lowest validation loss."""
    seed_everything(seed)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    n_chans, n_times = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))
    model = build_model(n_chans, n_times, n_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=30)

    best_state, best_val_loss, best_val_acc, best_epoch, no_improve = None, float("inf"), 0.0, 0, 0
    epochs_run = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch
        model.train()
        train_losses = []
        for xb, yb in batch_iter(X_tr, y_tr, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_acc = evaluate(model, X_val, y_val, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch = val_loss, val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"{tag} Ep {epoch:03d} | tr {np.mean(train_losses):.4f} | "
                  f"val {val_acc:.1%} | best_val_loss={best_val_loss:.4f}")
        if no_improve >= PATIENCE:
            print(f"{tag} Early stop @ ep {epoch} (best epoch {best_epoch})")
            break

    if best_state is None:
        raise RuntimeError("No best state found")
    model.load_state_dict(best_state)
    y_pred = predict(model, X_te)
    test_acc = float((y_pred == y_te).mean())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"test_acc": test_acc, "val_acc": float(best_val_acc), "best_epoch": best_epoch,
            "epochs_run": epochs_run, "y_true": y_te, "y_pred": y_pred,
            "best_state": best_state, "n_chans": n_chans, "n_times": n_times,
            "n_classes": n_classes}


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


def run_experiment(exp_name, class_mode, montages, collect_saliency=False,
                   methods=ATTR_METHODS, extra_meta=None):
    """Train Deep4Net for every subject / seed / fold on that subject's montage.

    montages: {subject_id: [channel names]}.
    If collect_saliency, attribution maps are computed on each fold's VALIDATION block
    and averaged over folds and seeds. Returns (records, subject_saliency) where
    subject_saliency = {sid: {method: {"all": vec, "by_class": {cid: vec}}}} (or {}).
    """
    out_dir = experiment_dir(exp_name, class_mode)
    records, subject_saliency = [], {}

    for sid in SUBJECT_IDS:
        channels = list(montages[sid])
        print("\n" + "=" * 80)
        print(f"SUBJECT {sid:02d} — {exp_name} — {class_mode} — {len(channels)} ch: {', '.join(channels)}")
        print("=" * 80)
        X, y, _info = load_subject_windows(sid, channels, class_mode)
        folds = make_blockwise_folds(len(y), n_blocks=4)

        sal_all = {m: [] for m in methods}
        sal_cls = {m: {} for m in methods}

        for seed in SEEDS:
            for fold_i, (tr, va, te) in enumerate(folds, start=1):
                tag = f"S{sid:02d} seed{seed} F{fold_i}"
                res = train_one_fold(X, y, tr, va, te, fold_seed(seed, sid, fold_i), tag=tag)
                records.append(make_fold_record(sid, seed, fold_i, class_mode, channels,
                                                res, tr, va, te))
                print(f"{tag}: test {res['test_acc']*100:.2f}% "
                      f"(val {res['val_acc']*100:.2f}%, best epoch {res['best_epoch']})")
                if collect_saliency:
                    model = build_model(res["n_chans"], res["n_times"], res["n_classes"])
                    model.load_state_dict(res["best_state"])
                    model.eval()
                    imps = compute_fold_channel_importances(model, X[va], y[va], methods)
                    for m in methods:
                        sal_all[m].append(imps[m]["all"])
                        for c, v in imps[m]["by_class"].items():
                            sal_cls[m].setdefault(c, []).append(v)
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                del res

        if collect_saliency:
            subject_saliency[sid] = {
                m: {"all": normalize_importance(np.mean(sal_all[m], axis=0)),
                    "by_class": {c: normalize_importance(np.mean(v, axis=0))
                                 for c, v in sal_cls[m].items()}}
                for m in methods
            }
        del X, y
        gc.collect()

    meta = {"experiment": exp_name, "class_mode": class_mode,
            "montages": {str(k): list(v) for k, v in montages.items()},
            "config": config_snapshot()}
    if extra_meta:
        meta.update(extra_meta)
    write_results(out_dir, records, meta)
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


def write_results(out_dir, records, meta):
    summary = summarize(records)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "records": records}, f)
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
    pre = int(round(-ERD_BASELINE_S[0] * TARGET_SFREQ))      # samples before the cue
    if SYNTHETIC:
        X, y = _synthetic_windows(list(channels), pre + int(4.5 * TARGET_SFREQ),
                                  seed=subject_id + 500, pre_samples=pre)
    else:
        dataset = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
        preprocessors = [
            Preprocessor("pick_channels", ch_names=list(channels), ordered=True),
            Preprocessor(scale_to_microvolts, apply_on_array=True),
            Preprocessor("resample", sfreq=TARGET_SFREQ),
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
    need = pre + int(ERD_ACTIVE_S[1] * TARGET_SFREQ)
    assert X.shape[2] >= need, (
        f"ERD windows have {X.shape[2]} samples, need >= {need} "
        f"(cue-0.5 s .. cue+3 s). Check the annotation duration / stop offset.")
    mask = np.isin(y, [LEFT_HAND_LABEL, RIGHT_HAND_LABEL])
    return X[mask], y[mask], TARGET_SFREQ


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
    key = hashlib.md5(("|".join(channels) + f"|{SYNTHETIC}").encode()).hexdigest()[:10]
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


def alignment_metrics(sal_all, sal_by_class, erd, class_mode):
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
    return out


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
