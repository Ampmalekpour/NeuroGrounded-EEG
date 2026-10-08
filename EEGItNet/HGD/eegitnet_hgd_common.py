# eegitnet_hgd_common.py
#
# Shared utilities for all EEG-ITNet / HGD experiments:
#   - eegitnet_hgd_baseline_full22.py   (full 22-channel baseline, "22ch" row)
#   - eegitnet_hgd_saliency_reduced.py  (saliency-guided reduction, "Ours" row)
#   - eegitnet_hgd_csp_reduced.py       (CSP-guided reduction, "CSP" row)
#   - eegitnet_hgd_mi_reduced.py        (Mutual-Information reduction, "MI" row)
#   - eegitnet_hgd_relieff_reduced.py   (ReliefF reduction, "RLF" row)
#
# Every script imports this module so the class-label mask, the LOSO montage
# logic, and the training loop cannot drift apart between methods the way
# they did before (the CSP/MI/ReliefF scripts previously masked on the wrong
# label pair — see apply_class_mode below).
#
# CLASS_MODE is set by each runner script before calling the functions here:
#   "2class"  -> left_hand vs right_hand only (labels remapped to 0/1)
#   "4class"  -> all four HGD classes (feet, left_hand, rest, right_hand)

import os
import gc
import copy
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
)

mne.set_log_level("WARNING")

# ────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────
SEED = 42


def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ────────────────────────────────────────────────
# Config
#
# NOTE: preprocessing constants (sample rate, filter band, window) are left
# EXACTLY as in the original scripts on purpose. The mismatches between this
# code and the paper's Methods section (Section 3.2: 160 Hz / 8-32 Hz /
# 0.5-2.5 s / exponential moving standardization) are being resolved by
# editing the paper text, not the code, per instruction.
# ────────────────────────────────────────────────
DATASET_NAME = "Schirrmeister2017"
SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0
TARGET_SFREQ = 100

TRIAL_START_OFFSET_S = 0.0
TRIAL_STOP_OFFSET_S = 4.0

MAX_EPOCHS = 600
PATIENCE = 150
BATCH_SIZE = 96
LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30
LABEL_SMOOTHING = 0.1

FULL_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Pz'
]

REDUCED_N_CHANNELS = 12

MONTAGE = mne.channels.make_standard_montage("standard_1005")

# ────────────────────────────────────────────────
# Class-mode handling
#
# After braindecode's alphabetical label mapping for HGD:
#   0 = feet, 1 = left_hand, 2 = rest, 3 = right_hand
#
# 2-class mode keeps ONLY left_hand (1) and right_hand (3), remapped to 0/1.
#
# *** BUG FIXED HERE ***
# The original EEGITNet_HGD_CSP.py / _MI.py / _RelieF.py scripts used
#   mask = np.isin(y, [0, 1])   # feet vs left_hand
# instead of the correct left-hand-vs-right-hand mask used by the saliency
# ("Ours") script. That meant the CSP/MI/ReliefF baselines you were
# comparing "Ours" against were trained on a completely different binary
# task. All five scripts now share this single definition.
# ────────────────────────────────────────────────
LEFT_HAND_LABEL = 1
RIGHT_HAND_LABEL = 3


def apply_class_mode(X, y, class_mode):
    if class_mode == "2class":
        mask = np.isin(y, [LEFT_HAND_LABEL, RIGHT_HAND_LABEL])
        X = X[mask]
        y = np.where(y[mask] == LEFT_HAND_LABEL, 0, 1).astype(np.int64)
        return X, y
    elif class_mode == "4class":
        return X, y.astype(np.int64)
    else:
        raise ValueError(f"Unknown class_mode: {class_mode!r} (use '2class' or '4class')")


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


def load_subject_windows(subject_id, channels, class_mode):
    """Loads one subject, applies the shared preprocessing pipeline, and
    returns (X, y) already filtered/remapped for the requested class_mode."""
    dataset = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
    preprocessors = [
        Preprocessor("pick_channels", ch_names=channels, ordered=True),
        Preprocessor(scale_to_microvolts, apply_on_array=True),
        Preprocessor("resample", sfreq=TARGET_SFREQ),
        Preprocessor("filter", l_freq=LOW_CUT_HZ, h_freq=HIGH_CUT_HZ),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocess(dataset, preprocessors, n_jobs=1)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(TRIAL_START_OFFSET_S * TARGET_SFREQ),
        trial_stop_offset_samples=int(TRIAL_STOP_OFFSET_S * TARGET_SFREQ),
        preload=True,
    )
    X, y = extract_xy(windows_dataset)
    X, y = apply_class_mode(X, y, class_mode)
    return X.astype(np.float32), y.astype(np.int64)


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


def corrcoef_safe(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) != len(b) or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def top_k_channels(scores, ch_names, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(ch_names[i], float(scores[i])) for i in idx]


def loso_montage(subject_scores, ch_names, k):
    """
    subject_scores: dict {subject_id: 1D score vector over ch_names}

    Returns dict {subject_id: [selected_channel_names]}, where each
    subject's montage is built ONLY from every OTHER subject's score
    vector (leave-one-subject-out). The held-out subject's own data never
    informs its own reduced montage, which is what a population-level
    montage intended for a *new* user actually requires. This directly
    addresses the reviewer's comment that the "global" montage must be
    estimated without the test subject's own data.
    """
    ids = list(subject_scores.keys())
    montages = {}
    for held_out in ids:
        others = [subject_scores[s] for s in ids if s != held_out]
        avg_score = normalize_importance(np.mean(np.stack(others, axis=0), axis=0))
        top = top_k_channels(avg_score, ch_names, k)
        montages[held_out] = [ch for ch, _ in top]
    return montages


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
    finite_vals = [v for v in vals if np.all(np.isfinite(v))]
    limit = float(np.max(np.abs(np.concatenate([v.ravel() for v in finite_vals])))) if finite_vals else 1.0
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Left Hand ERD", "Right Hand ERD", "Contrast (L-R)", "Combined"]
    for ax, data, title in zip(axes, vals, titles):
        mne.viz.plot_topomap(data, info, axes=ax, cmap="RdBu_r",
                              vlim=(-limit, limit), show=False)
        ax.set_title(title)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    plt.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-limit, limit)),
                 cax=cbar_ax)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topo_info(channels):
    info = mne.create_info(ch_names=list(channels), sfreq=TARGET_SFREQ, ch_types="eeg")
    info.set_montage(MONTAGE)
    return info


# ────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────
def build_model(n_chans, n_times, n_classes):
    return EEGITNet(
        n_outputs=n_classes, n_chans=n_chans, n_times=n_times, drop_prob=DROPOUT
    ).to(DEVICE)


def train_one_fold(X, y, train_idx, val_idx, test_idx, tag=""):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    n_chans, n_times = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))
    model = build_model(n_chans, n_times, n_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=30)

    best_state, best_val_loss, best_val_acc, no_improve = None, float("inf"), 0.0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in batch_iter(X_tr, y_tr, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_acc = evaluate(model, X_val, y_val, criterion)
        scheduler.step(val_loss)
        _, test_acc = evaluate(model, X_te, y_te, criterion)

        if val_loss < best_val_loss:
            best_val_loss, best_val_acc = val_loss, val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(f"{tag} Ep {epoch:03d} | tr {np.mean(train_losses):.4f} | "
              f"val {val_acc:.1%} | te {test_acc:.1%} | best_val_loss={best_val_loss:.4f}")

        if no_improve >= PATIENCE:
            print(f"{tag} Early stop @ ep {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state found")

    model.load_state_dict(best_state)
    _, final_test_acc = evaluate(model, X_te, y_te, criterion)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_test_acc, best_val_acc, best_state, n_chans, n_times, n_classes


def run_reduced_training_loso(subject_montages, class_mode, tag):
    """
    Shared "Phase 2" runner for CSP/MI/ReliefF/saliency reduced experiments.
    subject_montages: dict {subject_id: [channel_names]} (subject-specific,
    from loso_montage — never includes the subject's own data).
    Returns: fold_results dict, per-subject means, grand mean/std.
    """
    all_subject_means, all_subject_stds = [], []
    subject_fold_results = {}

    for sid in SUBJECT_IDS:
        channels = subject_montages[sid]
        print("\n" + "=" * 80)
        print(f"SUBJECT {sid:02d} — {tag} — {class_mode} — channels: {', '.join(channels)}")
        print("=" * 80)

        X, y = load_subject_windows(sid, channels, class_mode)
        print(f"Trials: {X.shape[0]}, shape={X.shape}, classes={np.unique(y)}")

        folds = make_blockwise_folds(len(y), n_blocks=4)
        fold_test_accs = []
        for fold_i, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
            test_acc, best_val_acc, _, _, _, _ = train_one_fold(
                X, y, tr_idx, val_idx, te_idx, tag=f"S{sid:02d} F{fold_i}"
            )
            fold_test_accs.append(test_acc)
            print(f"S{sid:02d} Fold {fold_i}: test {test_acc*100:.2f}% (best val {best_val_acc*100:.2f}%)")

        subj_mean = float(np.mean(fold_test_accs))
        subj_std = float(np.std(fold_test_accs))
        all_subject_means.append(subj_mean)
        all_subject_stds.append(subj_std)
        subject_fold_results[sid] = fold_test_accs
        print(f"Subject {sid:02d} mean: {subj_mean*100:.2f}% ± {subj_std*100:.2f}%")

        del X, y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grand_mean = float(np.mean(all_subject_means))
    grand_std = float(np.std(all_subject_means))
    return subject_fold_results, all_subject_means, all_subject_stds, grand_mean, grand_std


# ────────────────────────────────────────────────
# ERD/ERS reference maps
#
# Always computed on left-vs-right-hand trials specifically, regardless of
# CLASS_MODE, since ERD/ERS here is a fixed physiological reference map
# (lateralized hand movement), not the classification target itself.
# ────────────────────────────────────────────────
def load_subject_for_erd(subject_id, channels):
    ds = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
    preprocessors = [
        Preprocessor("pick_channels", ch_names=channels, ordered=True),
        Preprocessor(scale_to_microvolts, apply_on_array=True),
        Preprocessor("resample", sfreq=TARGET_SFREQ),
        Preprocessor("filter", l_freq=8.0, h_freq=30.0),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocess(ds, preprocessors, n_jobs=1)
    windows = create_windows_from_events(
        ds, trial_start_offset_samples=0, trial_stop_offset_samples=0, preload=True
    )
    X, y = extract_xy(windows)
    mask = np.isin(y, [LEFT_HAND_LABEL, RIGHT_HAND_LABEL])
    return X[mask], y[mask], TARGET_SFREQ


def compute_class_erd_map(X_class, sfreq, n_channels):
    if len(X_class) == 0:
        return np.full(n_channels, np.nan, dtype=np.float32)
    baseline = X_class[:, :, :int(0.5 * sfreq)]
    active = X_class[:, :, int(1.0 * sfreq):int(3.0 * sfreq)]
    p_base = np.mean(baseline ** 2, axis=2)
    p_act = np.mean(active ** 2, axis=2)
    erd = 100 * (p_act - p_base) / (p_base + 1e-12)
    return np.mean(erd, axis=0).astype(np.float32)


def compute_subject_erd_maps(subject_id, channels):
    X, y, sfreq = load_subject_for_erd(subject_id, channels)
    erd_l = compute_class_erd_map(X[y == LEFT_HAND_LABEL], sfreq, len(channels))
    erd_r = compute_class_erd_map(X[y == RIGHT_HAND_LABEL], sfreq, len(channels))
    return {
        "ERD_L": erd_l,
        "ERD_R": erd_r,
        "ERD_(L-R)": erd_l - erd_r,
        "ERD_comb": 0.5 * (erd_l + erd_r),
    }


def correlations_report(vec, erd_maps):
    """All four correlations, for the report only. As instructed, these
    extra correlations (ERD_L, ERD_R, ERD_comb) are logged but NEVER used
    to pick attribution methods or channels — only ERD_(L-R) still drives
    those decisions, exactly as in the original pipeline."""
    return {
        "ERD_L": corrcoef_safe(vec, erd_maps["ERD_L"]),
        "ERD_R": corrcoef_safe(vec, erd_maps["ERD_R"]),
        "ERD_(L-R)": corrcoef_safe(vec, erd_maps["ERD_(L-R)"]),
        "ERD_comb": corrcoef_safe(vec, erd_maps["ERD_comb"]),
    }


# ────────────────────────────────────────────────
# Gradient-based attribution (used by the saliency script only)
# ────────────────────────────────────────────────
IG_STEPS = 32
SMOOTHGRAD_SAMPLES = 12
SMOOTHGRAD_NOISE_STD = 0.10
ATTR_METHODS = ["grad", "grad_x_input", "integrated_gradients", "smoothgrad"]


def _gather_true_class_score(logits, targets):
    return logits.gather(1, targets.view(-1, 1)).sum()


def compute_attr_batch(model, xb, yb, method, ig_steps=IG_STEPS,
                        sg_samples=SMOOTHGRAD_SAMPLES, sg_noise_std=SMOOTHGRAD_NOISE_STD):
    model.eval()
    if method == "grad":
        x = xb.clone().detach().requires_grad_(True)
        out = model(x)
        score = _gather_true_class_score(out, yb)
        score.backward()
        return x.grad.detach()
    elif method == "grad_x_input":
        x = xb.clone().detach().requires_grad_(True)
        out = model(x)
        score = _gather_true_class_score(out, yb)
        score.backward()
        return x.grad.detach() * x.detach()
    elif method == "integrated_gradients":
        baseline = torch.zeros_like(xb)
        total_grad = torch.zeros_like(xb)
        for alpha in torch.linspace(0., 1., ig_steps + 1, device=xb.device)[1:]:
            xi = baseline + alpha * (xb - baseline)
            xi.requires_grad_(True)
            out = model(xi)
            score = _gather_true_class_score(out, yb)
            score.backward()
            total_grad += xi.grad.detach()
        return ((xb - baseline) * (total_grad / ig_steps)).detach()
    elif method == "smoothgrad":
        total = torch.zeros_like(xb)
        noise_std = sg_noise_std * xb.detach().std().clamp(min=1e-8)
        for _ in range(sg_samples):
            xn = xb + torch.randn_like(xb) * noise_std
            xn.requires_grad_(True)
            out = model(xn)
            score = _gather_true_class_score(out, yb)
            score.backward()
            total += xn.grad.detach().abs()
        return (total / sg_samples).detach()
    else:
        raise ValueError(method)


def compute_fold_channel_importances(model, X, y, methods, batch_size=32):
    """
    *** LEAKAGE FIX ***
    Call this with the VALIDATION set (X_val, y_val), never the test set.
    The original script computed saliency from X_te/y_te — the same trials
    later used to report "test" accuracy for the reduced-channel model —
    which is exactly the test-time information leakage the reviewer
    flagged (selecting channels from data that also serves as the
    evaluation set). Section 3.5 of the paper already claims saliency
    comes from the validation set; this now matches that claim.
    """
    n_chans = X.shape[1]
    sums = {m: np.zeros(n_chans, dtype=np.float64) for m in methods}
    n_total = 0
    for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
        bsz = xb.shape[0]
        n_total += bsz
        for m in methods:
            attr = compute_attr_batch(model, xb, yb, m)
            scores = attr.abs().mean(dim=(0, 2)).cpu().numpy()
            sums[m] += scores * bsz
    return {m: normalize_importance(sums[m] / max(1, n_total)) for m in methods}


# ────────────────────────────────────────────────
# Report helpers
# ────────────────────────────────────────────────
def write_comparison_block(f, full_means, reduced_means, grand_mean, red_grand_mean, tag):
    f.write("\n" + "#" * 80 + "\n")
    f.write(f"{tag}\n")
    f.write("#" * 80 + "\n\n")
    for i, (fa, ra) in enumerate(zip(full_means, reduced_means), 1):
        sid = SUBJECT_IDS[i - 1]
        f.write(f"S{sid:02d}: 22ch {fa*100:5.2f}%  ->  reduced {ra*100:5.2f}%\n")
    f.write(f"\nGRAND: 22ch {grand_mean*100:.2f}%  ->  reduced {red_grand_mean*100:.2f}%\n")
