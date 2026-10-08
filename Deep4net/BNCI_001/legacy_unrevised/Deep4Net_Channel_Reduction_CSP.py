#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep4Net on BNCI2014_001 with fold-wise CSP-based channel reduction.

Protocol:
- Dataset: BNCI2014_001
- Subjects: 1..9
- Within-subject 4-fold stratified CV
- Preprocessing matches the original Deep4Net script:
    * channel selection
    * microvolt scaling
    * resample to 250 Hz
    * 4-38 Hz bandpass (IIR Chebyshev-I, order 6, rp=0.5)
    * exponential moving standardization
    * epoch from -0.5 s to 4.0 - 1/sfreq
- Model/training matches the original regimen:
    * Deep4Net
    * AdamW
    * CosineAnnealingLR
    * SWA
    * EEG augmentations
    * Mixup
    * Label smoothing
    * Gradient clipping
    * TTA evaluation
- Channel reduction:
    * CSP channel importance is computed ONLY on the training fold
    * top-K channels are selected per fold
    * train/test are rebuilt using only those channels
"""

import os
import gc
import copy
import random
import warnings

import numpy as np
import mne
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP

from braindecode.datasets import MOABBDataset
from braindecode.models import Deep4Net
from braindecode.preprocessing import exponential_moving_standardize

from torch.optim.swa_utils import AveragedModel, update_bn

warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level("WARNING")

# =========================================================
# 1) Reproducibility
# =========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =========================================================
# 2) Constants
# =========================================================
DATASET_NAME = "BNCI2014_001"
SUBJECT_IDS = list(range(1, 10))

SFREQ = 250
BAND = (4.0, 38.0)

EPOCH_TMIN = -0.5
EPOCH_TMAX = 4.0 - 1.0 / SFREQ
N_SAMPLES_TRIAL = int(np.round((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)) + 1
N_SAMPLES_RECEPTIVE = 1000

FULL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

N_REDUCED_CHANNELS = 12

BATCH_SIZE = 64
EPOCHS = 350
SWA_START = 250
LR = 1e-3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2

N_SPLITS = 4

OUT_ROOT = "Deep4Net_BNCI2014_001_4Fold_CSP"
os.makedirs(OUT_ROOT, exist_ok=True)

# =========================================================
# 3) Utilities
# =========================================================
def normalize_importance(vec):
    vec = np.asarray(vec, dtype=np.float64)
    m = np.max(np.abs(vec))
    if m < 1e-12:
        return vec.astype(np.float32)
    return (vec / m).astype(np.float32)


def prepare_raw(raw: mne.io.BaseRaw, selected_channels):
    raw = raw.copy()
    raw.load_data()

    ch_map = {ch.upper(): ch for ch in raw.ch_names}
    actual_channels = [ch_map[c.upper()] for c in selected_channels if c.upper() in ch_map]
    raw.pick_channels(actual_channels, ordered=True)

    raw.apply_function(lambda x: x * 1e6, channel_wise=False)

    if int(raw.info["sfreq"]) != SFREQ:
        raw.resample(SFREQ, npad="auto", verbose=False)

    iir_params = dict(order=6, ftype="cheby1", rp=0.5)
    raw.filter(
        l_freq=BAND[0],
        h_freq=BAND[1],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )

    raw.apply_function(
        lambda x: exponential_moving_standardize(
            x, factor_new=1e-3, init_block_size=1000
        ),
        channel_wise=False
    )

    return raw


def _build_events_from_class_annotations(raw: mne.io.BaseRaw):
    event_id = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
    sf = raw.info["sfreq"]

    rows = []
    for ann in raw.annotations:
        desc = str(ann["description"])
        if desc in event_id:
            sample = int(np.round(float(ann["onset"]) * sf))
            code = int(event_id[desc])
            rows.append((sample, code))

    if not rows:
        raise RuntimeError("No class cue annotations found in raw annotations.")

    rows.sort(key=lambda t: t[0])
    samples = np.array([r[0] for r in rows], dtype=int)
    codes = np.array([r[1] for r in rows], dtype=int)

    events = np.column_stack((samples, np.zeros(len(samples), dtype=int), codes))
    return events, event_id


def epochs_from_cue_onsets(raw: mne.io.BaseRaw):
    events, event_id = _build_events_from_class_annotations(raw)

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )

    X = epochs.get_data().astype(np.float32)
    code_to_label = {1: 0, 2: 1, 3: 2, 4: 3}
    y = np.array([code_to_label[int(c)] for c in epochs.events[:, 2]], dtype=np.int64)

    if X.shape[2] != N_SAMPLES_TRIAL:
        if X.shape[2] > N_SAMPLES_TRIAL:
            X = X[:, :, :N_SAMPLES_TRIAL]
        else:
            pad = N_SAMPLES_TRIAL - X.shape[2]
            X = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="edge")

    return X, y


def build_subject_arrays(subject_id, selected_channels):
    ds = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=[subject_id])

    X_all, y_all = [], []
    for bd in ds.datasets:
        raw = prepare_raw(bd.raw, selected_channels)
        X, y = epochs_from_cue_onsets(raw)
        X_all.append(X)
        y_all.append(y)

    del ds
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all


# =========================================================
# 4) CSP channel selection
# =========================================================
def compute_csp_importance(X, y, n_components=4):
    """
    X: [N, C, T], y: [N]
    Returns per-channel importance.
    """
    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False
    )
    csp.fit(X.astype(np.float64), y.astype(np.int64))
    importance = np.abs(csp.patterns_).sum(axis=0)
    return normalize_importance(importance)


def select_top_channels_csp(X_train_full, y_train, full_channels, k):
    importance = compute_csp_importance(X_train_full, y_train)
    idx = np.argsort(-importance)[:k]
    selected_channels = [full_channels[i] for i in idx]
    return selected_channels, importance


# =========================================================
# 5) Training helpers
# =========================================================
def apply_eeg_augmentations(xb):
    xb = xb.clone()

    # additive gaussian noise (~50%)
    if random.random() > 0.5:
        noise_std = xb.std() * 0.05
        xb = xb + torch.randn_like(xb) * noise_std

    # amplitude scaling (~50%)
    if random.random() > 0.5:
        scales = torch.empty(xb.size(0), 1, 1, device=xb.device).uniform_(0.8, 1.2)
        xb = xb * scales

    # random channel dropout (~30%)
    if random.random() > 0.7:
        num_drop = random.randint(1, min(3, xb.size(1)))
        channels = torch.randperm(xb.size(1), device=xb.device)[:num_drop]
        xb[:, channels, :] = 0.0

    return xb


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def compute_smoothed_loss(log_probs, targets, smoothing=0.1):
    """
    log_probs: [B, C, T]
    targets:   [B, T]
    """
    n_classes = log_probs.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def mixup_criterion(pred, y_a, y_b, lam, smoothing=0.1):
    """
    pred: [B, C, T]
    y_a, y_b: [B]
    """
    log_probs = F.log_softmax(pred, dim=1)

    y_a_rep = y_a.unsqueeze(1).expand(-1, pred.size(2))
    y_b_rep = y_b.unsqueeze(1).expand(-1, pred.size(2))

    loss_a = compute_smoothed_loss(log_probs, y_a_rep, smoothing=smoothing)
    loss_b = compute_smoothed_loss(log_probs, y_b_rep, smoothing=smoothing)
    return lam * loss_a + (1 - lam) * loss_b


@torch.no_grad()
def evaluate_tta(model, loader):
    model.eval()
    preds = []
    ys = []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits_1 = model(xb).view(xb.size(0), 4, -1)

        xb_noise = xb + torch.randn_like(xb) * (xb.std() * 0.02)
        logits_2 = model(xb_noise).view(xb.size(0), 4, -1)

        xb_scale = xb * 0.90
        logits_3 = model(xb_scale).view(xb.size(0), 4, -1)

        p1 = F.softmax(logits_1, dim=1).mean(dim=2)
        p2 = F.softmax(logits_2, dim=1).mean(dim=2)
        p3 = F.softmax(logits_3, dim=1).mean(dim=2)

        final_probs = (p1 + p2 + p3) / 3.0
        pred = torch.argmax(final_probs, dim=1)

        preds.append(pred.cpu().numpy())
        ys.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return float((preds == ys).mean())


def build_deep4net_model(n_chans, n_classes=4):
    model = Deep4Net(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=N_SAMPLES_RECEPTIVE,
        final_conv_length="auto",
        drop_prob=0.5,
    )
    return model.to(DEVICE)


# =========================================================
# 6) Fold training
# =========================================================
def train_one_fold(
    X_train, y_train,
    X_test, y_test,
    subject_id,
    fold_idx,
    fold_dir
):
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    swa_bn_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    model = build_deep4net_model(n_chans=X_train.shape[1], n_classes=4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    swa_model = AveragedModel(model).to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            xb = apply_eeg_augmentations(xb)

            if random.random() > 0.5:
                xb_mix, y_a, y_b, lam = mixup_data(xb, yb, alpha=MIXUP_ALPHA)
            else:
                xb_mix = xb
                y_a, y_b, lam = yb, yb, 1.0

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb_mix)
            logits = logits.view(logits.size(0), logits.size(1), -1)

            loss = mixup_criterion(
                logits,
                y_a,
                y_b,
                lam,
                smoothing=LABEL_SMOOTHING
            )
            loss.backward()

            nn_utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

            with torch.no_grad():
                pred = logits.mean(dim=2).argmax(dim=1)
                running_correct += (pred == yb).sum().item()
                running_total += yb.size(0)

        scheduler.step()

        if epoch >= SWA_START:
            swa_model.update_parameters(model)

        if epoch % 50 == 0 or epoch == 1 or epoch == EPOCHS:
            train_loss = running_loss / max(running_total, 1)
            train_acc = running_correct / max(running_total, 1)
            print(
                f"Subject {subject_id:02d} | Fold {fold_idx} | "
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"loss={train_loss:.4f} | train_acc={train_acc*100:.2f}%"
            )

    update_bn(swa_bn_loader, swa_model, device=DEVICE)
    final_model = swa_model.module
    final_model.eval()

    fold_acc = evaluate_tta(final_model, test_loader)

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "subject": subject_id,
            "fold": fold_idx,
            "fold_acc": fold_acc,
        },
        os.path.join(fold_dir, "best_model_swa.pth")
    )

    del train_ds, test_ds, train_loader, test_loader, swa_bn_loader
    del model, optimizer, scheduler, swa_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_acc


# =========================================================
# 7) Subject pipeline
# =========================================================
def run_subject(subject_id):
    print(f"\n{'='*80}")
    print(f"Starting subject {subject_id}")
    print(f"{'='*80}")

    subject_dir = os.path.join(OUT_ROOT, f"subj_{subject_id:02d}")
    os.makedirs(subject_dir, exist_ok=True)

    # Build full 22-channel data once for this subject
    print("Building full-channel subject arrays...")
    X_full, y = build_subject_arrays(subject_id, FULL_CHANNELS)

    print(f"Subject {subject_id}: X_full shape = {X_full.shape}, y shape = {y.shape}")

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_accs = []
    fold_selected_channels = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y), start=1):
        print(f"\n--- Subject {subject_id} | Fold {fold_idx}/{N_SPLITS} ---")
        fold_dir = os.path.join(subject_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train_full = X_full[train_idx]
        y_train = y[train_idx]
        X_test_full = X_full[test_idx]
        y_test = y[test_idx]

        # CSP selection on training fold only
        selected_channels, importance = select_top_channels_csp(
            X_train_full, y_train, FULL_CHANNELS, N_REDUCED_CHANNELS
        )
        fold_selected_channels.append(selected_channels)

        print("Selected channels:", ", ".join(selected_channels))

        np.save(os.path.join(fold_dir, "csp_importance.npy"), importance)
        with open(os.path.join(fold_dir, "selected_channels.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(selected_channels) + "\n")

        selected_idx = [FULL_CHANNELS.index(ch) for ch in selected_channels]
        X_train = X_train_full[:, selected_idx, :]
        X_test = X_test_full[:, selected_idx, :]

        fold_acc = train_one_fold(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            subject_id=subject_id,
            fold_idx=fold_idx,
            fold_dir=fold_dir
        )

        print(f"Subject {subject_id} | Fold {fold_idx} accuracy = {fold_acc*100:.2f}%")
        fold_accs.append(fold_acc)

    subject_mean = float(np.mean(fold_accs))
    subject_std = float(np.std(fold_accs))

    with open(os.path.join(subject_dir, "subject_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Subject: {subject_id}\n")
        for i, acc in enumerate(fold_accs, start=1):
            f.write(f"Fold {i} Accuracy (%): {acc*100:.2f}\n")
        f.write(f"Mean Accuracy (%): {subject_mean*100:.2f}\n")
        f.write(f"Std Accuracy (%): {subject_std*100:.2f}\n\n")

        for i, chs in enumerate(fold_selected_channels, start=1):
            f.write(f"Fold {i} selected channels: {', '.join(chs)}\n")

    del X_full, y
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "subject_id": subject_id,
        "fold_accs": fold_accs,
        "mean_acc": subject_mean,
        "std_acc": subject_std,
        "selected_channels_per_fold": fold_selected_channels,
    }


# =========================================================
# 8) Main
# =========================================================
def main():
    all_results = []

    for subject_id in SUBJECT_IDS:
        result = run_subject(subject_id)
        all_results.append(result)

    means = [r["mean_acc"] for r in all_results]
    grand_mean = float(np.mean(means))
    grand_std = float(np.std(means))

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    for r in all_results:
        print(
            f"Subject {r['subject_id']:02d} | "
            f"mean={r['mean_acc']*100:.2f}% | "
            f"std={r['std_acc']*100:.2f}%"
        )
    print(f"Overall mean across subjects = {grand_mean*100:.2f}%")
    print(f"Overall std across subjects  = {grand_std*100:.2f}%")

    summary_path = os.path.join(OUT_ROOT, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Deep4Net + 4-fold CV + CSP-based channel reduction\n\n")
        for r in all_results:
            f.write(
                f"Subject {r['subject_id']:02d} | "
                f"mean={r['mean_acc']*100:.2f}% | "
                f"std={r['std_acc']*100:.2f}%\n"
            )
            for i, acc in enumerate(r["fold_accs"], start=1):
                f.write(f"  Fold {i}: {acc*100:.2f}%\n")
            for i, chs in enumerate(r["selected_channels_per_fold"], start=1):
                f.write(f"  Fold {i} channels: {', '.join(chs)}\n")
            f.write("\n")

        f.write(f"Overall mean across subjects = {grand_mean*100:.2f}%\n")
        f.write(f"Overall std across subjects  = {grand_std*100:.2f}%\n")

    print(f"\nSaved results to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
