# eegitnet_hgd_mutual_information_global12_reduced_only.py
# EEG-ITNet on HGD using Mutual-Information-selected GLOBAL 12 channels
# Training regimen aligned to:
#   eegitnet_hgd_paper_ready_full_vs_reduced_no_occlusion.py
# - Computes global MI ranking from full 22-channel data
# - Selects top 12 channels globally
# - Trains EEG-ITNet only on the selected 12 channels

import os
import gc
import copy
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as nn_utils

from sklearn.feature_selection import mutual_info_classif

import mne
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
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ────────────────────────────────────────────────
# Config — aligned to EEG-ITNet reference code
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
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Pz"
]

REDUCED_N_CHANNELS = 12
OUTPUT_BASE = "EEGITNet_HGD_MI_Global12_ReducedOnly_Output"

# ────────────────────────────────────────────────
# Utilities
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
        yield (
            torch.tensor(X[sl], dtype=torch.float32, device=DEVICE),
            torch.tensor(y[sl], dtype=torch.long, device=DEVICE),
        )


@torch.no_grad()
def evaluate(model, X, y, criterion, batch_size=256):
    model.eval()

    losses = []
    preds = []

    for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
        out = model(xb)
        loss = criterion(out, yb)
        losses.append(loss.item())
        preds.append(torch.argmax(out, dim=1).cpu().numpy())

    preds = np.concatenate(preds)
    acc = (preds == y).mean()

    return float(np.mean(losses)), float(acc)


def normalize_importance(vec):
    vec = np.asarray(vec, dtype=np.float64)
    m = np.max(np.abs(vec))
    return (vec / m).astype(np.float32) if m > 1e-12 else vec.astype(np.float32)


def top_k_channels(scores, ch_names, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(ch_names[i], float(scores[i])) for i in idx]


# ────────────────────────────────────────────────
# Data loading — aligned to EEG-ITNet reference
# ────────────────────────────────────────────────
def load_subject_windows(subject_id, channels):
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

    # Same binary setup as reference EEG-ITNet script
    mask = np.isin(y, [0, 1])
    X = X[mask]
    y = y[mask]

    return X.astype(np.float32), y.astype(np.int64)


# ────────────────────────────────────────────────
# Mutual Information channel ranking
# ────────────────────────────────────────────────
def compute_mi_importance(X, y):
    # Channel-level features: temporal variance per trial/channel
    features = np.var(X, axis=2).astype(np.float64)  # shape: trials x channels

    scores = mutual_info_classif(
        features,
        y,
        discrete_features=False,
        random_state=SEED,
    )
    return normalize_importance(scores)


def compute_global_mi_top_channels():
    print("\n" + "=" * 80)
    print("PHASE 1: GLOBAL MUTUAL INFORMATION CHANNEL RANKING FROM FULL 22-CHANNEL DATA")
    print("=" * 80)

    subject_scores = []

    for sid in SUBJECT_IDS:
        print(f"\nComputing MI scores for Subject {sid:02d}")

        X, y = load_subject_windows(sid, FULL_CHANNELS)

        print(f"Subject {sid:02d}: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

        scores = compute_mi_importance(X, y)
        subject_scores.append(scores)

        top12 = top_k_channels(scores, FULL_CHANNELS, 12)
        print(
            f"Subject {sid:02d} MI Top-12: "
            + ", ".join(f"{ch} ({v:.4f})" for ch, v in top12)
        )

        del X, y
        gc.collect()

    global_scores = normalize_importance(np.mean(np.stack(subject_scores, axis=0), axis=0))
    global_top12 = top_k_channels(global_scores, FULL_CHANNELS, REDUCED_N_CHANNELS)
    selected_channels = [ch for ch, _ in global_top12]

    print("\n" + "=" * 80)
    print("GLOBAL MI TOP-12 CHANNELS")
    print("=" * 80)
    print(", ".join(f"{ch} ({v:.4f})" for ch, v in global_top12))

    return selected_channels, global_scores


# ────────────────────────────────────────────────
# Model — same EEG-ITNet as reference
# ────────────────────────────────────────────────
def build_model(n_chans, n_times, n_classes):
    return EEGITNet(
        n_outputs=n_classes,
        n_chans=n_chans,
        n_times=n_times,
        drop_prob=DROPOUT,
    ).to(DEVICE)


# ────────────────────────────────────────────────
# Training — directly aligned to reference code
# ────────────────────────────────────────────────
def train_one_fold(X, y, train_idx, val_idx, test_idx, sid, fold_i):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))

    model = build_model(n_chans, n_times, n_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=30,
        verbose=True,
    )

    best_state = None
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

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

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"S{sid:02d} | Fold {fold_i}/4 | "
            f"Ep {epoch:03d} | tr loss {np.mean(train_losses):.4f} | "
            f"val {val_acc:.1%} | te {test_acc:.1%} | "
            f"best_val_loss={best_val_loss:.4f} | lr={current_lr:.2e}"
        )

        if epochs_no_improve >= PATIENCE:
            print(f"S{sid:02d} | Fold {fold_i} | Early stop @ ep {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state found")

    model.load_state_dict(best_state)
    _, final_test_acc = evaluate(model, X_te, y_te, criterion)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_test_acc, best_val_acc


# ────────────────────────────────────────────────
# Reduced global-12 MI training only
# ────────────────────────────────────────────────
def run_reduced_global12_training(selected_channels):
    print("\n" + "=" * 80)
    print("PHASE 2: EEG-ITNet TRAINING ON MI GLOBAL-12 CHANNELS ONLY")
    print("=" * 80)
    print("Selected channels:", ", ".join(selected_channels))

    all_subject_means = []
    all_subject_stds = []
    subject_fold_results = {}

    for sid in SUBJECT_IDS:
        print("\n" + "=" * 80)
        print(f"SUBJECT {sid:02d} — EEG-ITNet — MI GLOBAL-12")
        print("=" * 80)

        X, y = load_subject_windows(sid, selected_channels)
        print(f"Trials (left+right only): {X.shape[0]}, shape={X.shape}")

        folds = make_blockwise_folds(len(y), n_blocks=4)
        fold_test_accs = []

        for fold_i, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
            print(
                f"\nS{sid:02d} | Fold {fold_i}/4 | "
                f"train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}"
            )

            test_acc, best_val_acc = train_one_fold(
                X, y, tr_idx, val_idx, te_idx, sid=sid, fold_i=fold_i
            )
            fold_test_accs.append(test_acc)

            print(
                f"S{sid:02d} | Fold {fold_i} finished | "
                f"Test Acc: {test_acc * 100:.2f}% | "
                f"Best Val Acc: {best_val_acc * 100:.2f}%"
            )

        subj_mean = float(np.mean(fold_test_accs))
        subj_std = float(np.std(fold_test_accs))

        all_subject_means.append(subj_mean)
        all_subject_stds.append(subj_std)
        subject_fold_results[sid] = fold_test_accs

        print("\n" + "-" * 80)
        print(f"Subject {sid:02d} RESULTS — MI GLOBAL-12")
        for i, acc in enumerate(fold_test_accs, start=1):
            print(f"Fold {i}: {acc * 100:.2f}%")
        print(f"Mean: {subj_mean * 100:.2f}% ± {subj_std * 100:.2f}%")
        print("-" * 80)

        del X, y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grand_mean = float(np.mean(all_subject_means))
    grand_std = float(np.std(all_subject_means))

    print("\n" + "#" * 80)
    print("FINAL RESULTS — EEG-ITNet — MI GLOBAL-12 REDUCED ONLY")
    print(f"Grand mean: {grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%")
    print("#" * 80)

    return subject_fold_results, all_subject_means, all_subject_stds, grand_mean, grand_std


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    selected_channels, global_mi_scores = compute_global_mi_top_channels()

    with open(os.path.join(OUTPUT_BASE, "global_12_mi_channels.txt"), "w", encoding="utf-8") as f:
        for ch in selected_channels:
            f.write(ch + "\n")

    np.save(
        os.path.join(OUTPUT_BASE, "global_mi_scores_full22.npy"),
        global_mi_scores.astype(np.float32),
    )

    (
        subject_fold_results,
        subject_means,
        subject_stds,
        grand_mean,
        grand_std,
    ) = run_reduced_global12_training(selected_channels)

    report_path = os.path.join(OUTPUT_BASE, "EEGITNet_MI_Global12_Report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EEG-ITNet — HGD Schirrmeister2017 — Mutual Information GLOBAL-12 REDUCED ONLY\n")
        f.write("=" * 80 + "\n\n")

        f.write("Selected Global 12 Channels by Mutual Information:\n")
        f.write(", ".join(selected_channels) + "\n\n")

        f.write("Global MI Ranking Full 22 Channels:\n")
        for ch, score in top_k_channels(global_mi_scores, FULL_CHANNELS, len(FULL_CHANNELS)):
            f.write(f"  {ch:>4}: {score:.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Subject Results\n")
        f.write("=" * 80 + "\n\n")

        for sid, mean_acc, std_acc in zip(SUBJECT_IDS, subject_means, subject_stds):
            f.write(f"Subject {sid:02d}\n")
            for fold_i, acc in enumerate(subject_fold_results[sid], start=1):
                f.write(f"  Fold {fold_i}: {acc * 100:.2f}%\n")
            f.write(f"  Mean: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"Grand Mean Accuracy: {grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%\n")

    print("\nDone.")
    print(f"Selected channels saved to: {os.path.join(OUTPUT_BASE, 'global_12_mi_channels.txt')}")
    print(f"Report saved to: {report_path}")
