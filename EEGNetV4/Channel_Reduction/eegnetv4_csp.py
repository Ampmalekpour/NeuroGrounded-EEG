# eegnetv4_hgd_csp_channel_reduction.py

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

from mne.decoding import CSP

from braindecode.datasets import MOABBDataset
from braindecode.models import EEGNetv4
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
)

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
# Config
# ────────────────────────────────────────────────
DATASET_NAME = "Schirrmeister2017"
SUBJECT_IDS = [1, 2, 3, 4, 5]

FULL_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Pz'
]

# channel reduction
N_IMPORTANT_CHANNELS = 12
USE_LOCAL_SELECTION = False   # False = global top-n, True = local per-subject top-n

# preprocessing / windowing: from first script
LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0
TARGET_SFREQ = 100
TRIAL_START_OFFSET_S = 0.0
TRIAL_STOP_OFFSET_S = 4.0

# EEGNetv4 regimen: from first script
MAX_EPOCHS = 600
PATIENCE = 150
BATCH_SIZE = 96
LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30
LABEL_SMOOTHING = 0.1

OUTPUT_DIR = "EEGNetV4_HGD_CSP_Reduced"

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


def normalize_importance(vec):
    vec = np.asarray(vec, dtype=np.float64)
    m = np.max(np.abs(vec))
    return vec / m if m > 1e-12 else vec


def top_k_channels(scores, ch_names, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(ch_names[i], float(scores[i])) for i in idx]


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
    losses, preds = [], []
    for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
        out = model(xb)
        losses.append(criterion(out, yb).item())
        preds.append(torch.argmax(out, dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    return np.mean(losses), (preds == y).mean()


# ────────────────────────────────────────────────
# Data loading with first-script preprocessing
# ────────────────────────────────────────────────
def load_subject_data(subject_id, channels):
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

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert sfreq == TARGET_SFREQ

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(TRIAL_START_OFFSET_S * sfreq),
        trial_stop_offset_samples=int(TRIAL_STOP_OFFSET_S * sfreq),
        preload=True,
    )

    X, y = extract_xy(windows_dataset)

    # keep same label handling as first script
    mask = np.isin(y, [0, 1, 2, 3])
    X = X[mask]
    y = y[mask]

    return X, y


# ────────────────────────────────────────────────
# CSP channel importance
# ────────────────────────────────────────────────
def compute_csp_importance(X, y, n_components=4):
    """
    X shape: [trials, channels, time]
    y shape: [trials]
    Returns channel importance vector of shape [channels]
    """
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    # CSP expects float64
    X64 = X.astype(np.float64)
    csp.fit(X64, y)
    imp = np.abs(csp.patterns_).sum(axis=0)
    return normalize_importance(imp)


# ────────────────────────────────────────────────
# EEGNetv4 model and training: first-script regimen
# ────────────────────────────────────────────────
def build_eegnet_model(n_chans, n_times, n_classes):
    return EEGNetv4(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=n_times,
        final_conv_length="auto",
        F1=8, D=2, F2=16,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=DROPOUT,
    ).to(DEVICE)


def train_one_fold(X, y, train_idx, val_idx, test_idx):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))

    model = build_eegnet_model(n_chans, n_times, n_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=30,
        verbose=True
    )

    best_state = None
    best_val_loss = float('inf')
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
        test_loss, test_acc = evaluate(model, X_te, y_te, criterion)

        scheduler.step(val_loss)

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
            f"Ep {epoch:03d} | tr loss {np.mean(train_losses):.4f} | "
            f"val {val_acc:.1%} | te {test_acc:.1%} | "
            f"best_val_loss={best_val_loss:.4f} | lr={current_lr:.2e}"
        )

        if epochs_no_improve >= PATIENCE:
            print(f"Early stop @ ep {epoch}")
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
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: CSP-BASED CHANNEL REDUCTION ON FULL 22-CHANNEL DATA")
    print("=" * 80)

    data_cache = {}
    subject_importances = {}

    for sid in SUBJECT_IDS:
        print(f"\nLoading subject {sid:02d} full 22-channel data...")
        X, y = load_subject_data(sid, FULL_CHANNELS)
        data_cache[sid] = (X, y)

        print(f"Computing CSP importance for subject {sid:02d}...")
        imp = compute_csp_importance(X, y, n_components=4)
        subject_importances[sid] = imp

        print(f"Top-{N_IMPORTANT_CHANNELS} local channels for S{sid:02d}:")
        print(", ".join(
            f"{ch} ({score:.4f})"
            for ch, score in top_k_channels(imp, FULL_CHANNELS, N_IMPORTANT_CHANNELS)
        ))

    global_importance = normalize_importance(
        np.mean(np.stack([subject_importances[sid] for sid in SUBJECT_IDS], axis=0), axis=0)
    )

    global_channels = [ch for ch, _ in top_k_channels(global_importance, FULL_CHANNELS, N_IMPORTANT_CHANNELS)]
    local_channels = {
        sid: [ch for ch, _ in top_k_channels(subject_importances[sid], FULL_CHANNELS, N_IMPORTANT_CHANNELS)]
        for sid in SUBJECT_IDS
    }

    print("\n" + "=" * 80)
    print(f"GLOBAL TOP-{N_IMPORTANT_CHANNELS} CHANNELS")
    print("=" * 80)
    print(", ".join(global_channels))

    print("\n" + "=" * 80)
    print("PHASE 2: EEGNETV4 TRAINING WITH REDUCED CHANNELS")
    print("=" * 80)
    print("Selection mode:", "LOCAL" if USE_LOCAL_SELECTION else "GLOBAL")

    all_subject_means = []
    subject_results = []

    for sid in SUBJECT_IDS:
        print("\n" + "-" * 80)
        print(f"SUBJECT {sid:02d}")
        print("-" * 80)

        X_full, y = data_cache[sid]

        if USE_LOCAL_SELECTION:
            selected_channels = local_channels[sid]
        else:
            selected_channels = global_channels

        selected_indices = [FULL_CHANNELS.index(ch) for ch in selected_channels]
        X = X_full[:, selected_indices, :]

        print(f"Selected {len(selected_channels)} channels: {selected_channels}")
        print(f"Reduced data shape: {X.shape}")

        folds = make_blockwise_folds(len(y), n_blocks=4)
        fold_test_accs = []

        for fold_i, (tr_idx, val_idx, te_idx) in enumerate(folds, 1):
            print(f"\nFold {fold_i}/4 | train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")
            test_acc, val_acc = train_one_fold(X, y, tr_idx, val_idx, te_idx)
            fold_test_accs.append(test_acc)
            print(f"Fold {fold_i} final test acc: {test_acc*100:.2f}%")

        subj_mean = np.mean(fold_test_accs)
        subj_std = np.std(fold_test_accs)
        all_subject_means.append(subj_mean)

        print(f"\nSubject {sid:02d} mean accuracy: {subj_mean*100:.2f}% ± {subj_std*100:.2f}%")

        subject_results.append({
            "subject": sid,
            "channels": selected_channels,
            "fold_test_accs": fold_test_accs,
            "mean_acc": subj_mean,
            "std_acc": subj_std
        })

    grand_mean = np.mean(all_subject_means)
    grand_std = np.std(all_subject_means)

    print("\n" + "#" * 80)
    print("FINAL RESULTS")
    print("#" * 80)
    for res in subject_results:
        print(f"S{res['subject']:02d}: {res['mean_acc']*100:.2f}% ± {res['std_acc']*100:.2f}%")
    print(f"Grand mean: {grand_mean*100:.2f}% ± {grand_std*100:.2f}%")

    report_path = os.path.join(
        OUTPUT_DIR,
        f"eegnetv4_csp_reduced_top{N_IMPORTANT_CHANNELS}_{'local' if USE_LOCAL_SELECTION else 'global'}.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EEGNetv4 + CSP Channel Reduction Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"N important channels: {N_IMPORTANT_CHANNELS}\n")
        f.write(f"Selection mode: {'local' if USE_LOCAL_SELECTION else 'global'}\n\n")

        f.write("Global channel ranking:\n")
        for ch, score in top_k_channels(global_importance, FULL_CHANNELS, len(FULL_CHANNELS)):
            f.write(f"  {ch}: {score:.6f}\n")

        f.write("\nSelected global channels:\n")
        f.write(", ".join(global_channels) + "\n\n")

        f.write("Local channel selections:\n")
        for sid in SUBJECT_IDS:
            f.write(f"S{sid:02d}: " + ", ".join(local_channels[sid]) + "\n")

        f.write("\nSubject results:\n")
        for res in subject_results:
            f.write(f"\nSubject {res['subject']:02d}\n")
            f.write(f"Channels: {', '.join(res['channels'])}\n")
            for i, acc in enumerate(res["fold_test_accs"], 1):
                f.write(f"  Fold {i}: {acc*100:.2f}%\n")
            f.write(f"  Mean: {res['mean_acc']*100:.2f}% ± {res['std_acc']*100:.2f}%\n")

        f.write(f"\nGrand mean accuracy: {grand_mean*100:.2f}% ± {grand_std*100:.2f}%\n")

    print(f"\nReport saved to: {report_path}")
