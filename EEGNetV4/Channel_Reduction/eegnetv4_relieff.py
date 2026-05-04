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
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Pz"
]

CHANNEL_COUNTS = [12, 10]
CONDITIONS = ["G-12ch", "G-10ch", "L-12ch", "L-10ch"]

# Preprocessing from first EEGNetv4 script
LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0
TARGET_SFREQ = 100

TRIAL_START_OFFSET_S = 0.0
TRIAL_STOP_OFFSET_S = 4.0

# EEGNetv4 regimen from first script
MAX_EPOCHS = 600
PATIENCE = 150
BATCH_SIZE = 96

LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM = 10.0

N_FOLDS = 4

OUTPUT_DIR = "EEGNetV4_HGD_ReliefF_Reduced"

# ReliefF config from provided script
RELIEFF_NEIGHBORS = 10


# ────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────
def scale_to_microvolts(x):
    return x * 1e6


def extract_xy(windows_dataset):
    xs, ys = [], []
    for i in range(len(windows_dataset)):
        item = windows_dataset[i]
        x, y = item[0], item[1]
        xs.append(np.asarray(x, dtype=np.float32))
        ys.append(int(y))
    X = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    denom = np.max(np.abs(scores))
    if denom < 1e-12:
        return scores
    return scores / denom


def top_k_channels(scores, ch_names, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(ch_names[i], float(scores[i])) for i in idx]


def make_blockwise_folds(n_samples, n_blocks=4):
    """
    Same 4-fold blockwise CV structure as first EEGNetv4 script:
    - contiguous blocks
    - test block = current block
    - val block = previous block
    - train = remaining blocks
    """
    assert n_samples >= n_blocks
    blocks = np.array_split(np.arange(n_samples), n_blocks)
    folds = []

    for test_i in range(n_blocks):
        val_i = (test_i - 1) % n_blocks
        train_blocks = [b for b in range(n_blocks) if b not in [val_i, test_i]]

        train_idx = np.concatenate([blocks[b] for b in train_blocks])
        val_idx = blocks[val_i]
        test_idx = blocks[test_i]

        folds.append((train_idx, val_idx, test_idx))

    return folds


def batch_iter(X, y, batch_size, shuffle=True):
    idx = np.arange(len(y))
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        bidx = idx[start:start + batch_size]
        xb = torch.tensor(X[bidx], dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(y[bidx], dtype=torch.long, device=DEVICE)
        yield xb, yb


@torch.no_grad()
def evaluate(model, X, y, criterion, batch_size=256):
    model.eval()

    losses = []
    preds = []

    for xb, yb in batch_iter(X, y, batch_size=batch_size, shuffle=False):
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())

    preds = np.concatenate(preds)
    acc = np.mean(preds == y)
    mean_loss = float(np.mean(losses))

    return mean_loss, acc


def standardize_from_train(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


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

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(TRIAL_START_OFFSET_S * sfreq),
        trial_stop_offset_samples=int(TRIAL_STOP_OFFSET_S * sfreq),
        preload=True,
    )

    X, y = extract_xy(windows_dataset)

    # Keep MI classes only, consistent with first script
    mask = np.isin(y, [0, 1, 2, 3])
    X = X[mask]
    y = y[mask]

    return X, y


# ────────────────────────────────────────────────
# ReliefF channel importance
# ────────────────────────────────────────────────
def compute_relieff_importance(X, y, n_neighbors=10):
    """
    ReliefF using per-channel temporal variance as features.

    Matches the logic of your provided EEGConformer script:
      features = var over time for each channel
      normalize each feature dimension to [0, 1]
      ReliefF score = avg(diff to misses - diff to hits)
    """
    n_trials, n_chans, _ = X.shape

    features = np.var(X, axis=2)  # [n_trials, n_chans]

    feat_min = features.min(axis=0, keepdims=True)
    feat_max = features.max(axis=0, keepdims=True)
    features = (features - feat_min) / (feat_max - feat_min + 1e-8)

    weights = np.zeros(n_chans, dtype=np.float64)

    for i in range(n_trials):
        dists = np.linalg.norm(features - features[i], axis=1)
        dists[i] = np.inf

        hits_idx = np.where(y == y[i])[0]
        near_hits = hits_idx[np.argsort(dists[hits_idx])[:n_neighbors]]

        misses_idx = np.where(y != y[i])[0]
        near_misses = misses_idx[np.argsort(dists[misses_idx])[:n_neighbors]]

        diff_hits = np.mean(np.abs(features[i] - features[near_hits]), axis=0)
        diff_misses = np.mean(np.abs(features[i] - features[near_misses]), axis=0)

        weights += diff_misses - diff_hits

    weights = weights / n_trials
    return normalize_scores(weights)


# ────────────────────────────────────────────────
# EEGNetv4 model/training
# ────────────────────────────────────────────────
def build_eegnet_model(n_chans, n_times, n_classes):
    model = EEGNetv4(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=n_times,
        final_conv_length="auto",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=DROPOUT,
    )
    return model.to(DEVICE)


def train_one_fold(X, y, train_idx, val_idx, test_idx, sid, condition_name, fold_idx):
    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    X_train, X_val, X_test = standardize_from_train(X_train, X_val, X_test)

    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))

    model = build_eegnet_model(n_chans=n_chans, n_times=n_times, n_classes=n_classes)

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

        for xb, yb in batch_iter(X_train, y_train, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            train_losses.append(loss.item())

        val_loss, val_acc = evaluate(model, X_val, y_val, criterion)
        test_loss, test_acc = evaluate(model, X_test, y_test, criterion)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"S{sid:02d} | {condition_name} | Fold {fold_idx} | "
            f"Ep {epoch:03d} | "
            f"TrainLoss {np.mean(train_losses):.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"ValAcc {val_acc * 100:5.2f}% | "
            f"TestAcc {test_acc * 100:5.2f}% | "
            f"BestValLoss {best_val_loss:.4f} | "
            f"LR {current_lr:.2e}"
        )

        if epochs_no_improve >= PATIENCE:
            print(
                f"S{sid:02d} | {condition_name} | Fold {fold_idx} | "
                f"Early stopping at epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)
    _, final_test_acc = evaluate(model, X_test, y_test, criterion)

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
    print("PHASE 1: RELIEFF CHANNEL IMPORTANCE")
    print("=" * 80)

    data_cache = {}
    subject_importances = {}

    for sid in SUBJECT_IDS:
        print(f"\nLoading subject {sid:02d} full 22-channel data...")
        X, y = load_subject_data(sid, FULL_CHANNELS)

        data_cache[sid] = (X, y)

        print(f"Subject {sid:02d} data shape: X={X.shape}, y={y.shape}")
        print(f"Computing ReliefF importance for subject {sid:02d}...")

        relieff_scores = compute_relieff_importance(
            X, y, n_neighbors=RELIEFF_NEIGHBORS
        )
        subject_importances[sid] = relieff_scores

        print(f"Subject {sid:02d} top-12 ReliefF channels:")
        print(
            ", ".join(
                f"{ch} ({score:.4f})"
                for ch, score in top_k_channels(relieff_scores, FULL_CHANNELS, 12)
            )
        )

    global_importance = normalize_scores(
        np.mean(
            np.stack([subject_importances[sid] for sid in SUBJECT_IDS], axis=0),
            axis=0,
        )
    )

    global_12 = [ch for ch, _ in top_k_channels(global_importance, FULL_CHANNELS, 12)]
    global_10 = [ch for ch, _ in top_k_channels(global_importance, FULL_CHANNELS, 10)]

    local_12 = {
        sid: [ch for ch, _ in top_k_channels(subject_importances[sid], FULL_CHANNELS, 12)]
        for sid in SUBJECT_IDS
    }
    local_10 = {
        sid: [ch for ch, _ in top_k_channels(subject_importances[sid], FULL_CHANNELS, 10)]
        for sid in SUBJECT_IDS
    }

    print("\n" + "=" * 80)
    print("GLOBAL RELIEFF RANKING")
    print("=" * 80)
    for ch, score in top_k_channels(global_importance, FULL_CHANNELS, len(FULL_CHANNELS)):
        print(f"{ch:>4s}: {score:.6f}")

    print("\nGlobal top-12:", global_12)
    print("Global top-10:", global_10)

    print("\nLocal top channels:")
    for sid in SUBJECT_IDS:
        print(f"S{sid:02d} local top-12: {local_12[sid]}")
        print(f"S{sid:02d} local top-10: {local_10[sid]}")

    conditions = {
        "G-12ch": lambda sid: global_12,
        "G-10ch": lambda sid: global_10,
        "L-12ch": lambda sid: local_12[sid],
        "L-10ch": lambda sid: local_10[sid],
    }

    print("\n" + "=" * 80)
    print("PHASE 2: EEGNETV4 TRAINING ON RELIEFF-REDUCED CHANNEL SETS")
    print("=" * 80)

    all_results = {cond: [] for cond in conditions.keys()}
    detailed_results = {cond: [] for cond in conditions.keys()}

    for condition_name, get_channels in conditions.items():
        print("\n" + "#" * 80)
        print(f"CONDITION: {condition_name}")
        print("#" * 80)

        for sid in SUBJECT_IDS:
            print("\n" + "-" * 80)
            print(f"Subject {sid:02d} | Condition {condition_name}")
            print("-" * 80)

            X_full, y = data_cache[sid]

            selected_channels = get_channels(sid)
            selected_indices = [FULL_CHANNELS.index(ch) for ch in selected_channels]
            X = X_full[:, selected_indices, :]

            print(f"Selected channels ({len(selected_channels)}): {selected_channels}")
            print(f"Reduced X shape: {X.shape}")

            folds = make_blockwise_folds(len(y), n_blocks=N_FOLDS)

            fold_accs = []
            fold_val_accs = []

            for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
                print(
                    f"\nS{sid:02d} | {condition_name} | Fold {fold_idx}/{N_FOLDS} | "
                    f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
                )

                test_acc, val_acc = train_one_fold(
                    X=X,
                    y=y,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    sid=sid,
                    condition_name=condition_name,
                    fold_idx=fold_idx,
                )

                fold_accs.append(test_acc)
                fold_val_accs.append(val_acc)

                print(
                    f"S{sid:02d} | {condition_name} | Fold {fold_idx} "
                    f"final test accuracy: {test_acc * 100:.2f}%"
                )

            subject_mean = float(np.mean(fold_accs))
            subject_std = float(np.std(fold_accs))

            all_results[condition_name].append(subject_mean)

            detailed_results[condition_name].append(
                {
                    "subject": sid,
                    "channels": selected_channels,
                    "fold_test_accs": fold_accs,
                    "fold_val_accs": fold_val_accs,
                    "mean_acc": subject_mean,
                    "std_acc": subject_std,
                }
            )

            print(
                f"\nS{sid:02d} | {condition_name} mean accuracy: "
                f"{subject_mean * 100:.2f}% ± {subject_std * 100:.2f}%"
            )

    print("\n" + "=" * 80)
    print("FINAL RELIEFF + EEGNETV4 REPORT")
    print("=" * 80)

    for condition_name in conditions.keys():
        subject_means = all_results[condition_name]
        grand_mean = float(np.mean(subject_means))
        grand_std = float(np.std(subject_means))

        print("\n" + "-" * 80)
        print(f"Condition: {condition_name}")
        print("-" * 80)

        for res in detailed_results[condition_name]:
            print(
                f"S{res['subject']:02d}: "
                f"{res['mean_acc'] * 100:.2f}% ± {res['std_acc'] * 100:.2f}%"
            )

        print(
            f"Grand mean for {condition_name}: "
            f"{grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%"
        )

    report_path = os.path.join(
        OUTPUT_DIR,
        "eegnetv4_relieff_channel_reduction_report.txt",
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EEGNetv4 + ReliefF Channel Reduction Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Dataset: Schirrmeister2017 / HGD\n")
        f.write(f"Subjects: {SUBJECT_IDS}\n")
        f.write(f"Full channels: {FULL_CHANNELS}\n\n")

        f.write("Preprocessing:\n")
        f.write(f"  Resample: {TARGET_SFREQ} Hz\n")
        f.write(f"  Bandpass: {LOW_CUT_HZ}-{HIGH_CUT_HZ} Hz\n")
        f.write(f"  Trial window: {TRIAL_START_OFFSET_S}-{TRIAL_STOP_OFFSET_S} s\n\n")

        f.write("EEGNetv4 training configuration:\n")
        f.write(f"  Max epochs: {MAX_EPOCHS}\n")
        f.write(f"  Patience: {PATIENCE}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  LR: {LR}\n")
        f.write(f"  Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"  Dropout: {DROPOUT}\n")
        f.write(f"  Label smoothing: {LABEL_SMOOTHING}\n")
        f.write(f"  Gradient clipping norm: {GRAD_CLIP_NORM}\n")
        f.write(f"  ReliefF neighbors: {RELIEFF_NEIGHBORS}\n\n")

        f.write("Global ReliefF ranking:\n")
        for ch, score in top_k_channels(global_importance, FULL_CHANNELS, len(FULL_CHANNELS)):
            f.write(f"  {ch}: {score:.6f}\n")

        f.write("\nGlobal top-12:\n")
        f.write(", ".join(global_12) + "\n")

        f.write("\nGlobal top-10:\n")
        f.write(", ".join(global_10) + "\n")

        f.write("\nLocal selections:\n")
        for sid in SUBJECT_IDS:
            f.write(f"S{sid:02d} local top-12: {', '.join(local_12[sid])}\n")
            f.write(f"S{sid:02d} local top-10: {', '.join(local_10[sid])}\n")

        f.write("\n\nResults:\n")
        f.write("=" * 80 + "\n")

        for condition_name in conditions.keys():
            subject_means = all_results[condition_name]
            grand_mean = float(np.mean(subject_means))
            grand_std = float(np.std(subject_means))

            f.write(f"\nCondition: {condition_name}\n")
            f.write("-" * 80 + "\n")

            for res in detailed_results[condition_name]:
                f.write(f"\nSubject {res['subject']:02d}\n")
                f.write(f"Channels: {', '.join(res['channels'])}\n")

                for i, acc in enumerate(res["fold_test_accs"], start=1):
                    f.write(f"  Fold {i}: {acc * 100:.2f}%\n")

                f.write(
                    f"  Mean: {res['mean_acc'] * 100:.2f}% "
                    f"± {res['std_acc'] * 100:.2f}%\n"
                )

            f.write(
                f"\nGrand mean for {condition_name}: "
                f"{grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%\n"
            )

    print(f"\nReport saved to: {report_path}")
