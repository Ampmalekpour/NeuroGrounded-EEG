import os
import gc
import json
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
from braindecode.models import Deep4Net
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
)

# =========================================================
# Reproducibility & Device
# =========================================================
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

# =========================================================
# Config
# =========================================================
DATASET_NAME = "Schirrmeister2017"
SUBJECT_IDS = [1, 2, 3, 4, 5]

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
GRAD_CLIP_NORM = 10.0

N_FOLDS = 4

# CSP-based channel reduction
N_BEST_CHANNELS = 10
CSP_REG = None
CSP_LOG = True
CSP_NORM_TRACE = False

CANDIDATE_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Pz"
]

OUTPUT_DIR = "Deep4Net_HGD_CSP_ReducedOnly"

# =========================================================
# Helpers
# =========================================================
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
    blocks = np.array_split(np.arange(n_samples), n_blocks)
    folds = []
    for test_i in range(n_blocks):
        val_i = (test_i - 1) % n_blocks
        train_blocks = [b for b in range(n_blocks) if b not in (val_i, test_i)]
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
        sl = idx[start:start + batch_size]
        xb = torch.tensor(X[sl], dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(y[sl], dtype=torch.long, device=DEVICE)
        yield xb, yb


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
    acc = float((preds == y).mean())
    return float(np.mean(losses)), acc


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    m = np.max(np.abs(scores))
    return scores if m < 1e-12 else scores / m


def get_top_k(scores, names, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(names[i], float(scores[i])) for i in idx]


# =========================================================
# Data Loading
# =========================================================
def load_subject_data(subject_id, channels):
    dataset = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])

    preprocessors = [
        Preprocessor("pick_channels", ch_names=channels, ordered=True),
        Preprocessor(scale_to_microvolts, apply_on_array=True),
        Preprocessor("resample", sfreq=TARGET_SFREQ),
        Preprocessor("filter", l_freq=LOW_CUT_HZ, h_freq=HIGH_CUT_HZ),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
        ),
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

    mask = np.isin(y, [0, 1, 2, 3])
    X = X[mask]
    y = y[mask]

    del dataset, windows_dataset
    gc.collect()

    return X, y


# =========================================================
# CSP Channel Ranking
# =========================================================
def compute_csp_channel_scores(X, y, n_components=None):
    n_channels = X.shape[1]
    if n_components is None:
        n_components = n_channels

    csp = CSP(
        n_components=n_components,
        reg=CSP_REG,
        log=CSP_LOG,
        norm_trace=CSP_NORM_TRACE,
    )
    csp.fit(X, y)

    patterns = np.asarray(csp.patterns_, dtype=np.float64)
    scores = np.abs(patterns).sum(axis=0)
    return normalize_scores(scores)


def select_channels_via_csp(subject_id, n_best):
    X_rank, y_rank = load_subject_data(subject_id, CANDIDATE_CHANNELS)

    X_rank_csp = X_rank.astype(np.float64, copy=False)
    y_rank_csp = y_rank.astype(np.int64, copy=False)

    scores = compute_csp_channel_scores(
        X_rank_csp,
        y_rank_csp,
        n_components=len(CANDIDATE_CHANNELS),
    )
    ranked = get_top_k(scores, CANDIDATE_CHANNELS, len(CANDIDATE_CHANNELS))
    selected = [name for name, _ in ranked[:n_best]]
    return selected, scores, ranked



# =========================================================
# Model
# =========================================================
def build_model(n_chans, n_times, n_classes):
    model = Deep4Net(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length="auto",
        drop_prob=DROPOUT,
    )
    return model.to(DEVICE)


# =========================================================
# Training
# =========================================================
def train_one_fold(X, y, train_idx, val_idx, test_idx, subject_id, fold_id):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))

    print(f"S{subject_id:02d} | Fold {fold_id} | channels={n_chans} | classes={n_classes}")

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
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in batch_iter(X_tr, y_tr, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_acc = evaluate(model, X_val, y_val, criterion)
        _, test_acc = evaluate(model, X_te, y_te, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"S{subject_id:02d} | Fold {fold_id} | "
                f"Epoch {epoch:03d} | "
                f"Train loss={np.mean(train_losses):.4f} | "
                f"Val loss={val_loss:.4f} | "
                f"Val acc={val_acc:.4f} | "
                f"Test acc={test_acc:.4f}"
            )

        if no_improve >= PATIENCE:
            print(f"S{subject_id:02d} | Fold {fold_id} | Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, X_te, y_te, criterion)

    print(
        f"S{subject_id:02d} | Fold {fold_id} | "
        f"BEST val loss={best_val_loss:.4f} | "
        f"BEST val acc={best_val_acc:.4f} | "
        f"FINAL test acc={test_acc:.4f}"
    )

    fold_result = {
        "subject": int(subject_id),
        "fold": int(fold_id),
        "n_channels": int(n_chans),
        "n_times": int(n_times),
        "n_classes": int(n_classes),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }

    del model, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result


# =========================================================
# Main
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for subject_id in SUBJECT_IDS:
        print("=" * 80)
        print(f"Subject {subject_id:02d}: CSP-based channel reduction")
        print("=" * 80)

        selected_channels, csp_scores, ranked_channels = select_channels_via_csp(
            subject_id=subject_id,
            n_best=N_BEST_CHANNELS,
        )

        print(f"S{subject_id:02d} selected channels:")
        print(selected_channels)

        print(f"S{subject_id:02d} CSP ranking:")
        for i, (ch, score) in enumerate(ranked_channels, start=1):
            print(f"{i:02d}. {ch:>4s} | score={score:.6f}")

        subject_dir = os.path.join(OUTPUT_DIR, f"subject_{subject_id:02d}")
        os.makedirs(subject_dir, exist_ok=True)

        csp_info = {
            "subject": int(subject_id),
            "candidate_channels": CANDIDATE_CHANNELS,
            "selected_channels": selected_channels,
            "ranked_channels": [
                {"channel": ch, "score": float(score)}
                for ch, score in ranked_channels
            ],
            "n_best_channels": int(N_BEST_CHANNELS),
            "csp_reg": CSP_REG,
            "csp_log": bool(CSP_LOG),
            "csp_norm_trace": bool(CSP_NORM_TRACE),
        }

        with open(os.path.join(subject_dir, "csp_channel_selection.json"), "w") as f:
            json.dump(csp_info, f, indent=2)

        print("=" * 80)
        print(f"Subject {subject_id:02d}: loading reduced-channel dataset")
        print("=" * 80)

        X, y = load_subject_data(subject_id, selected_channels)

        print(
            f"S{subject_id:02d} data shape: X={X.shape}, y={y.shape}, "
            f"classes={np.unique(y)}"
        )

        folds = make_blockwise_folds(len(y), n_blocks=N_FOLDS)
        subject_results = []

        for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
            result = train_one_fold(
                X=X,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                subject_id=subject_id,
                fold_id=fold_id,
            )
            result["selected_channels"] = selected_channels
            subject_results.append(result)
            all_results.append(result)

            with open(os.path.join(subject_dir, f"fold_{fold_id:02d}_result.json"), "w") as f:
                json.dump(result, f, indent=2)

        subject_test_accs = [r["test_acc"] for r in subject_results]
        subject_summary = {
            "subject": int(subject_id),
            "selected_channels": selected_channels,
            "mean_test_acc": float(np.mean(subject_test_accs)),
            "std_test_acc": float(np.std(subject_test_accs)),
            "fold_test_accs": [float(a) for a in subject_test_accs],
        }

        print("=" * 80)
        print(f"S{subject_id:02d} SUMMARY")
        print(f"Selected channels: {selected_channels}")
        print(f"Fold test accuracies: {subject_test_accs}")
        print(
            f"Mean test acc={subject_summary['mean_test_acc']:.4f} "
            f"+/- {subject_summary['std_test_acc']:.4f}"
        )
        print("=" * 80)

        with open(os.path.join(subject_dir, "subject_summary.json"), "w") as f:
            json.dump(subject_summary, f, indent=2)

        del X, y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_test_accs = [r["test_acc"] for r in all_results]

    final_summary = {
        "dataset": DATASET_NAME,
        "subjects": SUBJECT_IDS,
        "n_best_channels": int(N_BEST_CHANNELS),
        "candidate_channels": CANDIDATE_CHANNELS,
        "mean_test_acc": float(np.mean(all_test_accs)),
        "std_test_acc": float(np.std(all_test_accs)),
        "all_fold_results": all_results,
        "config": {
            "low_cut_hz": float(LOW_CUT_HZ),
            "high_cut_hz": float(HIGH_CUT_HZ),
            "target_sfreq": int(TARGET_SFREQ),
            "trial_start_offset_s": float(TRIAL_START_OFFSET_S),
            "trial_stop_offset_s": float(TRIAL_STOP_OFFSET_S),
            "max_epochs": int(MAX_EPOCHS),
            "patience": int(PATIENCE),
            "batch_size": int(BATCH_SIZE),
            "lr": float(LR),
            "weight_decay": float(WEIGHT_DECAY),
            "dropout": float(DROPOUT),
            "label_smoothing": float(LABEL_SMOOTHING),
            "grad_clip_norm": float(GRAD_CLIP_NORM),
            "n_folds": int(N_FOLDS),
            "csp_reg": CSP_REG,
            "csp_log": bool(CSP_LOG),
            "csp_norm_trace": bool(CSP_NORM_TRACE),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "final_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    print("#" * 80)
    print("FINAL SUMMARY")
    print("#" * 80)
    print(f"All fold test accuracies: {all_test_accs}")
    print(
        f"Overall mean test acc={final_summary['mean_test_acc']:.4f} "
        f"+/- {final_summary['std_test_acc']:.4f}"
    )
    print("#" * 80)


if __name__ == "__main__":
    main()
