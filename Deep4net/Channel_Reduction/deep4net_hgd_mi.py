import os
import gc
import copy
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from sklearn.feature_selection import mutual_info_classif
from torch.optim.lr_scheduler import ReduceLROnPlateau

from braindecode.datasets import MOABBDataset
from braindecode.models import Deep4Net
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
)

# =========================================================
# Reproducibility
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
N_BEST_CHANNELS = 10

CANDIDATE_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Pz'
]

OUTPUT_DIR = "Deep4Net_HGD_MI_OnlyReduced"

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
# Data
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
# Mutual Information channel ranking
# =========================================================
def compute_mi_scores(X, y):
    """
    X shape: [n_trials, n_channels, n_times]
    Channel feature per trial = temporal variance.
    MI is computed between per-channel variance and class label.
    """
    features = np.var(X, axis=2)  # [n_trials, n_channels]

    scores = mutual_info_classif(
        features,
        y,
        discrete_features=False,
        random_state=SEED,
    )
    return normalize_scores(scores)


def select_channels_via_mi(subject_id, n_best):
    X_rank, y_rank = load_subject_data(subject_id, CANDIDATE_CHANNELS)
    scores = compute_mi_scores(X_rank, y_rank)
    ranked = get_top_k(scores, CANDIDATE_CHANNELS, len(CANDIDATE_CHANNELS))
    selected = [name for name, _ in ranked[:n_best]]
    return selected, scores, ranked

# =========================================================
# Model / training
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


def train_one_fold(X, y, train_idx, val_idx, test_idx, subject_id, fold_id):
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

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"S{subject_id:02d} | Fold {fold_id} | Ep {epoch:03d} | "
            f"tr_loss={np.mean(train_losses):.4f} | "
            f"val_acc={val_acc:.2%} | te_acc={test_acc:.2%} | "
            f"best_val_loss={best_val_loss:.4f} | lr={lr_now:.2e}"
        )

        if no_improve >= PATIENCE:
            print(f"S{subject_id:02d} | Fold {fold_id} | Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    _, final_test_acc = evaluate(model, X_te, y_te, criterion)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_test_acc, best_val_acc


def run_subject(subject_id):
    print("\n" + "=" * 80)
    print(f"SUBJECT {subject_id:02d}")
    print("=" * 80)

    selected_channels, scores, ranked = select_channels_via_mi(
        subject_id,
        N_BEST_CHANNELS
    )

    print("Mutual Information ranking:")
    for ch, sc in ranked:
        print(f"  {ch:>4s}: {sc:.6f}")

    print(f"\nSelected top-{N_BEST_CHANNELS} channels:")
    print(selected_channels)

    # Train only on reduced channel set
    X, y = load_subject_data(subject_id, selected_channels)
    print(f"Reduced training data shape: {X.shape}")

    folds = make_blockwise_folds(len(y), N_FOLDS)

    fold_test_accs = []
    fold_val_accs = []

    for fold_id, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
        print("\n" + "-" * 80)
        print(
            f"Subject {subject_id:02d} | Fold {fold_id}/{N_FOLDS} | "
            f"train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}"
        )
        print("-" * 80)

        test_acc, val_acc = train_one_fold(
            X, y, tr_idx, val_idx, te_idx, subject_id, fold_id
        )
        fold_test_accs.append(test_acc)
        fold_val_accs.append(val_acc)

    return {
        "subject": subject_id,
        "selected_channels": selected_channels,
        "scores": scores,
        "ranking": ranked,
        "fold_test_accs": fold_test_accs,
        "fold_val_accs": fold_val_accs,
        "mean_test_acc": float(np.mean(fold_test_accs)),
        "std_test_acc": float(np.std(fold_test_accs)),
    }

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    print("=" * 80)
    print("Deep4Net + Mutual Information channel selection + reduced-channel training only")
    print("=" * 80)

    for subject_id in SUBJECT_IDS:
        result = run_subject(subject_id)
        all_results.append(result)

    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Deep4Net + Mutual Information reduced-channel experiment\n")
        f.write("=" * 80 + "\n\n")

        subject_means = []

        for res in all_results:
            subject_means.append(res["mean_test_acc"])

            f.write(f"Subject {res['subject']:02d}\n")
            f.write("-" * 80 + "\n")
            f.write("Selected channels:\n")
            f.write(", ".join(res["selected_channels"]) + "\n\n")

            f.write("Ranking:\n")
            for ch, sc in res["ranking"]:
                f.write(f"  {ch:>4s}: {sc:.6f}\n")

            f.write("\nFold test accuracies:\n")
            for i, acc in enumerate(res["fold_test_accs"], start=1):
                f.write(f"  Fold {i}: {acc * 100:.2f}%\n")

            f.write(f"\nMean test accuracy: {res['mean_test_acc'] * 100:.2f}%\n")
            f.write(f"Std test accuracy : {res['std_test_acc'] * 100:.2f}%\n")
            f.write("\n" + "=" * 80 + "\n\n")

        f.write("Overall summary\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average across subjects: {np.mean(subject_means) * 100:.2f}%\n")
        f.write(f"Std across subjects    : {np.std(subject_means) * 100:.2f}%\n")

    print("\nSaved report to:", report_path)
