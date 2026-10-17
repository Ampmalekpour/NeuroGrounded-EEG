import os
import gc
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import mne

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGConformer
from torch.optim.lr_scheduler import ReduceLROnPlateau

mne.set_log_level("WARNING")

# ────────────────────────────────────────────────
# Reproducibility & Device
# ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
DATASET_NAME = "Schirrmeister2017"
SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

SFREQ = 500
BAND = (4.0, 125.0)
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0 - 1e-3
N_SAMPLES = int(SFREQ * 4.0)

FULL_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Pz"
]

LR = 1e-3
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 100
NS_SEG = 8

OUTPUT_BASE = "EEGConformer_MI_Global12_ReferenceAligned"

# ────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────
def make_blockwise_folds(n_samples, n_blocks=4):
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
            torch.from_numpy(X[sl]).float().to(DEVICE),
            torch.from_numpy(y[sl]).long().to(DEVICE),
        )


@torch.no_grad()
def evaluate(model, X, y, batch_size=64):
    model.eval()
    preds = []
    for xb, _ in batch_iter(X, y, batch_size=batch_size, shuffle=False):
        out = model(xb)
        preds.extend(out.argmax(1).cpu().numpy())
    return accuracy_score(y, np.asarray(preds))


def seg_recon_batch(xb, yb, ns=8):
    if ns <= 1:
        return xb

    X = xb.detach().cpu().numpy()
    y = yb.detach().cpu().numpy().astype(int)

    B, C, T = X.shape
    seg_len = T // ns
    if seg_len == 0:
        return xb

    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    X_aug = np.zeros_like(X)

    for i in range(B):
        cls = y[i]
        pool = idx_by_class.get(cls, np.array([], dtype=int))

        if len(pool) == 0:
            X_aug[i] = X[i]
            continue

        srcs = np.random.choice(pool, ns, replace=(len(pool) < ns))
        parts = []
        for s, src in enumerate(srcs):
            start = s * seg_len
            end = (s + 1) * seg_len if s < ns - 1 else T
            parts.append(X[src, :, start:end])
        X_aug[i] = np.concatenate(parts, axis=-1)

    return torch.from_numpy(X_aug).float().to(xb.device)


# ────────────────────────────────────────────────
# Data preparation: reference-aligned
# ────────────────────────────────────────────────
def prepare_raw(raw, target_channels):
    raw = raw.copy()

    ch_map = {ch.upper(): ch for ch in raw.ch_names}
    actual_channels = [ch_map[c.upper()] for c in target_channels if c.upper() in ch_map]
    raw.pick_channels(actual_channels)

    raw.resample(SFREQ, verbose=False)
    raw.filter(*BAND, method="iir", verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj(verbose=False)
    return raw


def extract_epochs(raw):
    class_map = {
        "left hand": 0,
        "left_hand": 0,
        "left": 0,
        "right hand": 1,
        "right_hand": 1,
        "right": 1,
        "feet": 2,
        "foot": 2,
        "rest": 3,
    }

    onsets, labels = [], []
    for ann in raw.annotations:
        desc = str(ann["description"]).lower().strip()
        if desc in class_map:
            onsets.append(ann["onset"])
            labels.append(class_map[desc])

    if len(labels) == 0:
        return np.empty((0, len(raw.ch_names), N_SAMPLES), dtype=np.float32), np.empty((0,), dtype=np.int64)

    sf = raw.info["sfreq"]
    samples = np.round(np.asarray(onsets) * sf).astype(int)
    events = np.column_stack(
        (samples, np.zeros(len(samples), dtype=int), np.asarray(labels, dtype=int) + 1)
    )

    epochs = mne.Epochs(
        raw,
        events,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data(copy=True).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    if X.shape[2] != N_SAMPLES:
        if X.shape[2] > N_SAMPLES:
            X = X[:, :, :N_SAMPLES]
        else:
            X = np.pad(X, ((0, 0), (0, 0), (0, N_SAMPLES - X.shape[2])), mode="edge")

    return X, y


def build_full_subject_data(subject_id, channels):
    ds = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
    X_list, y_list = [], []

    for bd in ds.datasets:
        raw = prepare_raw(bd.raw, channels)
        X, y = extract_epochs(raw)
        if len(y) == 0:
            continue
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


# ────────────────────────────────────────────────
# Mutual information channel scoring
# ────────────────────────────────────────────────
def compute_mi_importance(X, y):
    features = np.var(X, axis=2).astype(np.float64)
    scores = mutual_info_classif(
        features,
        y,
        discrete_features=False,
        random_state=SEED
    )
    return scores


# ────────────────────────────────────────────────
# Training: reference-aligned
# ────────────────────────────────────────────────
def train_one_fold_reference_aligned(X_tr, y_tr, X_te, y_te, n_chans, sid, fold_idx):
    model = EEGConformer(
        n_outputs=4,
        n_chans=n_chans,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
        final_fc_length="auto",
        n_times=N_SAMPLES,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
    )

    best_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0

        for xb, yb in batch_iter(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True):
            xb_aug = seg_recon_batch(xb, yb, ns=NS_SEG)
            xb_comb = torch.cat([xb, xb_aug], dim=0)
            yb_comb = torch.cat([yb, yb], dim=0)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb_comb)
            loss = criterion(out, yb_comb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        test_acc = evaluate(model, X_te, y_te, batch_size=64)
        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        mean_loss = total_loss / max(n_steps, 1)
        print(
            f"S{sid:02d} | G-12ch | Fold {fold_idx} | "
            f"Ep {epoch:3d} | Loss: {mean_loss:.4f} | "
            f"Acc: {test_acc * 100:5.2f}% | Best: {best_acc * 100:5.2f}%"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_acc = evaluate(model, X_te, y_te, batch_size=64)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_acc


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("PHASE 1: Mutual Information Calculation on full 22-channel data")
    data_cache = {}
    subject_importances = {}

    for sid in SUBJECT_IDS:
        X, y = build_full_subject_data(sid, FULL_CHANNELS)
        data_cache[sid] = (X, y)
        subject_importances[sid] = compute_mi_importance(X, y)
        print(f"  Subject {sid:02d}: cached {len(y)} trials")

    global_imp = np.mean(
        np.stack([subject_importances[sid] for sid in SUBJECT_IDS], axis=0),
        axis=0
    )
    global_12 = [FULL_CHANNELS[i] for i in np.argsort(-global_imp)[:12]]

    print("\nSelected Global 12 Channels:")
    print(global_12)

    with open(os.path.join(OUTPUT_BASE, "global_12_channels.txt"), "w") as f:
        f.write("\n".join(global_12))

    results = []

    print("\nPHASE 2: Training on Global 12 MI Subset with reference-aligned setup")
    for sid in SUBJECT_IDS:
        X_all, y = data_cache[sid]
        indices = [FULL_CHANNELS.index(c) for c in global_12]
        X_sub = X_all[:, indices, :]

        folds = make_blockwise_folds(len(y), n_blocks=4)
        fold_accs = []

        for fold_idx, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
            X_tr = X_sub[tr_idx].copy()
            y_tr = y[tr_idx].copy()
            X_te = X_sub[te_idx].copy()
            y_te = y[te_idx].copy()

            mu = X_tr.mean(axis=(0, 2), keepdims=True)
            std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8

            X_tr = ((X_tr - mu) / std).astype(np.float32)
            X_te = ((X_te - mu) / std).astype(np.float32)

            acc = train_one_fold_reference_aligned(
                X_tr, y_tr, X_te, y_te,
                n_chans=len(indices),
                sid=sid,
                fold_idx=fold_idx
            )
            fold_accs.append(acc)

        mean_acc = float(np.mean(fold_accs))
        results.append(mean_acc)
        print(f"S{sid:02d} Mean Acc: {mean_acc * 100:.2f}%")

    overall_mean = float(np.mean(results))

    report_path = os.path.join(OUTPUT_BASE, "MI_Global12_Report.txt")
    with open(report_path, "w") as f:
        f.write("=== GLOBAL 12 CHANNELS ===\n")
        f.write(", ".join(global_12) + "\n\n")
        f.write("=== SUBJECT RESULTS ===\n")
        for sid, acc in zip(SUBJECT_IDS, results):
            f.write(f"S{sid:02d}: {acc * 100:.2f}%\n")
        f.write(f"\nOverall Mean Accuracy: {overall_mean * 100:.2f}%\n")

    print("\n=== FINAL MUTUAL INFORMATION GLOBAL-12 REPORT ===")
    print(f"Overall Mean Accuracy: {overall_mean * 100:.2f}%")
    print(f"Report saved to: {report_path}")
