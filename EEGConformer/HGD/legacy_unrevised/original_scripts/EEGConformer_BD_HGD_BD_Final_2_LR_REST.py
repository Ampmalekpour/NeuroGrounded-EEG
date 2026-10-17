import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import mne
from sklearn.metrics import accuracy_score
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGConformer
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
SUBJECT_IDS = [10, 11, 12, 13, 14]

SFREQ = 500
BAND = (4.0, 125.0)
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0 - 1e-3
N_SAMPLES = int(SFREQ * 4.0)  # 2000

CHANNELS = [
    'Cz', 'C3', 'C2', 'C4', 'C1', 'CP2', 'Pz', 'CPz', 'CP6', 'CP3', 'FC1', 'FC4'
]

LR = 1e-3
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 100
NS_SEG = 8


# ────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────
def batch_iter(X, y, batch_size, shuffle=True):
    idx = np.arange(len(y))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sl = idx[start:start + batch_size]
        yield (torch.from_numpy(X[sl]).float().to(DEVICE),
               torch.from_numpy(y[sl]).long().to(DEVICE))


@torch.no_grad()
def evaluate(model, X, y, batch_size=64):
    model.eval()
    preds = []
    for xb, _ in batch_iter(X, y, batch_size, shuffle=False):
        out = model(xb)
        preds.extend(out.argmax(1).cpu().numpy())
    return accuracy_score(y, np.array(preds))


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


# ────────────────────────────────────────────────
# Data Preparation
# ────────────────────────────────────────────────
def prepare_raw(raw):
    raw = raw.copy()
    ch_map = {ch.upper(): ch for ch in raw.ch_names}
    selected = [ch_map.get(c.upper()) for c in CHANNELS if ch_map.get(c.upper())]
    selected = [ch for ch in selected if ch is not None]
    if len(selected) < 10:
        print("Warning: missing expected channels")
    raw.pick_channels(selected)
    raw.resample(SFREQ, verbose=False)
    raw.filter(*BAND, method='iir', verbose=False)
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj(verbose=False)
    return raw


def extract_epochs(raw):
    class_map = {
        "left hand": 0, "left_hand": 0, "left": 0,
        "right hand": 1, "right_hand": 1, "right": 1,
        "feet": 2, "foot": 2, "rest": 3
    }
    onsets, labels = [], []
    for ann in raw.annotations:
        desc = str(ann["description"]).lower().strip()
        if desc in class_map:
            onsets.append(ann["onset"])
            labels.append(class_map[desc])
    if not onsets:
        raise RuntimeError("No recognized class annotations")
    onsets = np.array(onsets)
    labels = np.array(labels)
    sf = raw.info["sfreq"]
    samples = np.round(onsets * sf).astype(int)
    codes = labels + 1
    events = np.column_stack((samples, np.zeros(len(samples), int), codes))
    events = events[np.argsort(events[:, 0])]
    event_id = {"left": 1, "right": 2, "feet": 3, "rest": 4}
    epochs = mne.Epochs(
        raw, events, event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False
    )
    X = epochs.get_data().astype(np.float32)
    y = labels.astype(np.int64)
    if X.shape[2] != N_SAMPLES:
        if X.shape[2] > N_SAMPLES:
            X = X[:, :, :N_SAMPLES]
        else:
            pad = N_SAMPLES - X.shape[2]
            X = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode='edge')
    return X, y


def build_full_subject_data(subject_id):
    ds = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
    X_list, y_list = [], []
    for bd in ds.datasets:
        raw = prepare_raw(bd.raw)
        X, y = extract_epochs(raw)
        X_list.append(X)
        y_list.append(y)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    mask = np.isin(y, [0, 1, 2, 3])
    X = X[mask]
    y = y[mask]
    print(f"Subject {subject_id} — Trials (4 classes): {X.shape[0]}")
    return X, y


# ────────────────────────────────────────────────
# Augmentation
# ────────────────────────────────────────────────
def seg_recon_batch(xb, yb, ns=8):
    if ns <= 1:
        return xb
    X = xb.cpu().numpy()
    y = yb.cpu().numpy().astype(int)
    B, C, T = X.shape
    seg_len = T // ns
    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    X_aug = np.zeros_like(X)
    for i in range(B):
        cls = y[i]
        pool = idx_by_class.get(cls, np.array([]))
        if len(pool) == 0:
            X_aug[i] = X[i]
            continue
        replace = len(pool) < ns
        srcs = np.random.choice(pool, ns, replace=replace)
        parts = [X[src, :, s * seg_len:(s + 1) * seg_len] for s, src in enumerate(srcs)]
        X_aug[i] = np.concatenate(parts, axis=-1)
    return torch.from_numpy(X_aug).float().to(xb.device)


# ────────────────────────────────────────────────
# One-fold training
# ────────────────────────────────────────────────
def train_one_fold(X_tr, y_tr, X_te, y_te):
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = EEGConformer(
        n_outputs=4,
        n_chans=X_tr.shape[1],
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
        final_fc_length='auto',
        n_times=N_SAMPLES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=False
    )

    best_acc = 0.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb_aug = seg_recon_batch(xb, yb, ns=NS_SEG)
            xb_comb = torch.cat([xb, xb_aug])
            yb_comb = torch.cat([yb, yb])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_comb)
            loss = criterion(logits, yb_comb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * yb_comb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb_comb).sum().item()
            total += yb_comb.size(0)

        train_acc = correct / total if total > 0 else 0.0
        test_acc = evaluate(model, X_te, y_te)
        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()

        print(f"Ep {epoch:3d} | tr {train_acc:.3%} | te {test_acc:.3%} | best {best_acc:.3%}")

    if best_state is None:
        raise RuntimeError("No best state found during training")

    model.load_state_dict(best_state)
    final_test_acc = evaluate(model, X_te, y_te)

    return final_test_acc


# ────────────────────────────────────────────────
# Experiment Runner
# ────────────────────────────────────────────────
def run_experiment():
    all_subject_means = []

    for sid in SUBJECT_IDS:
        print("\n" + "═" * 80)
        print(f"SUBJECT {sid} — EEGConformer — 4-fold block-wise CV")
        print("═" * 80)

        X, y = build_full_subject_data(sid)

        if len(X) < 20:
            print("Too few trials — skipping")
            continue

        folds = make_blockwise_folds(len(y), n_blocks=4)
        fold_test_accs = []

        for fold_i, (tr_idx, val_idx, te_idx) in enumerate(folds, 1):
            print(f"\nFold {fold_i}/4 | train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            mu = X_tr.mean(axis=(0, 2), keepdims=True)
            std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
            X_tr = (X_tr - mu) / std
            X_te = (X_te - mu) / std

            test_acc = train_one_fold(X_tr, y_tr, X_te, y_te)
            fold_test_accs.append(test_acc)

            torch.cuda.empty_cache()
            gc.collect()

        subj_mean = np.mean(fold_test_accs)
        subj_std = np.std(fold_test_accs)
        all_subject_means.append(subj_mean)

        print(f"\nSubject {sid} RESULTS")
        for i, acc in enumerate(fold_test_accs, 1):
            print(f"Fold {i}: {acc * 100:.2f}%")
        print(f"Mean: {subj_mean * 100:.2f}% ± {subj_std * 100:.2f}%")

        del X, y
        gc.collect()

    grand_mean = np.mean(all_subject_means)
    grand_std = np.std(all_subject_means)

    print("\n" + "#" * 80)
    print("FINAL RESULTS — 4-fold block-wise CV")
    for s, acc in zip(SUBJECT_IDS, all_subject_means):
        print(f"S{s:02d}: {acc * 100:.2f}%")
    print(f"Grand mean: {grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%")
    print("#" * 80)


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"=== TRAINING ON FIXED CHANNELS ({len(CHANNELS)}ch) ===")
    print("Channels:", ", ".join(CHANNELS))
    print("Subjects:", SUBJECT_IDS)

    run_experiment()
    print("\nDone.")
