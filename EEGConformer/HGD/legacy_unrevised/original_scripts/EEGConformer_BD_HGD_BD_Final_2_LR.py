# EEGConformer_HGD_paper_ready_full_vs_reduced_blockwise_cv.py
# Paper-ready version — consistent with EEGNetv4 / Deep4Net / EEG-ITNet
# - 4-fold block-wise CV per subject
# - Full 22ch baseline + reduced-channel re-training
# - Detailed report with fold accuracies, 3 correlation variants, top channels
# - Outputs in EEGConformer_HGD_Output/

import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mne
from sklearn.metrics import accuracy_score
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGConformer
from scipy.stats import pearsonr
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
# Config — aligned with other models
# ────────────────────────────────────────────────
DATASET_NAME = "Schirrmeister2017"
SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14]

SFREQ = 500
BAND = (4.0, 125.0)
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0 - 1e-3
N_SAMPLES = int(SFREQ * 4.0)  # 2000

FULL_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Pz'
]

LR = 1e-3
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 100                           # longer training — can be increased further
NS_SEG = 8

IG_STEPS = 20
SMOOTH_SAMPLES = 12
SMOOTH_NOISE_REL = 0.10

ATTR_METHODS = ["grad", "grad_x_input", "ig", "smoothgrad"]

# Reduced experiment
REDUCED_N_CHANNELS = 12
REDUCED_SELECTION = "best_methods"     # "all_methods" or "best_methods"
BEST_METHODS_COUNT = 3                 # top N by avg -ERD_combined correlation

OUTPUT_BASE = "EEGConformer_HGD_Output"

# ────────────────────────────────────────────────
# MNE topomap (dynamic)
# ────────────────────────────────────────────────
montage = mne.channels.make_standard_montage("standard_1005")
CHANNELS = None
TOPO_INFO = None

# ────────────────────────────────────────────────
# Utility functions (including the missing top_k_channels)
# ────────────────────────────────────────────────
def top_k_channels(scores, ch_names, k):
    """
    Returns list of (channel_name, score) tuples for the top-k highest scores.
    """
    scores = np.asarray(scores)
    idx = np.argsort(scores)[::-1][:k]  # descending order
    return [(ch_names[i], float(scores[i])) for i in idx]


def normalize_importance(vec):
    vec = np.asarray(vec, dtype=np.float64)
    m = np.max(np.abs(vec))
    return vec / m if m > 1e-12 else vec.astype(np.float32)


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
        print("Warning: few motor channels → using all EEG")
        raw.pick_types(eeg=True)
    else:
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
    event_id = {"left":1, "right":2, "feet":3, "rest":4}
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
            X = np.pad(X, ((0,0),(0,0),(0,pad)), mode='edge')
    return X, y, epochs.info


def build_full_subject_data(subject_id):
    ds = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
    X_list, y_list = [], []
    info = None
    for bd in ds.datasets:
        raw = prepare_raw(bd.raw)
        X, y, block_info = extract_epochs(raw)
        X_list.append(X)
        y_list.append(y)
        if info is None:
            info = block_info
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    mask = np.isin(y, [0, 1, 2, 3])
    X = X[mask]
    y = y[mask]
    print(f"Subject {subject_id} — Trials (4 classes): {X.shape[0]}")
    return X, y, info


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
    if seg_len * ns != T:
        raise ValueError(f"Time {T} not divisible by ns={ns}")
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
        parts = [X[src, :, s*seg_len:(s+1)*seg_len] for s, src in enumerate(srcs)]
        X_aug[i] = np.concatenate(parts, axis=-1)
    return torch.from_numpy(X_aug).float().to(xb.device)


# ────────────────────────────────────────────────
# ERD/ERS (full subject data)
# ────────────────────────────────────────────────
def compute_erd_ers(X, y, info):
    n_trials = len(y)
    event_samples = np.arange(n_trials) * N_SAMPLES
    event_ids = (np.array(y) + 1).astype(np.int32)
    events = np.column_stack([event_samples.astype(np.int32), np.zeros(n_trials, dtype=np.int32), event_ids])
    epochs = mne.EpochsArray(
        X, info, events=events,
        event_id={"left":1, "right":2, "feet":3, "rest":4},
        tmin=EPOCH_TMIN, baseline=None, verbose=False
    )
    psd_base = epochs.compute_psd(tmin=0.0, tmax=0.4, fmin=8, fmax=30, verbose=False).get_data().mean(axis=-1)
    psd_act  = epochs.compute_psd(tmin=0.5, tmax=3.0, fmin=8, fmax=30, verbose=False).get_data().mean(axis=-1)
    erd_ers_db = 10 * np.log10((psd_act + 1e-10) / (psd_base + 1e-10))
    left_idx = epochs.events[:, 2] == 1
    right_idx = epochs.events[:, 2] == 2
    erd_left = erd_ers_db[left_idx].mean(axis=0) if left_idx.sum() > 0 else np.zeros(len(CHANNELS))
    erd_right = erd_ers_db[right_idx].mean(axis=0) if right_idx.sum() > 0 else np.zeros(len(CHANNELS))
    erd_contrast = erd_left - erd_right
    erd_combined = (erd_left + erd_right) / 2
    erd_magnitude = np.abs(erd_combined)
    return {
        "ERD_L": erd_left,
        "ERD_R": erd_right,
        "ERD_contrast": erd_contrast,
        "ERD_combined": erd_combined,
        "ERD_magnitude": erd_magnitude
    }


def plot_erd_four_panel(erd_dict, info, out_png):
    vals = [erd_dict["ERD_L"], erd_dict["ERD_R"], erd_dict["ERD_contrast"], erd_dict["ERD_combined"]]
    limit = max([np.abs(v).max() for v in vals if v is not None] + [1e-6])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Left Hand ERD", "Right Hand ERD", "Contrast (L-R)", "Combined"]
    for ax, data, title in zip(axes, vals, titles):
        mne.viz.plot_topomap(data, info, axes=ax, cmap='RdBu_r', vlim=(-limit, limit), show=False)
        ax.set_title(title)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────
# Saliency
# ────────────────────────────────────────────────
@torch.enable_grad()
def compute_saliency(method, model, test_loader):
    was_training = model.training
    model.eval()  # important for stable saliency, disables dropout

    accum = []

    for xb, yb in test_loader:
        xb = xb.to(DEVICE).float()
        yb = yb.to(DEVICE).long()

        if method == "grad":
            x = xb.detach().clone().requires_grad_(True)

            outputs = model(x)
            score = outputs.gather(1, yb.view(-1, 1)).sum()

            model.zero_grad(set_to_none=True)
            score.backward()

            imp = x.grad.abs().mean(dim=2).detach().cpu().numpy()

        elif method == "grad_x_input":
            x = xb.detach().clone().requires_grad_(True)

            outputs = model(x)
            score = outputs.gather(1, yb.view(-1, 1)).sum()

            model.zero_grad(set_to_none=True)
            score.backward()

            imp = (x.grad * x).abs().mean(dim=2).detach().cpu().numpy()

        elif method == "ig":
            baseline = torch.zeros_like(xb)
            total_grad = torch.zeros_like(xb)

            for alpha in np.linspace(0, 1, IG_STEPS + 1)[1:]:
                xi = baseline + alpha * (xb - baseline)

                # Important: make xi a leaf tensor
                xi = xi.detach().clone().requires_grad_(True)

                outputs = model(xi)
                score = outputs.gather(1, yb.view(-1, 1)).sum()

                model.zero_grad(set_to_none=True)
                grad = torch.autograd.grad(score, xi, retain_graph=False, create_graph=False)[0]

                total_grad += grad.detach()

            avg_grad = total_grad / IG_STEPS
            ig_attr = (xb - baseline) * avg_grad

            imp = ig_attr.abs().mean(dim=2).detach().cpu().numpy()

        elif method == "smoothgrad":
            sigma = SMOOTH_NOISE_REL * xb.std()
            grads_sum = torch.zeros_like(xb)

            for _ in range(SMOOTH_SAMPLES):
                xn = xb + torch.randn_like(xb) * sigma

                # Important: make xn a leaf tensor
                xn = xn.detach().clone().requires_grad_(True)

                outputs = model(xn)
                score = outputs.gather(1, yb.view(-1, 1)).sum()

                model.zero_grad(set_to_none=True)
                grad = torch.autograd.grad(score, xn, retain_graph=False, create_graph=False)[0]

                grads_sum += grad.abs().detach()

            imp = (grads_sum / SMOOTH_SAMPLES).mean(dim=2).detach().cpu().numpy()

        else:
            raise ValueError(f"Unknown saliency method: {method}")

        accum.append(imp)

    if was_training:
        model.train()

    if accum:
        imp = np.concatenate(accum, axis=0).mean(axis=0)
        return normalize_importance(imp)

    return np.zeros(len(CHANNELS))



def plot_saliency(imp, info, title, fname):
    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(imp, info, axes=ax, cmap='viridis', show=False)
    ax.set_title(title)
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────
# One-fold training — same style as others
# ────────────────────────────────────────────────
def train_one_fold(X_tr, y_tr, X_te, y_te):
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds  = torch.utils.data.TensorDataset(torch.from_numpy(X_te),  torch.from_numpy(y_te))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    model = EEGConformer(
        n_outputs=4,  # left vs right only
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
        mode='max',  # because we monitor accuracy (higher is better)
        factor=0.5,  # halve LR when plateau
        patience=10,  # wait 10 epochs without improvement
        verbose=True
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

    return final_test_acc, best_state, model, train_loader, test_loader


# ────────────────────────────────────────────────
# Experiment Runner
# ────────────────────────────────────────────────
def run_experiment(channels_list, subdir_name):
    global CHANNELS, TOPO_INFO
    CHANNELS = channels_list[:]
    TOPO_INFO = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ, ch_types="eeg")
    TOPO_INFO.set_montage(montage)

    output_subdir = os.path.join(OUTPUT_BASE, subdir_name)
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(f"{output_subdir}/saliency", exist_ok=True)
    os.makedirs(f"{output_subdir}/erd_ers", exist_ok=True)

    all_subject_means = []
    subject_summaries = []

    for sid in SUBJECT_IDS:
        print("\n" + "═"*80)
        print(f"SUBJECT {sid} — EEGConformer — 4-fold block-wise CV")
        print("═"*80)

        X, y, mne_info = build_full_subject_data(sid)

        if len(X) < 20:
            print("Too few trials — skipping")
            continue

        folds = make_blockwise_folds(len(y), n_blocks=4)

        fold_test_accs = []
        subject_importance_accum = {m: np.zeros(len(CHANNELS), dtype=np.float64) for m in ATTR_METHODS}

        for fold_i, (tr_idx, val_idx, te_idx) in enumerate(folds, 1):
            print(f"\nFold {fold_i}/4 | train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            mu = X_tr.mean(axis=(0, 2), keepdims=True)
            std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
            X_tr = (X_tr - mu) / std
            X_te = (X_te - mu) / std

            test_acc, best_state, model, train_loader, test_loader = train_one_fold(X_tr, y_tr, X_te, y_te)
            fold_test_accs.append(test_acc)

            # Accumulate saliency across folds
            for method in ATTR_METHODS:
                imp = compute_saliency(method, model, test_loader)
                subject_importance_accum[method] += imp

            del model
            torch.cuda.empty_cache()
            gc.collect()

        subj_mean = np.mean(fold_test_accs)
        subj_std = np.std(fold_test_accs)
        all_subject_means.append(subj_mean)

        print(f"\nSubject {sid} RESULTS")
        for i, acc in enumerate(fold_test_accs, 1):
            print(f"Fold {i}: {acc*100:.2f}%")
        print(f"Mean: {subj_mean*100:.2f}% ± {subj_std*100:.2f}%")

        # Average importance
        subject_importances = {
            m: normalize_importance(subject_importance_accum[m] / len(folds))
            for m in ATTR_METHODS
        }

        # ERD/ERS on full data
        erd_dict = compute_erd_ers(X, y, mne_info)
        plot_erd_four_panel(erd_dict, mne_info, f"{output_subdir}/erd_ers/erd_four_panel_S{sid:02d}.png")

        # Correlations
        subject_corrs = {}
        for m, imp in subject_importances.items():
            if np.all(imp == 0):
                subject_corrs[m] = {k: np.nan for k in ["-erd_combined", "contrast", "magnitude"]}
                continue
            r_neg_comb = pearsonr(imp, -erd_dict["ERD_combined"])[0] if np.std(erd_dict["ERD_combined"]) > 1e-8 else np.nan
            r_contrast = pearsonr(imp, erd_dict["ERD_contrast"])[0]   if np.std(erd_dict["ERD_contrast"])   > 1e-8 else np.nan
            r_magn     = pearsonr(imp, erd_dict["ERD_magnitude"])[0]  if np.std(erd_dict["ERD_magnitude"])  > 1e-8 else np.nan
            subject_corrs[m] = {
                "-erd_combined": r_neg_comb,
                "contrast": r_contrast,
                "magnitude": r_magn
            }

        subject_summaries.append({
            "subject": sid,
            "fold_test_accs": fold_test_accs,
            "subject_mean_acc": subj_mean,
            "subject_std_acc": subj_std,
            "importance": subject_importances,
            "erd": erd_dict,
            "correlations": subject_corrs
        })

        # Save per-subject saliency
        for m, imp in subject_importances.items():
            plot_saliency(imp, mne_info, f"{m.upper()} S{sid:02d}", f"{output_subdir}/saliency/{m}_S{sid:02d}.png")

        del X, y
        gc.collect()

    grand_mean = np.mean(all_subject_means)
    grand_std = np.std(all_subject_means)

    print("\n" + "#" * 80)
    print("FINAL RESULTS — 4-fold block-wise CV")
    for s, acc in zip(SUBJECT_IDS, all_subject_means):
        print(f"S{s}: {acc*100:.2f}%")
    print(f"Grand mean: {grand_mean*100:.2f}% ± {grand_std*100:.2f}%")
    print("#" * 80)

    return subject_summaries, all_subject_means, grand_mean


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=== FULL 22-CHANNEL BASELINE — 4-fold block-wise CV ===")
    full_summaries, full_means, grand_mean = run_experiment(FULL_CHANNELS, "full_22ch")

    # ───── Best methods by average -ERD_combined correlation ─────
    method_avg_corrs = {}
    for m in ATTR_METHODS:
        vals = [s["correlations"][m]["-erd_combined"] for s in full_summaries]
        method_avg_corrs[m] = np.nanmean(vals)

    ranked = sorted(method_avg_corrs, key=method_avg_corrs.get, reverse=True)
    best_methods = ranked[:BEST_METHODS_COUNT]
    print(f"\nBest methods (top {BEST_METHODS_COUNT} by avg -ERD_combined corr): {best_methods}")

    # ───── Paper report ─────
    report_path = os.path.join(OUTPUT_BASE, "paper_summary_report_eegconformer.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EEGConformer — HGD — PAPER SUMMARY REPORT\n")
        f.write("4-fold block-wise CV — full 22ch + reduced channels\n\n")
        f.write("="*80 + "\n\n")

        for subj in full_summaries:
            sid = subj["subject"]
            f.write(f"SUBJECT {sid:02d}\n")
            f.write("-"*60 + "\n")
            for i, acc in enumerate(subj["fold_test_accs"], 1):
                f.write(f"  Fold {i}: {acc*100:.2f}%\n")
            std = subj["subject_std_acc"] * 100
            f.write(f"  Mean: {subj['subject_mean_acc']*100:.2f}% ± {std:.2f}%\n\n")

            f.write("CORRELATIONS:\n")
            for m in ATTR_METHODS:
                c = subj["correlations"][m]
                f.write(f"  {m:>14} | -ERD_comb {c['-erd_combined']:+.3f} | contrast {c['contrast']:+.3f} | magnitude {c['magnitude']:+.3f}\n")
            f.write("\n")

            avg_all = np.mean([subj["importance"][m] for m in ATTR_METHODS], axis=0)
            avg_all = normalize_importance(avg_all)
            f.write("Top channels — avg ALL methods:\n")
            f.write("  Top-8 : " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_all, FULL_CHANNELS, 8)) + "\n")
            f.write("  Top-12: " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_all, FULL_CHANNELS, 12)) + "\n\n")

            avg_best = np.mean([subj["importance"][m] for m in best_methods], axis=0)
            avg_best = normalize_importance(avg_best)
            f.write("Top channels — avg BEST methods:\n")
            f.write("  Top-8 : " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_best, FULL_CHANNELS, 8)) + "\n")
            f.write("  Top-12: " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_best, FULL_CHANNELS, 12)) + "\n")
            f.write("\n" + "="*80 + "\n\n")

        f.write(f"GRAND MEAN ACCURACY (22ch): {grand_mean*100:.2f}%\n\n")

        f.write("AVERAGED CORRELATIONS:\n")
        for m in ATTR_METHODS:
            c_neg = np.nanmean([s["correlations"][m]["-erd_combined"] for s in full_summaries])
            f.write(f"  {m:>14} → -ERD_combined {c_neg:+.3f}\n")
        f.write("\n")

        grand_avg_all = np.mean([np.mean([s["importance"][m] for m in ATTR_METHODS], axis=0) for s in full_summaries], axis=0)
        grand_avg_all = normalize_importance(grand_avg_all)
        f.write("GRAND TOP CHANNELS — ALL methods (Top-12):\n")
        f.write(", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(grand_avg_all, FULL_CHANNELS, 12)) + "\n\n")

        grand_avg_best = np.mean([np.mean([s["importance"][m] for m in best_methods], axis=0) for s in full_summaries], axis=0)
        grand_avg_best = normalize_importance(grand_avg_best)
        f.write("GRAND TOP CHANNELS — BEST methods (Top-12):\n")
        f.write(", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(grand_avg_best, FULL_CHANNELS, 12)) + "\n\n")

    print(f"Report saved: {report_path}")

    # ───── Reduced-channel round ─────
    if REDUCED_SELECTION == "all_methods":
        imp_for_selection = grand_avg_all
    else:
        imp_for_selection = grand_avg_best

    idx = np.argsort(-imp_for_selection)
    REDUCED_CHANNELS = [FULL_CHANNELS[i] for i in idx[:REDUCED_N_CHANNELS]]

    print(f"\n=== REDUCED TRAINING ({REDUCED_N_CHANNELS}ch — {REDUCED_SELECTION}) ===")
    print("Selected:", ", ".join(REDUCED_CHANNELS))

    reduced_summaries, reduced_means, red_grand_mean = run_experiment(
        REDUCED_CHANNELS, f"reduced_{REDUCED_N_CHANNELS}ch_{REDUCED_SELECTION}"
    )

    # Append comparison
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + "#" * 80 + "\n")
        f.write(f"REDUCED RESULTS ({REDUCED_N_CHANNELS} channels — {REDUCED_SELECTION})\n")
        f.write("#" * 80 + "\n\n")
        for sid, f_acc, r_acc in zip(SUBJECT_IDS, full_means, reduced_means):
            f.write(f"S{sid:02d}: 22ch {f_acc*100:.2f}%  →  reduced {r_acc*100:.2f}%\n")
        f.write(f"\nGRAND: 22ch {grand_mean*100:.2f}%  →  reduced {red_grand_mean*100:.2f}%\n")

    print("\n" + "="*80)
    print("FINAL COMPARISON — 22ch vs Reduced")
    print("="*80)
    for sid, f, r in zip(SUBJECT_IDS, full_means, reduced_means):
        print(f"S{sid:02d}: {f*100:6.2f}% → {r*100:6.2f}%")
    print(f"Grand: {grand_mean*100:6.2f}% → {red_grand_mean*100:6.2f}%")
    print("="*80)

    print(f"\nAll outputs in: {OUTPUT_BASE}/")
    print("Report:", report_path)
    print("\nDone.")