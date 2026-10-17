# deep4net_hgd_paper_ready_full_vs_reduced_no_occlusion.py
# Paper-ready version for Deep4Net on HGD (Schirrmeister2017)
# - Full 22-channel baseline
# - Reduced-channel re-training using top channels
# - No occlusion — only gradient-based attribution methods
# - Detailed text report with accuracies, correlations, best channels
# - Outputs saved in EEGDeep4Net_HGD_Output/...

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

import mne
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from braindecode.datasets import MOABBDataset
from braindecode.models import Deep4Net
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
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
# Config — Paper-ready for Deep4Net
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

IG_STEPS = 32
SMOOTHGRAD_SAMPLES = 12
SMOOTHGRAD_NOISE_STD = 0.10

# Only gradient-based methods
ATTR_METHODS = ["grad", "grad_x_input", "integrated_gradients", "smoothgrad"]

# Full channel set (22 channels)
FULL_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Pz'
]

# Reduced channel experiment config
REDUCED_N_CHANNELS = 12          # ← change to 8 if preferred
REDUCED_SELECTION = "best_methods"  # "all_methods" or "best_methods"
BEST_METHODS_COUNT = 3           # top N methods by ERD_(L-R) correlation

OUTPUT_BASE = "EEGDeep4Net_HGD_Output"   # distinct folder for Deep4Net

# ────────────────────────────────────────────────
# MNE topomap setup (dynamic)
# ────────────────────────────────────────────────
montage = mne.channels.make_standard_montage("standard_1005")
CHANNELS = None      # set in run_experiment
TOPO_INFO = None

# ────────────────────────────────────────────────
# Utilities (same as before)
# ────────────────────────────────────────────────
def scale_to_microvolts(x): return x * 1e6


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
    if shuffle: np.random.shuffle(idx)
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
    return np.mean(losses), (preds == y).mean()


def _normalize_importance(vec):
    vec = np.asarray(vec, dtype=np.float64)
    m = np.max(np.abs(vec))
    return vec / m if m > 1e-12 else vec.astype(np.float32)


def corrcoef_safe(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) != len(b) or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def top_k_channels(scores, ch_names, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(ch_names[i], float(scores[i])) for i in idx]


def print_top_channels(sid, method, scores, ch_names):
    top8 = top_k_channels(scores, ch_names, 8)
    top12 = top_k_channels(scores, ch_names, 12)
    print(f"\n[Subject {sid:02d}] Top-8  | {method}")
    print(", ".join(f"{ch} ({v:.4f})" for ch, v in top8))
    print(f"[Subject {sid:02d}] Top-12 | {method}")
    print(", ".join(f"{ch} ({v:.4f})" for ch, v in top12))


def save_array(path, arr):
    np.save(path, arr.astype(np.float32))


# ────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────
def plot_saliency_topomap(vec, info, title, out_png, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(vec, info, axes=ax, show=False, cmap=cmap)
    ax.set_title(title)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_erd_ers_topomaps(erd_l, erd_r, contrast, comb, info, out_png):
    vals = [erd_l, erd_r, contrast, comb]
    limit = float(np.max(np.abs(np.concatenate([v.ravel() for v in vals])))) if any(len(v) > 0 for v in vals) else 1.0
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


# ────────────────────────────────────────────────
# Attribution (gradient-based only)
# ────────────────────────────────────────────────
def _gather_true_class_score(logits, targets):
    return logits.gather(1, targets.view(-1, 1)).sum()


def compute_attr_batch(model, xb, yb, method, ig_steps=32, sg_samples=12, sg_noise_std=0.10):
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
        return (x.grad.detach() * x.detach())
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
        attr = (xb - baseline) * (total_grad / ig_steps)
        return attr.detach()
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
    n_chans = X.shape[1]
    out = {}
    sums = {m: np.zeros(n_chans, dtype=np.float64) for m in methods}
    n_total = 0
    for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
        bsz = xb.shape[0]
        n_total += bsz
        for m in methods:
            attr = compute_attr_batch(model, xb, yb, m,
                                      ig_steps=IG_STEPS,
                                      sg_samples=SMOOTHGRAD_SAMPLES,
                                      sg_noise_std=SMOOTHGRAD_NOISE_STD)
            scores = attr.abs().mean(dim=(0, 2)).cpu().numpy()
            sums[m] += scores * bsz
    for m in methods:
        out[m] = _normalize_importance(sums[m] / max(1, n_total))
    return out


# ────────────────────────────────────────────────
# ERD / ERS
# ────────────────────────────────────────────────
def load_subject_for_erd(subject_id):
    ds = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])
    preprocessors = [
        Preprocessor("pick_channels", ch_names=CHANNELS, ordered=True),
        Preprocessor(scale_to_microvolts, apply_on_array=True),
        Preprocessor("resample", sfreq=TARGET_SFREQ),
        Preprocessor("filter", l_freq=8.0, h_freq=30.0),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocess(ds, preprocessors, n_jobs=1)

    windows = create_windows_from_events(
        ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
    )
    X, y = extract_xy(windows)
    sfreq = TARGET_SFREQ
    return X, y, sfreq


def compute_class_erd_map(X_class, sfreq):
    if len(X_class) == 0:
        return np.full(len(CHANNELS), np.nan, dtype=np.float32)
    baseline = X_class[:, :, :int(0.5 * sfreq)]
    active = X_class[:, :, int(1.0 * sfreq):int(3.0 * sfreq)]
    p_base = np.mean(baseline ** 2, axis=2)
    p_act = np.mean(active ** 2, axis=2)
    erd = 100 * (p_act - p_base) / (p_base + 1e-12)
    return np.mean(erd, axis=0).astype(np.float32)


def compute_subject_erd_maps(subject_id):
    X, y, sfreq = load_subject_for_erd(subject_id)
    LEFT_HAND_LABEL = 0
    RIGHT_HAND_LABEL = 1
    erd_l = compute_class_erd_map(X[y == LEFT_HAND_LABEL], sfreq)
    erd_r = compute_class_erd_map(X[y == RIGHT_HAND_LABEL], sfreq)
    return {
        "ERD_L": erd_l,
        "ERD_R": erd_r,
        "ERD_(L-R)": erd_l - erd_r,
        "ERD_comb": 0.5 * (erd_l + erd_r),
    }


# ────────────────────────────────────────────────
# Model — Deep4Net
# ────────────────────────────────────────────────
def build_model(n_chans, n_times, n_classes):
    return Deep4Net(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length='auto',
        drop_prob=DROPOUT,
    ).to(DEVICE)


# ────────────────────────────────────────────────
# Training function
# ────────────────────────────────────────────────
def train_one_fold(X, y, train_idx, val_idx, test_idx):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))
    print(f"num channels {n_chans} num out {n_classes}")

    model = build_model(n_chans, n_times, n_classes)

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

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:03d} | tr loss {np.mean(train_losses):.4f} | "
              f"val {val_acc:.1%} | te {test_acc:.1%} | "
              f"best_val_loss={best_val_loss:.4f} | lr={current_lr:.2e}")

        if epochs_no_improve >= PATIENCE:
            print(f"Early stop @ ep {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state found")

    model.load_state_dict(best_state)
    _, final_test_acc = evaluate(model, X_te, y_te, criterion)
    return final_test_acc, best_val_acc, best_state, n_chans, n_times, n_classes


# ────────────────────────────────────────────────
# Experiment runner (full or reduced)
# ────────────────────────────────────────────────
def run_experiment(channels_list, subdir_name):
    global CHANNELS, TOPO_INFO
    CHANNELS = channels_list[:]
    TOPO_INFO = mne.create_info(ch_names=CHANNELS, sfreq=TARGET_SFREQ, ch_types="eeg")
    TOPO_INFO.set_montage(montage)

    output_subdir = os.path.join(OUTPUT_BASE, subdir_name)
    os.makedirs(output_subdir, exist_ok=True)

    all_subject_means = []
    subject_summaries = []

    for subject_id in SUBJECT_IDS:
        print("\n" + "=" * 80)
        print(f"SUBJECT {subject_id:02d} — HGD — Deep4Net")
        print("=" * 80)

        dataset = MOABBDataset(DATASET_NAME, subject_ids=[subject_id])

        preprocessors = [
            Preprocessor("pick_channels", ch_names=CHANNELS, ordered=True),
            Preprocessor(scale_to_microvolts, apply_on_array=True),
            Preprocessor("resample", sfreq=TARGET_SFREQ),
            Preprocessor("filter", l_freq=LOW_CUT_HZ, h_freq=HIGH_CUT_HZ),
            Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000),
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
        mask = np.isin(y, [0, 1])
        X = X[mask]
        y = y[mask]

        unique_y = np.unique(y)
        label_mapping = {original_val: seq_idx for seq_idx, original_val in enumerate(unique_y)}
        y = np.array([label_mapping[val] for val in y], dtype=np.int64)

        print(f"Trials (left+right only): {X.shape[0]}, shape {X.shape}")

        folds = make_blockwise_folds(len(y), n_blocks=4)

        fold_test_accs = []
        subject_importance_accum = {m: np.zeros(len(CHANNELS), dtype=np.float64) for m in ATTR_METHODS}

        for fold_i, (tr_idx, val_idx, te_idx) in enumerate(folds, 1):
            print(f"\nFold {fold_i}/4 | train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")
            test_acc, _, best_state, n_ch, n_t, n_cl = train_one_fold(X, y, tr_idx, val_idx, te_idx)
            fold_test_accs.append(test_acc)

            model = build_model(n_ch, n_t, n_cl)
            model.load_state_dict(best_state)
            model.eval()

            X_te, y_te = X[te_idx], y[te_idx]
            fold_imps = compute_fold_channel_importances(model, X_te, y_te, ATTR_METHODS, batch_size=32)

            for m in ATTR_METHODS:
                subject_importance_accum[m] += fold_imps.get(m, 0)

            del model
            gc.collect()

        subj_mean = np.mean(fold_test_accs)
        subj_std = np.std(fold_test_accs)
        all_subject_means.append(subj_mean)

        print(f"\nSubject {subject_id:02d} RESULTS")
        for i, acc in enumerate(fold_test_accs, 1):
            print(f"Fold {i}: {acc * 100:.2f}%")
        print(f"Mean: {subj_mean * 100:.2f}% ± {subj_std * 100:.2f}%")

        subject_importances = {
            m: _normalize_importance(subject_importance_accum[m] / len(folds))
            for m in ATTR_METHODS
        }

        erd_maps = compute_subject_erd_maps(subject_id)

        # Correlations (focus on ERD_(L-R))
        subject_corrs = {}
        for m, vec in subject_importances.items():
            c_lr = corrcoef_safe(vec, erd_maps["ERD_(L-R)"])
            subject_corrs[m] = {"ERD_(L-R)": c_lr}

        subject_summaries.append({
            "subject": subject_id,
            "fold_test_accs": fold_test_accs,
            "subject_mean_acc": subj_mean,
            "subject_std_acc": subj_std,
            "importance": subject_importances,
            "erd_maps": erd_maps,
            "correlations": subject_corrs
        })

        # Save visuals
        for m, scores in subject_importances.items():
            png = os.path.join(output_subdir, f"saliency_{m}_S{subject_id:02d}.png")
            npy = os.path.join(output_subdir, f"saliency_{m}_S{subject_id:02d}.npy")
            plot_saliency_topomap(scores, TOPO_INFO, f"S{subject_id:02d} {m}", png)
            save_array(npy, scores)

        erd_png = os.path.join(output_subdir, f"erd_S{subject_id:02d}.png")
        plot_erd_ers_topomaps(
            erd_maps["ERD_L"], erd_maps["ERD_R"],
            erd_maps["ERD_(L-R)"], erd_maps["ERD_comb"],
            TOPO_INFO, erd_png
        )

        for k, v in erd_maps.items():
            save_array(os.path.join(output_subdir, f"{k}_S{subject_id:02d}.npy"), v)

        del dataset, windows_dataset, X, y
        gc.collect()

    # Grand averages
    grand_mean = np.mean(all_subject_means)
    grand_std = np.std(all_subject_means)

    print("\n" + "#" * 80)
    print(f"FINAL HGD RESULTS — Deep4Net — {subdir_name}")
    print(f"Grand mean: {grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%")
    print("#" * 80)

    # Save combined saliency & ERD
    for method in ATTR_METHODS:
        vecs = [s["importance"][method] for s in subject_summaries]
        avg = _normalize_importance(np.mean(vecs, axis=0))
        png_path = os.path.join(output_subdir, f"final_combined_{method}.png")
        plot_saliency_topomap(avg, TOPO_INFO, f"Combined {method}", png_path)

    final_erd_l = np.mean([s["erd_maps"]["ERD_L"] for s in subject_summaries], axis=0)
    final_erd_r = np.mean([s["erd_maps"]["ERD_R"] for s in subject_summaries], axis=0)
    final_erd_lr = np.mean([s["erd_maps"]["ERD_(L-R)"] for s in subject_summaries], axis=0)
    final_erd_c = np.mean([s["erd_maps"]["ERD_comb"] for s in subject_summaries], axis=0)

    plot_erd_ers_topomaps(final_erd_l, final_erd_r, final_erd_lr, final_erd_c,
                          TOPO_INFO, os.path.join(output_subdir, "final_combined_erd.png"))

    return subject_summaries, all_subject_means, grand_mean, grand_std


# ────────────────────────────────────────────────
# MAIN — Paper pipeline
# ────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=== FULL 22-CHANNEL BASELINE — Deep4Net ===")
    full_summaries, full_means, grand_mean, grand_std = run_experiment(FULL_CHANNELS, "full_22ch")

    # Select best methods by average ERD_(L-R) correlation
    method_avg_corrs = {}
    for m in ATTR_METHODS:
        corrs = [s["correlations"][m]["ERD_(L-R)"] for s in full_summaries]
        method_avg_corrs[m] = np.nanmean(corrs)

    ranked_methods = sorted(method_avg_corrs, key=method_avg_corrs.get, reverse=True)
    best_methods = ranked_methods[:BEST_METHODS_COUNT]
    print(f"\nBest attribution methods (top {BEST_METHODS_COUNT} by avg ERD_(L-R) corr): {best_methods}")

    # ───── Paper report ─────
    report_path = os.path.join(OUTPUT_BASE, "paper_summary_report_deep4net.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Deep4Net — HGD (Schirrmeister2017) — PAPER SUMMARY REPORT\n")
        f.write("Full 22ch baseline + reduced channels\n\n")
        f.write("=" * 80 + "\n\n")

        for subj in full_summaries:
            sid = subj["subject"]
            f.write(f"SUBJECT {sid:02d}\n")
            f.write("-" * 60 + "\n")
            for i, acc in enumerate(subj["fold_test_accs"], 1):
                f.write(f"  Fold {i}: {acc * 100:.2f}%\n")
            std = subj["subject_std_acc"] * 100
            f.write(f"  Mean: {subj['subject_mean_acc'] * 100:.2f}% ± {std:.2f}%\n\n")

            f.write("CORRELATIONS with ERD_(L-R):\n")
            for m in ATTR_METHODS:
                c = subj["correlations"][m]["ERD_(L-R)"]
                f.write(f"  {m:>20} → {c:+.3f}\n")
            f.write("\n")

            # Avg all
            avg_all = np.mean([subj["importance"][m] for m in ATTR_METHODS], axis=0)
            avg_all = _normalize_importance(avg_all)
            f.write("Top channels — avg ALL methods:\n")
            f.write("  Top-8 : " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_all, FULL_CHANNELS, 8)) + "\n")
            f.write("  Top-12: " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_all, FULL_CHANNELS, 12)) + "\n\n")

            # Avg best
            avg_best = np.mean([subj["importance"][m] for m in best_methods], axis=0)
            avg_best = _normalize_importance(avg_best)
            f.write("Top channels — avg BEST methods:\n")
            f.write("  Top-8 : " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_best, FULL_CHANNELS, 8)) + "\n")
            f.write("  Top-12: " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_best, FULL_CHANNELS, 12)) + "\n")
            f.write("\n" + "=" * 80 + "\n\n")

        f.write(f"GRAND MEAN ACCURACY (22ch): {grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%\n\n")

        f.write("AVERAGED CORRELATIONS (ERD_(L-R)):\n")
        for m in ATTR_METHODS:
            c_lr = np.mean([s["correlations"][m]["ERD_(L-R)"] for s in full_summaries])
            f.write(f"  {m:>20} → {c_lr:+.3f}\n")
        f.write("\n")

        grand_avg_all = np.mean([np.mean([s["importance"][m] for m in ATTR_METHODS], axis=0) for s in full_summaries], axis=0)
        grand_avg_all = _normalize_importance(grand_avg_all)
        f.write("GRAND TOP CHANNELS — ALL methods (Top-12):\n")
        f.write(", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(grand_avg_all, FULL_CHANNELS, 12)) + "\n\n")

        grand_avg_best = np.mean([np.mean([s["importance"][m] for m in best_methods], axis=0) for s in full_summaries], axis=0)
        grand_avg_best = _normalize_importance(grand_avg_best)
        f.write("GRAND TOP CHANNELS — BEST methods (Top-12):\n")
        f.write(", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(grand_avg_best, FULL_CHANNELS, 12)) + "\n\n")

    print(f"Report saved: {report_path}")

    # ───── Reduced channel experiment ─────
    if REDUCED_SELECTION == "all_methods":
        grand_avg_imp = grand_avg_all
    else:
        grand_avg_imp = grand_avg_best

    selected = top_k_channels(grand_avg_imp, FULL_CHANNELS, REDUCED_N_CHANNELS)
    REDUCED_CHANNELS = [ch for ch, _ in selected]

    print(f"\n=== REDUCED TRAINING: {REDUCED_N_CHANNELS} channels ({REDUCED_SELECTION}) ===")
    print("Selected:", ", ".join(REDUCED_CHANNELS))

    reduced_summaries, reduced_means, red_grand_mean, red_grand_std = run_experiment(
        REDUCED_CHANNELS,
        f"reduced_{REDUCED_N_CHANNELS}ch_{REDUCED_SELECTION}"
    )

    # Append comparison to report
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + "#" * 80 + "\n")
        f.write(f"REDUCED ({REDUCED_N_CHANNELS} ch — {REDUCED_SELECTION})\n")
        f.write("#" * 80 + "\n\n")
        for i, (f_acc, r_acc) in enumerate(zip(full_means, reduced_means), 1):
            f.write(f"S{i:02d}: 22ch {f_acc*100:5.2f}%  →  reduced {r_acc*100:5.2f}%\n")
        f.write(f"\nGRAND: 22ch {grand_mean*100:.2f}%  →  reduced {red_grand_mean*100:.2f}%\n")

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    for i, (f, r) in enumerate(zip(full_means, reduced_means), 1):
        print(f"S{i:02d}:  {f*100:5.2f}%  →  {r*100:5.2f}%")
    print(f"Grand: {grand_mean*100:5.2f}%  →  {red_grand_mean*100:5.2f}%")
    print("=" * 80)

    print(f"\nResults saved in: {OUTPUT_BASE}/")
    print("Report: ", report_path)
    print("\nDone.")