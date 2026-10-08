# deep4net_bnci2014_001_paper_ready_full_vs_reduced.py
# Paper-ready attribution / reduced-channel pipeline using:
# - Dataset: BNCI2014_001
# - Model: Deep4Net
# - Training regimen from Train_All_Subjects_Braindecode_Deep4Net_4Fold_Ultimate.py

import os
import gc
import copy
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.model_selection import StratifiedKFold

import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from braindecode.datasets import MOABBDataset
from braindecode.models import Deep4Net
from braindecode.preprocessing import exponential_moving_standardize

warnings.filterwarnings("ignore", category=UserWarning, module="braindecode")

# ────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────
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

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
DATASET_NAME = "BNCI2014_001"
SUBJECT_IDS = list(range(1, 10))

SFREQ = 250
BAND = (4.0, 38.0)

EPOCH_TMIN = -0.5
EPOCH_TMAX = 4.0 - 1.0 / SFREQ
N_SAMPLES_RECEPTIVE = 1000
N_SAMPLES_TRIAL = int(np.round((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)) + 1

FULL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

BATCH_SIZE = 64
EPOCHS = 350
SWA_START = 250
N_FOLDS = 4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2

LR = 1e-3
WEIGHT_DECAY = 1e-3

IG_STEPS = 32
SMOOTHGRAD_SAMPLES = 12
SMOOTHGRAD_NOISE_STD = 0.10
ATTR_METHODS = ["grad", "grad_x_input", "integrated_gradients", "smoothgrad"]

REDUCED_N_CHANNELS = 12
REDUCED_SELECTION = "best_methods"   # "all_methods" or "best_methods"
BEST_METHODS_COUNT = 3

OUTPUT_DIR = "Deep4Net_BNCI2014_001_Output"

# ────────────────────────────────────────────────
# MNE topomap setup
# ────────────────────────────────────────────────
montage = mne.channels.make_standard_montage("standard_1005")
CHANNELS = None
TOPO_INFO = None

# ────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────
def save_array(path, arr):
    np.save(path, np.asarray(arr, dtype=np.float32))


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
    top8 = top_k_channels(scores, ch_names, min(8, len(ch_names)))
    top12 = top_k_channels(scores, ch_names, min(12, len(ch_names)))
    print(f"\n[Subject {sid:02d}] Top-8  | {method}")
    print(", ".join(f"{ch} ({v:.4f})" for ch, v in top8))
    print(f"[Subject {sid:02d}] Top-12 | {method}")
    print(", ".join(f"{ch} ({v:.4f})" for ch, v in top12))


def batch_iter(X, y, batch_size, shuffle=True):
    idx = np.arange(len(y))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sl = idx[start:start + batch_size]
        yield (
            torch.tensor(X[sl], dtype=torch.float32, device=DEVICE),
            torch.tensor(y[sl], dtype=torch.long, device=DEVICE)
        )

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
# Training regimen from Deep4Net script
# ────────────────────────────────────────────────
def apply_eeg_augmentations(xb):
    xb = xb.clone()

    if random.random() > 0.5:
        noise_std = xb.std() * 0.05
        xb = xb + torch.randn_like(xb) * noise_std

    if random.random() > 0.5:
        scales = torch.empty(xb.size(0), 1, 1, device=xb.device).uniform_(0.8, 1.2)
        xb = xb * scales

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
    """
    log_probs = F.log_softmax(pred, dim=1)
    y_a_broad = y_a.unsqueeze(1).expand(-1, pred.size(2))
    y_b_broad = y_b.unsqueeze(1).expand(-1, pred.size(2))

    loss_a = compute_smoothed_loss(log_probs, y_a_broad, smoothing)
    loss_b = compute_smoothed_loss(log_probs, y_b_broad, smoothing)
    return lam * loss_a + (1 - lam) * loss_b

# ────────────────────────────────────────────────
# Data prep from Deep4Net script
# ────────────────────────────────────────────────
def prepare_raw(raw: mne.io.BaseRaw, selected_channels) -> mne.io.BaseRaw:
    raw = raw.copy()
    raw.load_data()
    raw = raw.pick(selected_channels)
    raw.apply_function(lambda x: x * 1e6, channel_wise=False)

    if int(raw.info["sfreq"]) != SFREQ:
        raw.resample(SFREQ, npad="auto", verbose=False)

    iir_params = dict(order=6, ftype="cheby1", rp=0.5)
    raw.filter(
        l_freq=BAND[0],
        h_freq=BAND[1],
        method="iir",
        iir_params=iir_params,
        verbose=False
    )

    raw.apply_function(
        lambda x: exponential_moving_standardize(
            x, factor_new=1e-3, init_block_size=1000
        ),
        channel_wise=False
    )
    return raw


def epochs_from_cue_onsets(raw: mne.io.BaseRaw):
    class_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    event_id = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}

    onsets = []
    codes = []
    for ann in raw.annotations:
        desc = str(ann["description"])
        if desc in class_map:
            onsets.append(ann["onset"])
            codes.append(event_id[desc])

    onsets = np.array(onsets)
    codes = np.array(codes, dtype=int)
    sf = raw.info["sfreq"]
    samples = np.round(onsets * sf).astype(int)

    events = np.column_stack((samples, np.zeros(len(samples), dtype=int), codes))
    events = events[np.argsort(events[:, 0])]

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False
    )

    X = epochs.get_data().astype(np.float32)
    code_to_label = {1: 0, 2: 1, 3: 2, 4: 3}
    y = np.array([code_to_label[c] for c in epochs.events[:, 2]], dtype=np.int64)

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

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all

# ────────────────────────────────────────────────
# ERD maps
# ────────────────────────────────────────────────
def compute_class_erd_map(X_class, sfreq):
    if len(X_class) == 0:
        return np.full(len(CHANNELS), np.nan, dtype=np.float32)

    baseline = X_class[:, :, :int(0.5 * sfreq)]
    active = X_class[:, :, int(1.5 * sfreq):int(3.5 * sfreq)]

    p_base = np.mean(baseline ** 2, axis=2)
    p_act = np.mean(active ** 2, axis=2)
    erd = 100 * (p_act - p_base) / (p_base + 1e-12)
    return np.mean(erd, axis=0).astype(np.float32)


def compute_subject_erd_maps(X, y):
    LEFT_HAND_LABEL = 0
    RIGHT_HAND_LABEL = 1

    erd_l = compute_class_erd_map(X[y == LEFT_HAND_LABEL], SFREQ)
    erd_r = compute_class_erd_map(X[y == RIGHT_HAND_LABEL], SFREQ)

    return {
        "ERD_L": erd_l,
        "ERD_R": erd_r,
        "ERD_(L-R)": erd_l - erd_r,
        "ERD_comb": 0.5 * (erd_l + erd_r),
        "ERS_L": np.maximum(erd_l, 0),
        "ERS_R": np.maximum(erd_r, 0),
    }


def print_subject_erd_summary(sid, erd_maps):
    print(f"\n=== SUBJECT {sid:02d} | ERD / ERS SUMMARY ===")
    for key in ["ERD_L", "ERD_R"]:
        vec = erd_maps[key]
        if np.all(np.isnan(vec)):
            print(f"  {key}: no trials available")
            continue
        most_erd = np.argsort(vec)[:8]
        most_ers = np.argsort(vec)[::-1][:8]
        print(f"\n{key} strongest ERD:")
        print(", ".join(f"{CHANNELS[i]} ({vec[i]:.2f}%)" for i in most_erd))
        print(f"{key} strongest ERS:")
        print(", ".join(f"{CHANNELS[i]} ({vec[i]:.2f}%)" for i in most_ers))


def print_subject_correlations(sid, imp_dict, erd_maps):
    print(f"\n=== SUBJECT {sid:02d} | CORRELATIONS ===")
    grad_ref = imp_dict.get("grad", np.zeros(len(CHANNELS)))
    for m, vec in imp_dict.items():
        c_l = corrcoef_safe(vec, erd_maps["ERD_L"])
        c_r = corrcoef_safe(vec, erd_maps["ERD_R"])
        c_lr = corrcoef_safe(vec, erd_maps["ERD_(L-R)"])
        c_comb = corrcoef_safe(vec, erd_maps["ERD_comb"])
        c_g = corrcoef_safe(vec, grad_ref)
        print(f"{m:>20} | ERD_L {c_l:+.3f} | ERD_R {c_r:+.3f} | "
              f"(L-R) {c_lr:+.3f} | comb {c_comb:+.3f} | vsGrad {c_g:+.3f}")

# ────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────
def build_deep4net_model(n_chans, n_classes):
    return Deep4Net(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=N_SAMPLES_RECEPTIVE,
        final_conv_length="auto",
        drop_prob=0.5,
    ).to(DEVICE)

# ────────────────────────────────────────────────
# Evaluation with TTA
# ────────────────────────────────────────────────
@torch.no_grad()
def evaluate_tta(model, X, y, batch_size=256):
    model.eval()
    preds = []

    for xb, _ in batch_iter(X, y, batch_size, shuffle=False):
        logits_1 = model(xb).view(xb.size(0), 4, -1)

        xb_noise = xb + torch.randn_like(xb) * (xb.std() * 0.02)
        logits_2 = model(xb_noise).view(xb.size(0), 4, -1)

        xb_scale = xb * 0.90
        logits_3 = model(xb_scale).view(xb.size(0), 4, -1)

        p1 = F.softmax(logits_1, dim=1).mean(dim=2)
        p2 = F.softmax(logits_2, dim=1).mean(dim=2)
        p3 = F.softmax(logits_3, dim=1).mean(dim=2)

        final_probs = (p1 + p2 + p3) / 3.0
        preds.append(torch.argmax(final_probs, dim=1).cpu().numpy())

    preds = np.concatenate(preds)
    return (preds == y).mean()

# ────────────────────────────────────────────────
# Train one fold with Deep4Net regimen
# ────────────────────────────────────────────────
def train_one_fold(X, y, train_idx, test_idx):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr)
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_te),
        torch.from_numpy(y_te)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    swa_bn_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    n_chans = X.shape[1]
    n_classes = len(np.unique(y))

    model = build_deep4net_model(n_chans, n_classes)
    swa_model = AveragedModel(model).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            xb = apply_eeg_augmentations(xb)

            if random.random() > 0.5:
                xb, y_a, y_b, lam = mixup_data(xb, yb, alpha=MIXUP_ALPHA)
            else:
                y_a, y_b, lam = yb, yb, 1.0

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            logits = logits.view(logits.size(0), logits.size(1), -1)

            loss = mixup_criterion(
                logits, y_a, y_b, lam,
                smoothing=LABEL_SMOOTHING
            )

            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        scheduler.step()

        if epoch >= SWA_START:
            swa_model.update_parameters(model)

        if epoch % 50 == 0:
            print(f"    Epoch {epoch:03d}/{EPOCHS} complete...")

    update_bn(swa_bn_loader, swa_model, device=DEVICE)
    final_test_acc = evaluate_tta(swa_model, X_te, y_te, batch_size=256)

    state = copy.deepcopy(swa_model.module.state_dict())

    del model, swa_model, optimizer, scheduler, train_loader, test_loader, swa_bn_loader, train_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_test_acc, state, n_chans, n_classes

# ────────────────────────────────────────────────
# Attribution methods for Deep4Net temporal output
# ────────────────────────────────────────────────
def _gather_true_class_score(logits, targets):
    """
    logits: [B, C, T]
    targets: [B]
    """
    tdim = logits.size(2)
    targets_b = targets.unsqueeze(1).expand(-1, tdim)
    gathered = logits.gather(1, targets_b.unsqueeze(1)).squeeze(1)  # [B, T]
    return gathered.sum()


def compute_attr_batch(model, xb, yb, method, ig_steps=32, sg_samples=12, sg_noise_std=0.10):
    model.eval()

    if method == "grad":
        x = xb.clone().detach().requires_grad_(True)
        out = model(x).view(x.size(0), 4, -1)
        score = _gather_true_class_score(out, yb)
        score.backward()
        return x.grad.detach()

    elif method == "grad_x_input":
        x = xb.clone().detach().requires_grad_(True)
        out = model(x).view(x.size(0), 4, -1)
        score = _gather_true_class_score(out, yb)
        score.backward()
        return x.grad.detach() * x.detach()

    elif method == "integrated_gradients":
        baseline = torch.zeros_like(xb)
        total_grad = torch.zeros_like(xb)
        for alpha in torch.linspace(0., 1., ig_steps + 1, device=xb.device)[1:]:
            xi = baseline + alpha * (xb - baseline)
            xi.requires_grad_(True)
            out = model(xi).view(xi.size(0), 4, -1)
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
            out = model(xn).view(xn.size(0), 4, -1)
            score = _gather_true_class_score(out, yb)
            score.backward()
            total += xn.grad.detach().abs()
        return (total / sg_samples).detach()

    else:
        raise ValueError(method)


def compute_fold_channel_importances(model, X, y, methods, batch_size=32):
    n_chans = X.shape[1]
    sums = {m: np.zeros(n_chans, dtype=np.float64) for m in methods}
    n_total = 0

    for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
        bsz = xb.shape[0]
        n_total += bsz
        for m in methods:
            attr = compute_attr_batch(
                model, xb, yb, m,
                ig_steps=IG_STEPS,
                sg_samples=SMOOTHGRAD_SAMPLES,
                sg_noise_std=SMOOTHGRAD_NOISE_STD
            )
            scores = attr.abs().mean(dim=(0, 2)).cpu().numpy()
            sums[m] += scores * bsz

    out = {}
    for m in methods:
        out[m] = _normalize_importance(sums[m] / max(1, n_total))
    return out

# ────────────────────────────────────────────────
# Main experiment runner
# ────────────────────────────────────────────────
def run_experiment(channels_list, subdir_name):
    global CHANNELS, TOPO_INFO
    CHANNELS = channels_list[:]
    TOPO_INFO = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ, ch_types="eeg")
    TOPO_INFO.set_montage(montage)

    output_subdir = os.path.join(OUTPUT_DIR, subdir_name)
    os.makedirs(output_subdir, exist_ok=True)

    all_subject_means = []
    subject_summaries = []

    for subject_id in SUBJECT_IDS:
        print("\n" + "=" * 80)
        print(f"SUBJECT {subject_id:02d} — BNCI2014_001 — Deep4Net")
        print("=" * 80)

        X, y = build_subject_arrays(subject_id, CHANNELS)
        print(f"Trials: {X.shape[0]}, shape {X.shape}")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        fold_test_accs = []
        subject_importance_accum = {
            m: np.zeros(len(CHANNELS), dtype=np.float64) for m in ATTR_METHODS
        }

        for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold_i}/{N_FOLDS} | train={len(tr_idx)} test={len(te_idx)}")
            test_acc, best_state, n_ch, n_cl = train_one_fold(X, y, tr_idx, te_idx)
            fold_test_accs.append(test_acc)

            model = build_deep4net_model(n_ch, n_cl)
            model.load_state_dict(best_state)
            model.eval()

            X_te, y_te = X[te_idx], y[te_idx]
            fold_imps = compute_fold_channel_importances(
                model, X_te, y_te, ATTR_METHODS, batch_size=32
            )

            for m in ATTR_METHODS:
                subject_importance_accum[m] += fold_imps[m]

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        subj_mean = np.mean(fold_test_accs)
        subj_std = np.std(fold_test_accs)
        all_subject_means.append(subj_mean)

        print(f"\nSubject {subject_id:02d} RESULTS")
        for i, acc in enumerate(fold_test_accs, 1):
            print(f"Fold {i}: {acc * 100:.2f}%")
        print(f"Mean: {subj_mean * 100:.2f}% ± {subj_std * 100:.2f}%")

        subject_importances = {
            m: _normalize_importance(subject_importance_accum[m] / N_FOLDS)
            for m in ATTR_METHODS
        }

        erd_maps = compute_subject_erd_maps(X, y)
        print_subject_erd_summary(subject_id, erd_maps)
        print_subject_correlations(subject_id, subject_importances, erd_maps)

        print(f"\n=== SUBJECT {subject_id:02d} IMPORTANT CHANNELS ===")
        for m, scores in subject_importances.items():
            print_top_channels(subject_id, m, scores, CHANNELS)

        subject_corrs = {}
        for m, vec in subject_importances.items():
            c_l = corrcoef_safe(vec, erd_maps["ERD_L"])
            c_r = corrcoef_safe(vec, erd_maps["ERD_R"])
            c_lr = corrcoef_safe(vec, erd_maps["ERD_(L-R)"])
            c_comb = corrcoef_safe(vec, erd_maps["ERD_comb"])
            c_g = corrcoef_safe(vec, subject_importances.get("grad", np.zeros(len(CHANNELS))))
            subject_corrs[m] = {
                "ERD_L": c_l,
                "ERD_R": c_r,
                "ERD_(L-R)": c_lr,
                "ERD_comb": c_comb,
                "vsGrad": c_g
            }

        print(f"\nSaving saliency topomaps for S{subject_id:02d} ...")
        for m, scores in subject_importances.items():
            png = os.path.join(output_subdir, f"saliency_{m}_S{subject_id:02d}.png")
            npy = os.path.join(output_subdir, f"saliency_{m}_S{subject_id:02d}.npy")
            plot_saliency_topomap(scores, TOPO_INFO, f"S{subject_id:02d} {m}", png)
            save_array(npy, scores)
            print(f"  {png}")

        erd_png = os.path.join(output_subdir, f"erd_S{subject_id:02d}.png")
        plot_erd_ers_topomaps(
            erd_maps["ERD_L"], erd_maps["ERD_R"],
            erd_maps["ERD_(L-R)"], erd_maps["ERD_comb"],
            TOPO_INFO, erd_png
        )
        print(f"  {erd_png}")

        for k, v in erd_maps.items():
            save_array(os.path.join(output_subdir, f"{k}_S{subject_id:02d}.npy"), v)

        subject_summaries.append({
            "subject": subject_id,
            "fold_test_accs": fold_test_accs,
            "subject_mean_acc": subj_mean,
            "subject_std_acc": subj_std,
            "importance": subject_importances,
            "erd_maps": erd_maps,
            "correlations": subject_corrs
        })

        del X, y
        gc.collect()

    grand_mean = np.mean(all_subject_means)
    grand_std = np.std(all_subject_means)

    print("\n" + "#" * 80)
    print(f"FINAL BNCI2014_001 WITHIN-SUBJECT RESULTS — {subdir_name}")
    for s, acc in enumerate(all_subject_means, 1):
        print(f"S{s}: {acc * 100:.2f}%")
    print(f"Grand mean: {grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%")
    print("#" * 80)

    print("\nFINAL COMBINED SALIENCY ACROSS SUBJECTS")
    for method in ATTR_METHODS:
        vecs = [s["importance"][method] for s in subject_summaries]
        avg = _normalize_importance(np.mean(vecs, axis=0))
        npy_path = os.path.join(output_subdir, f"final_combined_{method}.npy")
        png_path = os.path.join(output_subdir, f"final_combined_{method}.png")
        save_array(npy_path, avg)
        plot_saliency_topomap(
            avg, TOPO_INFO,
            f"Combined {method} (mean {grand_mean * 100:.1f}%)",
            png_path
        )
        print(f"Saved: {png_path}")
        print("Top channels:")
        print(", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg, CHANNELS, min(12, len(CHANNELS)))))

    final_erd_l = np.mean([s["erd_maps"]["ERD_L"] for s in subject_summaries], axis=0)
    final_erd_r = np.mean([s["erd_maps"]["ERD_R"] for s in subject_summaries], axis=0)
    final_erd_lr = np.mean([s["erd_maps"]["ERD_(L-R)"] for s in subject_summaries], axis=0)
    final_erd_c = np.mean([s["erd_maps"]["ERD_comb"] for s in subject_summaries], axis=0)

    final_erd_png = os.path.join(output_subdir, "final_combined_erd.png")
    plot_erd_ers_topomaps(final_erd_l, final_erd_r, final_erd_lr, final_erd_c,
                          TOPO_INFO, final_erd_png)
    print(f"Saved final ERD: {final_erd_png}")

    for name, arr in [
        ("ERD_L", final_erd_l),
        ("ERD_R", final_erd_r),
        ("ERD_LR", final_erd_lr),
        ("ERD_comb", final_erd_c)
    ]:
        save_array(os.path.join(output_subdir, f"final_combined_{name}.npy"), arr)

    print(f"\nDone for {subdir_name}. Files are in: {output_subdir}")
    return subject_summaries, all_subject_means

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== PAPER-READY RUN: FULL 22-CHANNEL BASELINE ===")
    full_summaries, full_means = run_experiment(FULL_CHANNELS, "full_22ch")

    method_avg_corrs = {}
    for m in ATTR_METHODS:
        corrs = [s["correlations"][m]["ERD_(L-R)"] for s in full_summaries]
        method_avg_corrs[m] = np.nanmean(corrs)

    ranked_methods = sorted(method_avg_corrs, key=method_avg_corrs.get, reverse=True)
    best_methods = ranked_methods[:BEST_METHODS_COUNT]
    print(f"\nBest gradient methods (top {BEST_METHODS_COUNT} by ERD_(L-R) correlation): {best_methods}")

    report_path = os.path.join(OUTPUT_DIR, "paper_summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Deep4Net_BNCI2014_001 — PAPER SUMMARY REPORT\n")
        f.write("Full 22-channel baseline + Reduced-channel comparison\n\n")
        f.write("=" * 80 + "\n\n")

        for subj in full_summaries:
            sid = subj["subject"]
            f.write(f"SUBJECT {sid:02d}\n")
            f.write("-" * 60 + "\n")
            for i, acc in enumerate(subj["fold_test_accs"], 1):
                f.write(f"  Fold {i}: {acc * 100:.2f}%\n")
            std = subj["subject_std_acc"] * 100
            f.write(f"  Subject accuracy: {subj['subject_mean_acc'] * 100:.2f}% ± {std:.2f}%\n\n")

            f.write("CORRELATION RESULTS (with ERD maps):\n")
            for m in ATTR_METHODS:
                c = subj["correlations"][m]
                f.write(f"  {m:>20} | ERD_L {c['ERD_L']:+.3f} | ERD_R {c['ERD_R']:+.3f} | "
                        f"(L-R) {c['ERD_(L-R)']:+.3f} | comb {c['ERD_comb']:+.3f} | vsGrad {c['vsGrad']:+.3f}\n")
            f.write("\n")

            avg_all = np.mean([subj["importance"][m] for m in ATTR_METHODS], axis=0)
            avg_all = _normalize_importance(avg_all)
            f.write("BEST CHANNELS — Average of ALL methods:\n")
            f.write("  Top-8 : " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_all, FULL_CHANNELS, 8)) + "\n")
            f.write("  Top-12: " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_all, FULL_CHANNELS, 12)) + "\n\n")

            avg_best = np.mean([subj["importance"][m] for m in best_methods], axis=0)
            avg_best = _normalize_importance(avg_best)
            f.write("BEST CHANNELS — Average of BEST methods:\n")
            f.write("  Top-8 : " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_best, FULL_CHANNELS, 8)) + "\n")
            f.write("  Top-12: " + ", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(avg_best, FULL_CHANNELS, 12)) + "\n")
            f.write("\n" + "=" * 80 + "\n\n")

        grand_mean = np.mean(full_means)
        grand_std = np.std(full_means)
        f.write(f"GRAND MEAN ACCURACY (22 channels): {grand_mean * 100:.2f}% ± {grand_std * 100:.2f}%\n\n")

        f.write("AVERAGED CORRELATIONS (over all subjects):\n")
        for m in ATTR_METHODS:
            c_lr = np.nanmean([s["correlations"][m]["ERD_(L-R)"] for s in full_summaries])
            f.write(f"  {m:>20} → ERD_(L-R) corr = {c_lr:+.3f}\n")
        f.write("\n")

        grand_avg_all = np.mean(
            [np.mean([s["importance"][m] for m in ATTR_METHODS], axis=0) for s in full_summaries],
            axis=0
        )
        grand_avg_all = _normalize_importance(grand_avg_all)
        f.write("GRAND BEST CHANNELS — Average of ALL methods (Top-12):\n")
        f.write(", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(grand_avg_all, FULL_CHANNELS, 12)) + "\n\n")

        grand_avg_best = np.mean(
            [np.mean([s["importance"][m] for m in best_methods], axis=0) for s in full_summaries],
            axis=0
        )
        grand_avg_best = _normalize_importance(grand_avg_best)
        f.write("GRAND BEST CHANNELS — Average of BEST methods (Top-12):\n")
        f.write(", ".join(f"{ch} ({v:.4f})" for ch, v in top_k_channels(grand_avg_best, FULL_CHANNELS, 12)) + "\n\n")

    print(f"\nFull paper report saved → {report_path}")

    if REDUCED_SELECTION == "all_methods":
        grand_avg_imp = grand_avg_all
    else:
        grand_avg_imp = grand_avg_best

    selected_top = top_k_channels(grand_avg_imp, FULL_CHANNELS, REDUCED_N_CHANNELS)
    REDUCED_CHANNELS = [ch for ch, _ in selected_top]

    print(f"\n=== STARTING REDUCED TRAINING ({REDUCED_N_CHANNELS} channels, selection={REDUCED_SELECTION}) ===")
    print(f"Selected channels: {REDUCED_CHANNELS}")

    reduced_summaries, reduced_means = run_experiment(
        REDUCED_CHANNELS,
        f"reduced_{REDUCED_N_CHANNELS}ch_{REDUCED_SELECTION}"
    )

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + "#" * 80 + "\n")
        f.write(f"REDUCED-CHANNEL RESULTS ({REDUCED_N_CHANNELS} channels — {REDUCED_SELECTION})\n")
        f.write("#" * 80 + "\n\n")
        for i, (full_m, red_m) in enumerate(zip(full_means, reduced_means), 1):
            f.write(f"Subject {i:02d}: 22ch = {full_m * 100:.2f}% | Reduced = {red_m * 100:.2f}%\n")
        grand_red = np.mean(reduced_means)
        f.write(f"\nGRAND MEAN: 22ch = {grand_mean * 100:.2f}% | Reduced = {grand_red * 100:.2f}%\n")

    print("\n" + "=" * 80)
    print("FINAL COMPARISON — 22ch vs Reduced")
    print("=" * 80)
    for i, (f_m, r_m) in enumerate(zip(full_means, reduced_means), 1):
        print(f"S{i:02d}: 22ch {f_m*100:6.2f}%  →  Reduced {r_m*100:6.2f}%")
    print(f"GRAND : 22ch {grand_mean*100:6.2f}%  →  Reduced {grand_red*100:6.2f}%")
    print("=" * 80)

    print("\nAll outputs organized in:")
    print(f"   {OUTPUT_DIR}/full_22ch/")
    print(f"   {OUTPUT_DIR}/reduced_{REDUCED_N_CHANNELS}ch_{REDUCED_SELECTION}/")
    print(f"Paper report: {report_path}")
    print("\nDone. Ready for submission.")
