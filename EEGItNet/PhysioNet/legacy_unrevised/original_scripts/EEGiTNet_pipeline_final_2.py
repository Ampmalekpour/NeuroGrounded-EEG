# EEGNetV$_Physionet_paper_ready_full_vs_reduced.py
# Paper-ready attribution / reduced-channel pipeline using:
# - Dataset: Physionet
# - Model: EEGNetV$
# - Training regimen from Train_All_Subjects_Braindecode_EEGNetV$_4Fold_Ultimate.py

import os
import gc
import copy
import random
import warnings

import numpy as np
import torch
import sys
import json
import hashlib
import argparse
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr, wilcoxon, binom, t as t_dist
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from braindecode.models import EEGITNet
try:
    from braindecode.models import EEGNetv4
except ImportError:  # نسخه‌های جدیدتر braindecode
    from braindecode.models import EEGNet as EEGNetv4
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.datasets import MOABBDataset

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
# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
DATASET_NAME = "PhysionetMI"
RUN_TYPE = "imagery"                       # فقط imagery؛ executed فقط با تصمیم صریح
EXCLUDED_SUBJECTS = [88, 92, 100, 104]     # پس از تأیید audit (فاز ۱) نهایی شود
SUBJECT_IDS = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]
SMOKE_SUBJECTS = [1, 2, 7, 42, 72]

# تعریف صریح تسک‌ها (mapping همیشه از همین‌جا ساخته می‌شود)
TASKS = {
    "4c_LRHF": ["left_hand", "right_hand", "hands", "feet"],   # تسک فعلی مقاله
    "4c_LR0F": ["left_hand", "right_hand", "rest", "feet"],    # تعریف رایج ادبیات
    "2c_LR":   ["left_hand", "right_hand"],
    "4c_OLDBUG": ["rest", "right_hand", "feet", "hands"],      # بازسازی باگ قدیمی: left_hand (id=0) حذف می‌شد
}
BALANCE_REST_TASKS = ["4c_LR0F"]           # rest زیرنمونه‌گیری می‌شود
TASK = "4c_LRHF"
MODEL_NAME = "eegitnet"                    # "eegitnet" | "eegnetv4"
MODEL_OUTPUTS_PROBS = False                # اگر stage=audit نشان داد خروجی مدل softmax است → True

# پیش‌پردازش و پنجره‌ها
LOW_CUT_HZ = 4.0                           # اصلی؛ برای ablation: 8.0 و 32.0
HIGH_CUT_HZ = 40.0
TARGET_SFREQ = 160
PRE_CUE_S = 1.5                            # epoch از cue-1.5 s (۰.۵ s اضافه برای اثر لبه‌ی فیلتر ERD)
POST_CUE_S = 3.0                           # تا cue+3.0 s
CROP = 320                                 # 2 s
CROP_STRIDE = 16
VAL_CROP_STRIDE = 40

# آموزش
OUTER_FOLDS = 5
INNER_VAL_FRAC = 0.2
SPLIT_SEED = 0
SEEDS = [0, 1, 2]
MAX_EPOCHS = 300
PATIENCE = 50
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.25
AUG_SCALE = (0.9, 1.1)
AUG_NOISE = 0.05

# Transfer learning
N_GROUPS = 5
GROUP_SEED = 0
PRETRAIN_EPOCHS = 100
PRETRAIN_PATIENCE = 15
PRETRAIN_VAL_FRAC = 0.1
FT_LR = 1e-4
FT_EPOCHS = 100
FT_PATIENCE = 20

# Attribution
ATTR_METHODS = ["grad", "grad_x_input", "integrated_gradients", "smoothgrad"]
PRIMARY_ATTR = "integrated_gradients"      # a-priori؛ انتخاب داده‌محور ممنوع
IG_STEPS = 50
SMOOTHGRAD_SAMPLES = 25
SMOOTHGRAD_NOISE_STD = 0.2

# ERD
ERD_REF = (-1.0, -0.1)
ERD_ACT = (0.5, 2.5)
BANDS = {"mu": (8.0, 12.0), "beta": (13.0, 30.0)}
LR_PAIRS = [("C3", "C4"), ("CP3", "CP4"), ("FC3", "FC4"), ("C5", "C6"), ("C1", "C2")]

# Faithfulness
DEL_KS = [4, 8, 12, 16]
N_RANDOM_DEL = 20

# انتخاب کانال
K_LIST = [8, 12, 16]
K_MAIN = 12
FIXED_SENSORIMOTOR = ["FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4"]
N_RANDOM_MONTAGES = 20
REDUCED_SPECS = [                          # (روش، پروتکل)
    ("saliency_local", "scratch"), ("erd_only", "scratch"), ("csp", "scratch"),
    ("mi", "scratch"), ("relieff", "scratch"), ("fixed", "scratch"), ("random", "scratch"),
    ("saliency_global", "ss_tl"), ("fixed", "ss_tl"),
]

# آمار
TOST_MARGIN_PP = 5.0                       # حاشیه‌ی هم‌ارزی (درصد)؛ باید در مقاله توجیه شود
N_BOOT = 5000

RESULTS_DIR = "results"
CACHE_DIR = "cache"
OUTPUT_DIR = "EEGitNet_Attribution_Physionet_Output"

CANONICAL_CHANNELS = None                  # در main پر می‌شود
GROUP_OF = None                            # در main پر می‌شود


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
# Training regimen from EEGITNET script
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
# Model and Training
# ────────────────────────────────────────────────
# ── مسیرها، seed، لاگ ─────────────────────────────────────────────
def rdir(*parts):
    p = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def rfile(*parts):
    p = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run_seed(seed, sid, fold):
    return int(seed) * 1_000_003 + int(sid) * 101 + int(fold)


def norm_ch(s):
    return s.strip(".").upper()


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def log_metrics(exp, task, model_name, protocol, montage, n_ch, sid, fold, seed, y_true, probs, test_idx):
    pred = probs.argmax(1)
    acc = float((pred == y_true).mean())
    kappa = float(cohen_kappa_score(y_true, pred))
    np.savez_compressed(rfile("preds", exp, f"S{sid:03d}_f{fold}_seed{seed}.npz"),
                        y_true=y_true, y_prob=probs.astype(np.float32), trial_idx=test_idx)
    append_row(rfile("metrics.csv"), {
        "exp": exp, "task": task, "model": model_name, "protocol": protocol, "n_ch": n_ch,
        "montage": montage, "subject": sid, "fold": fold, "seed": seed,
        "acc": acc, "kappa": kappa, "n_test": int(len(y_true))})
    return acc


def subject_done(exp, sid, n_expected):
    p = os.path.join(RESULTS_DIR, "metrics.csv")
    if not os.path.exists(p):
        return False
    df = pd.read_csv(p, usecols=["exp", "subject"])
    return int(((df["exp"] == exp) & (df["subject"] == sid)).sum()) >= n_expected


# ── مدل ───────────────────────────────────────────────────────────
def build_model(n_chans, n_times, n_classes, name=None):
    name = name or MODEL_NAME
    if name == "eegitnet":
        m = EEGITNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, drop_prob=DROPOUT)
    elif name == "eegnetv4":
        m = EEGNetv4(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    else:
        raise ValueError(name)
    return m.to(DEVICE)


def model_logits(model, x):
    """همیشه logit برمی‌گرداند (اگر مدل softmax دارد، log می‌گیرد)."""
    out = model(x)
    if out.dim() == 3:
        out = out.mean(dim=-1)
    if MODEL_OUTPUTS_PROBS:
        out = torch.log(out.clamp_min(1e-8))
    return out


@torch.no_grad()
def predict_proba(model, X, stride=None, bs=64):
    """میانگین softmax روی cropهای با stride ثابت. X: [n, C, T] نرمال‌شده."""
    stride = stride or CROP_STRIDE
    model.eval()
    starts = list(range(0, X.shape[2] - CROP + 1, stride))
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).float().to(DEVICE)
        p = 0
        for s in starts:
            p = p + torch.softmax(model_logits(model, xb[:, :, s:s + CROP]), dim=1)
        out.append((p / len(starts)).cpu().numpy())
    return np.concatenate(out)


_INIT_REG = {}


def _check_reinit(model, key):
    sig = round(sum(float(p.detach().double().sum()) for p in model.parameters()), 8)
    for k, v in _INIT_REG.items():
        if k[0] == key[0] and k != key and v == sig:
            raise AssertionError(f"مدل دوباره‌مقداردهی نشده: {k} == {key}")
    _INIT_REG[key] = sig


def train_model(X_tr, y_tr, X_va, y_va, n_classes, seed, model_name=None, key=None,
                init_state=None, lr=None, max_epochs=None, patience=None):
    """آموزش با AdamW + cosine + crop تصادفی + augmentation سبک + early stopping روی NLL مجموعه‌ی val."""
    lr = LR if lr is None else lr
    max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs
    patience = PATIENCE if patience is None else patience
    set_seed(seed)
    model = build_model(X_tr.shape[1], CROP, n_classes, model_name)
    if init_state is None:
        if key is not None:
            _check_reinit(model, key)
    else:
        model.load_state_dict(init_state)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    Xt = torch.from_numpy(np.ascontiguousarray(X_tr)).float()
    yt = torch.from_numpy(np.asarray(y_tr)).long()
    T = Xt.shape[2]

    best, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            if len(b) < 2:
                continue
            xb, yb = Xt[b].to(DEVICE), yt[b].to(DEVICE)
            starts = torch.randint(0, T - CROP + 1, (len(b),)).tolist()
            xb = torch.stack([xb[j, :, s:s + CROP] for j, s in enumerate(starts)])
            scale = torch.empty(len(b), 1, 1, device=DEVICE).uniform_(*AUG_SCALE)
            xb = xb * scale + AUG_NOISE * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model_logits(model, xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
        sched.step()

        p = predict_proba(model, X_va, stride=VAL_CROP_STRIDE)
        vloss = float(-np.log(p[np.arange(len(y_va)), y_va] + 1e-8).mean())
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_state


# ── نرمال‌سازی و Euclidean Alignment ─────────────────────────────
def zfit(X):
    return (X.mean(axis=(0, 2), keepdims=True).astype(np.float32),
            (X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32))


def zapply(X, st):
    return ((X - st[0]) / st[1]).astype(np.float32)


def ea_matrix(X, eps=1e-6):
    """R = میانگین کوواریانس trialها؛ خروجی R^(-1/2)."""
    Z = X.transpose(1, 0, 2).reshape(X.shape[1], -1).astype(np.float64)
    R = Z @ Z.T / Z.shape[1]
    R = R + eps * np.trace(R) / R.shape[0] * np.eye(R.shape[0])
    w, V = np.linalg.eigh(R)
    return ((V * (w ** -0.5)) @ V.T).astype(np.float32)


def ea_apply(X, W):
    return np.einsum("cd,ndt->nct", W, X).astype(np.float32)


# ────────────────────────────────────────────────
# Data prep from Physionet script
# ────────────────────────────────────────────────
def scale_to_microvolts(x): return x * 1e6


def extract_xy(ds):
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i][:2]
        xs.append(np.asarray(x, dtype=np.float32))
        ys.append(int(y))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def make_dataset(subject_id):
    kw = {"imagined": RUN_TYPE == "imagery", "executed": RUN_TYPE != "imagery"}
    try:
        return MOABBDataset(DATASET_NAME, subject_ids=[subject_id], dataset_kwargs=kw)
    except TypeError:
        print("WARNING: این نسخه‌ی braindecode پارامتر dataset_kwargs را نمی‌شناسد؛ نوع run را با audit بررسی کن.")
        return MOABBDataset(DATASET_NAME, subject_ids=[subject_id])


def get_canonical_channels():
    p = os.path.join(RESULTS_DIR, "canonical_channels.json")
    if os.path.exists(p):
        return json.load(open(p))
    ds = make_dataset(SUBJECT_IDS[0])
    raw = ds.datasets[0].raw
    chs = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, stim=False)]
    json.dump(chs, open(rfile("canonical_channels.json"), "w"))
    return chs


def get_topo_info(ch_names):
    """نام کانال‌ها را برای montage استاندارد 1005 نرمال می‌کند (فقط برای رسم)."""
    def fix(c):
        u = c.strip(".").upper()
        if u.startswith("FP"):
            u = "Fp" + u[2:]
        if u.endswith("Z"):
            u = u[:-1] + "z"
        return u
    info = mne.create_info(ch_names=[fix(c) for c in ch_names], sfreq=TARGET_SFREQ, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    return info


def balance_rest(X, y, rest_label, seed):
    others = [np.sum(y == c) for c in np.unique(y) if c != rest_label]
    n_keep = int(max(others))
    rest_idx = np.where(y == rest_label)[0]
    if len(rest_idx) > n_keep:
        rng = np.random.RandomState(seed)
        keep_rest = rng.choice(rest_idx, n_keep, replace=False)
        keep = np.sort(np.concatenate([np.where(y != rest_label)[0], keep_rest]))
        return X[keep], y[keep]
    return X, y


def load_subject(subject_id, task):
    """یک‌بار window با cue-1.5 s تا cue+3.0 s؛ X_cls همان بازه‌ی cue..cue+3.0 s است (هم‌ترازی تضمین‌شده)."""
    sf = TARGET_SFREQ
    pre = int(PRE_CUE_S * sf)
    cache = os.path.join(CACHE_DIR, task, f"S{subject_id:03d}.npz")
    if not os.path.exists(cache):
        names = TASKS[task]
        mapping = {n: i for i, n in enumerate(names)}
        ds = make_dataset(subject_id)
        pre_ops = [
            Preprocessor("pick_channels", ch_names=CANONICAL_CHANNELS, ordered=True),
            Preprocessor(scale_to_microvolts, apply_on_array=True),
            Preprocessor("resample", sfreq=sf),
            Preprocessor("filter", l_freq=LOW_CUT_HZ, h_freq=HIGH_CUT_HZ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocess(ds, pre_ops, n_jobs=1)
        win = int((PRE_CUE_S + POST_CUE_S) * sf)
        w = create_windows_from_events(
            ds, trial_start_offset_samples=-pre, trial_stop_offset_samples=0,
            window_size_samples=win, window_stride_samples=win,
            drop_last_window=True, preload=True, mapping=mapping)
        X, y = extract_xy(w)
        assert X.shape[2] == win, X.shape
        if task in BALANCE_REST_TASKS:
            X, y = balance_rest(X, y, mapping["rest"], seed=subject_id)
        counts = np.bincount(y, minlength=len(names))
        print(f"[S{subject_id:03d}] {task} trials={len(y)} per-class={dict(zip(names, counts.tolist()))}")
        if "rest" not in names and counts.max() > 40:
            print(f"WARNING S{subject_id}: بیش از 40 trial در یک کلاس؛ احتمال لود شدن runهای executed. audit را ببین.")
        if len(np.unique(y)) != len(names):
            print(f"SKIP S{subject_id}: همه‌ی کلاس‌ها حاضر نیستند")
            return None
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X_erd=X.astype(np.float32), y=y)
    z = np.load(cache)
    X_erd, y = z["X_erd"], z["y"]
    return {"X_erd": X_erd, "X_cls": X_erd[:, :, pre:], "y": y}


# ── Audit (فاز ۱) ─────────────────────────────────────────────────
def run_audit(subjects):
    rows = []
    for sid in subjects:
        ds = make_dataset(sid)
        for i, d in enumerate(ds.datasets):
            u, c = np.unique(d.raw.annotations.description, return_counts=True)
            rows.append({"subject": sid, "ds_index": i,
                         "session": d.description.get("session", ""), "run": d.description.get("run", ""),
                         "sfreq": d.raw.info["sfreq"], "n_samples": d.raw.n_times,
                         "events": json.dumps(dict(zip(u.tolist(), c.tolist())))})
        print(f"audit S{sid} done")
    pd.DataFrame(rows).to_csv(rfile("audit.csv"), index=False)
    print("saved", rfile("audit.csv"))


def mapping_proof(sid):
    """اثبات باگ نسخه‌ی قدیمی: بدون mapping برچسب‌ها الفبایی می‌شوند و y != 0 کلاس feet را حذف می‌کند."""
    ds = make_dataset(sid)
    ops = [Preprocessor("pick_channels", ch_names=CANONICAL_CHANNELS, ordered=True),
           Preprocessor(scale_to_microvolts, apply_on_array=True),
           Preprocessor("filter", l_freq=LOW_CUT_HZ, h_freq=HIGH_CUT_HZ)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocess(ds, ops, n_jobs=1)
    sf = TARGET_SFREQ
    kw = dict(trial_start_offset_samples=int(0.5 * sf), window_size_samples=2 * sf,
              window_stride_samples=2 * sf, drop_last_window=True, preload=True)
    _, y_old = extract_xy(create_windows_from_events(ds, **kw))
    allnames = ["left_hand", "right_hand", "hands", "feet", "rest"]
    _, y_new = extract_xy(create_windows_from_events(ds, mapping={n: i for i, n in enumerate(allnames)}, **kw))
    c_old = np.bincount(y_old)
    c_new = {n: int(np.sum(y_new == i)) for i, n in enumerate(allnames)}
    guess = {int(k): [n for n, v in c_new.items() if v == c_old[k]] for k in range(len(c_old))}
    kept = len(y_old) - c_old[0]
    lines = [f"subject {sid}", f"counts WITHOUT mapping by integer id: {c_old.tolist()}",
             f"counts WITH explicit mapping by name: {c_new}",
             f"integer id -> candidate names (matched by count; الفبایی انتظار می‌رود feet,hands,left_hand,rest,right_hand): {guess}",
             f"old code `y != 0` removes id 0 -> kept {kept} trials",
             f"rest fraction among kept trials (majority-class accuracy): {c_new['rest'] / max(1, kept):.3f}"]
    open(rfile("old_mapping_proof.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


def check_model_output(n_chans=64):
    """بررسی اینکه لایه‌ی آخر مدل softmax دارد یا نه."""
    m = build_model(n_chans, CROP, 4).eval()
    with torch.no_grad():
        o = m(torch.randn(8, n_chans, CROP, device=DEVICE))
    s = o.sum(1) if o.dim() == 2 else o.sum(1).mean(-1)
    print("output shape:", tuple(o.shape), "| row sums:", s.cpu().numpy().round(3))
    print("اگر همه‌ی جمع‌ها ≈ 1 و خروجی ≥ 0 است → MODEL_OUTPUTS_PROBS = True")


# ── Splitها (فاز ۳) ───────────────────────────────────────────────
def get_splits(sid, task, y):
    path = rfile("splits", task, f"S{sid:03d}.json")
    if os.path.exists(path):
        return json.load(open(path))
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    out = []
    for trv, te in outer.split(np.zeros(len(y)), y):
        tr, va = train_test_split(trv, test_size=INNER_VAL_FRAC, stratify=y[trv], random_state=SPLIT_SEED)
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == len(y)
        out.append({"tr": tr.tolist(), "va": va.tolist(), "te": te.tolist()})
    json.dump(out, open(path, "w"))
    return out


def get_groups():
    global GROUP_OF
    rng = np.random.RandomState(GROUP_SEED)
    perm = rng.permutation(SUBJECT_IDS)
    GROUP_OF = {int(s): int(i % N_GROUPS) for i, s in enumerate(perm)}
    json.dump(GROUP_OF, open(rfile("groups.json"), "w"))
    return GROUP_OF
# ERD/ERS مرجع فیزیولوژیک (فاز ۷) — mu/beta جدا، توان کلاس میانگین‌گیری‌شده، baseline پیش از cue
def _band_power(X, band, sfreq):
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, X, axis=-1), axis=-1)) ** 2


def erd_maps_from_epochs(X_erd, y, n_classes, sfreq=TARGET_SFREQ):
    """X_erd: [n, C, T] با cue در نمونه‌ی PRE_CUE_S*sfreq. خروجی: {band: {'erd': [K,C], 'mag': [K,C]}}."""
    t0 = int(PRE_CUE_S * sfreq)
    r0, r1 = t0 + int(ERD_REF[0] * sfreq), t0 + int(ERD_REF[1] * sfreq)
    a0, a1 = t0 + int(ERD_ACT[0] * sfreq), t0 + int(ERD_ACT[1] * sfreq)
    out = {}
    for band, rng_hz in BANDS.items():
        P = _band_power(X_erd, rng_hz, sfreq)
        erd = np.full((n_classes, X_erd.shape[1]), np.nan)
        for c in range(n_classes):
            m = y == c
            if m.sum() == 0:
                continue
            Pc = P[m].mean(0)                                   # ابتدا میانگین روی trialها
            ref, act = Pc[:, r0:r1].mean(1), Pc[:, a0:a1].mean(1)
            erd[c] = 100.0 * (act - ref) / (ref + 1e-12)        # منفی = ERD
        out[band] = {"erd": erd, "mag": np.maximum(-erd, 0.0)}
    return out


def save_erd_reference(task, sid):
    d = load_subject(sid, task)
    if d is None:
        return
    em = erd_maps_from_epochs(d["X_erd"], d["y"], len(TASKS[task]))
    np.savez_compressed(rfile("erd", task, f"S{sid:03d}.npz"),
                        **{f"{b}_{k}": em[b][k] for b in em for k in ("erd", "mag")})


# معیارهای Fidelity (فاز ۸): فقط همپوشانی با «بزرگی» تغییر وابسته به تسک
def fid_m1(A_abs, erd_mag):
    vals = []
    for c in range(A_abs.shape[0]):
        if np.all(np.isnan(A_abs[c])) or np.all(np.isnan(erd_mag[c])) or np.std(erd_mag[c]) < 1e-12:
            continue
        vals.append(spearmanr(A_abs[c], erd_mag[c])[0])
    return float(np.nanmean(vals)) if vals else np.nan


def fid_m3(a_all, erd_all, k=12):
    a = set(np.argsort(a_all)[::-1][:k])
    b = set(np.argsort(erd_all)[::-1][:k])
    return len(a & b) / len(a | b)


def fid_m2(A_abs, erd_mag, ch_names):
    """لترالیته (کلاس 0=left و 1=right). d = چپ − راست روی جفت‌های (C3,C4)...؛ «چپ» = الکترود نیمکره‌ی چپ."""
    names = [norm_ch(c) for c in ch_names]
    ds, de = [], []
    for c in (0, 1):
        for l, r in LR_PAIRS:
            if l in names and r in names:
                il, ir = names.index(l), names.index(r)
                ds.append(A_abs[c, il] - A_abs[c, ir])
                de.append(erd_mag[c, il] - erd_mag[c, ir])
    ds, de = np.array(ds), np.array(de)
    if len(ds) < 3:
        return np.nan, np.nan
    sp = spearmanr(ds, de)[0] if np.std(ds) > 0 and np.std(de) > 0 else np.nan
    return float(sp), float(np.mean(np.sign(ds) == np.sign(de)))
# ────────────────────────────────────────────────
# Attribution methods for EEGITNET temporal output
# ────────────────────────────────────────────────
def compute_attr_batch(model, xb, yb, method, ig_steps=None, sg_samples=None, sg_noise=None):
    """attribution علامت‌دار نسبت به logit کلاس هدف. خروجی [B, C, T]."""
    ig_steps = IG_STEPS if ig_steps is None else ig_steps
    sg_samples = SMOOTHGRAD_SAMPLES if sg_samples is None else sg_samples
    sg_noise = SMOOTHGRAD_NOISE_STD if sg_noise is None else sg_noise
    model.eval()
    idx = torch.arange(xb.size(0), device=xb.device)

    def grad_at(x):
        x = x.clone().detach().requires_grad_(True)
        score = model_logits(model, x)[idx, yb].sum()
        g, = torch.autograd.grad(score, x)
        return g.detach()

    if method == "grad":
        return grad_at(xb)
    if method == "grad_x_input":
        return grad_at(xb) * xb
    if method == "integrated_gradients":
        tot = torch.zeros_like(xb)
        for a in torch.linspace(0.0, 1.0, ig_steps + 1, device=xb.device)[1:]:
            tot += grad_at(a * xb)
        return xb * tot / ig_steps
    if method == "smoothgrad":
        std = xb.std(dim=(1, 2), keepdim=True).clamp_min(1e-8) * sg_noise
        tot = torch.zeros_like(xb)
        for _ in range(sg_samples):
            tot += grad_at(xb + torch.randn_like(xb) * std)
        return tot / sg_samples
    raise ValueError(method)


def compute_attr(model, X, y, methods, n_cls, bs=16):
    """X: [n, C, T≥CROP] نرمال‌شده؛ crop مرکزی. خروجی: {method: {signed[K,C], abs[K,C], abs_all[C]}}."""
    C = X.shape[1]
    s0 = (X.shape[2] - CROP) // 2
    S = {m: np.zeros((n_cls, C)) for m in methods}
    A = {m: np.zeros((n_cls, C)) for m in methods}
    TOT = {m: np.zeros(C) for m in methods}
    cnt = np.zeros(n_cls)
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs, :, s0:s0 + CROP])).float().to(DEVICE)
        yb_np = np.asarray(y[i:i + bs])
        yb = torch.from_numpy(yb_np).long().to(DEVICE)
        for m in methods:
            a = compute_attr_batch(model, xb, yb, m)
            s = a.mean(dim=2).cpu().numpy()
            ab = a.abs().mean(dim=2).cpu().numpy()
            for c in range(n_cls):
                msk = yb_np == c
                if msk.any():
                    S[m][c] += s[msk].sum(0)
                    A[m][c] += ab[msk].sum(0)
            TOT[m] += ab.sum(0)
        for c in range(n_cls):
            cnt[c] += (yb_np == c).sum()
    out = {}
    for m in methods:
        cc = np.maximum(cnt, 1)[:, None]
        sg, ab = S[m] / cc, A[m] / cc
        sg[cnt == 0] = np.nan
        ab[cnt == 0] = np.nan
        out[m] = {"signed": sg, "abs": ab, "abs_all": TOT[m] / max(1, cnt.sum())}
    return out


# Faithfulness (فاز ۹)
def model_randomization_rows(model, Xv, yv, n_cls, method=None):
    method = method or PRIMARY_ATTR
    base = compute_attr(model, Xv, yv, [method], n_cls)[method]["abs_all"]
    m2 = copy.deepcopy(model)
    leaves = [m for m in m2.modules()
              if len(list(m.children())) == 0 and len(list(m.parameters(recurse=False))) > 0]
    rows = []
    for step, mod in enumerate(reversed(leaves), 1):       # از لایه‌ی آخر به اول
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
        else:
            for p in mod.parameters(recurse=False):
                torch.nn.init.normal_(p, 0.0, 0.05)
        a = compute_attr(m2, Xv, yv, [method], n_cls)[method]["abs_all"]
        rows.append({"step": step, "n_layers_total": len(leaves),
                     "spearman_vs_original": float(spearmanr(base, a)[0])})
    return rows


def channel_deletion_rows(model, Xt, yt, scores, rng):
    """حذف (صفر کردن) k کانال بدون بازآموزی؛ ترتیب top / bottom / random."""
    C = Xt.shape[1]
    rows = []
    for k in DEL_KS:
        if k >= C:
            continue
        for order in ("top", "bottom", "random"):
            if order == "random":
                accs = []
                for _ in range(N_RANDOM_DEL):
                    idx = rng.choice(C, k, replace=False)
                    Xd = Xt.copy(); Xd[:, idx, :] = 0.0
                    accs.append(float((predict_proba(model, Xd).argmax(1) == yt).mean()))
                acc = float(np.mean(accs))
            else:
                idx = np.argsort(-scores)[:k] if order == "top" else np.argsort(scores)[:k]
                Xd = Xt.copy(); Xd[:, idx, :] = 0.0
                acc = float((predict_proba(model, Xd).argmax(1) == yt).mean())
            rows.append({"k": k, "order": order, "acc": acc})
    return rows

# ────────────────────────────────────────────────
# Main experiment runner
# ────────────────────────────────────────────────
def exp_name(task, model_name, protocol, montage):
    return f"{task}__{model_name}__{protocol}__{montage}"


# ── Pretrain گروهی (فاز ۵) ────────────────────────────────────────
def get_pretrained(task, g, ch_idx, montage, model_name):
    """مدل global گروه g: آموزش روی همه‌ی سوژه‌های گروه‌های دیگر (+EA هر سوژه)."""
    path = rfile("pretrained", f"{task}__{model_name}__{montage}__g{g}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    n_cls = len(TASKS[task])
    Xtr, ytr, Xva, yva = [], [], [], []
    for sid in SUBJECT_IDS:
        if GROUP_OF[sid] == g:
            continue
        d = load_subject(sid, task)
        if d is None:
            continue
        X = d["X_cls"][:, ch_idx]
        X = ea_apply(X, ea_matrix(X))
        y = d["y"]
        try:
            tr, va = train_test_split(np.arange(len(y)), test_size=PRETRAIN_VAL_FRAC, stratify=y, random_state=0)
        except ValueError:
            tr, va = train_test_split(np.arange(len(y)), test_size=PRETRAIN_VAL_FRAC, random_state=0)
        Xtr.append(X[tr]); ytr.append(y[tr]); Xva.append(X[va]); yva.append(y[va])
    X_tr, y_tr = np.concatenate(Xtr), np.concatenate(ytr)
    X_va, y_va = np.concatenate(Xva), np.concatenate(yva)
    del Xtr, Xva
    mu, sd = zfit(X_tr)
    X_tr -= mu; X_tr /= sd
    X_va = zapply(X_va, (mu, sd))
    print(f"pretrain g={g} montage={montage}: train={len(y_tr)} val={len(y_va)}")
    model, best = train_model(X_tr, y_tr, X_va, y_va, n_cls, seed=run_seed(0, 999, g), model_name=model_name,
                              key=(f"pre_{task}_{model_name}_{montage}", 0, g, 0),
                              max_epochs=PRETRAIN_EPOCHS, patience=PRETRAIN_PATIENCE)
    obj = {"state": best, "mu": mu, "sd": sd, "X_va": X_va, "y_va": y_va}
    torch.save(obj, path)
    return obj


def global_scores(task, g, model_name):
    """saliency مدل global گروه g روی trialهای val سوژه‌های pretrain (سوژه‌ی هدف نقشی ندارد)."""
    p = rfile("global_saliency", f"{task}__{model_name}__g{g}.npy")
    if os.path.exists(p):
        return np.load(p)
    n_cls = len(TASKS[task])
    full = np.arange(len(CANONICAL_CHANNELS))
    pre = get_pretrained(task, g, full, "full64", model_name)
    model = build_model(len(full), CROP, n_cls, model_name)
    model.load_state_dict(pre["state"])
    s = compute_attr(model, pre["X_va"], pre["y_va"], [PRIMARY_ATTR], n_cls)[PRIMARY_ATTR]["abs_all"]
    np.save(p, s)
    return s


# ── انتخاب کانال (فاز ۱۰) ─────────────────────────────────────────
def _bandpass(X, band, sfreq=TARGET_SFREQ):
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def csp_channel_scores(X, y, n_cls, n_comp=6):
    from mne.decoding import CSP
    Xf = _bandpass(X, (8.0, 30.0)).astype(np.float64)
    score = np.zeros(X.shape[1])
    Cov = np.mean([x @ x.T / x.shape[1] for x in Xf], axis=0)
    for c in range(n_cls):                                   # one-vs-rest
        csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True, cov_est="concat")
        csp.fit(Xf, (y == c).astype(int))
        W = csp.filters_[:n_comp]                            # [n_comp, C]
        A = Cov @ W.T                                        # pattern ها
        A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
        score += np.abs(A).sum(axis=1)
    return score


def rank_scores(method, task, sid, fold, d, sp, attr_exp):
    n_cls = len(TASKS[task])
    trva = np.concatenate([sp["tr"], sp["va"]])
    if method == "saliency_local":
        z = np.load(rfile("attr", attr_exp, f"S{sid:03d}_f{fold}_seed{SEEDS[0]}.npz"))
        return z[f"{PRIMARY_ATTR}__val_all"]                 # فقط inner-val
    if method == "erd_only":
        em = erd_maps_from_epochs(d["X_erd"][trva], d["y"][trva], n_cls)   # فقط outer-train
        return np.nanmean(em["mu"]["mag"] + em["beta"]["mag"], axis=0)
    X, y = d["X_cls"][trva], d["y"][trva]
    if method == "csp":
        return csp_channel_scores(X, y, n_cls)
    F_ = np.log(_bandpass(X, (8.0, 30.0)).var(axis=2) + 1e-12)
    if method == "mi":
        return mutual_info_classif(F_, y, n_neighbors=3, random_state=SPLIT_SEED)
    if method == "relieff":
        from skrebate import ReliefF
        Fz = (F_ - F_.mean(0)) / (F_.std(0) + 1e-9)
        r = ReliefF(n_features_to_select=Fz.shape[1], n_neighbors=10, n_jobs=1)
        r.fit(Fz, y)
        return r.feature_importances_
    raise ValueError(method)


def make_get_ch(method, K, task, model_name, protocol, m_idx=0):
    attr_exp = exp_name(task, model_name, protocol, "full64")
    names = [norm_ch(c) for c in CANONICAL_CHANNELS]

    def get_ch(sid, k, d, sp):
        if method == "fixed":
            miss = [c for c in FIXED_SENSORIMOTOR if norm_ch(c) not in names]
            assert not miss, f"کانال‌های fixed در داده نیستند: {miss}"
            return np.sort([names.index(norm_ch(c)) for c in FIXED_SENSORIMOTOR])
        if method == "random":
            rng = np.random.RandomState(1000 * m_idx + K)
            return np.sort(rng.choice(len(names), K, replace=False))
        if method == "saliency_global":
            sc = global_scores(task, GROUP_OF[sid], model_name)
        else:
            sc = rank_scores(method, task, sid, k, d, sp, attr_exp)
        return np.sort(np.argsort(-sc)[:K])
    return get_ch


# ── اجرای یک آزمایش روی یک سوژه (baseline / SS-TL / reduced) ─────
def attr_and_faith(model, Xn, y, va, te, n_cls, exp, sid, k, seed, faith):
    A_val = compute_attr(model, Xn[va], y[va], ATTR_METHODS, n_cls)    # برای انتخاب کانال
    A_te = compute_attr(model, Xn[te], y[te], ATTR_METHODS, n_cls)     # فقط برای گزارش fidelity
    pack = {}
    for m in ATTR_METHODS:
        for tag, A in (("val", A_val), ("te", A_te)):
            pack[f"{m}__{tag}_signed"] = A[m]["signed"]
            pack[f"{m}__{tag}_abs"] = A[m]["abs"]
            pack[f"{m}__{tag}_all"] = A[m]["abs_all"]
    np.savez_compressed(rfile("attr", exp, f"S{sid:03d}_f{k}_seed{seed}.npz"), **pack)
    if not faith:
        return
    rng = np.random.RandomState(run_seed(seed, sid, k))
    for r in channel_deletion_rows(model, Xn[te], y[te], A_val[PRIMARY_ATTR]["abs_all"], rng):
        append_row(rfile("faithfulness.csv"), {"exp": exp, "subject": sid, "fold": k, "seed": seed,
                                               "test": "deletion", "step": np.nan, "k": r["k"],
                                               "order": r["order"], "value": r["acc"]})
    if k == 0:
        for r in model_randomization_rows(model, Xn[va], y[va], n_cls):
            append_row(rfile("faithfulness.csv"), {"exp": exp, "subject": sid, "fold": k, "seed": seed,
                                                   "test": "randomization", "step": r["step"], "k": np.nan,
                                                   "order": "", "value": r["spearman_vs_original"]})


def run_subject_exp(task, sid, protocol, exp, montage, get_ch, model_name, seeds, save_attr=False, faith=False):
    assert protocol in ("scratch", "ss_tl")
    n_cls = len(TASKS[task])
    d = load_subject(sid, task)
    if d is None:
        return
    y = d["y"]
    splits = get_splits(sid, task, y)
    if subject_done(exp, sid, len(splits) * len(seeds)):
        print(f"[skip] {exp} S{sid}")
        return
    for k, sp in enumerate(splits):
        tr, va, te = np.array(sp["tr"]), np.array(sp["va"]), np.array(sp["te"])
        trva = np.concatenate([tr, va])
        ch_idx = get_ch(sid, k, d, sp)
        X = d["X_cls"][:, ch_idx]
        n_ch = len(ch_idx)
        for seed in seeds:
            rs = run_seed(seed, sid, k)
            key = (exp, sid, k, seed)
            if protocol == "scratch":
                Xn = zapply(X, zfit(X[tr]))                       # آمار فقط از inner-train
                model, _ = train_model(Xn[tr], y[tr], Xn[va], y[va], n_cls, rs, model_name, key=key)
            else:
                pre = get_pretrained(task, GROUP_OF[sid], ch_idx, montage, model_name)
                Xa = ea_apply(X, ea_matrix(X[trva]))              # R فقط از outer-train سوژه‌ی هدف
                Xn = zapply(Xa, (pre["mu"], pre["sd"]))           # آمار z-score از pretrain
                gm = build_model(n_ch, CROP, n_cls, model_name)
                gm.load_state_dict(pre["state"])
                pg = predict_proba(gm, Xn[te])
                log_metrics(exp.replace("__ss_tl__", "__global__"), task, model_name, "global", montage,
                            n_ch, sid, k, seed, y[te], pg, te)
                model, _ = train_model(Xn[tr], y[tr], Xn[va], y[va], n_cls, rs, model_name, key=key,
                                       init_state=pre["state"], lr=FT_LR, max_epochs=FT_EPOCHS,
                                       patience=FT_PATIENCE)
            probs = predict_proba(model, Xn[te])
            acc = log_metrics(exp, task, model_name, protocol, montage, n_ch, sid, k, seed, y[te], probs, te)
            print(f"{exp} S{sid:03d} f{k} seed{seed}: acc={acc * 100:.1f}%")
            if save_attr and seed == seeds[0]:
                attr_and_faith(model, Xn, y, va, te, n_cls, exp, sid, k, seed, faith)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ── Stageها ───────────────────────────────────────────────────────
def stage_baseline(task, model_name):
    exp = exp_name(task, model_name, "scratch", "full64")
    full = np.arange(len(CANONICAL_CHANNELS))
    for sid in SUBJECT_IDS:
        run_subject_exp(task, sid, "scratch", exp, "full64", lambda s, k, d, sp: full,
                        model_name, SEEDS, save_attr=True, faith=True)
        save_erd_reference(task, sid)


def stage_sstl(task, model_name):
    exp = exp_name(task, model_name, "ss_tl", "full64")
    full = np.arange(len(CANONICAL_CHANNELS))
    for sid in SUBJECT_IDS:
        run_subject_exp(task, sid, "ss_tl", exp, "full64", lambda s, k, d, sp: full,
                        model_name, SEEDS, save_attr=True, faith=False)


def stage_reduced(task, model_name):
    for K in K_LIST:
        for method, protocol in REDUCED_SPECS:
            if method == "fixed" and K != len(FIXED_SENSORIMOTOR):
                continue
            if method == "random" and K != K_MAIN:
                continue
            for m_idx in range(N_RANDOM_MONTAGES if method == "random" else 1):
                montage = f"{method}{m_idx if method == 'random' else ''}_k{K}"
                exp = exp_name(task, model_name, protocol, montage)
                get_ch = make_get_ch(method, K, task, model_name, protocol, m_idx)
                seeds = SEEDS[:1] if method == "random" else SEEDS
                print(f"\n=== {exp} ===")
                for sid in SUBJECT_IDS:
                    run_subject_exp(task, sid, protocol, exp, montage, get_ch, model_name, seeds)


def stage_fidelity(task, model_name):
    rows = []
    for protocol in ("scratch", "ss_tl"):
        exp = exp_name(task, model_name, protocol, "full64")
        for sid in SUBJECT_IDS:
            pe = os.path.join(RESULTS_DIR, "erd", task, f"S{sid:03d}.npz")
            if not os.path.exists(pe):
                continue
            zr = np.load(pe)
            for k in range(OUTER_FOLDS):
                pa = os.path.join(RESULTS_DIR, "attr", exp, f"S{sid:03d}_f{k}_seed{SEEDS[0]}.npz")
                if not os.path.exists(pa):
                    continue
                z = np.load(pa)
                for m in ATTR_METHODS:
                    for band in BANDS:
                        mag = zr[f"{band}_mag"]
                        m2s, m2g = fid_m2(z[f"{m}__te_abs"], mag, CANONICAL_CHANNELS)
                        rows.append({"exp": exp, "protocol": protocol, "subject": sid, "fold": k,
                                     "method": m, "band": band,
                                     "m1_spearman": fid_m1(z[f"{m}__te_abs"], mag),
                                     "m2_spearman": m2s, "m2_sign": m2g,
                                     "m3_jaccard": fid_m3(z[f"{m}__te_all"], np.nanmean(mag, axis=0), K_MAIN)})
    pd.DataFrame(rows).to_csv(rfile("fidelity.csv"), index=False)
    print("saved", rfile("fidelity.csv"), len(rows), "rows")


def stage_figures(task, model_name):
    info = get_topo_info(CANONICAL_CHANNELS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for protocol in ("scratch", "ss_tl"):
        exp = exp_name(task, model_name, protocol, "full64")
        vecs = []
        for sid in SUBJECT_IDS:
            fs = [np.load(os.path.join(RESULTS_DIR, "attr", exp, f"S{sid:03d}_f{k}_seed{SEEDS[0]}.npz"))
                  [f"{PRIMARY_ATTR}__te_all"] for k in range(OUTER_FOLDS)
                  if os.path.exists(os.path.join(RESULTS_DIR, "attr", exp, f"S{sid:03d}_f{k}_seed{SEEDS[0]}.npz"))]
            if fs:
                vecs.append(_normalize_importance(np.mean(fs, axis=0)))
        if vecs:
            plot_saliency_topomap(np.mean(vecs, axis=0), info,
                                  f"GROUP mean |{PRIMARY_ATTR}| ({protocol}, n={len(vecs)}; per-subject max-normalised)",
                                  os.path.join(OUTPUT_DIR, f"group_saliency_{protocol}.png"))
    er = [np.load(os.path.join(RESULTS_DIR, "erd", task, f"S{s:03d}.npz"))["mu_erd"] for s in SUBJECT_IDS
          if os.path.exists(os.path.join(RESULTS_DIR, "erd", task, f"S{s:03d}.npz"))]
    if er:
        er = np.nanmean(er, axis=0)
        plot_erd_ers_topomaps(er[0], er[1], er[0] - er[1], 0.5 * (er[0] + er[1]), info,
                              os.path.join(OUTPUT_DIR, "group_erd_mu.png"))


def stage_bugctl(model_name):
    """کنترل فاز ۱: mapping قدیمی (hands/left/rest/right) در برابر mapping جدید روی سوژه‌های smoke."""
    full = np.arange(len(CANONICAL_CHANNELS))
    for task in ("4c_OLDBUG", "4c_LRHF"):
        exp = exp_name(task, model_name, "scratch", "full64")
        for sid in SUBJECT_IDS:
            run_subject_exp(task, sid, "scratch", exp, "full64", lambda s, k, d, sp: full,
                            model_name, SEEDS[:1])
    df = pd.read_csv(rfile("metrics.csv"))
    for task in ("4c_OLDBUG", "4c_LRHF"):
        sub = df[df["task"] == task]
        print(task, "mean acc = %.1f%%" % (100 * sub["acc"].mean()))
        for sid in SUBJECT_IDS:
            d = load_subject(sid, task)
            if d is not None:
                print(f"   S{sid}: majority-class rate = {np.bincount(d['y']).max() / len(d['y']) * 100:.1f}%")


# ── آمار نهایی (فاز ۱۱) ───────────────────────────────────────────
def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def boot_ci(x, n=None, seed=0):
    n = N_BOOT if n is None else n
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tost_paired(d_pp, margin):
    d = np.asarray(d_pp, float)
    n, m = len(d), d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    p_low = 1 - t_dist.cdf((m + margin) / se, n - 1)
    p_up = t_dist.cdf((m - margin) / se, n - 1)
    h = t_dist.ppf(0.95, n - 1) * se
    return m, m - h, m + h, max(p_low, p_up)


def check_duplicate_predictions():
    seen, dup = {}, []
    base = os.path.join(RESULTS_DIR, "preds")
    for exp in sorted(os.listdir(base)) if os.path.exists(base) else []:
        for f in sorted(os.listdir(os.path.join(base, exp))):
            z = np.load(os.path.join(base, exp, f))
            h = hashlib.md5(z["y_prob"].tobytes() + z["y_true"].tobytes()).hexdigest()
            if h in seen and seen[h][0] != exp:
                dup.append({"hash": h, "a": "/".join(seen[h]), "b": f"{exp}/{f}"})
            seen.setdefault(h, (exp, f))
    pd.DataFrame(dup).to_csv(rfile("duplicate_predictions.csv"), index=False)
    print("prediction hashes: duplicates across experiments =", len(dup))
    return dup


def stage_stats(task, model_name):
    df = pd.read_csv(rfile("metrics.csv"))
    df = df[df["task"] == task].copy()
    df["correct"] = df["acc"] * df["n_test"]
    g1 = df.groupby(["exp", "protocol", "montage", "n_ch", "subject", "seed"], as_index=False).agg(
        correct=("correct", "sum"), n=("n_test", "sum"))
    g1["acc"] = g1["correct"] / g1["n"]
    per = g1.groupby(["exp", "protocol", "montage", "n_ch", "subject"], as_index=False).agg(
        acc=("acc", "mean"), n=("n", "mean"), acc_sd_over_seeds=("acc", "std"))
    per.to_csv(rfile("subject_accuracy.csv"), index=False)
    summ = per.groupby(["exp", "protocol", "montage", "n_ch"]).agg(
        mean_acc=("acc", "mean"), sd_acc=("acc", "std"), n_subjects=("subject", "nunique")).reset_index()
    for i, r in summ.iterrows():                                   # بازمحاسبه‌ی میانگین از ردیف‌های per-subject
        chk = per[per["exp"] == r["exp"]]["acc"].mean()
        assert abs(chk - r["mean_acc"]) < 1e-12
    summ.to_csv(rfile("summary_accuracy.csv"), index=False)
    n_cls = len(TASKS[task])

    # chance و آستانه‌ی معناداری هر سوژه
    ch = per.copy()
    ch["thr"] = [binom.ppf(0.95, int(round(n)), 1 / n_cls) / int(round(n)) for n in ch["n"]]
    ch["above"] = ch["acc"] > ch["thr"]
    ch.groupby("exp").agg(frac_above_chance=("above", "mean"), mean_acc=("acc", "mean")).reset_index() \
        .to_csv(rfile("chance_summary.csv"), index=False)

    # TOST: reduced در برابر full64 (همان پروتکل)
    rows = []
    for exp in per["exp"].unique():
        if exp.endswith("__full64"):
            continue
        full_exp = exp.rsplit("__", 1)[0] + "__full64"
        if full_exp not in set(per["exp"]):
            continue
        a = per[per["exp"] == exp].set_index("subject")["acc"]
        b = per[per["exp"] == full_exp].set_index("subject")["acc"]
        j = pd.concat([a, b], axis=1, keys=["red", "full"]).dropna()
        if len(j) < 3:
            continue
        d = (j["red"] - j["full"]).values * 100
        m, lo, hi, p = tost_paired(d, TOST_MARGIN_PP)
        rows.append({"exp": exp, "vs": full_exp, "n": len(j), "mean_diff_pp": m, "ci90_lo": lo, "ci90_hi": hi,
                     "p_tost": p, "equivalent": bool(p < 0.05),
                     "pct_subjects_loss_gt_margin": float(np.mean(d < -TOST_MARGIN_PP) * 100)})
    pd.DataFrame(rows).to_csv(rfile("tost.csv"), index=False)

    # مقایسه‌ی روش‌های انتخاب (جفتی، Holm)
    comp = []
    for protocol in per["protocol"].unique():
        sub = per[(per["protocol"] == protocol) & (per["montage"] != "full64")].copy()
        sub["method"] = sub["montage"].str.replace(r"\d+_k", "_k", regex=True)
        sub["K"] = sub["montage"].str.extract(r"_k(\d+)")[0].astype(int)
        for K, sk in sub.groupby("K"):
            wide = sk.groupby(["subject", "method"])["acc"].mean().unstack()
            for a_, b_ in __import__("itertools").combinations(wide.columns, 2):
                jj = wide[[a_, b_]].dropna()
                if len(jj) >= 6 and (jj[a_] - jj[b_]).abs().sum() > 0:
                    comp.append({"protocol": protocol, "K": K, "A": a_, "B": b_, "n": len(jj),
                                 "mean_diff_pp": float((jj[a_] - jj[b_]).mean() * 100),
                                 "p": float(wilcoxon(jj[a_], jj[b_]).pvalue)})
    if comp:
        cdf = pd.DataFrame(comp)
        cdf["p_holm"] = holm(cdf["p"].values)
        cdf.to_csv(rfile("selection_comparison.csv"), index=False)

    # fidelity
    fp = os.path.join(RESULTS_DIR, "fidelity.csv")
    if os.path.exists(fp):
        fd = pd.read_csv(fp)
        ps = fd.groupby(["exp", "method", "band", "subject"], as_index=False)[
            ["m1_spearman", "m2_spearman", "m2_sign", "m3_jaccard"]].mean()
        null = {"m1_spearman": 0.0, "m2_spearman": 0.0, "m2_sign": 0.5,
                "m3_jaccard": K_MAIN / (2 * len(CANONICAL_CHANNELS) - K_MAIN)}
        out = []
        for (exp, m, band), gg in ps.groupby(["exp", "method", "band"]):
            for met, nv in null.items():
                x = gg[met].dropna().values
                if len(x) < 6:
                    continue
                zmean = np.tanh(np.mean(np.arctanh(np.clip(x, -0.999, 0.999)))) if "spearman" in met else x.mean()
                lo, hi = boot_ci(x)
                try:
                    p = float(wilcoxon(x - nv).pvalue)
                except ValueError:
                    p = np.nan
                out.append({"exp": exp, "method": m, "band": band, "metric": met, "n": len(x),
                            "mean": float(zmean), "ci95_lo": lo, "ci95_hi": hi, "null": nv, "p": p})
        if out:
            o = pd.DataFrame(out)
            o["p_holm"] = holm(np.nan_to_num(o["p"].values, nan=1.0))
            o.to_csv(rfile("fidelity_summary.csv"), index=False)

    # faithfulness
    pf = os.path.join(RESULTS_DIR, "faithfulness.csv")
    if os.path.exists(pf):
        ff = pd.read_csv(pf)
        dl = ff[ff["test"] == "deletion"].groupby(["exp", "order", "k"])["value"].agg(["mean", "std", "count"])
        dl.reset_index().to_csv(rfile("faithfulness_deletion_summary.csv"), index=False)
        rn = ff[ff["test"] == "randomization"].groupby(["exp", "step"])["value"].agg(["mean", "std", "count"])
        rn.reset_index().to_csv(rfile("faithfulness_randomization_summary.csv"), index=False)

    # confusion matrix تجمعی (seed اول) و توزیع predictionها
    for exp in per["exp"].unique():
        yt, yp = [], []
        pdir = os.path.join(RESULTS_DIR, "preds", exp)
        for f in sorted(os.listdir(pdir)):
            if f.endswith(f"seed{SEEDS[0]}.npz"):
                z = np.load(os.path.join(pdir, f))
                yt.append(z["y_true"]); yp.append(z["y_prob"].argmax(1))
        if yt:
            cm = confusion_matrix(np.concatenate(yt), np.concatenate(yp), labels=list(range(n_cls)))
            pd.DataFrame(cm, index=TASKS[task], columns=TASKS[task]).to_csv(rfile("confusion", f"{exp}.csv"))
    check_duplicate_predictions()
    print("آمار در", RESULTS_DIR, "ذخیره شد.")


# ── Self-tests (جایگزین pytest، داخل همین فایل) ───────────────────
def selftest():
    # 1) mapping صریح و پیوسته
    for t, names in TASKS.items():
        mp = {n: i for i, n in enumerate(names)}
        assert sorted(mp.values()) == list(range(len(names))), t
    # 2) ERD مصنوعی: دامنه بعد از cue نصف → ERD ≈ -75%
    sf = TARGET_SFREQ
    T = int((PRE_CUE_S + POST_CUE_S) * sf)
    t = np.arange(T) / sf - PRE_CUE_S
    sig = np.sin(2 * np.pi * 10 * t) * np.where(t < 0, 1.0, 0.5)
    X = np.tile(sig, (30, 1, 1)) + 0.01 * np.random.RandomState(0).randn(30, 1, T)
    em = erd_maps_from_epochs(X, np.zeros(30, int), 1)
    assert abs(em["mu"]["erd"][0, 0] + 75) < 5, em["mu"]["erd"]
    # 3) EA: میانگین کوواریانس بعد از EA ≈ I
    Xr = np.random.RandomState(1).randn(40, 8, 200) * np.linspace(1, 5, 8)[None, :, None]
    Xa = ea_apply(Xr, ea_matrix(Xr))
    R = np.mean([x @ x.T / x.shape[1] for x in Xa], axis=0)
    assert np.allclose(R, np.eye(8), atol=1e-3)
    # 4) splitها ناهم‌پوشان
    y = np.repeat(np.arange(4), 25)
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=0)
    for trv, te in outer.split(np.zeros(len(y)), y):
        tr, va = train_test_split(trv, test_size=INNER_VAL_FRAC, stratify=y[trv], random_state=0)
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == len(y)
    # 5) completeness انتگرال‌گیری گرادیان (IG)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4 * 32, 16), torch.nn.Tanh(),
                              torch.nn.Linear(16, 3)).to(DEVICE).eval()
    x = torch.randn(4, 4, 32, device=DEVICE)
    yb = torch.tensor([0, 1, 2, 0], device=DEVICE)
    attr = compute_attr_batch(net, x, yb, "integrated_gradients", ig_steps=400)
    lhs = attr.sum(dim=(1, 2))
    idx = torch.arange(4, device=DEVICE)
    rhs = net(x)[idx, yb] - net(torch.zeros_like(x))[idx, yb]
    assert torch.allclose(lhs, rhs, atol=5e-3), (lhs, rhs)
    # 6) شکل attribution برای n_cls دلخواه
    out = compute_attr_batch(net, x, yb, "grad")
    assert out.shape == x.shape
    print("selftest: همه‌ی تست‌ها گذشتند ✔")

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="audit",
                    choices=["audit", "selftest", "bugctl", "baseline", "sstl", "fidelity",
                             "figures", "reduced", "stats"])
    ap.add_argument("--task", default=TASK, choices=list(TASKS))
    ap.add_argument("--model", default=MODEL_NAME, choices=["eegitnet", "eegnetv4"])
    ap.add_argument("--smoke", action="store_true", help="۵ سوژه، epoch کم، پوشه‌ی results_smoke")
    ap.add_argument("--subjects", default="", help="مثلاً 1,2,3")
    args = ap.parse_args()

    TASK, MODEL_NAME = args.task, args.model
    if args.smoke:
        SUBJECT_IDS = SMOKE_SUBJECTS[:]
        SEEDS = [0]
        MAX_EPOCHS, PATIENCE = 20, 10
        PRETRAIN_EPOCHS, PRETRAIN_PATIENCE = 5, 5
        FT_EPOCHS, FT_PATIENCE = 5, 5
        IG_STEPS, SMOOTHGRAD_SAMPLES = 8, 4
        N_RANDOM_MONTAGES, N_RANDOM_DEL = 2, 3
        RESULTS_DIR = "results_smoke"
    elif args.subjects:
        SUBJECT_IDS = [int(s) for s in args.subjects.split(",")]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mne.set_log_level("ERROR")                     # حذف پیام‌های تکراری MNE

    if args.stage == "selftest":
        selftest()
        sys.exit(0)

    CANONICAL_CHANNELS = get_canonical_channels()
    get_groups()
    print(f"stage={args.stage} task={TASK} model={MODEL_NAME} subjects={len(SUBJECT_IDS)} "
          f"n_channels={len(CANONICAL_CHANNELS)} results={RESULTS_DIR}")

    if args.stage == "audit":
        run_audit(SUBJECT_IDS)
        mapping_proof(SUBJECT_IDS[0])
        check_model_output(len(CANONICAL_CHANNELS))
    elif args.stage == "bugctl":
        stage_bugctl(MODEL_NAME)
    elif args.stage == "baseline":
        stage_baseline(TASK, MODEL_NAME)
    elif args.stage == "sstl":
        stage_sstl(TASK, MODEL_NAME)
    elif args.stage == "fidelity":
        stage_fidelity(TASK, MODEL_NAME)
    elif args.stage == "figures":
        stage_figures(TASK, MODEL_NAME)
    elif args.stage == "reduced":
        stage_reduced(TASK, MODEL_NAME)
    elif args.stage == "stats":
        stage_stats(TASK, MODEL_NAME)

