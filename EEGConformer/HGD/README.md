# EEGConformer on HGD (High-Gamma Dataset, Schirrmeister2017) — second revision

14 subjects, 22 sensorimotor channels, **executed** movements (not imagined).
`2class` = left vs right hand, `4class` = feet / left hand / rest / right hand.

## Run (Windows PowerShell, repository root)

```powershell
cd D:\EEG\NeuroGrounded-EEG
python run_all.py --dataset hgd --models eegconformer
```

- Steps, both class modes in turn: tune, saliency, csp, relieff, mi, controls, then the tables.
- If it stops, run the same command again; finished steps are skipped.
- Results: `EEGConformer/HGD/report_session/REPORT.md` and `reports/HGD_session/SUMMARY.md`.
- Your first-revision HGD results stay in `<class>/results/` and are never touched (new runs write to
  `<class>/results_session/`).

## Protocol (default `session` = the official HGD split)

- Train on the official **training runs** (~880 trials in 4-class, ~440 in 2-class).
- Validation = the last 25 % of the training runs, in time order. It is used for early stopping, the
  learning-rate schedule, the profile choice and the saliency.
- The model is then refitted on all training runs, and the plateau learning rates are replayed.
- Test once on the official **test runs** (~160 / ~80 trials). Seeds 42, 43, 44 are averaged.
- The test runs are never used for any choice, and CSP/MI/ReliefF see the training runs only.
- Montages are leave-one-subject-out.

## What was wrong in the original scripts (`legacy_unrevised/original_scripts/`)

- **Test fold used for training choices:** the LR scheduler ran on test accuracy, and the "best" checkpoint
  came from the test fold. Some files also kept a shallow `state_dict().copy()`, so "best" was really the
  last epoch. Both now use **validation loss** only.
- **Test data in selection:** saliency on the test fold, one global montage, methods chosen by ERD
  correlation. The REST script re-used a hard-coded channel list picked on the same subjects.
- The first script's integrated gradients and SmoothGrad were all zeros (non-leaf tensor).
- **Post-cue ERD baseline** (0–0.4 s).
- The reduced-channel files re-referenced differently from the saliency files.

Full list: `docs/REVIEW_CROSSWALK.md`, section I.

## Training recipes (picked per subject on validation data only, by `*_tune.py`)

| Profile | Settings | Source |
|---|---|---|
| `original` | 22 channels average-referenced, 500 Hz, 4–125 Hz Butterworth, EMS (instead of a per-fold z-score), 0–4 s, AdamW 1e-3 / 1e-4, plateau (0.5, 10) on **val** loss, ≤100 epochs, batch 32 + segmentation-reconstruction, no smoothing; 4-class depth 6 / 10 heads / dropout 0.5, 2-class depth 2 / 4 heads / dropout 0.6 | your scripts, leakage fixed |
| `song2022` | 250 Hz, 4–40 Hz, Adam lr 2e-4, betas (0.5, 0.999), no weight decay, constant LR, batch 72 + segmentation-reconstruction, ≤300 epochs | EEGConformer paper (Song et al. 2022) |

CSP uses a small regulariser (1e-4) because average-referenced data are rank-deficient.
