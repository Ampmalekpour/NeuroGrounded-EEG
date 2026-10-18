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
  `<class>/results_session/`). The invalid first session run is kept in `*_invalid_v1/` folders.

## Protocol (default `session`)

- **Duplicated recordings are dropped.** For subjects 6–14, MOABB's "test" file is an exact copy of the
  training file (see `analysis/hgd_data_check.csv`). Testing on it would mean testing on training data.
- **Test set, one rule for all 14 subjects:**
  - subjects 1–5: the genuine official test run (160 trials in 4-class, 80 in 2-class);
  - subjects 6–14: the chronologically last 160 trials (80 in 2-class) of the single recording, at most
    20 % of the trials (subject 11 has only 160 trials, so 32).
- **Validation** = the last 25 % of the remaining training trials, in time order. It is used for
  checkpoints, the learning-rate schedule, the profile choice and the saliency.
- **Checkpoint** = the highest validation accuracy, never before epoch 20. The lowest-loss rule had kept
  the untrained epoch-1 model for ~15 % of the 2-class seeds.
- **Refit** on all training trials (the plateau learning rates are replayed), then test once. Seeds 42,
  43 and 44 are averaged.
- The test trials are never used for any choice. CSP/MI/ReliefF see only the training and validation
  trials. Montages are leave-one-subject-out.

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
