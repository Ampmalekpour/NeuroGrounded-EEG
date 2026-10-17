# EEG-ITNet on HGD (High-Gamma Dataset, Schirrmeister2017) — second revision

14 subjects, 22 sensorimotor channels, **executed** movements (not imagined).
`2class` = left vs right hand, `4class` = feet / left hand / rest / right hand.

## Run (Windows PowerShell, repository root)

```powershell
cd D:\EEG\NeuroGrounded-EEG
python run_all.py --dataset hgd --models eegitnet
```

- Steps, both class modes in turn: tune, saliency, csp, relieff, mi, controls, then the tables.
- If it stops, run the same command again; finished steps are skipped.
- Results: `EEGItNet/HGD/report_session/REPORT.md` and `reports/HGD_session/SUMMARY.md`.
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

- **No 4-class version:** the script is 2-class only, which explains why the paper's 4-class EEG-ITNet HGD
  table equals the 2-class one (reviewer comment 1). 4-class is now run properly.
- **Wrong classes:** the CSP/MI/ReliefF scripts train feet vs left hand (labels `[0, 1]`), so the baselines
  were not comparable with the saliency run.
- **8-s windows**, **test data in selection** (as for EEGNetv4), post-cue ERD baseline, no standardisation.

Full list: `docs/REVIEW_CROSSWALK.md`, section I.

## Training recipes (picked per subject on validation data only, by `*_tune.py`)

| Profile | Settings | Source |
|---|---|---|
| `original` | 100 Hz, 4–38 Hz, no standardisation, 0–4 s, dropout 0.3, AdamW 5e-4 / 1e-4, plateau (0.5, 30), ≤600 epochs, patience 150, batch 96, label smoothing 0.1, clip 10 | your script / paper, window fixed |
| `ems` | `original` + exponential moving standardisation | braindecode standard |
| `salami128` | 128 Hz (inception kernels 125/250/500 ms), EMS, dropout 0.4 | EEG-ITNet paper (Salami et al. 2022) |
