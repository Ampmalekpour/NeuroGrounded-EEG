# Deep4Net on HGD (High-Gamma Dataset, Schirrmeister2017) — second revision

14 subjects, 22 sensorimotor channels, **executed** movements (not imagined).
`2class` = left vs right hand, `4class` = feet / left hand / rest / right hand.

## Run (Windows PowerShell, repository root)

```powershell
cd D:\EEG\NeuroGrounded-EEG
python run_all.py --dataset hgd --models deep4net
```

- Steps, both class modes in turn: tune, saliency, csp, relieff, mi, controls, then the tables.
- If it stops, run the same command again; finished steps are skipped.
- Results: `Deep4net/HGD/report_session/REPORT.md` and `reports/HGD_session/SUMMARY.md`.
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

- **Wrong classes:** every file trains feet vs left hand (labels `[0, 1]`), not left vs right.
- **No 4-class version:** the main script is 2-class only.
- **8-s windows**, **test data in selection** (saliency on the test fold, one global montage, methods
  chosen by ERD correlation). CSP summed all 22 patterns, so its ranking was nearly uninformative.
- **Post-cue ERD baseline.**

Full list: `docs/REVIEW_CROSSWALK.md`, section I.

## Training recipes (picked per subject on validation data only, by `*_tune.py`)

| Profile | Settings | Source |
|---|---|---|
| `original` | 100 Hz, 4–38 Hz, EMS, 0–4 s, dropout 0.3, AdamW 5e-4 / 1e-4, plateau (0.5, 30), ≤600 epochs, patience 150, batch 96, label smoothing 0.1, clip 10 | your scripts / paper, window fixed |
| `schirrmeister` | 250 Hz, EMS, −0.5–4 s, dropout 0.5, AdamW 1e-2 / 5e-4, cosine, ≤300 epochs, batch 64 | Schirrmeister et al. 2017 / braindecode HGD setting |
| `cosine100` | the `schirrmeister` optimiser at 100 Hz (isolates the optimiser change) | — |
