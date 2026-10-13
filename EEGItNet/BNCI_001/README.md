# EEG-ITNet on BNCI2014-001 (BCI Competition IV-2a)

Same structure, steps, tables and physiology block as `EEGNetV4/BNCI_001` (see its README).
`2class` = left vs right hand, `4class` = feet / left hand / right hand / tongue.

## Run

```powershell
cd D:\EEG\NeuroGrounded-EEG
$env:EEG_SEEDS="42,43,44"
python run_all.py                      # EEGNetV4 first, then EEG-ITNet
python run_all.py --models eegitnet    # only EEG-ITNet
```

Results: `EEGItNet/BNCI_001/report/REPORT.md` and `reports/BNCI_001/SUMMARY.md` (both models side by side).

## What is kept from your original scripts (profile `original`)

| Setting | Value |
|---|---|
| Preprocessing | 250 Hz, Chebyshev-I (order 6, 0.5 dB) 4–38 Hz IIR, exponential moving standardisation |
| Input window | −0.5 … 4.0 s around the cue (1125 samples) |
| Training | 350 epochs, batch 64 (drop_last), AdamW lr 1e-3 / wd 1e-3, cosine annealing, grad clip 10 |
| Regularisation | noise / scaling / channel-dropout augmentation, mixup (p 0.5, α 0.2), label smoothing 0.1 |
| Final model | SWA average from epoch 250 + BatchNorm update |
| Prediction | test-time augmentation (input, +2 % noise, ×0.9; mean softmax) |
| Attributions | grad, grad×input, integrated gradients (32 steps), SmoothGrad (12 × 10 %) |

## What changed and why (details: `docs/REVIEW_CROSSWALK.md`, section G)

| Original | Now | Review |
|---|---|---|
| saliency on the **test** fold | saliency on the **validation** block | 8 |
| one global montage from all 9 subjects | leave-one-subject-out montage per subject | 5, 12 |
| top-3 attribution methods chosen by ERD_(L−R) correlation | all four methods averaged (choice does not depend on ERD) | 8 |
| shuffled StratifiedKFold(4), no validation set | 4 chronological blocks: train / validation / test | 4, 13 |
| ERD from band-passed, EMS-standardised data | ERD from raw µV data, pre-cue baseline, same maps for every model | 7 |
| 4-class only in `et_v66.py` (accuracy only) | 2-class and 4-class with the full pipeline | 1 |
| accuracy only | predictions, confusion matrices, class counts, several seeds, TOST tables | 10, 14, 15 |

**Note on accuracy:** the chronological blocks train on half of each subject's trials (the original used 75 %,
shuffled). Expect lower but more honest numbers. The tuning step can only choose among profiles; it cannot undo
this protocol change.

## Fair tuning

`*_tune.py` compares these profiles on **validation** accuracy only (leave-one-subject-out choice per subject),
and the chosen profile is used for every method:

| Profile | Change |
|---|---|
| `original` | your settings |
| `cue_0_4` | window 0–4 s (no pre-cue samples) |
| `no_aug` | no augmentation, no mixup |
| `drop_0.25` | dropout 0.25 instead of 0.4 |

EEG-ITNet has no early stopping (the SWA model is the final model), so tuning costs
4 profiles × 9 subjects × 4 folds × 350 epochs per class mode. To save time:
`$env:EEG_TUNE_CANDIDATES="original,no_aug"`, or skip tuning with `python run_all.py --no-tune`.
