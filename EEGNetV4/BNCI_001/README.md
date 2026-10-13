# EEGNetv4 on BNCI2014-001 (BCI Competition IV-2a)

9 subjects, 22 EEG channels, motor **imagery**, 576 trials per subject (2 sessions × 288).
`2class` = left vs right hand, `4class` = feet / left hand / right hand / tongue.

## Run everything (one command)

```powershell
cd D:\EEG\NeuroGrounded-EEG\EEGNetV4\BNCI_001
$env:EEG_SEEDS="42,43,44"
python run_all.py
```

`run_all.py` runs, for `2class` and then `4class`:

| Step | Script | What it produces |
|---|---|---|
| 1 tune | `*_tune.py` | fair choice of the training profile (validation data only) |
| 2 saliency | `*_saliency.py` | "22ch" + "Ours", ERD/ERS maps, alignment metrics, deletion + randomisation checks, topomaps |
| 3 csp | `*_csp.py` | "CSP" row |
| 4 relieff | `*_relieff.py` | "RLF" row |
| 5 mi | `*_mi.py` | "MI" row |
| 6 controls | `*_controls.py` | fixed sensorimotor, 5 random, ERD-only montages |
| 7 aggregate | `analysis/aggregate_results.py` | `report/REPORT.md` + CSV tables + figures |

- Logs: `<class>/logs/<step>.log`; progress: `run_all_status.json`.
- If it stops, run it again: finished steps are skipped (`--force` redoes them).
- Useful options: `--classes 4class`, `--steps saliency csp`, `--steps aggregate` (tables only), `--no-tune`.
- Dry run without data (minutes, CPU): `$env:EEG_SYNTHETIC="1"; $env:EEG_SUBJECTS="1,2,3"; $env:EEG_MAX_EPOCHS="2"`.

## Reviewer tables (`report/`)

| File | Content | Review |
|---|---|---|
| `T1_accuracy_per_subject.csv` | accuracy per subject × method, mean, SD | 1, 2, 11 |
| `T2_montages.csv` | channels and profile per subject × method | 5, 12 |
| `T3_equivalence_vs_22ch.csv` | paired TOST, 90% CI, worst subject, margin sensitivity | 10 |
| `T4_ours_vs_other_montages.csv` | Ours vs CSP/RLF/MI/controls, paired t + Wilcoxon, Holm | 11, 12 |
| `T5_confusion_*.csv` | confusion matrices, recall, majority baseline, collapsed folds | 15 |
| `T6_seed_stability.csv` | grand mean per seed | 14 |
| `T7_alignment.csv` | saliency–ERD alignment, participant level, 95% CI | 7, 9 |
| `T8_deletion_test.csv`, `T8_randomisation_check.csv` | faithfulness checks | 8 |
| `T9_config.json`, `selected_profile.json` | exact configuration and chosen profile | 13, 14 |

## Best accuracy, fairly

"Fair" here means three things:
- settings are chosen **without test data**;
- the **same settings are used for every method** (22ch, Ours, CSP, RLF, MI, controls);
- the choice is documented.

`*_tune.py` does exactly that. Each candidate profile is trained on 22 channels, and only its
**validation** accuracy is recorded; the test block is never scored. Each subject then gets the profile
that is best on the **other** subjects (leave-one-subject-out), so a subject's own data never shape its
settings. The choice is saved to `results/selected_profile.json` and used by every later step.

Candidates (`PROFILES` in `eegnetv4_bnci_common.py`):

| Profile | Change vs your original script | Why |
|---|---|---|
| `original` | — (4–40 Hz, 0.5–2.5 s, kernel 32, dropout 0.5, AdamW 2e-3, cosine, 800 ep / patience 120) | your settings |
| `k64` | temporal kernel 64 | EEGNet paper: kernel = fs/2 at 128 Hz |
| `k64_long` | + window 0.5–4.0 s | imagery lasts until 4 s after the cue |
| `k64_aug` | + mixup, amplitude scaling, noise, channel dropout | the augmentation your manuscript describes (Sec. 3.4) |
| `k64_long_aug` | both | |

What I deliberately did **not** do, because it would not be fair:
- tune on test accuracy;
- give different methods different settings;
- tune preprocessing to raise the saliency–ERD correlation. `test_for_cor.py` does this (8–30 Hz,
  average reference, dropout 0.25, active-window saliency); it optimises the quantity you then report,
  which is the circularity in review comment 8.

Bigger levers that would change the protocol (not done, your decision):
- **Session-wise evaluation:** the standard IV-2a split (train on session T, test on session E) makes
  results comparable with the literature.
- **More training data per fold:** currently 288 trials (2 of the 4 blocks).

Use **3 or more seeds** (`EEG_SEEDS`).

`run_all.py --no-tune` or `EEG_PROFILE=original` reproduces your original settings exactly.
