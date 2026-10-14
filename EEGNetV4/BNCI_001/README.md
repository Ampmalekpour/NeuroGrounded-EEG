# EEGNetv4 on BNCI2014-001 (BCI Competition IV-2a)

9 subjects, 22 EEG channels, motor **imagery**, 576 trials per subject (2 sessions × 288).
`2class` = left vs right hand, `4class` = feet / left hand / right hand / tongue.

## Run everything (one command, from the repository root)

```powershell
cd D:\EEG\NeuroGrounded-EEG
python run_all.py                      # EEGNetV4 -> EEG-ITNet -> Deep4Net -> EEGConformer, session protocol
python run_all.py --models eegnetv4    # only this model
```

**Protocol** (`--protocol`, default `session`):
- `session`: train on session 1, validate on its last 25 %, refit on all of session 1, test once on
  session 2, average the seeds (default seeds 42, 43, 44). This is the official BCI IV-2a split.
  Output goes to `<class>/results_session/`, `<class>/logs_session/`, `report_session/` and
  `reports/BNCI_001_session/`.
- `blocks`: the first-revision protocol (4 chronological blocks). Output goes to `results/`, `logs/`,
  `report/` and `reports/BNCI_001/`. The two never overwrite each other.

For each model, `2class` then `4class`:

| Step | Script | What it produces |
|---|---|---|
| 1 tune | `*_tune.py` | fair choice of the training profile (validation data only) |
| 2 saliency | `*_saliency.py` | "22ch" + "Ours", ERD/ERS maps, saliency–ERD correlations, deletion + randomisation checks, exploratory physiology correlations, topomaps |
| 3 csp | `*_csp.py` | "CSP" row |
| 4 relieff | `*_relieff.py` | "RLF" row |
| 5 mi | `*_mi.py` | "MI" row |
| 6 controls | `*_controls.py` | fixed sensorimotor, 5 random, ERD-only montages |
| 7 aggregate | `analysis/aggregate_results.py` | `report*/REPORT.md` + CSV tables + figures, and a cross-model summary |

- If it stops, run it again: finished steps are skipped (`--force` redoes them).
- Useful options: `--models eegnetv4 deep4net`, `--classes 4class`, `--steps saliency csp`, `--steps aggregate`, `--no-tune`.
- Dry run without data (CPU): `$env:EEG_SYNTHETIC="1"; $env:EEG_SUBJECTS="1,2,3"; $env:EEG_MAX_EPOCHS="2"`.

## Reviewer tables (`report/`)

| File | Content | Review |
|---|---|---|
| `T1_accuracy_per_subject.csv` | accuracy per subject × method, mean, SD | 1, 2, 11 |
| `T2_montages.csv` | channels and profile per subject × method | 5, 12 |
| `T3_equivalence_vs_22ch.csv` | paired TOST, 90% CI, worst subject, margin sensitivity | 10 |
| `T4_ours_vs_other_montages.csv` | Ours vs CSP/RLF/MI/controls, paired t + Wilcoxon, Holm | 11, 12 |
| `T5_confusion_*.csv` | confusion matrices, recall, majority baseline, collapsed folds | 15 |
| `T6_seed_stability.csv` | grand mean per seed | 14 |
| `T7_alignment.csv` | saliency–ERD alignment (sign-invariant), participant level, 95% CI | 7, 9 |
| `T7b_original_correlations.csv` | the original scripts' correlations: saliency vs ERD_L / ERD_R / ERD_(L−R) / ERD_comb and vs plain gradient | 7 |
| `T8_deletion_test.csv`, `T8_randomisation_check.csv` | faithfulness checks | 8 |
| `T9_config.json`, `selected_profile.json` | exact configuration and chosen profile | 13, 14 |
| `T10_physiology_*.csv` | **exploratory**: saliency vs other physiological maps (see below) | — |

## Exploratory physiology (curiosity only)

Besides the 8–30 Hz ERD used in the paper, the saliency step computes these maps per channel from the
raw recordings (left/right-hand trials, no model involved) and correlates them with the saliency maps.
**None of them is used for any choice**; the paper's alignment remains T7.

| Map | What it is |
|---|---|
| `ERD_theta/mu/beta/lowgamma_L/R/mag` | ERD% in 4–8, 8–13, 13–30, 30–40 Hz, 1–3 s after the cue vs pre-cue baseline |
| `ERS_beta_rebound_*` | beta (13–30 Hz) power 4–5 s after the cue (post-imagery rebound) vs baseline |
| `logpow_8_30` | absolute log 8–30 Hz power during imagery |
| `r2_signed`, `r2_abs` | how well each channel's log 8–30 Hz power separates left from right (r²) |
| `TFR_ERD_dB_*` | Morlet time–frequency ERD in dB, 8–30 Hz |
| `ITC_4_30` | inter-trial phase coherence 4–30 Hz, 0–1 s (cue-locked phase) |
| `ERP_abs_0.2_0.8` | size of the cue-evoked potential (< 5 Hz) |
| `SlowPot_0.5_3` | slow potential (< 5 Hz), 0.5–3 s (MRCP/CNV-like) |

Files: `results/full22/physiology_correlations.json`, `physiology_maps_SXX.npz`, `physiology_maps_group.png`.
Skip it with `$env:EEG_SKIP_PHYSIOLOGY="1"`.

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

Protocol levers (now in place, session protocol): the official session split (more training data,
comparable with the literature), refit on all of session 1, and a seed ensemble.

The session protocol uses seeds 42, 43, 44 by default (`EEG_SEEDS` changes it).

`run_all.py --no-tune` or `EEG_PROFILE=original` reproduces your original settings exactly.
