# Deep4Net on PhysioNet EEG Motor Movement/Imagery (MOABB `PhysionetMI`)

- **Data:** 109 subjects, 64 channels, 160 Hz, motor **imagery** runs only.
  - Runs 4, 8, 12: left fist vs right fist.
  - Runs 6, 10, 14: both fists vs both feet.
- **Classes:**
  - `2class` = left hand vs right hand (runs 4, 8, 12).
  - `4class` = left hand, right hand, both fists (`hands`), both feet (`feet`) (all six runs).
  - Rest (T0) is never a class.
  - Labels come from an explicit name → id mapping and are checked against the annotations.
- **Cohort:** 105 subjects. Subjects 88, 92, 100 and 104 are excluded by default because of irregular
  sampling rate or trial timing in the PhysioNet-MI literature. Check them yourself with
  `analysis/check_physionet_data.py`, and change the list with `EEG_PHYSIONET_EXCLUDE` (`none` = all 109).
- **Size:** about 22 trials per class and subject (≈45 trials in 2class, ≈90 in 4class).

## Run everything for this model (Windows PowerShell)

```powershell
cd D:\EEG\NeuroGrounded-EEG
D:\EEG\.venv\Scripts\Activate.ps1
python analysis\check_physionet_data.py      # once: downloads all subjects (~1.5 GB) and checks them
python Deep4net\PhysioNet\run_suite.py       # the whole Deep4Net suite, 2class and 4class
```

- `run_suite.py` is the same as `python run_all.py --dataset physionet --models deep4net`.
- It runs, for 2class and then 4class: tune → saliency → csp → relieff → mi → controls → tables.
- It is the same pipeline as for BNCI2014-001 and HGD.
- If it stops, start the same command again. Finished steps are skipped.
- **Results:** `Deep4net/PhysioNet/report_session/REPORT.md` and `reports/PhysioNet_session/SUMMARY.md`.

## Protocol (`session`, as for BNCI and HGD)

PhysioNet has no official train/test split, so the split is by run:
- **Test** = the last imagery run of each type (run 12, and run 14 in 4class).
- **Training** = the two earlier runs of each type.
- **Validation** = the last 25 % of each class's training trials, in time order. It is used for early
  stopping, the profile choice and the saliency maps.
- **Refit** on all training runs with the chosen number of epochs, then test once. Seeds 42, 43 and 44
  are averaged (seed ensemble).
- One model per subject, trained from scratch on that subject only.
- The test run is never used for any choice. CSP, MI and ReliefF see only the training runs.
- Montages and the profile choice are leave-one-subject-out.

## Training profiles (chosen per subject on validation data only, leave-one-subject-out, by `*_tune.py`)

Architecture: braindecode Deep4Net, filters 10 samples, max-pool 3 / stride 3, dropout 0.5 (Schirrmeister et al. 2017).

| Profile | Preprocessing | Notes |
|---|---|---|
| `zscore` (default) | 160 Hz, 4–40 Hz, 0–3 s after the cue, per-subject channel-wise z-score (training-run mean/SD) | literature architecture |
| `ems` | as `zscore` but exponential moving standardisation instead of the z-score | |
| `original` | your PhysioNet script, bugs fixed: 8–32 Hz, 0.5–2.5 s, no normalisation, Adam(W) lr 1e-2 / wd 1e-5, MultiStepLR [20, 50] ×0.1, 100 epochs, batch 64, no label smoothing, clip 10, checkpoint on val loss | architecture: filter 10, pooling 2 / stride 2, dropout 0.25 (manuscript Sec. 3.4) |

Training (all profiles except `original`): AdamW lr 1e-3 / wd 5e-4, cosine schedule, ≤ 300 epochs, patience 80,
batch 16, label smoothing 0.1, gradient clipping 10. Checkpoint: the highest validation accuracy
(validation loss breaks ties), not before epoch 20.

Tuning candidates: zscore, ems, original. Normalisation is **channel-wise only**. A spatial transform such as
Euclidean alignment would mix electrodes, so the channel saliency and the channel selection would no
longer refer to single electrodes.

## Outputs

- `<class>/results_session/<experiment>/results.json` holds, per subject and seed:
  - montages, profile, class counts, train/test trial counts;
  - per-trial predictions and confusion matrices;
  - the full configuration.
- Experiments: `full64` (all 64 channels, saliency, ERD maps, alignment, physiology), `saliency_reduced`
  (Ours), `csp_reduced`, `mi_reduced`, `relieff_reduced`, `control_sensorimotor`, `control_erd`,
  `control_random_draw0-4`.
- Preprocessed windows are cached in `cache/physionet/` at the repository root (about 1.2 GB per
  preprocessing setting, shared by all models). Delete it any time; it is rebuilt from the MOABB data.
- `report_session/REPORT.md` contains:
  - T1 accuracy per subject (all 105);
  - T2 montages and T3 TOST vs `64ch`;
  - T4 Ours vs the other montages and T5 confusion / class balance;
  - T6 seeds and T7 saliency–ERD alignment;
  - T8 deletion and randomisation checks, T9 configuration and T10 exploratory physiology.

## What was wrong in the original PhysioNet code (`EEGItNet/PhysioNet/legacy_unrevised/original_scripts/`)

- **Wrong classes.** The earlier code built windows without a mapping, so braindecode numbered the
  event names alphabetically (feet=0, hands=1, left_hand=2, rest=3, right_hand=4). It then removed
  label 0.
  - So "4-class" was hands / left / **rest** / right: feet was dropped and rest kept.
  - Rest is the most frequent event, which fits the repeated ≈48 % accuracies that reviewer 15 noticed.
  - `check_physionet_data.py` prints the rest share under the old rule.
- **Mislabelled baseline:** the full montage is 64 channels, not "22ch" (reviewer 5).
- **Shuffled K-fold with test-fold saliency.** Attribution methods were ranked by their ERD correlation,
  and one global 12-channel montage came from all subjects. That is circular and leaks test data. Now
  all four methods are averaged and montages are leave-one-subject-out.
- **4-class assumption in the attribution code:** `model(x).view(B, 4, -1)`.

Full list: `docs/REVIEW_CROSSWALK.md`, section J.
