# PhysioNet EEG Motor Movement/Imagery — how to run

One command runs the whole suite of one model: tuning, 64ch + Ours, CSP, ReliefF, MI, controls and the
reviewer tables, for 2class and 4class. Unfinished steps continue where they stopped.

## 1. Once: download and check the data

```powershell
cd D:\EEG\NeuroGrounded-EEG
D:\EEG\.venv\Scripts\Activate.ps1
python analysis\check_physionet_data.py
```

- Downloads all 109 subjects (about 1.5 GB) to your MNE data folder.
- Writes `analysis/physionet_data_check.csv`.
- Lists anything unusual: sampling rate, missing runs or classes, irregular cue timing.

## 2. One model, everything (start it and come back later)

```powershell
python EEGNetV4\PhysioNet\run_suite.py
python EEGItNet\PhysioNet\run_suite.py
python Deep4net\PhysioNet\run_suite.py
python EEGConformer\PhysioNet\run_suite.py
```

Or all four in order: `python run_all.py --dataset physionet`.

- **Results:** `<Model>/PhysioNet/report_session/REPORT.md`, plus `reports/PhysioNet_session/SUMMARY.md` across
  models.
- **Logs:** `<Model>/PhysioNet/<class>/logs_session/<step>.log`, and progress in
  `run_all_status_physionet_session.json`.
- **If it stops:** run the same command again. Use `--force` to redo finished steps.

## 3. Quick dry run without data (a few minutes, CPU)

```powershell
$env:EEG_SYNTHETIC="1"; $env:EEG_SUBJECTS="1,2,3,4"; $env:EEG_MAX_EPOCHS="3"; $env:EEG_SEEDS="42,43"
python EEGNetV4\PhysioNet\run_suite.py
Remove-Item Env:EEG_SYNTHETIC, Env:EEG_SUBJECTS, Env:EEG_MAX_EPOCHS, Env:EEG_SEEDS
```

Then **delete the dry-run output** before the real run. Otherwise the real run will skip those steps:

```powershell
Remove-Item -Recurse EEGNetV4\PhysioNet\*\results_session, EEGNetV4\PhysioNet\*\logs_session, EEGNetV4\PhysioNet\report_session, reports\PhysioNet_session, run_all_status_physionet_session.json
```

## What the pipeline does (details in `<Model>/PhysioNet/README.md` and `docs/REVIEW_CROSSWALK.md` J)

- **Cohort:** 105 subjects (88, 92, 100 and 104 excluded by default). Imagery runs only.
  - 2class = left vs right fist.
  - 4class = left fist, right fist, both fists, both feet.
  - Rest is never a class.
- **Channels:** all 64 channels ("64ch") vs 12-channel montages.
- **Protocol:** `session`, the same pipeline as BNCI and HGD.
  - One model per subject, trained from scratch.
  - Test = the last run of each type. Validation = the last 25 % of each class's training trials.
  - Then refit, and the 3 seeds are averaged.
- **No leakage:** montages and the profile choice are leave-one-subject-out. No test run is used for any
  choice.
- **Run time:** the tuning step plus 12 experiments × 105 subjects × 3 seeds per class mode. The first
  run also builds the window cache (about 20 min per preprocessing setting, reused by all models).
