# Deep4Net on HGD (Schirrmeister2017)

Full-22-channel baseline, four 12-channel ranking methods and three control montages, for
`2class` (left vs right hand) and `4class` (feet / left hand / rest / right hand).
HGD contains **executed** movements, not imagined ones.

| Script | Table row | What it does |
|---|---|---|
| `deep4net_hgd_baseline_full22.py` | `22ch` | Full 22-channel model + ERD/ERS reference maps |
| `deep4net_hgd_saliency_reduced.py` | `Ours` | Validation-set saliency -> LOSO 12-channel montage -> retrain; alignment metrics |
| `deep4net_hgd_csp_reduced.py` | `CSP` | CSP-ranked montage (4-class: one-vs-rest) |
| `deep4net_hgd_mi_reduced.py` | `MI` | Mutual-information-ranked montage |
| `deep4net_hgd_relieff_reduced.py` | `RLF` | ReliefF-ranked montage (multi-class, prior-weighted) |
| `deep4net_hgd_controls_reduced.py` | controls | `sensorimotor` fixed montage, `random` montages, `erd` ERD-only ranking |
| `deep4net_hgd_common.py` | – | Shared code; do not run |
| `../../analysis/equivalence_stats.py` | Table 7 | Paired TOST, 90% CI, per-subject losses, margin sensitivity |

All montages are **leave-one-subject-out**: subject S's 12 channels are computed only from the
other subjects. Every model is tested within subject with 4 chronological blocks
(test = block i, validation = block i-1, train = the rest).

## Run (Windows PowerShell shown; same on Linux/macOS)

```powershell
cd Deep4net\HGD
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch braindecode moabb mne scikit-learn scipy matplotlib

# 1. dry run on synthetic data (no download, a few minutes on CPU) - checks the install end to end
$env:EEG_SYNTHETIC="1"; $env:EEG_MAX_EPOCHS="3"; $env:EEG_SUBJECTS="1,2,3"
python deep4net_hgd_baseline_full22.py
Remove-Item Env:EEG_SYNTHETIC, Env:EEG_MAX_EPOCHS, Env:EEG_SUBJECTS

# 2. real run (downloads HGD on first use; GPU strongly recommended)
$env:EEG_SEEDS="42,43,44"
python deep4net_hgd_baseline_full22.py
python deep4net_hgd_saliency_reduced.py
python deep4net_hgd_csp_reduced.py
python deep4net_hgd_mi_reduced.py
python deep4net_hgd_relieff_reduced.py
python deep4net_hgd_controls_reduced.py sensorimotor random erd

# 3. equivalence statistics (from the repository root)
python analysis\equivalence_stats.py `
  Deep4net\HGD\results\baseline_full22\4class\results.json `
  Deep4net\HGD\results\saliency_reduced\4class\results.json --margin 6
```

Environment variables (all optional): `EEG_SEEDS`, `EEG_SUBJECTS`, `EEG_CLASS_MODES`
(e.g. `4class`), `EEG_MAX_EPOCHS`, `EEG_PATIENCE`, `EEG_RESULTS_DIR`, `EEG_SYNTHETIC`,
`EEG_ATTR_SELECTION` (`all` default | `best`, exploratory).

## Outputs (`results/<experiment>/<class_mode>/`)

* `results.json` - config snapshot, montages, summary, and one record per subject/seed/fold with
  accuracy, **y_true / y_pred, confusion matrix, class counts, majority-class accuracy,
  test indices**, channels, best epoch.
* `fold_results.csv`, `subject_summary.csv` - flat tables for the paper.
* `*_report.txt` - human-readable report.
* `saliency_full22/<mode>/alignment_metrics.json` - participant-level saliency-ERD alignment with CIs.

Every table in the manuscript should be generated from these files, not typed by hand.
Compute: 14 subjects x 4 folds x #seeds x 2 class modes per script, up to 600 epochs each.
The saliency script retrains the 22-channel models itself and, because seeds are fixed per
(seed, subject, fold), reproduces the baseline numbers, so the baseline script is optional if
you only need the saliency row.

See `REVIEW_CROSSWALK.md` for how each reviewer comment is handled.
