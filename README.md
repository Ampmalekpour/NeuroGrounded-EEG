# NeuroGrounded-EEG

Neurophysiological validation of deep learning models for motor imagery EEG using saliency
alignment with ERD/ERS patterns, and saliency-guided channel reduction.

## Layout

```
<Model>/<Dataset>/<2class|4class>/<scripts>

Deep4net/
  HGD/
    deep4net_hgd_common.py            shared code for this model + dataset (do not run)
    2class/                           left hand vs right hand
      deep4net_hgd_2class_saliency.py   MAIN: 22ch baseline + saliency-ranked 12ch  ("22ch", "Ours")
      deep4net_hgd_2class_csp.py        CSP-ranked 12ch                             ("CSP")
      deep4net_hgd_2class_relieff.py    ReliefF-ranked 12ch                         ("RLF")
      deep4net_hgd_2class_mi.py         mutual-information-ranked 12ch              ("MI")
      deep4net_hgd_2class_controls.py   fixed / random / ERD-only 12ch montages
    4class/                           feet / left hand / rest / right hand (same five scripts)
    legacy_unrevised/                 old pre-review scripts, kept for reference
  BNCI_001/legacy_unrevised/          old Deep4Net BNCI2014-001 scripts and outputs
EEGItNet/
  HGD/                                same structure as Deep4net/HGD (common + 2class/ + 4class/)
EEGNetV4/
  HGD/legacy_unrevised/               old pre-review scripts (not yet ported)
analysis/equivalence_stats.py         paired TOST / CI / per-subject losses from results.json files
docs/REVIEW_CROSSWALK.md              reviewer comments vs original code vs this code
```

Each `2class/` or `4class/` folder is self-contained to run: scripts import the `*_common.py`
one level up, fix the class mode themselves, and write to `<that folder>/results/`.

## Setup and run (Windows PowerShell)

```powershell
cd D:\EEG\NeuroGrounded-EEG
python -m venv D:\EEG\.venv            # keep the environment OUTSIDE the repository
D:\EEG\.venv\Scripts\Activate.ps1
pip install torch braindecode moabb mne scikit-learn scipy matplotlib

cd Deep4net\HGD\2class

# 1. dry run on synthetic data (no download, a few minutes on CPU)
$env:EEG_SYNTHETIC="1"; $env:EEG_MAX_EPOCHS="3"; $env:EEG_SUBJECTS="1,2,3"
python deep4net_hgd_2class_saliency.py
Remove-Item Env:EEG_SYNTHETIC, Env:EEG_MAX_EPOCHS, Env:EEG_SUBJECTS

# 2. real run (downloads HGD on first use; GPU strongly recommended)
$env:EEG_SEEDS="42,43,44"
python deep4net_hgd_2class_saliency.py
python deep4net_hgd_2class_csp.py
python deep4net_hgd_2class_relieff.py
python deep4net_hgd_2class_mi.py
python deep4net_hgd_2class_controls.py sensorimotor random erd
```

For the other settings run the same files from `Deep4net\HGD\4class`, `EEGItNet\HGD\2class`
or `EEGItNet\HGD\4class`.

Optional environment variables: `EEG_SEEDS`, `EEG_SUBJECTS`, `EEG_MAX_EPOCHS`, `EEG_PATIENCE`,
`EEG_RESULTS_DIR`, `EEG_SYNTHETIC`, `EEG_ATTR_SELECTION` (`all` default | `best`, exploratory).

## Outputs (`<class folder>/results/<experiment>/`)

`full22/` (22-channel baseline, saliency, alignment metrics, ERD maps), `saliency_reduced/`,
`csp_reduced/`, `mi_reduced/`, `relieff_reduced/`, `control_*/`. Each contains `results.json`
(config, montages, per-fold predictions, confusion matrices, class counts, chance and
majority-class accuracy), `fold_results.csv`, `subject_summary.csv` and a text report.
Manuscript tables should be generated from these files.

## Equivalence statistics

```powershell
cd D:\EEG\NeuroGrounded-EEG
python analysis\equivalence_stats.py `
  Deep4net\HGD\4class\results\full22\results.json `
  Deep4net\HGD\4class\results\saliency_reduced\results.json --margin 6
```

All 12-channel montages are leave-one-subject-out: subject S's channels are computed only from
the other subjects. See `docs/REVIEW_CROSSWALK.md` for details and open items.
HGD contains executed movements, not imagined ones.
