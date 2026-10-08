# EEG-ITNet on HGD (Schirrmeister2017)

Full-22-channel baseline and 12-channel reduced-montage experiments, for both
`2class` (left vs right hand) and `4class` (feet / left / rest / right).

| File | Table row | Role |
|---|---|---|
| `eegitnet_hgd_common.py` | – | Shared loading, class masks, folds, training loop, LOSO montage, ERD maps, attribution code |
| `eegitnet_hgd_baseline_full22.py` | `22ch` | Full-montage baseline + ERD/ERS reference maps |
| `eegitnet_hgd_csp_reduced.py` | `CSP` | CSP-ranked 12-channel montage (4-class via one-vs-rest) |
| `eegitnet_hgd_mi_reduced.py` | `MI` | Mutual-information-ranked montage |
| `eegitnet_hgd_relieff_reduced.py` | `RLF` | ReliefF-ranked montage |
| `eegitnet_hgd_saliency_reduced.py` | `Ours` | **Not yet in the repo** (referenced by `common`) |

Run from this directory (scripts import `eegitnet_hgd_common` and write their
outputs, git-ignored, to the working directory):

```bash
cd EEGItNet/HGD
python eegitnet_hgd_baseline_full22.py
python eegitnet_hgd_csp_reduced.py
python eegitnet_hgd_mi_reduced.py
python eegitnet_hgd_relieff_reduced.py
```

## What changed relative to the earlier per-method scripts

1. **Class-mask bug**: the old CSP/MI/ReliefF scripts used labels `[0, 1]`
   (feet vs left hand) for the "2-class" task. All scripts now share
   `apply_class_mode()` (left = 1, right = 3).
2. **Leave-one-subject-out montages**: subject S's reduced montage is built
   only from the other 13 subjects' channel scores.
3. **One shared pipeline** (`common`) so the 2-class and 4-class runs cannot
   share or copy results, and so no method can drift from the others.
4. **Saliency from validation data**: `compute_fold_channel_importances` is
   documented to take the validation split, not the test split.

See `REVIEW_CROSSWALK.md` for a point-by-point comparison of the original code,
the revised code and the reviewer's comments.

## Known open items (reviewer report)

- Preprocessing here (100 Hz, 4-38 Hz, 0-4 s, no EMS) differs from paper
  Section 3.2; the paper text must be updated to match.
- Only accuracies are written (text reports). No per-fold predictions,
  confusion matrices, class counts or machine-readable (JSON/CSV) results.
- Single seed (42); no repeated initialisations.
- ERD/ERS baseline window is the first 0.5 s *after* the cue (windows start at
  the cue), not the pre-cue interval described in the paper. ERD is signed,
  saliency is absolute-valued, and `ERD_(L-R)` is still the only criterion used
  to choose attribution methods.
- ReliefF is a simplified variant (k nearest misses pooled across classes, no
  class-prior weighting), so it does not exactly match Eq. 8 of the paper.
- `train_one_fold` evaluates the test fold every epoch (print only; it is not
  used for checkpoint selection, which uses validation loss).
- No fixed-sensorimotor-montage, random-montage, or ERD-only selection controls.
