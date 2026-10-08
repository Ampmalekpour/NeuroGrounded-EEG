# Reviewer comments vs. original and revised EEG-ITNet/HGD code

Status key: **FIXED** (changed in revised code), **PARTIAL**, **UNCHANGED** (code still has the issue),
**PAPER** (the fix is a manuscript edit), **UNVERIFIED** (depends on a file not in the repo).

"Original" = the old `eegitnet_hgd_paper_ready_full_vs_reduced_no_occlusion.py` (2-class and 4-class
copies) plus the old CSP/MI/ReliefF scripts. "Revised" = the five files in this folder.
The revised saliency script (`eegitnet_hgd_saliency_reduced.py`) is now included and has been read.

## A. What was wrong in the original code

| # | Problem in original code | Review # | Revised status |
|---|---|---|---|
| 1 | Old CSP/MI/ReliefF 2-class scripts masked labels `[0, 1]` (feet vs left hand) instead of left vs right, so the baselines solved a different task than "Ours". | 1, 11, 15 | **FIXED**: all scripts use `apply_class_mode()` (left=1, right=3 -> 0/1). |
| 2 | Saliency was computed on the **test fold** (`compute_fold_channel_importances(model, X_te, y_te)`), then the reduced model was scored on those same test trials. | 6 | **FIXED**: the saliency script passes `X[val_idx], y[val_idx]` to `compute_fold_channel_importances`. (The function itself still does not enforce this.) |
| 3 | One **global** montage averaged over all 14 subjects, including the one being tested. | 6, 16 | **FIXED** for CSP/MI/ReliefF and saliency (`loso_montage`: subject S's montage uses only the other 13), **except** the residual leak in section B (method ranking). |
| 4 | Attribution methods were chosen by correlation with ERD_(L-R) and that same correlation was then used as validation (circular). | 8 | **UNCHANGED**: `best_methods` is still ranked by mean ERD_(L-R) correlation (saliency script lines 141-146); the other three correlations are report-only. |
| 5 | Saliency is `abs()`-averaged and averaged over all classes; ERD is signed; L-R flips sign with subtraction order. Sign of r was then interpreted. | 7, 9 | **UNCHANGED** (same `abs().mean` code; extra correlations only logged). |
| 6 | ERD "baseline" is samples 0-0.5 s of a window that starts **at the cue** (post-cue), not the pre-cue interval stated in the paper. | 13 | **UNCHANGED** (`compute_class_erd_map`, `trial_start_offset_samples=0`). |
| 7 | Test accuracy evaluated and printed every epoch inside the training loop. Not used for checkpoint choice (val loss is), but invites leakage and costs compute. | 6 | **UNCHANGED**. |
| 8 | Only accuracies saved (text). No per-fold predictions, confusion matrices, class counts, or machine-readable results, so tables cannot be regenerated or audited. | 1, 2, 15 | **UNCHANGED**. Worse: the old 4-class script printed class counts and a LABEL CHECK; the revised loader prints neither and no longer asserts class count (2-class) or the label mapping (0=feet, 1=left, 2=rest, 3=right). |
| 9 | 2-class and 4-class were separate near-duplicate scripts (only `OUTPUT_BASE` differs), making copied/reused results easy. | 1 | **PARTIAL**: one shared pipeline removes the mechanism, but nothing proves the old tables were wrong until everything is re-run. |
| 10 | Single seed (42); no repeated initialisations. | 14 | **UNCHANGED**. |
| 11 | CSP/ReliefF/MI details undefined (CSP multiclass, #filters, features). | 12 | **PARTIAL**: now explicit in code (CSP: 4 components, no reg, one-vs-rest for 4-class; MI/ReliefF: per-channel temporal variance, k=10). Still needs to go into the paper. |
| 12 | No fixed-sensorimotor, random-montage or ERD-only selection controls. | 12 | **UNCHANGED**. |
| 13 | No code for TOST/equivalence, confidence intervals, per-subject losses. | 10 | **UNCHANGED** (absent). |
| 14 | Only global selection exists; paper describes Local and Global subsets. | 5, 16 | **PAPER**: remove "Local" or implement it. |

## B. New issues found in the revised code

- **Residual leak in the saliency montage.** `best_methods` is ranked by the ERD_(L-R) correlation
  averaged over **all 14** subjects, so held-out subject S's saliency and its ERD map (computed from all
  of S's trials, including the test blocks) influence which methods build S's "LOSO" montage. The effect
  is probably small, but the "never uses S's data" claim is not strictly true. Fix: rank methods per
  held-out subject from the other 13 only.
- **Two different 22ch runs.** The saliency script retrains its own 22-channel models and the baseline
  script trains another set. They will give different numbers (different RNG state), so decide which one
  fills the `22ch` table row, or the row and the "Ours" comparison will not share a baseline.
- Saliency uses the true-class logit averaged over all classes, but is compared only with the left-vs-right
  ERD reference, including in the 4-class run (feet/rest trials contribute to saliency, not to the reference).

- **ReliefF does not match paper Eq. 8**: nearest misses are pooled across all other classes with no
  per-class prior weighting. The header comment claiming it is correct for >2 classes overstates this.
- Unused imports (`TARGET_SFREQ` in baseline, `write_comparison_block` in CSP). Harmless.
- The CSP spatial-pattern indexing was checked on synthetic data with MNE 1.13.2 and is correct.

## C. Paper-vs-code mismatches (fix the paper, per the note in `common`)

| Paper | Code |
|---|---|
| 160 Hz, 8-32 Hz, 0.5-2.5 s, exponential moving standardisation | 100 Hz, 4-38 Hz, 0-4 s (400 samples), no standardisation |
| Checkpoint = best validation **accuracy** | best validation **loss** |
| SmoothGrad N = 50 | `SMOOTHGRAD_SAMPLES = 12` (IG uses 32 steps) |
| 10 and 12 channels mentioned | 12 only (`REDUCED_N_CHANNELS`) |
| ERD from "preprocessed" EEG, pre-cue baseline | ERD from separate 8-30 Hz filter (no standardisation), post-cue baseline |
| "HGD, 128 channels, motor imagery" | 22 pre-selected sensorimotor channels; executed movement |
| No architecture changes except output layer | (PhysioNet changes are in other scripts, not in this folder) |

## D. Needed before the results can answer the review

1. Re-run baseline and all reduced variants; save per-fold predictions, confusion matrices, class counts
   and a JSON/CSV; regenerate every table from those files.
2. Supply the saliency script and make it use validation data and LOSO montages.
3. Fix ERD to a pre-cue baseline; define the validity metric (magnitude vs lateralisation, class-specific).
4. Multiple seeds; fixed/random/ERD-only montage controls; TOST with confidence intervals and per-subject losses.
