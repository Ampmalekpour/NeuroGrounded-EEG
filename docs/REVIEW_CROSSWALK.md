# Reviewer comments vs. the original scripts and the current code

Applies to `Deep4net/HGD`, `EEGItNet/HGD`, `EEGNetV4/HGD` and `EEGConformer/HGD` (identical pipeline; EEG-ITNet
and EEGNetv4 differ only in the model and in not using exponential moving standardisation; EEGNetv4 keeps its
original 350-epoch budget; EEGConformer keeps its own preprocessing/training recipe, see section D).
"Original Deep4Net" = `Deep4Net_BD_HGD_BD_Final_2_Class_new.py` and `..._4_Class_old.py`.
The original EEGNetv4 2-class script had the same feet-vs-left-hand mask (`[0, 1]`) and ERD labels 0/1 as
the Deep4Net one; its 4-class script used labels 0/1 for the ERD references.
The original EEG-ITNet saliency scripts used the correct left/right labels (1/3); its old CSP/MI/ReliefF
scripts masked labels `[0, 1]` (feet vs left hand) instead.
Status: **FIXED** (changed here), **PAPER** (manuscript edit), **OPEN** (needs data or a decision).
Everything marked FIXED was exercised end to end on synthetic data (2- and 4-class, 2 seeds);
the real HGD run has not been executed from this environment.

## A. Problems found in the original scripts

| # | Original behaviour | Review # | Status |
|---|---|---|---|
| 1 | **Wrong class labels.** braindecode maps event names alphabetically (feet=0, left_hand=1, rest=2, right_hand=3; `np.unique` over annotation descriptions). The 2-class script kept labels `[0, 1]` = **feet vs left hand**, and both scripts built the ERD "left/right" references from labels 0/1 (feet / left hand). | 1, 11, 15 | **FIXED**: explicit mapping, left = 1, right = 3, asserted against the annotation names of every subject at load time (`_check_label_mapping`). |
| 2 | Saliency computed on the **test fold** (`X_te`) and used to pick channels that were then tested on the same trials. | 6 | **FIXED**: validation block only. |
| 3 | One global montage averaged over **all 14 subjects**, including the tested one. | 6, 16 | **FIXED**: leave-one-subject-out montages for every method. |
| 4 | Attribution methods chosen by their ERD_(L-R) correlation, then that correlation used as validation. | 8 | **FIXED**: default averages all four methods (no ERD in the choice). `EEG_ATTR_SELECTION=best` is available, labelled exploratory, and ranks methods from the other subjects only. |
| 5 | Unsigned saliency correlated with signed ERD and L-R contrast; sign of r interpreted. | 7, 9 | **FIXED**: sign-invariant metrics vs desynchronisation strength (-ERD): `magnitude`, `class_specific` (left saliency vs left ERD, right vs right), `lateralisation` (S_L - S_R vs D_L - D_R). Per-class saliency is computed. Pearson and Spearman, participant-level mean/SD/95% CI in `alignment_metrics.json`. The old statistic is kept as `legacy_signed_LR_*` and labelled uninterpretable. |
| 6 | ERD "baseline" was the first 0.5 s **after** the cue. | 13 | **FIXED**: windows start 0.5 s before the cue; baseline -0.5..0 s, active 1..3 s; asserted that the window is long enough. ERD is computed from 8-30 Hz data **without** EMS. Trial-averaged power ratio instead of the mean of per-trial ratios. |
| 7 | Test accuracy evaluated and printed every epoch. | 6 | **FIXED**: the test block is touched once, after the best-validation-loss checkpoint is restored. |
| 8 | Only accuracies printed/saved; no predictions, class counts or machine-readable results. | 1, 2, 15 | **FIXED**: `results.json` / `fold_results.csv` with y_true, y_pred, confusion matrix, class counts, chance and majority-class accuracy. |
| 9 | Single seed; weights not provably re-initialised per fold. | 14 | **FIXED**: `EEG_SEEDS` list; every (seed, subject, fold) has its own seed and a freshly built model. Run 3+ seeds for the paper. |
| 10 | CSP/MI/ReliefF settings undefined; ReliefF pooled misses over classes. | 12 | **FIXED**: all settings are in the script headers and `results.json`; ReliefF follows Eq. 8 (k misses per class, prior-weighted); features are log-variance. |
| 11 | No fixed / random / ERD-only montage controls. | 12 | **FIXED**: `*_controls.py`. |
| 12 | No equivalence analysis code; no CI or per-subject losses. | 10 | **FIXED**: `analysis/equivalence_stats.py` (90% CI, TOST p, margin sensitivity, worst loss, # subjects exceeding the margin). The margin itself still needs a justification in the paper. |
| 13 | `ReduceLROnPlateau(verbose=True)` fails on current PyTorch. | – | **FIXED**. |
| 14 | Two different 22-channel runs (baseline vs the one used for saliency) would give different numbers. | – | **FIXED**: per-fold seeding makes them identical, and the main script produces the 22ch row itself (the separate baseline script was removed). |

## B. Items that are not code fixes

| Review # | Item | Status |
|---|---|---|
| 3 | HGD is executed movement | **PAPER** (title, abstract, dataset section) |
| 4 | PhysioNet runs, class definitions, 109 vs 10 subjects | **OPEN** (separate dataset scripts) |
| 5 | 10 vs 12 channels; 22 vs 64 channel baselines | **PAPER** - code uses 12 of 22 pre-selected HGD channels everywhere; HGD has 128 |
| 13 | Preprocessing text vs code | **PAPER** - code: 100 Hz, 4-38 Hz, EMS(1e-3, 1000); all values are written to `results.json` |
| 14 | SmoothGrad N = 50 in the paper; code uses 12 | **PAPER** (or set `SMOOTHGRAD_SAMPLES`) |
| 14 | Checkpoint = best validation *loss* (paper says accuracy) | **PAPER** |
| 1, 2 | Tables 1-6 | **OPEN**: regenerate from the new `results.json` files |
| 13 | Signal-quality / artefact-screened sensitivity analysis | **OPEN** |
| 8 | Channel-removal (perturbation) test of saliency faithfulness | **OPEN** |
| 16 | New-user generalisation of a global montage | LOSO montages test exactly this; report them as such |

## C. Check these in your first real log

* `X=(trials, 22, n_times)`: with the unchanged windowing, `n_times` is printed at load time and stored
  in `results.json`. Confirm it matches what the manuscript states (400 samples = 4 s at 100 Hz).
* The label check passes silently; it raises if braindecode's mapping ever differs.


## D. EEGConformer-specific findings (original scripts)

Read from `EEGConformer_BD_HGD_BD_Final_2_LR.py`, `..._LR_REST.py` and the three `EEGConformer_Channel_Reduction_*.py`
scripts. These are in addition to the problems in section A.

| # | Original behaviour | Status |
|---|---|---|
| 1 | **Checkpoint and learning-rate schedule chosen on the TEST fold** (`best_acc` / `scheduler.step` use `test_acc`; the validation block is never used). Reported accuracy is therefore optimistically biased. | **FIXED**: validation block. |
| 2 | In the main script `best_state = model.state_dict().copy()` is a shallow copy, so the "best" checkpoint is just the live final weights (final-epoch test accuracy). The CSP/MI/ReliefF scripts used `copy.deepcopy` (true best-on-test). The "22ch/Ours" and CSP/MI/RLF rows were therefore **not selected the same way**, which favours the three baselines. (My reading of the code; consistent with the review's point that the comparison needs a uniform protocol.) | **FIXED**: one training function for every method. |
| 3 | Saliency computed on the test loader. | **FIXED**: validation block (z-scored like the model input). |
| 4 | CSP/MI/ReliefF: one global montage from all 14 subjects. | **FIXED**: leave-one-subject-out. |
| 5 | The reduced-montage experiment re-picked channels *before* the average reference, so its reference differed from the 22-channel model (CSP/MI/RLF subset the 22-channel data after referencing). | **FIXED**: referenced once over 22 channels; reduced montages are column subsets. |
| 6 | `raw.pick_channels` without `ordered=True` can return channels in recording order (MNE-version dependent), and labels were taken from the annotation list rather than from the epochs actually created (a dropped epoch would misalign X and y). | **FIXED**: ordered picks with an assertion; labels come from `epochs.events`. |
| 7 | ERD in dB from PSD, baseline 0-0.4 s *after* the cue, method choice by `-ERD_combined` correlation. | **FIXED**: shared pre-cue ERD (8-30 Hz, trial-averaged power ratio) and the sign-invariant alignment metrics. |
| 8 | 2-class and 4-class used different model/scheduler settings (depth 2/6, heads 4/10, dropout 0.6/0.5, patience 15/10). | **OPEN / PAPER**: kept as in the originals (`CONFORMER_CFG`), stored in `results.json`; choose one configuration or justify both. |
| 9 | The only 4-class script was a fixed 12-channel run for subjects 10-14 (Cz, C3, C2, C4, C1, CP2, Pz, CPz, CP6, CP3, FC1, FC4); there was no 4-class 22-channel/saliency script. | **OPEN**: the 4-class folder is new; its settings come from that script - please confirm. |
| 10 | CSP used `reg=1e-4`, `log=True`; MI/ReliefF used raw variance; no seeds beyond 42. | **FIXED**: log-variance features, prior-weighted ReliefF, `EEG_SEEDS`. CSP keeps `reg=1e-4`. |
| 11 | Fixed 100 epochs, no early stopping, no per-fold predictions saved. | Fixed budget kept (original); predictions, confusion matrices and class counts are now saved. |
