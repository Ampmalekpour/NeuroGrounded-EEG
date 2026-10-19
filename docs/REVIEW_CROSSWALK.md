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


## E. EEGNetv4 on BNCI2014-001 (original scripts `EN_BD_BNCI_BD_Final_1.py`, `eegnetv4_bnci_csp_2.py`, `eegnetv4_bnci_mutual_information.py`, `eegnetv4_bnci_relieff.py`, `test_for_cor.py`)

| # | Original behaviour | Status |
|---|---|---|
| 1 | **Wrong ERD labels.** `infer_target_name_mapping_from_y` assumes 0=left, 1=right, 2=feet, 3=tongue; braindecode maps BNCI events alphabetically (feet=0, left_hand=1, right_hand=2, tongue=3). The "left-hand ERD" was feet and the "right-hand ERD" was left hand, so the L-R contrast and the method ranking used the wrong maps. | **FIXED** (names asserted at load time) |
| 2 | Saliency computed on the test fold. | **FIXED**: validation block |
| 3 | CSP and MI: one global montage from all 9 subjects, including the tested one. | **FIXED**: leave-one-subject-out |
| 4 | ReliefF: per-subject montage computed from **all** of that subject's trials (including its test blocks), and 10 channels instead of 12. | **FIXED**: LOSO, 12 channels |
| 5 | Occlusion as a 5th attribution method (the manuscript lists four). | Dropped from the ranking; channel removal is now a separate faithfulness test (T8) |
| 6 | Test accuracy evaluated every epoch; one seed; only text reports. | **FIXED** |
| 7 | CSP: 8 components, MNE multi-class CSP, mean of absolute patterns. | Unified with the other datasets (4 components, one-vs-rest, sum of absolute patterns) |
| 8 | Manuscript Sec. 3.4 describes 350 epochs, lr 1e-3, wd 1e-3, mixup, scaling, noise, channel dropout, SWA and test-time augmentation; the code used 800 epochs, lr 2e-3, wd 5e-4 and no augmentation/SWA/TTA. | **PAPER**: describe what was run. Augmentation is a tuning candidate; SWA/TTA are not implemented |
| 9 | `test_for_cor.py` changes preprocessing (8-30 Hz, average reference, dropout 0.25, active-window saliency) to raise the saliency-ERD correlation. | Not used: tuning on the reported metric is circular (review 8) |
| 10 | No 2-class BNCI script. | **ADDED** (`2class/`) |

New in this folder: `*_tune.py` (validation-only, leave-one-subject-out profile choice applied to every method),
deletion and model-randomisation checks (review 8), and `run_all.py` + `analysis/aggregate_results.py` that
produce tables T1-T9.

## F. Found while porting BNCI: HGD windows are 8 s long

braindecode gives every MOABB event the dataset `interval` as its duration (verified in the installed
braindecode source: `annots.duration += offset - onset`). For Schirrmeister2017 the interval is [0, 4] s, so
the HGD scripts' `trial_stop_offset = +4 s` produces windows from the cue to **8 s** after it (800 samples at
100 Hz). With a 3-4 s inter-trial interval these windows can include the start of the next trial. The
manuscript says 0.5-2.5 s. This applies to the original HGD scripts of every model and to the HGD folders
here (which kept the original windowing on purpose); EEGConformer/HGD is not affected (0-4 s epochs).
**OPEN**: confirm with the printed `X=(..., n_times)` in a real HGD log, then decide whether to switch HGD to
a 0-4 s (or 0.5-2.5 s) window.

## G. EEG-ITNet on BNCI2014-001 (original scripts `EEGITNet_BNCI_Final_Pipeline.py`, `_CSP.py`, `_MI.py`, `_ReliefF.py`, `et_v66.py`)

| # | Original behaviour | Status |
|---|---|---|
| 1 | Saliency computed on the **test** fold of each StratifiedKFold split. | **FIXED**: validation block |
| 2 | Saliency montage: top-3 attribution methods chosen by their mean **ERD_(L-R) correlation**, then one global top-12 montage from all 9 subjects. Choosing by the reported metric is circular, and the tested subject's data shape its own montage. | **FIXED**: all four methods averaged by default (`EEG_ATTR_SELECTION=best` keeps an exploratory, LOSO version), leave-one-subject-out montages |
| 3 | CSP, MI and ReliefF: one **global** top-12 montage from all 9 subjects (including the tested one), computed from all trials. | **FIXED**: leave-one-subject-out |
| 4 | `StratifiedKFold(4, shuffle=True)` and **no validation set**: neighbouring trials of the same run end up in train and test, and nothing is held out for any choice. | **CHANGED**: 4 chronological blocks (train / validation / test), same folds as EEGNetV4 so the two models are directly comparable. The training set is half of the trials instead of 75 %, so accuracy will be lower than in the original logs |
| 5 | ERD maps from the band-passed, **EMS-standardised** training data (EMS removes slow power changes, which is exactly what ERD measures). | **FIXED**: ERD from raw µV data, pre-cue baseline, at 128 Hz for every model so EEGNetV4 and EEG-ITNet are compared with identical maps |
| 6 | The manuscript describes TTA on validation data; the code applies TTA to the test fold only (there is no validation set). | Now TTA is applied to both validation and test predictions; describe it as such in the paper |
| 7 | 4-class results only from `et_v66.py`, which reports accuracy and nothing else. | **ADDED**: full 4-class pipeline (`4class/`) |
| 8 | Correlations printed per subject (saliency vs ERD_L, ERD_R, ERD_(L-R), ERD_comb, and vs grad) but not summarised with CIs. | **KEPT and extended**: the same correlations (T7b, Pearson + Spearman) plus the sign-invariant ones (T7), participant level with Fisher-z CIs |
| 9 | Fixed recipe (350 epochs, SWA, TTA, mixup, augmentation, label smoothing 0.1, grad clip 10). | **KEPT** as profile `original`; `*_tune.py` chooses between it and three variants on validation data only |

Both BNCI models also run an **exploratory physiology block** (band-wise ERD, beta rebound, r², TFR ERD in dB,
ITC, ERP, slow potential) and report its correlation with saliency in T10 / `reports/BNCI_001/SUMMARY.md`.
It is curiosity only and is not used for any choice. The root `run_all.py` runs EEGNetV4 and then EEG-ITNet.

## H. Second revision on BNCI2014-001: session protocol and all four models

**Why.** In the first revision every fold trained on only half of a subject's trials (chronological blocks),
so accuracy dropped below the paper's (which used shuffled folds, i.e. leakage between neighbouring trials).
The fix that is both cleaner and more accurate is the dataset's own protocol.

| # | Change | Why it is clean |
|---|---|---|
| 1 | **Session protocol (default).** Train on session 1, test on session 2 (official BCI IV-2a split). | Session 2 is recorded on another day and is scored once, at the end; results are comparable with the literature |
| 2 | Validation = last 25 % of session 1 (chronological); used for early stopping, profile tuning and saliency. | Nothing from session 2 influences any choice |
| 3 | **Refit** on all of session 1 with the chosen number of epochs (EEGNetV4: best validation epoch; SWA recipes: fixed 350 epochs). | Uses only training-session data |
| 4 | **Seed ensemble**: softmax outputs of all seeds averaged (default seeds 42, 43, 44); single seeds still reported (T6). | All seeds are averaged, none is picked |
| 5 | CSP / MI / ReliefF scores computed from session 1 only (still leave-one-subject-out). | Session 2 never shapes a montage |
| 6 | Deletion (faithfulness) test on session 2 with the refitted model; ranking from validation saliency. | Evaluation only |
| 7 | **Deep4Net and EEGConformer added on BNCI**, with the paper's BNCI recipe (the legacy Deep4Net BNCI script used exactly this recipe). | Same recipe and tuning rule as EEG-ITNet |

Not used on purpose:
- **Euclidean alignment.** It improves cross-session accuracy, but it whitens the data with a spatial matrix,
  which mixes the channels. Channel saliency would then no longer belong to single electrodes, and that is
  what this paper interprets.
- **Cross-subject pretraining.** It would help weak subjects, but it changes the "subject-dependent" framing
  of the paper.

The first-revision protocol is still available (`--protocol blocks`, results in `results/`); the session
protocol writes to `results_session/`, `logs_session/`, `report_session/` and `reports/BNCI_001_session/`.

## I. HGD (Schirrmeister2017), second revision: all four models (original scripts in `<Model>/HGD/legacy_unrevised/original_scripts/`)

| # | Original behaviour | Models | Status |
|---|---|---|---|
| 1 | **Wrong classes.** Labels `[0, 1]` under braindecode's alphabetical mapping (feet=0, left_hand=1, rest=2, right_hand=3) are feet vs left hand, while the code and paper call them left vs right. All "left/right" ERD maps and the ERD-based method ranking used the feet−left contrast. | EEGNetv4, Deep4Net (all files); EEG-ITNet CSP/MI/ReliefF | **FIXED**: 2class = left vs right hand; names asserted at load time |
| 2 | **0–8 s windows.** `trial_stop_offset = +4 s` on a 4-s event gives windows to 8 s after the cue, into the inter-trial interval and the next trial. | EEGNetv4, Deep4Net, EEG-ITNet | **FIXED**: windows relative to the cue (0–4 s unless a profile says otherwise). Resolves section F |
| 3 | **No 4-class model.** The EEG-ITNet and Deep4Net HGD scripts are 2-class only, which explains the reviewer's finding that the 4-class EEG-ITNet HGD table equals the 2-class one. | EEG-ITNet, Deep4Net | **FIXED**: 2class and 4class for every model |
| 4 | **Test fold drove training.** EEGConformer's LR scheduler used test accuracy and its "best" checkpoint came from the test fold. Some files stored a shallow `state_dict().copy()`, so "best" was the last epoch. | EEGConformer | **FIXED**: validation loss only; deep copies |
| 5 | **Saliency on the test fold; one global montage from all 14 subjects (test trials included); attribution methods chosen by their ERD correlation.** CSP/MI/ReliefF were also fitted on all trials. | all | **FIXED**: validation saliency, leave-one-subject-out montages, all methods averaged, selection methods see the training runs only |
| 6 | **Official HGD split ignored.** Train and test runs were pooled into chronological blocks (train = 50 % of the trials). | all | **FIXED**: default = official training runs → test runs, with refit and seed ensemble (`--protocol blocks` still available) |
| 7 | **Post-cue ERD baseline** (0–0.5 s or 0–0.4 s after the cue). | all | **FIXED**: pre-cue baseline −0.5–0 s, raw µV, the same maps for every model |
| 8 | Inconsistent preprocessing between the saliency and the CSP/MI/ReliefF files (normalisation, re-referencing), and the EEGConformer first script's integrated gradients and SmoothGrad were zeros (non-leaf tensor). | EEGNetv4, EEGConformer | **FIXED**: one loader and one attribution code for every method |
| 10 | **Duplicated test recordings.** For subjects 6–14 the MOABB "1test" file is an exact copy of "0train" (`analysis/check_hgd_data.py`, `analysis/hgd_data_check.csv`: identical signal fingerprints, trial counts and durations). The original scripts pooled both files, so the same trials appeared in the training and test blocks. That is the most likely reason for the near-100 % accuracies of S06–S14 in Tables 1 and 4. | all | **FIXED**: duplicates dropped in every loader; those subjects hold out their chronologically last 160 trials (≤ 20 %). Subjects 1–5 keep the genuine test run |
| 11 | With the small HGD 2-class validation sets, lowest-validation-loss checkpoints kept the untrained epoch-1 model for ~15 % of seeds (one-class predictions). | (our first session run) | **FIXED**: checkpoint = highest validation accuracy, not before epoch 20 |
| 12 | Frequency band. All HGD profiles except the Conformer's stop at 38-40 Hz, while Schirrmeister et al. (2017) used a 4 Hz high-pass only on HGD (executed movements, strong high gamma). | all | **OPT-IN**: `python run_all.py --dataset hgd --protocol wide` runs 250 Hz, 4-100 Hz profiles into `results_session_wide/`, a sensitivity analysis next to the 4-38 Hz result (EMG caveat). See `docs/HGD_RESULTS_ANALYSIS.md` |
| 9 | One recipe for all models except EEGConformer. | — | **KEPT** as profile `original`. Each model also gets literature candidates (EEGNet: Lawhern 2018; EEG-ITNet: Salami 2022; Deep4Net: Schirrmeister 2017; EEGConformer: Song 2022), chosen per subject on validation data only |

HGD contains executed movements; the manuscript should say so. Physiology block for HGD adds a high-gamma
(60–90 Hz) ERD map, since Schirrmeister et al. (2017) found high gamma informative on this dataset.

## J. PhysioNet EEG Motor Movement/Imagery (MOABB `PhysionetMI`): all four models (original scripts in `EEGItNet/PhysioNet/legacy_unrevised/original_scripts/`)

Applies to `EEGNetV4/PhysioNet`, `EEGItNet/PhysioNet`, `Deep4net/PhysioNet` and `EEGConformer/PhysioNet`
(one pipeline; the models differ in architecture and training profiles). Original code:
`EEGiTNet-Physionet-SWT_Pipelline-_2th_try.py` (the paper's pipeline, labelled "EEGNetV$" but using
EEG-ITNet) and the later draft `EEGiTNet_pipeline_final_2.py`.

| # | Original behaviour | Reviewer | Status |
|---|---|---|---|
| 1 | **Classes not defined; wrong in the paper version.** Runs, event remapping and class definitions were not given. The paper version built windows without a mapping (braindecode then numbers the names alphabetically: feet=0, hands=1, left_hand=2, rest=3, right_hand=4) and removed label 0, so "4-class" was hands / left / **rest** / right: feet dropped, rest (the majority event) kept. | 4, 15 | **FIXED**: imagery runs only. 2class = left vs right fist (runs 4, 8, 12); 4class = left fist, right fist, both fists, both feet (runs 4–14). Rest is never a class. An explicit mapping is used everywhere and checked against the annotation names at load time |
| 2 | **Repeated ≈48 % accuracies** in the 4-class table are consistent with predicting the majority class. | 15 | **FIXED** at the source (item 1). T5 now reports per-class recall, predicted shares, the majority-class rate and a collapse count |
| 3 | **"22ch" label for a 64-channel baseline;** 10- vs 12-channel ambiguity. | 5 | **FIXED**: the baseline is `full64` / "64ch". Every reduced montage has 12 channels (81.25 % fewer). T2 lists every selected electrode per subject and method |
| 4 | **Ten subjects shown, N = 109 claimed;** means not matching the displayed values. | 4 | **FIXED**: the cohort is 105 subjects (88, 92, 100, 104 excluded by default and listed in `results.json`; `EEG_PHYSIONET_EXCLUDE` to change). T1 lists all of them; class and trial counts per subject are in `results.json` (`meta.class_counts`) and `analysis/check_physionet_data.py` |
| 5 | **Shuffled stratified K-fold; saliency on the test fold.** | 6, 11 | **FIXED**: session protocol as for BNCI and HGD, with a run-wise chronological split (no official split exists): test = last run of each type, validation = last 25 % of each class's training trials, refit, test once, 3-seed ensemble. Saliency on validation trials only |
| 6 | **Circular selection:** attribution methods ranked by their ERD correlation; one global top-12 montage from all subjects, then retrained on the same subjects. | 7, 11, 12 | **FIXED**: all four methods averaged (the choice does not depend on ERD). Montages are leave-one-subject-out, as for BNCI and HGD |
| 7 | **Attribution code assumed 4 classes and a temporal output** (`model(x).view(B, 4, -1)`), so it was wrong for 2-class. | 8 | **FIXED**: true-class logit of the model output, any number of classes |
| 8 | ERD timing in the paper version was already right (−0.5–3 s epoch: pre-cue baseline −0.5–0 s, active 1–3 s). Its left/right ids depended on the order of a re-mapping step. | 9, 13 | **KEPT / FIXED**: same intervals, now stated in `T9_config.json`. ERD comes from raw µV at 8–30 Hz, at one sampling rate for all models; left/right ids come from the explicit mapping |
| 9 | **PhysioNet-specific architecture changes** described inconsistently (Sec. 3.3 vs 3.4). | 14 | **FIXED**: each model's exact configuration is written to `T9_config.json`. The manuscript's PhysioNet setting (dropout 0.25; Deep4Net temporal filter 10, pooling 2) is the `original` profile next to the literature settings. The profile is chosen on validation data only. 3 seeds, ensemble and per-seed tables (T6) |
| 10 | Spatial normalisation (e.g. Euclidean alignment) was proposed in the later draft. | 7, 12 | **NOT USED**: it mixes electrodes, so channel saliency and selection would no longer refer to single electrodes. Normalisation is channel-wise (z-score with training-run statistics, or EMS) |

Comparison with the manuscript: the paper's PhysioNet 4-class numbers (Table 3) come from the
hands/left/rest/right task (item 1). They are not comparable with the corrected 4-class task, whose chance
level is 25 %. The 2-class definition in the paper is not documented. The new numbers should replace both
tables. Text values that can be read from the manuscript: EEGConformer 4-class 64ch 55.53 %, Ours 55.52 %,
CSP 48.39 %, ReliefF 48.42 %, MI 48.53 %.
