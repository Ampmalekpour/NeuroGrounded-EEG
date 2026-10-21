# Everything so far vs the paper (BNCI2014-001 and HGD)

Status: BNCI done for all 4 models; HGD done for EEGNetV4 and Deep4Net; HGD EEG-ITNet / EEGConformer and all of
PhysioNet not run yet.

**How to read the tables.** `ours` = session protocol (BNCI: session 1 → session 2; HGD: official training runs →
test runs), 3-seed ensemble, tuning on validation data only, leave-one-subject-out montages, 12 channels.
`paper` = the means printed in Tables 1, 2, 4 and 5 of the manuscript. TOST margin ±6 pp.


## BNCI, 4class

| Model | | 22ch | Ours (saliency) | CSP | RLF | MI | Ours − 22ch (TOST ±6) |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | paper | 63.75 | 61.54 | 59.65 | 62.75 | 61.55 | -2.21 |
| | **ours** | **62.73** | **60.88** | **57.68** | **57.64** | **59.84** | -1.85 (NOT equivalent, p=0.0920) |
| | Δ | -1.0 | -0.7 | -2.0 | -5.1 | -1.7 | |
| EEGItNet | paper | 54.96 | 53.95 | 52.89 | 52.28 | 55.17 | -1.01 |
| | **ours** | **50.97** | **50.58** | **50.04** | **49.31** | **50.08** | -0.39 (equivalent, p=0.0004) |
| | Δ | -4.0 | -3.4 | -2.9 | -3.0 | -5.1 | |
| Deep4net | paper | 61.96 | 59.57 | 57.62 | 52.14 | 53.18 | -2.39 |
| | **ours** | **45.14** | **45.33** | **44.37** | **44.98** | **44.75** | +0.20 (equivalent, p=0.0002) |
| | Δ | -16.8 | -14.2 | -13.3 | -7.2 | -8.4 | |
| EEGConformer | paper | 61.96 | 59.72 | 58.91 | 58.41 | 60.34 | -2.24 |
| | **ours** | **64.85** | **60.46** | **58.83** | **58.41** | **60.57** | -4.40 (equivalent, p=0.0149) |
| | Δ | +2.9 | +0.7 | -0.1 | +0.0 | +0.2 | |

## BNCI, 2class

| Model | | 22ch | Ours (saliency) | CSP | RLF | MI | Ours − 22ch (TOST ±6) |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | paper | 72.72 | 67.36 | 62.58 | 64.27 | 71.99 | -5.36 |
| | **ours** | **67.13** | **66.28** | **69.98** | **66.82** | **66.20** | -0.85 (equivalent, p=0.0013) |
| | Δ | -5.6 | -1.1 | +7.4 | +2.6 | -5.8 | |
| EEGItNet | paper | 71.68 | 72.18 | 64.62 | 72.11 | 72.30 | +0.50 |
| | **ours** | **71.06** | **71.60** | **69.06** | **68.06** | **66.67** | +0.54 (equivalent, p=0.0011) |
| | Δ | -0.6 | -0.6 | +4.4 | -4.1 | -5.6 | |
| Deep4net | paper | 72.61 | 74.07 | 69.75 | 72.22 | 72.07 | +1.46 |
| | **ours** | **65.59** | **65.28** | **66.44** | **63.66** | **63.27** | -0.31 (equivalent, p=0.0013) |
| | Δ | -7.0 | -8.8 | -3.3 | -8.6 | -8.8 | |
| EEGConformer | paper | 81.64 | 81.17 | 80.17 | 78.70 | 79.71 | -0.47 |
| | **ours** | **78.70** | **76.39** | **76.93** | **75.85** | **74.92** | -2.31 (equivalent, p=0.0119) |
| | Δ | -2.9 | -4.8 | -3.2 | -2.9 | -4.8 | |

## HGD, 4class

| Model | | 22ch | Ours (saliency) | CSP | RLF | MI | Ours − 22ch (TOST ±6) |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | paper | 95.77 | 96.23 | 94.89 | 97.22 | 96.98 | +0.46 |
| | **ours** | **87.05** | **85.36** | **86.38** | **85.62** | **86.29** | -1.70 (equivalent, p=0.0003) |
| | Δ | -8.7 | -10.9 | -8.5 | -11.6 | -10.7 | |
| EEGItNet | paper | 96.14 | 96.96 | 96.11 | 95.71 | 94.99 | +0.82 |
| | ours | not run yet | | | | | |
| Deep4net | paper | 96.90 | 96.78 | 93.26 | 95.89 | 96.36 | -0.12 |
| | **ours** | **80.67** | **80.35** | **79.24** | **80.18** | **80.04** | -0.31 (equivalent, p=<0.0001) |
| | Δ | -16.2 | -16.4 | -14.0 | -15.7 | -16.3 | |
| EEGConformer | paper | 93.58 | 93.45 | 91.13 | 94.22 | 94.06 | -0.13 |
| | ours | not run yet | | | | | |

## HGD, 2class

| Model | | 22ch | Ours (saliency) | CSP | RLF | MI | Ours − 22ch (TOST ±6) |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | paper | 99.69 | 99.56 | 99.41 | 99.56 | 99.60 | -0.13 |
| | **ours** | **83.04** | **81.07** | **83.93** | **84.02** | **80.36** | -1.96 (equivalent, p=0.0172) |
| | Δ | -16.7 | -18.5 | -15.5 | -15.5 | -19.2 | |
| EEGItNet | paper | 96.14 | 96.96 | 96.11 | 95.71 | 94.99 | +0.82 |
| | ours | not run yet | | | | | |
| Deep4net | paper | 98.62 | 99.16 | 98.94 | 99.15 | 99.17 | +0.54 |
| | **ours** | **75.62** | **76.52** | **75.62** | **75.80** | **75.71** | +0.89 (equivalent, p=0.0003) |
| | Δ | -23.0 | -22.6 | -23.3 | -23.3 | -23.5 | |
| EEGConformer | paper | 93.91 | 94.31 | 92.76 | 94.37 | 94.11 | +0.40 |
| | ours | not run yet | | | | | |


## What the numbers say

**1. The 12-channel saliency montage is statistically equivalent to the full montage almost everywhere.**
Ours − 22ch is within ±6 pp in 11 of 12 model/class combinations (TOST p < 0.05). The exception is BNCI EEGNetV4
4-class (−1.85 pp, p = 0.092). This is the paper's central claim, and it holds, in the narrower form the reviewer asked
for: "the mean difference fell within the chosen margin" (comment 10).

**2. The saliency montage is not better than the alternatives, and the paper should say so (comment 11).**
Among the four 12-channel methods (Ours, CSP, ReliefF, MI) Ours ranks 1st in 6 of 12 cases, 2nd in 3, 3rd in 2
and 4th in 1 (HGD EEGNetV4 4-class). The differences are small and not significant after Holm correction. The only raw p < 0.05
results are BNCI EEG-ITNet 2-class (Ours above MI, fixed sensorimotor and random). Compared with the controls, Ours is above the random montages in 11 of 12 cases (on average +1.7 pp) and about equal
to the fixed sensorimotor (7 of 12 above, +0.3 pp) and ERD-only montages (8 of 12 above, +0.3 pp).
The honest statement is "equivalent to 22 channels and comparable to CSP / ReliefF / MI", not "better".

**3. Where our accuracy is close to the paper, and where it is not.**
- BNCI EEGConformer (+2.9 on 22ch, 4-class), EEGNetV4 (−1.0) and EEG-ITNet 2-class (−0.6) are close.
- BNCI EEG-ITNet 4-class is 4 pp lower and BNCI EEGNetV4 / Deep4Net 2-class 6–7 pp lower (22ch). The paper used shuffled
  folds, where neighbouring trials of one session leak between training and test; our test is a different day.
- BNCI Deep4Net 4-class is 17 pp below the paper (45.1 vs 62.0). See point 5.
- HGD is 9–23 pp below the paper. This is expected: the paper's S06–S14 were scored on duplicated test recordings
  (≈ 100 % each), and its "2-class" task was feet vs left hand. Details in `docs/HGD_RESULTS_ANALYSIS.md` and
  `docs/HGD_EEGNetV4_vs_paper.md`. On S01–S05, which were never affected, EEGNetV4 matches the paper (88.0 vs 88.4).

**4. Faithfulness and alignment (reviewer comments 7–9), EEGNetV4 HGD 4-class as the example.**
- Deletion test: removing the 2 top-ranked channels costs 33 pp, random channels 15 pp, bottom-ranked 3 pp
  (p < 0.0001). The saliency ranking is faithful to what the model uses.
- Model-randomisation check: correlation with an untrained network about 0.
- Saliency vs ERD (sign-invariant, participant level with 95 % CI): magnitude r = 0.45 and class-specific r = 0.43 for
  plain gradients and SmoothGrad; lateralisation about 0 to −0.12. Saliency finds the active channels but not the side.

**5. Deep4Net is the weak spot.** HGD: 80.7 % vs EEGNetV4 87.1 % (4-class), 75.6 % vs 83.0 % (2-class). BNCI 4-class:
45.1 % vs 62.7 %. Several subjects stay at chance in every seed (HGD S09 4-class 55 %, S01 and S09 2-class below 50 %),
with early best epochs. The tuning step only offers learning rates of 1e-2 or 5e-4. Deep4Net trained by Schirrmeister et al.
saw many cropped windows per trial, which a trial-wise run does not. A lower learning rate (1e-3) is the obvious,
untested candidate. It is a tuning-candidate change, not a pipeline change.

## How each reviewer comment stands

| # | Comment | Status |
|---|---|---|
| 1, 2 | Duplicated EEGItNet tables, wrong means, tables not from saved results | **Done in code.** Every table is generated from `results.json` (per-trial predictions); EEG-ITNet HGD 2- and 4-class are separate runs (not run yet) |
| 3 | HGD is executed movement | **Paper text.** Noted in every HGD config and README |
| 4 | PhysioNet runs, classes, N = 109 vs 10 | **Code ready, not run.** 2- and 4-class definitions, 105 subjects, exclusions and trial counts (T0) are defined |
| 5 | 10 vs 12 channels, 22 vs 64 baseline | **Done.** 12 channels everywhere; baselines labelled 22ch (BNCI, HGD) / 64ch (PhysioNet); T2 lists every montage per subject and method |
| 6 | Nested evaluation, test-set saliency | **Done.** Saliency and checkpoints from validation trials; test touched once; montages leave-one-subject-out |
| 7 | Unsigned saliency vs signed ERD | **Done.** Sign-invariant magnitude / class-specific / lateralisation metrics; the old statistic is shown only for comparison |
| 8 | ERD as "ground truth", faithfulness, circular selection | **Done.** Deletion and randomisation tests; attribution methods are not chosen by their ERD correlation. Wording ("reference", not "ground truth") is a paper edit |
| 9 | Participant-level correlations, correction | **Done.** T7, T7b and cross-model C2 with Fisher-z CIs; Holm in T4. Remove the unreported temporal / centre-of-mass analyses from Methods |
| 10 | Margin of 6 pp, individual losses | **Mostly done.** T3 gives the 90 % CI, worst subject, subjects losing > 6 pp and the margins at which equivalence holds. The justification of 6 pp is still the authors' to give; narrow the claims |
| 11 | "Outperforms" not supported | **Results confirm the reviewer** (point 2 above). Paper text must change |
| 12 | CSP / ReliefF / MI specs, fixed and random montages, ERD-only control | **Done.** All settings in script headers and `results.json`; three controls run for every model |
| 13 | ERD from unstandardised data, intervals, artefact check | **ERD part done** (raw µV, −0.5–0 s baseline, 1–3 s active, 8–30 Hz). **Artefact-screened sensitivity analysis is still open** |
| 14 | Seeds, final configurations | **Done.** 3 seeds with ensemble and per-seed table (T6); full configuration per run (T9) |
| 15 | Class balance, chance, majority, confusions, re-initialisation | **Done.** T0 (counts per subject), T5 (confusions, per subject in CSV), models rebuilt per seed |
| 16 | Deployment, calibration, new-user transfer | **Partly.** Leave-one-subject-out montages test transfer to a new user; BNCI tests montages across sessions. The claims about clinical hardware must be toned down in the paper |

## Still open
1. HGD: EEG-ITNet and EEGConformer. PhysioNet: all four models.
2. Artefact-screened sensitivity analysis (comment 13).
3. Deep4Net: a lower-learning-rate candidate, then a re-run of the tuning and saliency steps.
4. Paper: replace Tables 1–6 with the regenerated tables and apply the wording changes listed above.
