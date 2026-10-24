# Results report — EEGItNet\PhysioNet

Generated 2026-10-22 03:57 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Protocol: run-wise chronological split per subject: test = the last imagery run of each type (run 12 left/right, run 14 hands/feet); training = runs 4, 8 (and 6, 10); validation = the chronologically last 25% of the training trials of EACH class; refit on all training runs. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects (n = 105). Random = mean of the random-montage draws.

| Subject | 64ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|
| S01 | 46.67 | 33.33 | 60.00 | 60.00 | 53.33 | 46.67 | 40.00 | 44.00 |
| S02 | 66.67 | 60.00 | 46.67 | 53.33 | 66.67 | 46.67 | 53.33 | 54.67 |
| S03 | 40.00 | 33.33 | 40.00 | 53.33 | 60.00 | 33.33 | 20.00 | 38.67 |
| S04 | 73.33 | 53.33 | 53.33 | 73.33 | 60.00 | 53.33 | 46.67 | 56.00 |
| S05 | 53.33 | 40.00 | 46.67 | 53.33 | 33.33 | 60.00 | 53.33 | 46.67 |
| S06 | 60.00 | 46.67 | 66.67 | 66.67 | 60.00 | 53.33 | 66.67 | 64.00 |
| S07 | 86.67 | 66.67 | 60.00 | 33.33 | 73.33 | 60.00 | 46.67 | 62.67 |
| S08 | 46.67 | 33.33 | 40.00 | 40.00 | 33.33 | 46.67 | 33.33 | 36.00 |
| S09 | 46.67 | 46.67 | 46.67 | 53.33 | 40.00 | 46.67 | 53.33 | 56.00 |
| S10 | 40.00 | 60.00 | 60.00 | 26.67 | 46.67 | 40.00 | 40.00 | 45.33 |
| S11 | 40.00 | 33.33 | 46.67 | 46.67 | 33.33 | 40.00 | 40.00 | 44.00 |
| S12 | 53.33 | 46.67 | 60.00 | 40.00 | 46.67 | 60.00 | 60.00 | 50.67 |
| S13 | 40.00 | 46.67 | 33.33 | 53.33 | 53.33 | 60.00 | 46.67 | 53.33 |
| S14 | 46.67 | 60.00 | 53.33 | 60.00 | 60.00 | 53.33 | 53.33 | 53.33 |
| S15 | 40.00 | 60.00 | 53.33 | 60.00 | 66.67 | 40.00 | 80.00 | 56.00 |
| S16 | 53.33 | 53.33 | 46.67 | 53.33 | 53.33 | 53.33 | 53.33 | 48.00 |
| S17 | 53.33 | 46.67 | 60.00 | 53.33 | 53.33 | 46.67 | 53.33 | 58.67 |
| S18 | 93.33 | 66.67 | 86.67 | 73.33 | 93.33 | 66.67 | 73.33 | 84.00 |
| S19 | 40.00 | 40.00 | 66.67 | 66.67 | 53.33 | 46.67 | 40.00 | 50.67 |
| S20 | 66.67 | 53.33 | 60.00 | 66.67 | 60.00 | 60.00 | 46.67 | 64.00 |
| S21 | 53.33 | 46.67 | 66.67 | 53.33 | 46.67 | 53.33 | 53.33 | 53.33 |
| S22 | 46.67 | 66.67 | 53.33 | 46.67 | 46.67 | 53.33 | 53.33 | 61.33 |
| S23 | 46.67 | 60.00 | 46.67 | 53.33 | 66.67 | 53.33 | 46.67 | 56.00 |
| S24 | 33.33 | 53.33 | 53.33 | 53.33 | 46.67 | 40.00 | 53.33 | 41.33 |
| S25 | 46.67 | 60.00 | 60.00 | 60.00 | 40.00 | 60.00 | 53.33 | 57.33 |
| S26 | 60.00 | 46.67 | 33.33 | 46.67 | 40.00 | 26.67 | 46.67 | 41.33 |
| S27 | 46.67 | 60.00 | 40.00 | 46.67 | 53.33 | 60.00 | 60.00 | 49.33 |
| S28 | 53.33 | 40.00 | 46.67 | 73.33 | 53.33 | 53.33 | 60.00 | 58.67 |
| S29 | 53.33 | 46.67 | 60.00 | 46.67 | 60.00 | 53.33 | 53.33 | 49.33 |
| S30 | 73.33 | 60.00 | 66.67 | 60.00 | 60.00 | 53.33 | 60.00 | 60.00 |
| S31 | 60.00 | 40.00 | 46.67 | 46.67 | 60.00 | 46.67 | 53.33 | 50.67 |
| S32 | 53.33 | 60.00 | 93.33 | 73.33 | 73.33 | 53.33 | 80.00 | 72.00 |
| S33 | 46.67 | 40.00 | 40.00 | 40.00 | 60.00 | 53.33 | 46.67 | 41.33 |
| S34 | 40.00 | 40.00 | 46.67 | 60.00 | 53.33 | 53.33 | 20.00 | 41.33 |
| S35 | 46.67 | 53.33 | 46.67 | 46.67 | 46.67 | 46.67 | 46.67 | 53.33 |
| S36 | 80.00 | 80.00 | 73.33 | 73.33 | 73.33 | 66.67 | 73.33 | 73.33 |
| S37 | 40.00 | 33.33 | 33.33 | 40.00 | 33.33 | 40.00 | 33.33 | 34.67 |
| S38 | 53.33 | 46.67 | 46.67 | 46.67 | 46.67 | 60.00 | 46.67 | 53.33 |
| S39 | 53.33 | 73.33 | 66.67 | 66.67 | 66.67 | 53.33 | 66.67 | 57.33 |
| S40 | 60.00 | 66.67 | 80.00 | 66.67 | 73.33 | 80.00 | 40.00 | 72.00 |
| S41 | 60.00 | 53.33 | 60.00 | 53.33 | 53.33 | 46.67 | 66.67 | 58.67 |
| S42 | 60.00 | 53.33 | 53.33 | 66.67 | 53.33 | 73.33 | 53.33 | 68.00 |
| S43 | 66.67 | 53.33 | 46.67 | 46.67 | 53.33 | 60.00 | 53.33 | 52.00 |
| S44 | 26.67 | 60.00 | 40.00 | 46.67 | 53.33 | 53.33 | 60.00 | 57.33 |
| S45 | 66.67 | 46.67 | 40.00 | 60.00 | 40.00 | 46.67 | 46.67 | 48.00 |
| S46 | 53.33 | 46.67 | 46.67 | 40.00 | 46.67 | 53.33 | 53.33 | 48.00 |
| S47 | 66.67 | 53.33 | 66.67 | 60.00 | 46.67 | 53.33 | 60.00 | 54.67 |
| S48 | 46.67 | 60.00 | 66.67 | 66.67 | 53.33 | 53.33 | 53.33 | 56.00 |
| S49 | 66.67 | 60.00 | 53.33 | 66.67 | 60.00 | 73.33 | 53.33 | 64.00 |
| S50 | 73.33 | 26.67 | 46.67 | 60.00 | 46.67 | 46.67 | 73.33 | 49.33 |
| S51 | 60.00 | 46.67 | 60.00 | 66.67 | 66.67 | 60.00 | 53.33 | 54.67 |
| S52 | 40.00 | 26.67 | 53.33 | 33.33 | 53.33 | 46.67 | 46.67 | 42.67 |
| S53 | 40.00 | 46.67 | 46.67 | 46.67 | 60.00 | 33.33 | 66.67 | 50.67 |
| S54 | 53.33 | 60.00 | 46.67 | 53.33 | 60.00 | 53.33 | 73.33 | 57.33 |
| S55 | 73.33 | 73.33 | 46.67 | 60.00 | 60.00 | 73.33 | 60.00 | 65.33 |
| S56 | 60.00 | 40.00 | 46.67 | 53.33 | 53.33 | 46.67 | 66.67 | 48.00 |
| S57 | 86.67 | 86.67 | 86.67 | 86.67 | 80.00 | 73.33 | 60.00 | 82.67 |
| S58 | 66.67 | 73.33 | 53.33 | 66.67 | 60.00 | 60.00 | 66.67 | 52.00 |
| S59 | 53.33 | 53.33 | 53.33 | 66.67 | 66.67 | 53.33 | 60.00 | 53.33 |
| S60 | 66.67 | 53.33 | 53.33 | 53.33 | 73.33 | 60.00 | 66.67 | 60.00 |
| S61 | 53.33 | 46.67 | 53.33 | 40.00 | 33.33 | 40.00 | 40.00 | 41.33 |
| S62 | 40.00 | 60.00 | 46.67 | 73.33 | 40.00 | 60.00 | 60.00 | 52.00 |
| S63 | 33.33 | 40.00 | 60.00 | 46.67 | 66.67 | 53.33 | 53.33 | 48.00 |
| S64 | 80.00 | 86.67 | 60.00 | 93.33 | 73.33 | 73.33 | 93.33 | 80.00 |
| S65 | 60.00 | 53.33 | 53.33 | 46.67 | 60.00 | 46.67 | 46.67 | 53.33 |
| S66 | 60.00 | 60.00 | 60.00 | 66.67 | 60.00 | 46.67 | 60.00 | 50.67 |
| S67 | 80.00 | 73.33 | 73.33 | 73.33 | 66.67 | 53.33 | 66.67 | 62.67 |
| S68 | 53.33 | 46.67 | 60.00 | 66.67 | 46.67 | 40.00 | 33.33 | 49.33 |
| S69 | 73.33 | 66.67 | 46.67 | 66.67 | 73.33 | 53.33 | 73.33 | 64.00 |
| S70 | 40.00 | 60.00 | 66.67 | 46.67 | 46.67 | 53.33 | 60.00 | 48.00 |
| S71 | 53.33 | 40.00 | 40.00 | 26.67 | 46.67 | 60.00 | 53.33 | 57.33 |
| S72 | 66.67 | 66.67 | 73.33 | 73.33 | 73.33 | 80.00 | 66.67 | 66.67 |
| S73 | 66.67 | 40.00 | 46.67 | 46.67 | 40.00 | 60.00 | 33.33 | 45.33 |
| S74 | 46.67 | 60.00 | 46.67 | 46.67 | 60.00 | 20.00 | 40.00 | 49.33 |
| S75 | 66.67 | 60.00 | 60.00 | 60.00 | 66.67 | 53.33 | 46.67 | 56.00 |
| S76 | 46.67 | 46.67 | 46.67 | 46.67 | 46.67 | 60.00 | 46.67 | 52.00 |
| S77 | 46.67 | 53.33 | 60.00 | 53.33 | 53.33 | 60.00 | 53.33 | 53.33 |
| S78 | 53.33 | 53.33 | 53.33 | 53.33 | 46.67 | 53.33 | 46.67 | 52.00 |
| S79 | 40.00 | 46.67 | 53.33 | 46.67 | 40.00 | 46.67 | 46.67 | 44.00 |
| S80 | 33.33 | 40.00 | 60.00 | 53.33 | 53.33 | 26.67 | 46.67 | 50.67 |
| S81 | 53.33 | 66.67 | 46.67 | 60.00 | 60.00 | 53.33 | 66.67 | 52.00 |
| S82 | 60.00 | 46.67 | 66.67 | 60.00 | 60.00 | 53.33 | 46.67 | 50.67 |
| S83 | 53.33 | 33.33 | 53.33 | 46.67 | 46.67 | 53.33 | 46.67 | 53.33 |
| S84 | 40.00 | 60.00 | 40.00 | 53.33 | 40.00 | 46.67 | 60.00 | 48.00 |
| S85 | 60.00 | 53.33 | 66.67 | 53.33 | 60.00 | 73.33 | 53.33 | 57.33 |
| S86 | 60.00 | 60.00 | 60.00 | 60.00 | 53.33 | 46.67 | 46.67 | 61.33 |
| S87 | 66.67 | 60.00 | 53.33 | 60.00 | 66.67 | 66.67 | 66.67 | 65.33 |
| S89 | 60.00 | 60.00 | 60.00 | 46.67 | 53.33 | 73.33 | 53.33 | 57.33 |
| S90 | 60.00 | 60.00 | 40.00 | 40.00 | 60.00 | 46.67 | 46.67 | 57.33 |
| S91 | 53.33 | 60.00 | 66.67 | 53.33 | 60.00 | 60.00 | 53.33 | 54.67 |
| S93 | 66.67 | 53.33 | 66.67 | 53.33 | 53.33 | 33.33 | 60.00 | 49.33 |
| S94 | 46.67 | 46.67 | 60.00 | 80.00 | 66.67 | 53.33 | 66.67 | 62.67 |
| S95 | 40.00 | 53.33 | 60.00 | 60.00 | 66.67 | 40.00 | 40.00 | 60.00 |
| S96 | 60.00 | 60.00 | 86.67 | 66.67 | 66.67 | 66.67 | 80.00 | 62.67 |
| S97 | 60.00 | 46.67 | 53.33 | 60.00 | 60.00 | 40.00 | 53.33 | 53.33 |
| S98 | 66.67 | 66.67 | 53.33 | 73.33 | 66.67 | 53.33 | 73.33 | 70.67 |
| S99 | 60.00 | 60.00 | 60.00 | 66.67 | 46.67 | 53.33 | 53.33 | 57.33 |
| S101 | 46.67 | 53.33 | 86.67 | 66.67 | 73.33 | 53.33 | 73.33 | 57.33 |
| S102 | 53.33 | 53.33 | 53.33 | 73.33 | 73.33 | 66.67 | 53.33 | 64.00 |
| S103 | 26.67 | 46.67 | 40.00 | 46.67 | 46.67 | 60.00 | 46.67 | 41.33 |
| S105 | 46.67 | 53.33 | 53.33 | 66.67 | 46.67 | 53.33 | 53.33 | 52.00 |
| S106 | 46.67 | 40.00 | 53.33 | 40.00 | 53.33 | 46.67 | 60.00 | 49.33 |
| S107 | 26.67 | 40.00 | 26.67 | 53.33 | 33.33 | 33.33 | 40.00 | 30.67 |
| S108 | 46.67 | 46.67 | 46.67 | 46.67 | 53.33 | 53.33 | 53.33 | 49.33 |
| S109 | 53.33 | 60.00 | 73.33 | 60.00 | 40.00 | 53.33 | 53.33 | 58.67 |
| Mean | 54.54 | 53.02 | 55.24 | 56.32 | 55.62 | 53.02 | 54.35 | 54.46 |
| SD | 13.13 | 11.71 | 12.36 | 12.05 | 11.65 | 10.99 | 12.26 | 9.37 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 105 | P2 T10 AF4 FC5 P5 FC6 PO7 FC3 FT8 C2 C1 CP2 |
| CSP | 27 | Fpz Fp1 AF7 Fp2 AFz AF3 AF8 F8 AF4 F7 F1 F5 |
| RLF | 86 | F7 CP6 C4 CP4 F8 P4 FT7 C6 FT8 P6 FC4 PO4 |
| MI | 94 | F7 F8 F5 P6 FT7 C4 P4 F2 CP4 P2 CP2 FC4 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 8 | PO8 O2 O1 PO4 Oz POz PO7 PO3 Fpz P4 P2 Fp2 |

### T3. Equivalence of each 12-channel montage with all 64 channels (paired TOST)

Difference = montage − 64ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 105 | -1.52 | 12.72 | [-3.58, +0.54] | 0.0002 | yes | -46.67 | 49 | 4,5,6,8,10 |
| CSP | 105 | +0.70 | 14.20 | [-1.60, +3.00] | 0.0001 | yes | -26.67 | 41 | 3,4,5,6,8,10 |
| RLF | 105 | +1.78 | 13.75 | [-0.45, +4.01] | 0.0011 | yes | -53.33 | 37 | 5,6,8,10 |
| MI | 105 | +1.08 | 12.51 | [-0.95, +3.11] | <0.0001 | yes | -26.67 | 40 | 4,5,6,8,10 |
| Fixed sensorimotor | 105 | -1.52 | 12.92 | [-3.62, +0.57] | 0.0003 | yes | -33.33 | 40 | 4,5,6,8,10 |
| ERD-only | 105 | -0.19 | 14.16 | [-2.48, +2.10] | <0.0001 | yes | -40.00 | 41 | 3,4,5,6,8,10 |
| Random (mean of draws) | 105 | -0.08 | 10.18 | [-1.72, +1.57] | <0.0001 | yes | -24.00 | 28 | 2,3,4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 105 | -2.22 | [-4.73, +0.29] | 0.0820 | 0.3279 | 0.1102 | 0.4269 | 31 / 27 / 47 |
| RLF | 105 | -3.30 | [-5.64, -0.97] | 0.0060 | 0.0360 | 0.0037 | 0.0224 | 26 / 30 / 49 |
| MI | 105 | -2.60 | [-4.80, -0.41] | 0.0206 | 0.1028 | 0.0331 | 0.1655 | 33 / 26 / 46 |
| Fixed sensorimotor | 105 | +0.00 | [-2.32, +2.32] | 1.0000 | 1.0000 | 0.8742 | 0.8742 | 43 / 18 / 44 |
| ERD-only | 105 | -1.33 | [-3.63, +0.97] | 0.2531 | 0.5063 | 0.2922 | 0.5844 | 35 / 29 / 41 |
| Random (mean of draws) | 105 | -1.45 | [-3.09, +0.20] | 0.0835 | 0.3279 | 0.1067 | 0.4269 | 43 / 6 / 56 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 64ch | 54.54 | left_hand 53.3, right_hand 55.8 | left_hand 48.8, right_hand 51.2 | 53.33 | 13 / 105 |
| Ours | 53.02 | left_hand 57.1, right_hand 48.8 | left_hand 54.2, right_hand 45.8 | 53.33 | 17 / 105 |
| CSP | 55.24 | left_hand 57.4, right_hand 53.1 | left_hand 52.2, right_hand 47.8 | 53.33 | 13 / 105 |
| RLF | 56.32 | left_hand 62.3, right_hand 50.3 | left_hand 56.1, right_hand 43.9 | 53.33 | 12 / 105 |
| MI | 55.62 | left_hand 57.6, right_hand 53.6 | left_hand 52.1, right_hand 47.9 | 53.33 | 12 / 105 |
| Fixed sensorimotor | 53.02 | left_hand 53.7, right_hand 52.3 | left_hand 50.7, right_hand 49.3 | 53.33 | 17 / 105 |
| ERD-only | 54.35 | left_hand 57.3, right_hand 51.4 | left_hand 53.0, right_hand 47.0 | 53.33 | 14 / 105 |
| Random (mean of draws) | 54.46 | left_hand 55.2, right_hand 53.7 | left_hand 50.8, right_hand 49.2 | 53.33 | 72 / 525 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 64ch | 3 | 42: 53.46, 43: 53.33, 44: 51.81 | 52.87 | 0.92 |
| Ours | 3 | 42: 53.33, 43: 53.65, 44: 52.76 | 53.25 | 0.45 |
| CSP | 3 | 42: 54.48, 43: 54.03, 44: 54.54 | 54.35 | 0.28 |
| RLF | 3 | 42: 54.41, 43: 56.13, 44: 54.60 | 55.05 | 0.94 |
| MI | 3 | 42: 55.62, 43: 54.60, 44: 55.43 | 55.22 | 0.54 |
| Fixed sensorimotor | 3 | 42: 52.51, 43: 53.78, 44: 52.70 | 52.99 | 0.68 |
| ERD-only | 3 | 42: 55.56, 43: 55.05, 44: 52.76 | 54.46 | 1.49 |
| Random (mean of draws) | 3 | 42: 54.30, 43: 53.69, 44: 54.04 | 54.01 | 0.31 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full64/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | -0.013 | [-0.039, 0.012] | 0.130 | 105 |
| grad | class_specific_pearson | 0.013 | [-0.004, 0.030] | 0.087 | 105 |
| grad | lateralisation_pearson | -0.012 | [-0.038, 0.013] | 0.128 | 105 |
| grad | magnitude_spearman | -0.022 | [-0.049, 0.005] | 0.137 | 105 |
| grad | class_specific_spearman | 0.022 | [0.006, 0.039] | 0.084 | 105 |
| grad | lateralisation_spearman | 0.001 | [-0.023, 0.025] | 0.123 | 105 |
| grad | legacy_signed_LR_pearson | 0.008 | [-0.021, 0.037] | 0.147 | 105 |
| grad_x_input | magnitude_pearson | -0.052 | [-0.091, -0.012] | 0.196 | 105 |
| grad_x_input | class_specific_pearson | 0.009 | [-0.023, 0.042] | 0.162 | 105 |
| grad_x_input | lateralisation_pearson | -0.165 | [-0.235, -0.093] | 0.334 | 105 |
| grad_x_input | magnitude_spearman | -0.054 | [-0.095, -0.013] | 0.200 | 105 |
| grad_x_input | class_specific_spearman | 0.014 | [-0.019, 0.047] | 0.161 | 105 |
| grad_x_input | lateralisation_spearman | -0.148 | [-0.217, -0.078] | 0.327 | 105 |
| grad_x_input | legacy_signed_LR_pearson | -0.002 | [-0.041, 0.036] | 0.192 | 105 |
| integrated_gradients | magnitude_pearson | -0.053 | [-0.090, -0.016] | 0.185 | 105 |
| integrated_gradients | class_specific_pearson | 0.008 | [-0.022, 0.038] | 0.151 | 105 |
| integrated_gradients | lateralisation_pearson | -0.165 | [-0.236, -0.093] | 0.335 | 105 |
| integrated_gradients | magnitude_spearman | -0.056 | [-0.094, -0.017] | 0.191 | 105 |
| integrated_gradients | class_specific_spearman | 0.013 | [-0.018, 0.043] | 0.151 | 105 |
| integrated_gradients | lateralisation_spearman | -0.156 | [-0.224, -0.086] | 0.327 | 105 |
| integrated_gradients | legacy_signed_LR_pearson | -0.001 | [-0.039, 0.037] | 0.190 | 105 |
| smoothgrad | magnitude_pearson | -0.013 | [-0.039, 0.012] | 0.130 | 105 |
| smoothgrad | class_specific_pearson | 0.013 | [-0.004, 0.030] | 0.087 | 105 |
| smoothgrad | lateralisation_pearson | -0.012 | [-0.037, 0.013] | 0.128 | 105 |
| smoothgrad | magnitude_spearman | -0.022 | [-0.049, 0.005] | 0.137 | 105 |
| smoothgrad | class_specific_spearman | 0.022 | [0.006, 0.039] | 0.084 | 105 |
| smoothgrad | lateralisation_spearman | 0.001 | [-0.023, 0.025] | 0.122 | 105 |
| smoothgrad | legacy_signed_LR_pearson | 0.008 | [-0.021, 0.037] | 0.147 | 105 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.013 | [-0.035, 0.009] | 105 |
| grad | ERD_L_spearman | -0.020 | [-0.042, 0.003] | 105 |
| grad | ERD_R_pearson | -0.015 | [-0.039, 0.010] | 105 |
| grad | ERD_R_spearman | -0.028 | [-0.051, -0.004] | 105 |
| grad | ERD_LR_pearson | 0.008 | [-0.021, 0.037] | 105 |
| grad | ERD_LR_spearman | 0.007 | [-0.021, 0.036] | 105 |
| grad | ERD_comb_pearson | -0.019 | [-0.041, 0.002] | 105 |
| grad | ERD_comb_spearman | -0.030 | [-0.053, -0.007] | 105 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |
| grad_x_input | ERD_L_pearson | -0.040 | [-0.078, -0.001] | 105 |
| grad_x_input | ERD_L_spearman | -0.035 | [-0.074, 0.004] | 105 |
| grad_x_input | ERD_R_pearson | -0.029 | [-0.070, 0.013] | 105 |
| grad_x_input | ERD_R_spearman | -0.035 | [-0.076, 0.006] | 105 |
| grad_x_input | ERD_LR_pearson | -0.002 | [-0.041, 0.036] | 105 |
| grad_x_input | ERD_LR_spearman | 0.001 | [-0.037, 0.039] | 105 |
| grad_x_input | ERD_comb_pearson | -0.052 | [-0.092, -0.013] | 105 |
| grad_x_input | ERD_comb_spearman | -0.049 | [-0.090, -0.008] | 105 |
| grad_x_input | vs_grad_pearson | 0.926 | [0.913, 0.938] | 105 |
| grad_x_input | vs_grad_spearman | 0.917 | [0.903, 0.929] | 105 |
| integrated_gradients | ERD_L_pearson | -0.038 | [-0.075, -0.002] | 105 |
| integrated_gradients | ERD_L_spearman | -0.035 | [-0.072, 0.002] | 105 |
| integrated_gradients | ERD_R_pearson | -0.025 | [-0.065, 0.015] | 105 |
| integrated_gradients | ERD_R_spearman | -0.035 | [-0.073, 0.004] | 105 |
| integrated_gradients | ERD_LR_pearson | -0.001 | [-0.039, 0.037] | 105 |
| integrated_gradients | ERD_LR_spearman | 0.006 | [-0.031, 0.043] | 105 |
| integrated_gradients | ERD_comb_pearson | -0.049 | [-0.087, -0.012] | 105 |
| integrated_gradients | ERD_comb_spearman | -0.049 | [-0.087, -0.010] | 105 |
| integrated_gradients | vs_grad_pearson | 0.921 | [0.908, 0.932] | 105 |
| integrated_gradients | vs_grad_spearman | 0.909 | [0.896, 0.921] | 105 |
| smoothgrad | ERD_L_pearson | -0.013 | [-0.035, 0.009] | 105 |
| smoothgrad | ERD_L_spearman | -0.019 | [-0.042, 0.003] | 105 |
| smoothgrad | ERD_R_pearson | -0.015 | [-0.039, 0.010] | 105 |
| smoothgrad | ERD_R_spearman | -0.027 | [-0.051, -0.003] | 105 |
| smoothgrad | ERD_LR_pearson | 0.008 | [-0.021, 0.037] | 105 |
| smoothgrad | ERD_LR_spearman | 0.007 | [-0.021, 0.036] | 105 |
| smoothgrad | ERD_comb_pearson | -0.019 | [-0.040, 0.002] | 105 |
| smoothgrad | ERD_comb_spearman | -0.029 | [-0.052, -0.006] | 105 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.050 | [-0.085, -0.015] | -0.018 | -0.035 | -0.001 | 105 |
| ERD_theta_R | -0.043 | [-0.077, -0.009] | -0.040 | -0.035 | -0.037 | 105 |
| ERD_theta_mag | -0.036 | [-0.068, -0.004] | -0.036 | -0.028 | -0.028 | 105 |
| ERD_mu_L | -0.029 | [-0.058, -0.001] | -0.036 | -0.027 | -0.039 | 105 |
| ERD_mu_R | -0.030 | [-0.059, -0.001] | -0.013 | -0.038 | -0.008 | 105 |
| ERD_mu_mag | -0.031 | [-0.060, -0.001] | -0.031 | -0.031 | -0.031 | 105 |
| ERD_beta_L | -0.023 | [-0.053, 0.006] | -0.007 | -0.026 | -0.002 | 105 |
| ERD_beta_R | -0.004 | [-0.034, 0.027] | -0.017 | -0.010 | -0.008 | 105 |
| ERD_beta_mag | -0.012 | [-0.041, 0.017] | -0.012 | -0.007 | -0.007 | 105 |
| ERS_beta_rebound_L | -0.021 | [-0.049, 0.008] | 0.016 | -0.023 | 0.020 | 105 |
| ERS_beta_rebound_R | -0.036 | [-0.067, -0.005] | -0.009 | -0.038 | -0.007 | 105 |
| ERS_beta_rebound_mag | 0.006 | [-0.024, 0.036] | 0.006 | 0.009 | 0.009 | 105 |
| ERD_lowgamma_L | 0.010 | [-0.017, 0.036] | -0.008 | 0.014 | -0.004 | 105 |
| ERD_lowgamma_R | 0.001 | [-0.027, 0.029] | 0.004 | -0.006 | 0.005 | 105 |
| ERD_lowgamma_mag | -0.003 | [-0.032, 0.026] | -0.003 | -0.003 | -0.003 | 105 |
| logpow_8_30 | 0.003 | [-0.027, 0.033] | 0.004 | 0.016 | 0.016 | 105 |
| r2_signed | 0.008 | [-0.021, 0.038] | 0.004 | 0.001 | -0.001 | 105 |
| r2_abs | 0.004 | [-0.026, 0.033] | 0.004 | -0.001 | -0.001 | 105 |
| TFR_ERD_dB_L | -0.031 | [-0.059, -0.004] | -0.033 | -0.033 | -0.033 | 105 |
| TFR_ERD_dB_R | -0.018 | [-0.046, 0.011] | -0.013 | -0.025 | -0.011 | 105 |
| TFR_ERD_dB_mag | -0.030 | [-0.058, -0.002] | -0.030 | -0.029 | -0.029 | 105 |
| ITC_4_30 | -0.024 | [-0.055, 0.006] | -0.024 | -0.024 | -0.024 | 105 |
| ERP_abs_0.2_0.8 | -0.017 | [-0.046, 0.013] | -0.017 | -0.005 | -0.005 | 105 |
| SlowPot_0.5_3 | -0.032 | [-0.064, 0.001] | -0.029 | -0.038 | -0.025 | 105 |
| ERD_beta_classspecific | — | — | -0.007 [-0.029, 0.014] | — | -0.001 | 105 |
| ERD_lowgamma_classspecific | — | — | 0.000 [-0.021, 0.022] | — | 0.001 | 105 |
| ERD_mu_classspecific | — | — | -0.025 [-0.047, -0.003] | — | -0.024 | 105 |
| ERD_theta_classspecific | — | — | -0.029 [-0.055, -0.004] | — | -0.020 | 105 |
| ERS_beta_rebound_classspecific | — | — | 0.005 [-0.018, 0.028] | — | 0.009 | 105 |
| TFR_ERD_dB_classspecific | — | — | -0.022 [-0.043, -0.002] | — | -0.021 | 105 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +0.25 ± 4.93 | +0.07 ± 2.14 | +0.04 ± 2.56 | 0.6514 | 0.6725 |
| 4 | -0.13 ± 4.89 | +0.17 ± 2.42 | -0.44 ± 3.18 | 0.4365 | 0.5231 |
| 6 | +0.42 ± 6.19 | +0.00 ± 2.87 | -0.40 ± 3.51 | 0.3600 | 0.1617 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.010 | [-0.025, 0.005] | 105 |
| grad_x_input | 0.078 | [0.057, 0.099] | 105 |
| integrated_gradients | 0.071 | [0.049, 0.092] | 105 |
| smoothgrad | -0.010 | [-0.025, 0.005] | 105 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: False): S01=zscore, S02=zscore, S03=ems, S04=zscore, S05=ems, S06=zscore, S07=zscore, S08=ems, S09=zscore, S10=zscore, S11=ems, S12=zscore, S13=zscore, S14=zscore, S15=zscore, S16=zscore, S17=ems, S18=zscore, S19=zscore, S20=zscore, S21=ems, S22=zscore, S23=zscore, S24=zscore, S25=zscore, S26=zscore, S27=zscore, S28=zscore, S29=zscore, S30=zscore, S31=zscore, S32=zscore, S33=zscore, S34=ems, S35=zscore, S36=ems, S37=ems, S38=zscore, S39=zscore, S40=ems, S41=ems, S42=zscore, S43=zscore, S44=ems, S45=zscore, S46=zscore, S47=zscore, S48=ems, S49=ems, S50=zscore, S51=zscore, S52=zscore, S53=zscore, S54=ems, S55=zscore, S56=zscore, S57=ems, S58=zscore, S59=zscore, S60=zscore, S61=zscore, S62=zscore, S63=ems, S64=zscore, S65=ems, S66=zscore, S67=zscore, S68=zscore, S69=zscore, S70=ems, S71=zscore, S72=zscore, S73=zscore, S74=zscore, S75=zscore, S76=zscore, S77=zscore, S78=zscore, S79=zscore, S80=zscore, S81=zscore, S82=zscore, S83=ems, S84=zscore, S85=zscore, S86=zscore, S87=zscore, S89=zscore, S90=zscore, S91=zscore, S93=zscore, S94=zscore, S95=zscore, S96=zscore, S97=zscore, S98=zscore, S99=zscore, S101=zscore, S102=zscore, S103=zscore, S105=zscore, S106=zscore, S107=zscore, S108=zscore, S109=zscore

| Profile | Mean validation accuracy (%) |
|---|---|
| zscore | 65.60 |
| ems | 65.60 |
| original | 59.76 |

```
model: "EEGITNet"
dataset: "PhysionetMI"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"zscore": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": null, "align": "zscore", "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "ems": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": "ems", "align": null, "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "original": {"sfreq": 160, "band_hz": [8.0, 32.0], "window_s": [0.5, 2.5], "norm": null, "align": null, "car": false, "dropout": 0.25, "lr": 0.01, "weight_decay": 1e-05, "scheduler": "multistep", "milestones": [20, 50], "gamma": 0.1, "max_epochs": 100, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "augment": false, "select_on": "val_loss", "min_select_epoch": 1}}
checkpoint_rule: "highest validation accuracy (loss breaks ties), not before epoch 20 ('original': lowest validation loss); refit keeps the final weights"
folds: "run-wise chronological split per subject: test = the last imagery run of each type (run 12 left/right, run 14 hands/feet); training = runs 4, 8 (and 6, 10); validation = the chronologically last 25% of the training trials of EACH class; refit on all training runs"
seeds: [42, 43, 44]
```

Figures (530 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `2class\figures`.

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Protocol: run-wise chronological split per subject: test = the last imagery run of each type (run 12 left/right, run 14 hands/feet); training = runs 4, 8 (and 6, 10); validation = the chronologically last 25% of the training trials of EACH class; refit on all training runs. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects (n = 105). Random = mean of the random-montage draws.

| Subject | 64ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|
| S01 | 36.67 | 26.67 | 46.67 | 16.67 | 36.67 | 46.67 | 23.33 | 36.67 |
| S02 | 16.67 | 23.33 | 30.00 | 23.33 | 36.67 | 36.67 | 26.67 | 22.67 |
| S03 | 20.00 | 20.00 | 20.00 | 26.67 | 23.33 | 23.33 | 16.67 | 29.33 |
| S04 | 36.67 | 13.33 | 33.33 | 30.00 | 36.67 | 23.33 | 26.67 | 34.67 |
| S05 | 13.33 | 23.33 | 16.67 | 20.00 | 26.67 | 26.67 | 23.33 | 22.00 |
| S06 | 40.00 | 40.00 | 36.67 | 30.00 | 36.67 | 30.00 | 50.00 | 41.33 |
| S07 | 40.00 | 40.00 | 33.33 | 50.00 | 40.00 | 36.67 | 36.67 | 27.33 |
| S08 | 20.00 | 33.33 | 20.00 | 10.00 | 23.33 | 33.33 | 26.67 | 26.67 |
| S09 | 10.00 | 20.00 | 23.33 | 13.33 | 30.00 | 16.67 | 30.00 | 20.67 |
| S10 | 40.00 | 30.00 | 30.00 | 43.33 | 30.00 | 26.67 | 33.33 | 31.33 |
| S11 | 20.00 | 16.67 | 23.33 | 23.33 | 20.00 | 23.33 | 26.67 | 20.00 |
| S12 | 20.00 | 26.67 | 23.33 | 33.33 | 30.00 | 23.33 | 26.67 | 22.67 |
| S13 | 26.67 | 23.33 | 26.67 | 30.00 | 36.67 | 33.33 | 26.67 | 27.33 |
| S14 | 33.33 | 26.67 | 33.33 | 26.67 | 23.33 | 20.00 | 30.00 | 29.33 |
| S15 | 53.33 | 40.00 | 36.67 | 46.67 | 36.67 | 33.33 | 50.00 | 39.33 |
| S16 | 26.67 | 36.67 | 30.00 | 30.00 | 36.67 | 26.67 | 33.33 | 26.00 |
| S17 | 26.67 | 26.67 | 20.00 | 26.67 | 30.00 | 26.67 | 26.67 | 28.00 |
| S18 | 60.00 | 46.67 | 46.67 | 70.00 | 53.33 | 36.67 | 53.33 | 36.00 |
| S19 | 20.00 | 20.00 | 23.33 | 13.33 | 10.00 | 23.33 | 26.67 | 22.00 |
| S20 | 26.67 | 23.33 | 26.67 | 23.33 | 13.33 | 30.00 | 30.00 | 29.33 |
| S21 | 26.67 | 33.33 | 20.00 | 26.67 | 23.33 | 20.00 | 30.00 | 26.00 |
| S22 | 36.67 | 26.67 | 23.33 | 40.00 | 40.00 | 33.33 | 33.33 | 34.00 |
| S23 | 26.67 | 26.67 | 23.33 | 36.67 | 23.33 | 20.00 | 26.67 | 28.67 |
| S24 | 20.00 | 20.00 | 26.67 | 23.33 | 23.33 | 26.67 | 23.33 | 24.67 |
| S25 | 33.33 | 30.00 | 40.00 | 23.33 | 33.33 | 40.00 | 33.33 | 31.33 |
| S26 | 20.00 | 36.67 | 36.67 | 26.67 | 33.33 | 16.67 | 26.67 | 25.33 |
| S27 | 26.67 | 23.33 | 26.67 | 23.33 | 30.00 | 26.67 | 20.00 | 20.00 |
| S28 | 33.33 | 33.33 | 40.00 | 26.67 | 23.33 | 33.33 | 30.00 | 36.67 |
| S29 | 36.67 | 26.67 | 26.67 | 46.67 | 36.67 | 26.67 | 30.00 | 34.00 |
| S30 | 23.33 | 33.33 | 23.33 | 16.67 | 23.33 | 16.67 | 23.33 | 25.33 |
| S31 | 33.33 | 30.00 | 30.00 | 33.33 | 13.33 | 13.33 | 33.33 | 26.67 |
| S32 | 33.33 | 33.33 | 23.33 | 23.33 | 30.00 | 26.67 | 33.33 | 30.00 |
| S33 | 30.00 | 20.00 | 23.33 | 23.33 | 30.00 | 16.67 | 33.33 | 24.67 |
| S34 | 20.00 | 23.33 | 16.67 | 16.67 | 36.67 | 13.33 | 20.00 | 20.67 |
| S35 | 63.33 | 36.67 | 46.67 | 43.33 | 53.33 | 60.00 | 33.33 | 56.67 |
| S36 | 43.33 | 33.33 | 53.33 | 36.67 | 46.67 | 40.00 | 40.00 | 37.33 |
| S37 | 20.00 | 16.67 | 20.00 | 26.67 | 26.67 | 20.00 | 16.67 | 22.67 |
| S38 | 30.00 | 30.00 | 16.67 | 36.67 | 16.67 | 23.33 | 33.33 | 31.33 |
| S39 | 26.67 | 23.33 | 23.33 | 23.33 | 23.33 | 20.00 | 26.67 | 26.67 |
| S40 | 33.33 | 23.33 | 40.00 | 30.00 | 46.67 | 30.00 | 40.00 | 38.67 |
| S41 | 33.33 | 33.33 | 36.67 | 33.33 | 20.00 | 26.67 | 36.67 | 32.00 |
| S42 | 26.67 | 26.67 | 20.00 | 23.33 | 23.33 | 53.33 | 26.67 | 36.00 |
| S43 | 13.33 | 26.67 | 16.67 | 10.00 | 26.67 | 3.33 | 16.67 | 18.00 |
| S44 | 40.00 | 33.33 | 26.67 | 36.67 | 33.33 | 23.33 | 43.33 | 32.00 |
| S45 | 46.67 | 30.00 | 13.33 | 33.33 | 40.00 | 30.00 | 23.33 | 30.67 |
| S46 | 26.67 | 20.00 | 26.67 | 36.67 | 26.67 | 30.00 | 26.67 | 30.67 |
| S47 | 33.33 | 26.67 | 16.67 | 20.00 | 26.67 | 26.67 | 26.67 | 21.33 |
| S48 | 26.67 | 30.00 | 20.00 | 30.00 | 23.33 | 33.33 | 23.33 | 31.33 |
| S49 | 33.33 | 10.00 | 20.00 | 30.00 | 30.00 | 43.33 | 23.33 | 31.33 |
| S50 | 23.33 | 30.00 | 30.00 | 26.67 | 20.00 | 23.33 | 16.67 | 22.00 |
| S51 | 26.67 | 26.67 | 26.67 | 36.67 | 30.00 | 20.00 | 26.67 | 28.00 |
| S52 | 20.00 | 26.67 | 26.67 | 20.00 | 16.67 | 26.67 | 36.67 | 20.67 |
| S53 | 30.00 | 23.33 | 30.00 | 16.67 | 30.00 | 23.33 | 23.33 | 24.67 |
| S54 | 26.67 | 36.67 | 23.33 | 20.00 | 26.67 | 26.67 | 33.33 | 32.00 |
| S55 | 33.33 | 26.67 | 33.33 | 33.33 | 33.33 | 23.33 | 33.33 | 34.67 |
| S56 | 30.00 | 33.33 | 40.00 | 36.67 | 43.33 | 33.33 | 33.33 | 31.33 |
| S57 | 60.00 | 43.33 | 33.33 | 50.00 | 56.67 | 30.00 | 40.00 | 39.33 |
| S58 | 26.67 | 26.67 | 23.33 | 26.67 | 26.67 | 33.33 | 23.33 | 29.33 |
| S59 | 36.67 | 26.67 | 23.33 | 30.00 | 26.67 | 23.33 | 36.67 | 28.00 |
| S60 | 23.33 | 26.67 | 26.67 | 23.33 | 26.67 | 30.00 | 36.67 | 22.67 |
| S61 | 30.00 | 30.00 | 26.67 | 30.00 | 30.00 | 26.67 | 33.33 | 30.00 |
| S62 | 40.00 | 30.00 | 33.33 | 36.67 | 46.67 | 33.33 | 36.67 | 38.67 |
| S63 | 36.67 | 16.67 | 26.67 | 26.67 | 20.00 | 23.33 | 23.33 | 31.33 |
| S64 | 33.33 | 33.33 | 33.33 | 30.00 | 26.67 | 36.67 | 33.33 | 30.00 |
| S65 | 33.33 | 23.33 | 26.67 | 23.33 | 20.00 | 26.67 | 23.33 | 30.00 |
| S66 | 40.00 | 30.00 | 33.33 | 40.00 | 26.67 | 30.00 | 33.33 | 29.33 |
| S67 | 20.00 | 30.00 | 26.67 | 16.67 | 40.00 | 26.67 | 20.00 | 25.33 |
| S68 | 20.00 | 20.00 | 13.33 | 23.33 | 40.00 | 30.00 | 26.67 | 26.67 |
| S69 | 23.33 | 33.33 | 20.00 | 30.00 | 30.00 | 33.33 | 33.33 | 32.00 |
| S70 | 36.67 | 33.33 | 23.33 | 30.00 | 33.33 | 16.67 | 33.33 | 30.67 |
| S71 | 36.67 | 33.33 | 23.33 | 30.00 | 33.33 | 30.00 | 50.00 | 36.00 |
| S72 | 43.33 | 40.00 | 43.33 | 36.67 | 53.33 | 60.00 | 33.33 | 38.67 |
| S73 | 16.67 | 30.00 | 13.33 | 16.67 | 16.67 | 26.67 | 6.67 | 13.33 |
| S74 | 20.00 | 30.00 | 26.67 | 23.33 | 36.67 | 10.00 | 20.00 | 24.67 |
| S75 | 33.33 | 16.67 | 40.00 | 33.33 | 30.00 | 20.00 | 30.00 | 20.00 |
| S76 | 26.67 | 30.00 | 36.67 | 36.67 | 36.67 | 33.33 | 26.67 | 30.67 |
| S77 | 26.67 | 20.00 | 36.67 | 26.67 | 30.00 | 33.33 | 13.33 | 26.00 |
| S78 | 16.67 | 30.00 | 30.00 | 26.67 | 26.67 | 30.00 | 30.00 | 28.67 |
| S79 | 26.67 | 23.33 | 13.33 | 23.33 | 30.00 | 13.33 | 26.67 | 19.33 |
| S80 | 23.33 | 23.33 | 26.67 | 33.33 | 30.00 | 30.00 | 30.00 | 31.33 |
| S81 | 30.00 | 30.00 | 33.33 | 26.67 | 23.33 | 16.67 | 16.67 | 25.33 |
| S82 | 26.67 | 20.00 | 26.67 | 26.67 | 33.33 | 23.33 | 33.33 | 26.67 |
| S83 | 26.67 | 36.67 | 26.67 | 30.00 | 26.67 | 30.00 | 23.33 | 25.33 |
| S84 | 26.67 | 20.00 | 30.00 | 30.00 | 30.00 | 26.67 | 33.33 | 36.67 |
| S85 | 30.00 | 26.67 | 30.00 | 16.67 | 26.67 | 33.33 | 36.67 | 33.33 |
| S86 | 20.00 | 23.33 | 33.33 | 40.00 | 20.00 | 23.33 | 26.67 | 28.00 |
| S87 | 36.67 | 30.00 | 33.33 | 36.67 | 23.33 | 40.00 | 30.00 | 33.33 |
| S89 | 36.67 | 50.00 | 33.33 | 40.00 | 36.67 | 26.67 | 33.33 | 32.67 |
| S90 | 10.00 | 33.33 | 20.00 | 23.33 | 36.67 | 23.33 | 26.67 | 28.00 |
| S91 | 23.33 | 33.33 | 26.67 | 33.33 | 20.00 | 10.00 | 26.67 | 28.67 |
| S93 | 30.00 | 46.67 | 30.00 | 36.67 | 30.00 | 33.33 | 33.33 | 35.33 |
| S94 | 30.00 | 26.67 | 36.67 | 30.00 | 23.33 | 36.67 | 30.00 | 31.33 |
| S95 | 30.00 | 23.33 | 26.67 | 26.67 | 23.33 | 26.67 | 26.67 | 25.33 |
| S96 | 30.00 | 23.33 | 20.00 | 16.67 | 26.67 | 30.00 | 23.33 | 29.33 |
| S97 | 40.00 | 33.33 | 36.67 | 23.33 | 20.00 | 30.00 | 36.67 | 33.33 |
| S98 | 40.00 | 33.33 | 26.67 | 26.67 | 30.00 | 26.67 | 30.00 | 26.67 |
| S99 | 30.00 | 26.67 | 30.00 | 23.33 | 26.67 | 26.67 | 26.67 | 32.67 |
| S101 | 40.00 | 33.33 | 26.67 | 56.67 | 33.33 | 33.33 | 40.00 | 30.00 |
| S102 | 33.33 | 36.67 | 43.33 | 16.67 | 26.67 | 26.67 | 36.67 | 26.00 |
| S103 | 23.33 | 23.33 | 30.00 | 26.67 | 30.00 | 30.00 | 20.00 | 25.33 |
| S105 | 30.00 | 26.67 | 26.67 | 36.67 | 23.33 | 26.67 | 33.33 | 26.67 |
| S106 | 23.33 | 40.00 | 30.00 | 30.00 | 30.00 | 33.33 | 26.67 | 31.33 |
| S107 | 3.33 | 23.33 | 13.33 | 20.00 | 10.00 | 16.67 | 30.00 | 20.00 |
| S108 | 20.00 | 33.33 | 26.67 | 23.33 | 36.67 | 40.00 | 33.33 | 36.00 |
| S109 | 23.33 | 16.67 | 26.67 | 26.67 | 26.67 | 33.33 | 23.33 | 29.33 |
| Mean | 29.49 | 28.44 | 28.10 | 28.95 | 29.71 | 27.97 | 29.49 | 29.14 |
| SD | 9.94 | 7.21 | 8.00 | 9.54 | 8.86 | 8.98 | 7.63 | 6.05 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 103 | FT8 F8 P8 T10 P2 POz TP7 Iz P5 P1 CP6 C6 |
| CSP | 22 | F1 AFz Fpz F3 AF3 Fp1 F5 Fz F4 F2 Fp2 AF4 |
| RLF | 100 | O2 F7 F8 T10 PO8 Iz T9 FT7 Oz AF8 F5 AF7 |
| MI | 104 | F7 Oz P6 Fz F8 O2 T9 C3 C6 F5 TP8 F4 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 8 | PO8 O2 O1 PO4 Oz POz PO7 PO3 Fpz P4 P2 Fp2 |

### T3. Equivalence of each 12-channel montage with all 64 channels (paired TOST)

Difference = montage − 64ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 105 | -1.05 | 9.51 | [-2.59, +0.49] | <0.0001 | yes | -26.67 | 36 | 3,4,5,6,8,10 |
| CSP | 105 | -1.40 | 8.79 | [-2.82, +0.03] | <0.0001 | yes | -33.33 | 31 | 3,4,5,6,8,10 |
| RLF | 105 | -0.54 | 8.13 | [-1.86, +0.78] | <0.0001 | yes | -20.00 | 31 | 2,3,4,5,6,8,10 |
| MI | 105 | +0.22 | 9.27 | [-1.28, +1.72] | <0.0001 | yes | -20.00 | 28 | 2,3,4,5,6,8,10 |
| Fixed sensorimotor | 105 | -1.52 | 10.14 | [-3.17, +0.12] | <0.0001 | yes | -30.00 | 42 | 4,5,6,8,10 |
| ERD-only | 105 | -0.00 | 8.42 | [-1.36, +1.36] | <0.0001 | yes | -30.00 | 23 | 2,3,4,5,6,8,10 |
| Random (mean of draws) | 105 | -0.36 | 7.23 | [-1.53, +0.82] | <0.0001 | yes | -24.00 | 19 | 2,3,4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 105 | +0.35 | [-1.34, +2.04] | 0.6824 | 1.0000 | 0.4406 | 1.0000 | 45 / 14 / 46 |
| RLF | 105 | -0.51 | [-2.32, +1.31] | 0.5797 | 1.0000 | 0.7500 | 1.0000 | 45 / 13 / 47 |
| MI | 105 | -1.27 | [-3.06, +0.52] | 0.1625 | 0.9555 | 0.1956 | 0.9779 | 40 / 17 / 48 |
| Fixed sensorimotor | 105 | +0.48 | [-1.50, +2.45] | 0.6332 | 1.0000 | 0.4815 | 1.0000 | 46 / 16 / 43 |
| ERD-only | 105 | -1.05 | [-2.51, +0.42] | 0.1593 | 0.9555 | 0.1272 | 0.7631 | 34 / 21 / 50 |
| Random (mean of draws) | 105 | -0.69 | [-2.12, +0.74] | 0.3402 | 1.0000 | 0.4346 | 1.0000 | 46 / 3 / 56 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 64ch | 29.49 | left_hand 28.1, right_hand 23.3, hands 26.8, feet 39.8 | left_hand 22.7, right_hand 21.2, hands 22.6, feet 33.4 | 26.67 | 2 / 105 |
| Ours | 28.44 | left_hand 29.6, right_hand 20.3, hands 25.9, feet 37.9 | left_hand 25.6, right_hand 18.6, hands 22.0, feet 33.7 | 26.67 | 1 / 105 |
| CSP | 28.10 | left_hand 24.8, right_hand 20.6, hands 24.7, feet 42.2 | left_hand 20.7, right_hand 19.7, hands 21.6, feet 38.1 | 26.67 | 3 / 105 |
| RLF | 28.95 | left_hand 26.0, right_hand 26.6, hands 24.6, feet 38.6 | left_hand 22.1, right_hand 22.1, hands 21.0, feet 34.8 | 26.67 | 2 / 105 |
| MI | 29.71 | left_hand 28.1, right_hand 23.0, hands 29.1, feet 38.6 | left_hand 22.8, right_hand 19.9, hands 24.0, feet 33.3 | 26.67 | 3 / 105 |
| Fixed sensorimotor | 27.97 | left_hand 26.2, right_hand 24.4, hands 29.6, feet 31.6 | left_hand 24.1, right_hand 22.4, hands 26.4, feet 27.1 | 26.67 | 3 / 105 |
| ERD-only | 29.49 | left_hand 28.2, right_hand 22.8, hands 24.6, feet 42.3 | left_hand 23.7, right_hand 20.1, hands 20.8, feet 35.4 | 26.67 | 3 / 105 |
| Random (mean of draws) | 29.14 | left_hand 28.4, right_hand 24.0, hands 26.2, feet 37.8 | left_hand 23.8, right_hand 21.2, hands 21.9, feet 33.1 | 26.67 | 15 / 525 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 64ch | 3 | 42: 28.60, 43: 28.86, 44: 29.17 | 28.88 | 0.29 |
| Ours | 3 | 42: 27.02, 43: 26.67, 44: 28.10 | 27.26 | 0.74 |
| CSP | 3 | 42: 27.21, 43: 27.37, 44: 28.83 | 27.80 | 0.89 |
| RLF | 3 | 42: 28.89, 43: 27.75, 44: 28.54 | 28.39 | 0.59 |
| MI | 3 | 42: 28.00, 43: 27.90, 44: 28.38 | 28.10 | 0.25 |
| Fixed sensorimotor | 3 | 42: 27.97, 43: 26.95, 44: 27.75 | 27.56 | 0.53 |
| ERD-only | 3 | 42: 28.86, 43: 28.32, 44: 28.13 | 28.43 | 0.38 |
| Random (mean of draws) | 3 | 42: 28.23, 43: 28.18, 44: 27.92 | 28.11 | 0.17 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full64/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | -0.007 | [-0.032, 0.018] | 0.128 | 105 |
| grad | class_specific_pearson | 0.010 | [-0.008, 0.029] | 0.094 | 105 |
| grad | lateralisation_pearson | -0.018 | [-0.043, 0.007] | 0.126 | 105 |
| grad | magnitude_spearman | -0.005 | [-0.032, 0.022] | 0.137 | 105 |
| grad | class_specific_spearman | 0.017 | [0.000, 0.034] | 0.087 | 105 |
| grad | lateralisation_spearman | -0.013 | [-0.038, 0.012] | 0.126 | 105 |
| grad | legacy_signed_LR_pearson | -0.008 | [-0.036, 0.021] | 0.144 | 105 |
| grad_x_input | magnitude_pearson | -0.046 | [-0.081, -0.012] | 0.171 | 105 |
| grad_x_input | class_specific_pearson | -0.042 | [-0.073, -0.012] | 0.152 | 105 |
| grad_x_input | lateralisation_pearson | -0.113 | [-0.162, -0.062] | 0.241 | 105 |
| grad_x_input | magnitude_spearman | -0.040 | [-0.075, -0.005] | 0.172 | 105 |
| grad_x_input | class_specific_spearman | -0.037 | [-0.067, -0.007] | 0.151 | 105 |
| grad_x_input | lateralisation_spearman | -0.096 | [-0.148, -0.044] | 0.249 | 105 |
| grad_x_input | legacy_signed_LR_pearson | 0.003 | [-0.028, 0.035] | 0.161 | 105 |
| integrated_gradients | magnitude_pearson | -0.047 | [-0.081, -0.013] | 0.169 | 105 |
| integrated_gradients | class_specific_pearson | -0.042 | [-0.072, -0.012] | 0.149 | 105 |
| integrated_gradients | lateralisation_pearson | -0.098 | [-0.146, -0.050] | 0.231 | 105 |
| integrated_gradients | magnitude_spearman | -0.042 | [-0.077, -0.008] | 0.172 | 105 |
| integrated_gradients | class_specific_spearman | -0.038 | [-0.068, -0.009] | 0.148 | 105 |
| integrated_gradients | lateralisation_spearman | -0.081 | [-0.131, -0.031] | 0.240 | 105 |
| integrated_gradients | legacy_signed_LR_pearson | 0.005 | [-0.026, 0.035] | 0.154 | 105 |
| smoothgrad | magnitude_pearson | -0.007 | [-0.033, 0.018] | 0.128 | 105 |
| smoothgrad | class_specific_pearson | 0.010 | [-0.008, 0.028] | 0.094 | 105 |
| smoothgrad | lateralisation_pearson | -0.018 | [-0.043, 0.007] | 0.126 | 105 |
| smoothgrad | magnitude_spearman | -0.005 | [-0.032, 0.022] | 0.138 | 105 |
| smoothgrad | class_specific_spearman | 0.017 | [-0.000, 0.034] | 0.087 | 105 |
| smoothgrad | lateralisation_spearman | -0.013 | [-0.038, 0.012] | 0.126 | 105 |
| smoothgrad | legacy_signed_LR_pearson | -0.008 | [-0.036, 0.021] | 0.144 | 105 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.019 | [-0.042, 0.004] | 105 |
| grad | ERD_L_spearman | -0.024 | [-0.048, 0.000] | 105 |
| grad | ERD_R_pearson | -0.001 | [-0.027, 0.026] | 105 |
| grad | ERD_R_spearman | -0.008 | [-0.032, 0.016] | 105 |
| grad | ERD_LR_pearson | -0.008 | [-0.036, 0.021] | 105 |
| grad | ERD_LR_spearman | -0.004 | [-0.031, 0.024] | 105 |
| grad | ERD_comb_pearson | -0.013 | [-0.038, 0.013] | 105 |
| grad | ERD_comb_spearman | -0.020 | [-0.046, 0.006] | 105 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |
| grad_x_input | ERD_L_pearson | 0.019 | [-0.014, 0.052] | 105 |
| grad_x_input | ERD_L_spearman | 0.019 | [-0.015, 0.053] | 105 |
| grad_x_input | ERD_R_pearson | 0.023 | [-0.015, 0.060] | 105 |
| grad_x_input | ERD_R_spearman | 0.022 | [-0.013, 0.057] | 105 |
| grad_x_input | ERD_LR_pearson | 0.003 | [-0.028, 0.035] | 105 |
| grad_x_input | ERD_LR_spearman | 0.007 | [-0.023, 0.038] | 105 |
| grad_x_input | ERD_comb_pearson | 0.021 | [-0.016, 0.057] | 105 |
| grad_x_input | ERD_comb_spearman | 0.022 | [-0.014, 0.058] | 105 |
| grad_x_input | vs_grad_pearson | 0.963 | [0.954, 0.970] | 105 |
| grad_x_input | vs_grad_spearman | 0.955 | [0.945, 0.963] | 105 |
| integrated_gradients | ERD_L_pearson | 0.017 | [-0.015, 0.049] | 105 |
| integrated_gradients | ERD_L_spearman | 0.019 | [-0.014, 0.052] | 105 |
| integrated_gradients | ERD_R_pearson | 0.020 | [-0.017, 0.057] | 105 |
| integrated_gradients | ERD_R_spearman | 0.019 | [-0.016, 0.054] | 105 |
| integrated_gradients | ERD_LR_pearson | 0.005 | [-0.026, 0.035] | 105 |
| integrated_gradients | ERD_LR_spearman | 0.010 | [-0.018, 0.039] | 105 |
| integrated_gradients | ERD_comb_pearson | 0.019 | [-0.017, 0.055] | 105 |
| integrated_gradients | ERD_comb_spearman | 0.021 | [-0.015, 0.057] | 105 |
| integrated_gradients | vs_grad_pearson | 0.951 | [0.942, 0.958] | 105 |
| integrated_gradients | vs_grad_spearman | 0.940 | [0.931, 0.949] | 105 |
| smoothgrad | ERD_L_pearson | -0.019 | [-0.042, 0.004] | 105 |
| smoothgrad | ERD_L_spearman | -0.023 | [-0.048, 0.001] | 105 |
| smoothgrad | ERD_R_pearson | -0.001 | [-0.027, 0.026] | 105 |
| smoothgrad | ERD_R_spearman | -0.008 | [-0.032, 0.017] | 105 |
| smoothgrad | ERD_LR_pearson | -0.008 | [-0.036, 0.021] | 105 |
| smoothgrad | ERD_LR_spearman | -0.003 | [-0.031, 0.024] | 105 |
| smoothgrad | ERD_comb_pearson | -0.013 | [-0.038, 0.013] | 105 |
| smoothgrad | ERD_comb_spearman | -0.020 | [-0.046, 0.006] | 105 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.004 | [-0.035, 0.027] | -0.002 | -0.002 | 0.012 | 105 |
| ERD_theta_R | -0.002 | [-0.033, 0.028] | -0.020 | 0.002 | -0.017 | 105 |
| ERD_theta_mag | -0.014 | [-0.044, 0.016] | -0.014 | -0.000 | -0.000 | 105 |
| ERD_mu_L | 0.000 | [-0.030, 0.030] | -0.013 | -0.004 | -0.010 | 105 |
| ERD_mu_R | 0.004 | [-0.028, 0.036] | -0.028 | 0.001 | -0.021 | 105 |
| ERD_mu_mag | -0.030 | [-0.060, 0.000] | -0.030 | -0.024 | -0.024 | 105 |
| ERD_beta_L | 0.007 | [-0.019, 0.033] | -0.008 | 0.011 | -0.011 | 105 |
| ERD_beta_R | 0.017 | [-0.013, 0.047] | -0.003 | 0.010 | -0.000 | 105 |
| ERD_beta_mag | -0.008 | [-0.035, 0.020] | -0.008 | -0.013 | -0.013 | 105 |
| ERS_beta_rebound_L | -0.015 | [-0.043, 0.012] | -0.002 | -0.009 | 0.002 | 105 |
| ERS_beta_rebound_R | -0.014 | [-0.046, 0.019] | 0.004 | -0.019 | -0.002 | 105 |
| ERS_beta_rebound_mag | 0.005 | [-0.024, 0.033] | 0.005 | 0.005 | 0.005 | 105 |
| ERD_lowgamma_L | 0.003 | [-0.025, 0.031] | 0.007 | 0.002 | -0.003 | 105 |
| ERD_lowgamma_R | 0.019 | [-0.008, 0.046] | 0.006 | 0.013 | -0.003 | 105 |
| ERD_lowgamma_mag | 0.003 | [-0.026, 0.032] | 0.003 | -0.015 | -0.015 | 105 |
| logpow_8_30 | -0.034 | [-0.065, -0.002] | -0.033 | -0.018 | -0.018 | 105 |
| r2_signed | 0.003 | [-0.026, 0.033] | 0.019 | 0.002 | 0.010 | 105 |
| r2_abs | 0.019 | [-0.007, 0.045] | 0.019 | 0.010 | 0.010 | 105 |
| TFR_ERD_dB_L | 0.004 | [-0.022, 0.031] | -0.014 | 0.006 | -0.013 | 105 |
| TFR_ERD_dB_R | 0.010 | [-0.018, 0.038] | -0.018 | 0.004 | -0.018 | 105 |
| TFR_ERD_dB_mag | -0.027 | [-0.054, -0.000] | -0.027 | -0.026 | -0.026 | 105 |
| ITC_4_30 | -0.013 | [-0.038, 0.013] | -0.013 | -0.009 | -0.009 | 105 |
| ERP_abs_0.2_0.8 | -0.014 | [-0.041, 0.012] | -0.014 | -0.010 | -0.010 | 105 |
| SlowPot_0.5_3 | -0.027 | [-0.061, 0.006] | -0.013 | -0.031 | -0.014 | 105 |
| ERD_beta_classspecific | — | — | -0.001 [-0.021, 0.018] | — | 0.001 | 105 |
| ERD_lowgamma_classspecific | — | — | 0.005 [-0.017, 0.026] | — | -0.007 | 105 |
| ERD_mu_classspecific | — | — | -0.019 [-0.040, 0.003] | — | -0.013 | 105 |
| ERD_theta_classspecific | — | — | -0.009 [-0.034, 0.016] | — | -0.004 | 105 |
| ERS_beta_rebound_classspecific | — | — | 0.006 [-0.016, 0.027] | — | 0.006 | 105 |
| TFR_ERD_dB_classspecific | — | — | -0.016 [-0.035, 0.004] | — | -0.015 | 105 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +0.17 ± 2.67 | +0.19 ± 1.53 | +0.23 ± 2.09 | 0.9091 | 0.8163 |
| 4 | +0.53 ± 3.66 | +0.36 ± 2.00 | +0.42 ± 2.19 | 0.5341 | 0.7511 |
| 6 | +0.74 ± 3.88 | +0.52 ± 2.13 | +0.46 ± 2.46 | 0.4516 | 0.3893 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.010 | [-0.025, 0.006] | 105 |
| grad_x_input | 0.041 | [0.020, 0.062] | 105 |
| integrated_gradients | 0.039 | [0.019, 0.060] | 105 |
| smoothgrad | -0.010 | [-0.025, 0.006] | 105 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=ems, S02=ems, S03=ems, S04=ems, S05=ems, S06=ems, S07=ems, S08=ems, S09=ems, S10=ems, S11=ems, S12=ems, S13=ems, S14=ems, S15=ems, S16=ems, S17=ems, S18=ems, S19=ems, S20=ems, S21=ems, S22=ems, S23=ems, S24=ems, S25=ems, S26=ems, S27=ems, S28=ems, S29=ems, S30=ems, S31=ems, S32=ems, S33=ems, S34=ems, S35=ems, S36=ems, S37=ems, S38=ems, S39=ems, S40=ems, S41=ems, S42=ems, S43=ems, S44=ems, S45=ems, S46=ems, S47=ems, S48=ems, S49=ems, S50=ems, S51=ems, S52=ems, S53=ems, S54=ems, S55=ems, S56=ems, S57=ems, S58=ems, S59=ems, S60=ems, S61=ems, S62=ems, S63=ems, S64=ems, S65=ems, S66=ems, S67=ems, S68=ems, S69=ems, S70=ems, S71=ems, S72=ems, S73=ems, S74=ems, S75=ems, S76=ems, S77=ems, S78=ems, S79=ems, S80=ems, S81=ems, S82=ems, S83=ems, S84=ems, S85=ems, S86=ems, S87=ems, S89=ems, S90=ems, S91=ems, S93=ems, S94=ems, S95=ems, S96=ems, S97=ems, S98=ems, S99=ems, S101=ems, S102=ems, S103=ems, S105=ems, S106=ems, S107=ems, S108=ems, S109=ems

| Profile | Mean validation accuracy (%) |
|---|---|
| zscore | 40.24 |
| ems | 40.83 |
| original | 31.79 |

```
model: "EEGITNet"
dataset: "PhysionetMI"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"zscore": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": null, "align": "zscore", "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "ems": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": "ems", "align": null, "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "original": {"sfreq": 160, "band_hz": [8.0, 32.0], "window_s": [0.5, 2.5], "norm": null, "align": null, "car": false, "dropout": 0.25, "lr": 0.01, "weight_decay": 1e-05, "scheduler": "multistep", "milestones": [20, 50], "gamma": 0.1, "max_epochs": 100, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "augment": false, "select_on": "val_loss", "min_select_epoch": 1}}
checkpoint_rule: "highest validation accuracy (loss breaks ties), not before epoch 20 ('original': lowest validation loss); refit keeps the final weights"
folds: "run-wise chronological split per subject: test = the last imagery run of each type (run 12 left/right, run 14 hands/feet); training = runs 4, 8 (and 6, 10); validation = the chronologically last 25% of the training trials of EACH class; refit on all training runs"
seeds: [42, 43, 44]
```

Figures (530 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `4class\figures`.
