# Results report — EEGNetV4\PhysioNet

Generated 2026-10-24 06:00 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Protocol: run-wise chronological split per subject: test = the last imagery run of each type (run 12 left/right, run 14 hands/feet); training = runs 4, 8 (and 6, 10); validation = the chronologically last 25% of the training trials of EACH class; refit on all training runs. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects (n = 105). Random = mean of the random-montage draws.

| Subject | 64ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|
| S01 | 73.33 | 53.33 | 80.00 | 80.00 | 60.00 | 40.00 | 40.00 | 54.67 |
| S02 | 86.67 | 66.67 | 53.33 | 66.67 | 66.67 | 53.33 | 60.00 | 53.33 |
| S03 | 53.33 | 33.33 | 53.33 | 53.33 | 60.00 | 53.33 | 46.67 | 54.67 |
| S04 | 60.00 | 46.67 | 73.33 | 53.33 | 53.33 | 66.67 | 46.67 | 53.33 |
| S05 | 66.67 | 26.67 | 26.67 | 60.00 | 53.33 | 60.00 | 46.67 | 45.33 |
| S06 | 73.33 | 66.67 | 66.67 | 60.00 | 73.33 | 46.67 | 86.67 | 68.00 |
| S07 | 80.00 | 86.67 | 86.67 | 66.67 | 80.00 | 86.67 | 60.00 | 62.67 |
| S08 | 33.33 | 33.33 | 33.33 | 53.33 | 60.00 | 40.00 | 33.33 | 44.00 |
| S09 | 46.67 | 46.67 | 46.67 | 53.33 | 53.33 | 46.67 | 46.67 | 46.67 |
| S10 | 60.00 | 53.33 | 60.00 | 46.67 | 53.33 | 26.67 | 26.67 | 48.00 |
| S11 | 33.33 | 46.67 | 40.00 | 40.00 | 53.33 | 46.67 | 53.33 | 40.00 |
| S12 | 66.67 | 60.00 | 60.00 | 53.33 | 60.00 | 53.33 | 53.33 | 58.67 |
| S13 | 53.33 | 46.67 | 26.67 | 40.00 | 33.33 | 46.67 | 33.33 | 41.33 |
| S14 | 60.00 | 60.00 | 53.33 | 60.00 | 46.67 | 53.33 | 53.33 | 60.00 |
| S15 | 60.00 | 86.67 | 53.33 | 73.33 | 66.67 | 46.67 | 46.67 | 62.67 |
| S16 | 46.67 | 53.33 | 53.33 | 40.00 | 53.33 | 53.33 | 53.33 | 52.00 |
| S17 | 66.67 | 80.00 | 73.33 | 66.67 | 73.33 | 60.00 | 60.00 | 61.33 |
| S18 | 93.33 | 93.33 | 93.33 | 86.67 | 93.33 | 80.00 | 66.67 | 93.33 |
| S19 | 26.67 | 33.33 | 60.00 | 33.33 | 53.33 | 33.33 | 40.00 | 38.67 |
| S20 | 40.00 | 60.00 | 73.33 | 40.00 | 40.00 | 60.00 | 53.33 | 64.00 |
| S21 | 46.67 | 46.67 | 66.67 | 46.67 | 40.00 | 40.00 | 40.00 | 49.33 |
| S22 | 53.33 | 46.67 | 53.33 | 46.67 | 60.00 | 53.33 | 60.00 | 50.67 |
| S23 | 53.33 | 53.33 | 60.00 | 60.00 | 66.67 | 66.67 | 60.00 | 60.00 |
| S24 | 60.00 | 33.33 | 46.67 | 40.00 | 53.33 | 26.67 | 40.00 | 44.00 |
| S25 | 40.00 | 66.67 | 33.33 | 60.00 | 40.00 | 46.67 | 73.33 | 49.33 |
| S26 | 46.67 | 60.00 | 46.67 | 60.00 | 53.33 | 46.67 | 60.00 | 56.00 |
| S27 | 73.33 | 46.67 | 53.33 | 53.33 | 53.33 | 53.33 | 53.33 | 53.33 |
| S28 | 33.33 | 53.33 | 33.33 | 60.00 | 40.00 | 40.00 | 40.00 | 58.67 |
| S29 | 66.67 | 53.33 | 53.33 | 46.67 | 60.00 | 73.33 | 46.67 | 53.33 |
| S30 | 73.33 | 86.67 | 73.33 | 66.67 | 66.67 | 53.33 | 53.33 | 61.33 |
| S31 | 53.33 | 66.67 | 53.33 | 46.67 | 26.67 | 73.33 | 53.33 | 58.67 |
| S32 | 80.00 | 73.33 | 80.00 | 80.00 | 73.33 | 73.33 | 86.67 | 65.33 |
| S33 | 46.67 | 46.67 | 46.67 | 60.00 | 40.00 | 60.00 | 40.00 | 45.33 |
| S34 | 53.33 | 60.00 | 46.67 | 73.33 | 60.00 | 80.00 | 60.00 | 56.00 |
| S35 | 86.67 | 46.67 | 80.00 | 86.67 | 80.00 | 46.67 | 46.67 | 64.00 |
| S36 | 66.67 | 86.67 | 73.33 | 73.33 | 66.67 | 73.33 | 86.67 | 76.00 |
| S37 | 33.33 | 46.67 | 46.67 | 46.67 | 46.67 | 40.00 | 46.67 | 41.33 |
| S38 | 33.33 | 40.00 | 26.67 | 26.67 | 40.00 | 40.00 | 53.33 | 44.00 |
| S39 | 60.00 | 46.67 | 66.67 | 46.67 | 53.33 | 53.33 | 46.67 | 53.33 |
| S40 | 86.67 | 40.00 | 86.67 | 73.33 | 80.00 | 53.33 | 53.33 | 64.00 |
| S41 | 66.67 | 53.33 | 53.33 | 46.67 | 40.00 | 46.67 | 53.33 | 44.00 |
| S42 | 73.33 | 46.67 | 46.67 | 46.67 | 46.67 | 60.00 | 46.67 | 50.67 |
| S43 | 66.67 | 53.33 | 66.67 | 53.33 | 66.67 | 60.00 | 66.67 | 58.67 |
| S44 | 73.33 | 53.33 | 40.00 | 46.67 | 40.00 | 46.67 | 46.67 | 48.00 |
| S45 | 60.00 | 40.00 | 33.33 | 60.00 | 46.67 | 26.67 | 40.00 | 50.67 |
| S46 | 46.67 | 20.00 | 53.33 | 46.67 | 53.33 | 60.00 | 60.00 | 46.67 |
| S47 | 60.00 | 40.00 | 73.33 | 53.33 | 60.00 | 66.67 | 46.67 | 45.33 |
| S48 | 26.67 | 53.33 | 40.00 | 53.33 | 53.33 | 46.67 | 53.33 | 50.67 |
| S49 | 73.33 | 73.33 | 60.00 | 60.00 | 73.33 | 66.67 | 66.67 | 61.33 |
| S50 | 33.33 | 53.33 | 46.67 | 33.33 | 53.33 | 33.33 | 46.67 | 48.00 |
| S51 | 60.00 | 73.33 | 66.67 | 66.67 | 60.00 | 60.00 | 46.67 | 62.67 |
| S52 | 40.00 | 66.67 | 46.67 | 46.67 | 40.00 | 53.33 | 53.33 | 37.33 |
| S53 | 53.33 | 46.67 | 46.67 | 53.33 | 46.67 | 60.00 | 66.67 | 49.33 |
| S54 | 60.00 | 66.67 | 46.67 | 73.33 | 66.67 | 66.67 | 66.67 | 60.00 |
| S55 | 40.00 | 60.00 | 46.67 | 40.00 | 46.67 | 46.67 | 46.67 | 40.00 |
| S56 | 53.33 | 60.00 | 53.33 | 66.67 | 66.67 | 46.67 | 46.67 | 52.00 |
| S57 | 86.67 | 86.67 | 80.00 | 86.67 | 80.00 | 80.00 | 80.00 | 78.67 |
| S58 | 86.67 | 73.33 | 73.33 | 80.00 | 80.00 | 60.00 | 66.67 | 73.33 |
| S59 | 60.00 | 46.67 | 46.67 | 60.00 | 53.33 | 60.00 | 60.00 | 56.00 |
| S60 | 53.33 | 53.33 | 60.00 | 46.67 | 46.67 | 46.67 | 66.67 | 60.00 |
| S61 | 53.33 | 60.00 | 66.67 | 53.33 | 80.00 | 73.33 | 46.67 | 62.67 |
| S62 | 60.00 | 53.33 | 53.33 | 40.00 | 46.67 | 53.33 | 46.67 | 44.00 |
| S63 | 53.33 | 53.33 | 60.00 | 53.33 | 66.67 | 40.00 | 53.33 | 52.00 |
| S64 | 80.00 | 80.00 | 53.33 | 46.67 | 66.67 | 80.00 | 60.00 | 73.33 |
| S65 | 46.67 | 46.67 | 60.00 | 80.00 | 53.33 | 60.00 | 60.00 | 53.33 |
| S66 | 66.67 | 73.33 | 66.67 | 66.67 | 53.33 | 60.00 | 53.33 | 56.00 |
| S67 | 53.33 | 73.33 | 53.33 | 60.00 | 80.00 | 53.33 | 80.00 | 54.67 |
| S68 | 66.67 | 66.67 | 53.33 | 60.00 | 46.67 | 53.33 | 46.67 | 50.67 |
| S69 | 46.67 | 53.33 | 40.00 | 60.00 | 46.67 | 66.67 | 53.33 | 53.33 |
| S70 | 60.00 | 53.33 | 66.67 | 60.00 | 60.00 | 60.00 | 60.00 | 49.33 |
| S71 | 60.00 | 46.67 | 66.67 | 46.67 | 46.67 | 53.33 | 60.00 | 46.67 |
| S72 | 86.67 | 86.67 | 80.00 | 80.00 | 80.00 | 86.67 | 93.33 | 77.33 |
| S73 | 46.67 | 53.33 | 46.67 | 60.00 | 33.33 | 73.33 | 53.33 | 52.00 |
| S74 | 73.33 | 46.67 | 66.67 | 40.00 | 40.00 | 46.67 | 53.33 | 57.33 |
| S75 | 53.33 | 46.67 | 53.33 | 46.67 | 60.00 | 46.67 | 53.33 | 46.67 |
| S76 | 46.67 | 53.33 | 60.00 | 60.00 | 53.33 | 53.33 | 46.67 | 53.33 |
| S77 | 20.00 | 53.33 | 66.67 | 53.33 | 60.00 | 53.33 | 46.67 | 50.67 |
| S78 | 60.00 | 53.33 | 53.33 | 53.33 | 53.33 | 53.33 | 53.33 | 54.67 |
| S79 | 60.00 | 46.67 | 46.67 | 46.67 | 46.67 | 46.67 | 46.67 | 46.67 |
| S80 | 40.00 | 40.00 | 40.00 | 53.33 | 46.67 | 53.33 | 46.67 | 45.33 |
| S81 | 80.00 | 60.00 | 53.33 | 66.67 | 53.33 | 53.33 | 60.00 | 56.00 |
| S82 | 53.33 | 53.33 | 53.33 | 53.33 | 53.33 | 46.67 | 60.00 | 48.00 |
| S83 | 53.33 | 66.67 | 46.67 | 53.33 | 53.33 | 53.33 | 53.33 | 42.67 |
| S84 | 46.67 | 53.33 | 26.67 | 66.67 | 46.67 | 60.00 | 66.67 | 46.67 |
| S85 | 60.00 | 53.33 | 80.00 | 60.00 | 73.33 | 80.00 | 53.33 | 60.00 |
| S86 | 33.33 | 33.33 | 53.33 | 40.00 | 40.00 | 46.67 | 33.33 | 49.33 |
| S87 | 53.33 | 60.00 | 60.00 | 73.33 | 66.67 | 60.00 | 73.33 | 65.33 |
| S89 | 53.33 | 60.00 | 53.33 | 60.00 | 60.00 | 40.00 | 53.33 | 52.00 |
| S90 | 40.00 | 33.33 | 26.67 | 26.67 | 26.67 | 40.00 | 33.33 | 40.00 |
| S91 | 73.33 | 53.33 | 66.67 | 60.00 | 66.67 | 40.00 | 40.00 | 56.00 |
| S93 | 33.33 | 33.33 | 40.00 | 46.67 | 33.33 | 46.67 | 53.33 | 42.67 |
| S94 | 60.00 | 66.67 | 66.67 | 66.67 | 46.67 | 53.33 | 46.67 | 50.67 |
| S95 | 53.33 | 60.00 | 60.00 | 60.00 | 66.67 | 60.00 | 40.00 | 48.00 |
| S96 | 80.00 | 80.00 | 80.00 | 60.00 | 80.00 | 53.33 | 60.00 | 64.00 |
| S97 | 60.00 | 33.33 | 46.67 | 53.33 | 40.00 | 40.00 | 46.67 | 46.67 |
| S98 | 66.67 | 60.00 | 66.67 | 60.00 | 53.33 | 60.00 | 46.67 | 62.67 |
| S99 | 53.33 | 53.33 | 53.33 | 40.00 | 53.33 | 60.00 | 46.67 | 54.67 |
| S101 | 73.33 | 66.67 | 53.33 | 66.67 | 66.67 | 40.00 | 53.33 | 62.67 |
| S102 | 60.00 | 66.67 | 60.00 | 80.00 | 66.67 | 60.00 | 60.00 | 73.33 |
| S103 | 53.33 | 33.33 | 40.00 | 66.67 | 60.00 | 53.33 | 53.33 | 50.67 |
| S105 | 46.67 | 60.00 | 60.00 | 53.33 | 60.00 | 60.00 | 80.00 | 57.33 |
| S106 | 46.67 | 46.67 | 66.67 | 26.67 | 53.33 | 53.33 | 40.00 | 45.33 |
| S107 | 33.33 | 26.67 | 40.00 | 33.33 | 13.33 | 26.67 | 46.67 | 33.33 |
| S108 | 53.33 | 53.33 | 46.67 | 53.33 | 66.67 | 33.33 | 60.00 | 56.00 |
| S109 | 60.00 | 60.00 | 53.33 | 73.33 | 66.67 | 46.67 | 40.00 | 41.33 |
| Mean | 57.27 | 55.43 | 56.00 | 56.38 | 56.13 | 54.03 | 53.78 | 53.96 |
| SD | 15.41 | 14.83 | 14.37 | 13.33 | 13.83 | 12.87 | 12.24 | 9.81 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 105 | PO8 F8 FC6 PO4 P2 F7 FC1 TP8 Pz T10 T9 CP3 |
| CSP | 15 | Fpz Fp1 AF7 Fp2 AFz AF3 AF8 AF4 F8 F7 F5 F1 |
| RLF | 70 | F7 CP6 C4 CP4 FT7 F8 P6 C6 P4 T9 FC4 PO4 |
| MI | 94 | F7 F8 F5 P6 F2 F3 FC4 CP4 CP6 CP2 P4 F4 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 8 | PO8 O2 O1 PO4 Oz POz PO7 PO3 Fpz P4 P2 Fp2 |

### T3. Equivalence of each 12-channel montage with all 64 channels (paired TOST)

Difference = montage − 64ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 105 | -1.84 | 15.18 | [-4.30, +0.62] | 0.0030 | yes | -46.67 | 43 | 5,6,8,10 |
| CSP | 105 | -1.27 | 13.84 | [-3.51, +0.97] | 0.0003 | yes | -40.00 | 42 | 4,5,6,8,10 |
| RLF | 105 | -0.89 | 13.59 | [-3.09, +1.31] | 0.0001 | yes | -33.33 | 45 | 4,5,6,8,10 |
| MI | 105 | -1.14 | 13.57 | [-3.34, +1.06] | 0.0002 | yes | -33.33 | 46 | 4,5,6,8,10 |
| Fixed sensorimotor | 105 | -3.24 | 15.80 | [-5.80, -0.68] | 0.0381 | yes | -40.00 | 50 | 6,8,10 |
| ERD-only | 105 | -3.49 | 16.07 | [-6.09, -0.89] | 0.0564 | no | -40.00 | 53 | 8,10 |
| Random (mean of draws) | 105 | -3.31 | 11.93 | [-5.25, -1.38] | 0.0115 | yes | -33.33 | 43 | 6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 105 | -0.57 | [-3.42, +2.28] | 0.6919 | 1.0000 | 0.7849 | 1.0000 | 41 / 27 / 37 |
| RLF | 105 | -0.95 | [-3.57, +1.66] | 0.4716 | 1.0000 | 0.6328 | 1.0000 | 39 / 25 / 41 |
| MI | 105 | -0.70 | [-3.35, +1.95] | 0.6028 | 1.0000 | 0.7171 | 1.0000 | 36 / 27 / 42 |
| Fixed sensorimotor | 105 | +1.40 | [-1.33, +4.13] | 0.3125 | 1.0000 | 0.3359 | 1.0000 | 45 / 25 / 35 |
| ERD-only | 105 | +1.65 | [-0.95, +4.25] | 0.2115 | 1.0000 | 0.3679 | 1.0000 | 45 / 24 / 36 |
| Random (mean of draws) | 105 | +1.47 | [-0.61, +3.55] | 0.1636 | 0.9814 | 0.2077 | 1.0000 | 51 / 10 / 44 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 64ch | 57.27 | left_hand 57.9, right_hand 56.6 | left_hand 50.7, right_hand 49.3 | 53.33 | 8 / 105 |
| Ours | 55.43 | left_hand 56.5, right_hand 54.3 | left_hand 51.1, right_hand 48.9 | 53.33 | 9 / 105 |
| CSP | 56.00 | left_hand 56.6, right_hand 55.4 | left_hand 50.7, right_hand 49.3 | 53.33 | 18 / 105 |
| RLF | 56.38 | left_hand 57.8, right_hand 55.0 | left_hand 51.4, right_hand 48.6 | 53.33 | 17 / 105 |
| MI | 56.13 | left_hand 59.9, right_hand 52.3 | left_hand 53.8, right_hand 46.2 | 53.33 | 16 / 105 |
| Fixed sensorimotor | 54.03 | left_hand 52.5, right_hand 55.6 | left_hand 48.4, right_hand 51.6 | 53.33 | 24 / 105 |
| ERD-only | 53.78 | left_hand 58.1, right_hand 49.4 | left_hand 54.4, right_hand 45.6 | 53.33 | 23 / 105 |
| Random (mean of draws) | 53.96 | left_hand 56.0, right_hand 51.9 | left_hand 52.1, right_hand 47.9 | 53.33 | 89 / 525 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 64ch | 3 | 42: 55.68, 43: 55.37, 44: 55.11 | 55.39 | 0.29 |
| Ours | 3 | 42: 54.79, 43: 53.71, 44: 54.29 | 54.26 | 0.54 |
| CSP | 3 | 42: 52.57, 43: 55.11, 44: 52.89 | 53.52 | 1.38 |
| RLF | 3 | 42: 54.92, 43: 52.89, 44: 54.79 | 54.20 | 1.14 |
| MI | 3 | 42: 54.48, 43: 53.14, 44: 53.14 | 53.59 | 0.77 |
| Fixed sensorimotor | 3 | 42: 50.98, 43: 51.81, 44: 53.08 | 51.96 | 1.06 |
| ERD-only | 3 | 42: 53.27, 43: 53.59, 44: 52.70 | 53.19 | 0.45 |
| Random (mean of draws) | 3 | 42: 52.98, 43: 53.27, 44: 52.79 | 53.01 | 0.24 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full64/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.017 | [-0.010, 0.044] | 0.137 | 105 |
| grad | class_specific_pearson | -0.013 | [-0.033, 0.008] | 0.103 | 105 |
| grad | lateralisation_pearson | -0.015 | [-0.038, 0.008] | 0.117 | 105 |
| grad | magnitude_spearman | 0.010 | [-0.017, 0.038] | 0.138 | 105 |
| grad | class_specific_spearman | -0.012 | [-0.031, 0.007] | 0.098 | 105 |
| grad | lateralisation_spearman | -0.007 | [-0.031, 0.016] | 0.120 | 105 |
| grad | legacy_signed_LR_pearson | 0.011 | [-0.012, 0.033] | 0.117 | 105 |
| grad_x_input | magnitude_pearson | -0.008 | [-0.049, 0.033] | 0.206 | 105 |
| grad_x_input | class_specific_pearson | -0.002 | [-0.036, 0.032] | 0.170 | 105 |
| grad_x_input | lateralisation_pearson | -0.103 | [-0.150, -0.056] | 0.229 | 105 |
| grad_x_input | magnitude_spearman | -0.007 | [-0.048, 0.034] | 0.203 | 105 |
| grad_x_input | class_specific_spearman | -0.005 | [-0.037, 0.026] | 0.159 | 105 |
| grad_x_input | lateralisation_spearman | -0.081 | [-0.127, -0.036] | 0.221 | 105 |
| grad_x_input | legacy_signed_LR_pearson | 0.005 | [-0.034, 0.044] | 0.193 | 105 |
| integrated_gradients | magnitude_pearson | -0.007 | [-0.048, 0.033] | 0.204 | 105 |
| integrated_gradients | class_specific_pearson | -0.002 | [-0.035, 0.032] | 0.168 | 105 |
| integrated_gradients | lateralisation_pearson | -0.100 | [-0.146, -0.054] | 0.226 | 105 |
| integrated_gradients | magnitude_spearman | -0.006 | [-0.047, 0.034] | 0.201 | 105 |
| integrated_gradients | class_specific_spearman | -0.006 | [-0.037, 0.026] | 0.157 | 105 |
| integrated_gradients | lateralisation_spearman | -0.079 | [-0.123, -0.034] | 0.217 | 105 |
| integrated_gradients | legacy_signed_LR_pearson | 0.007 | [-0.032, 0.045] | 0.192 | 105 |
| smoothgrad | magnitude_pearson | 0.017 | [-0.010, 0.044] | 0.137 | 105 |
| smoothgrad | class_specific_pearson | -0.013 | [-0.033, 0.007] | 0.103 | 105 |
| smoothgrad | lateralisation_pearson | -0.016 | [-0.039, 0.007] | 0.117 | 105 |
| smoothgrad | magnitude_spearman | 0.010 | [-0.017, 0.037] | 0.138 | 105 |
| smoothgrad | class_specific_spearman | -0.012 | [-0.031, 0.007] | 0.099 | 105 |
| smoothgrad | lateralisation_spearman | -0.007 | [-0.031, 0.016] | 0.120 | 105 |
| smoothgrad | legacy_signed_LR_pearson | 0.011 | [-0.012, 0.033] | 0.117 | 105 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | 0.016 | [-0.010, 0.041] | 105 |
| grad | ERD_L_spearman | 0.009 | [-0.016, 0.034] | 105 |
| grad | ERD_R_pearson | 0.010 | [-0.014, 0.033] | 105 |
| grad | ERD_R_spearman | 0.018 | [-0.005, 0.041] | 105 |
| grad | ERD_LR_pearson | 0.011 | [-0.012, 0.033] | 105 |
| grad | ERD_LR_spearman | 0.002 | [-0.019, 0.024] | 105 |
| grad | ERD_comb_pearson | 0.014 | [-0.013, 0.040] | 105 |
| grad | ERD_comb_spearman | 0.017 | [-0.010, 0.044] | 105 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |
| grad_x_input | ERD_L_pearson | -0.027 | [-0.068, 0.015] | 105 |
| grad_x_input | ERD_L_spearman | -0.020 | [-0.060, 0.020] | 105 |
| grad_x_input | ERD_R_pearson | -0.017 | [-0.060, 0.025] | 105 |
| grad_x_input | ERD_R_spearman | -0.004 | [-0.043, 0.035] | 105 |
| grad_x_input | ERD_LR_pearson | 0.005 | [-0.034, 0.044] | 105 |
| grad_x_input | ERD_LR_spearman | 0.002 | [-0.033, 0.036] | 105 |
| grad_x_input | ERD_comb_pearson | -0.036 | [-0.080, 0.008] | 105 |
| grad_x_input | ERD_comb_spearman | -0.017 | [-0.060, 0.025] | 105 |
| grad_x_input | vs_grad_pearson | 0.922 | [0.909, 0.934] | 105 |
| grad_x_input | vs_grad_spearman | 0.914 | [0.900, 0.926] | 105 |
| integrated_gradients | ERD_L_pearson | -0.026 | [-0.067, 0.015] | 105 |
| integrated_gradients | ERD_L_spearman | -0.019 | [-0.058, 0.020] | 105 |
| integrated_gradients | ERD_R_pearson | -0.018 | [-0.060, 0.023] | 105 |
| integrated_gradients | ERD_R_spearman | -0.005 | [-0.043, 0.034] | 105 |
| integrated_gradients | ERD_LR_pearson | 0.007 | [-0.032, 0.045] | 105 |
| integrated_gradients | ERD_LR_spearman | 0.003 | [-0.031, 0.038] | 105 |
| integrated_gradients | ERD_comb_pearson | -0.036 | [-0.080, 0.008] | 105 |
| integrated_gradients | ERD_comb_spearman | -0.017 | [-0.058, 0.025] | 105 |
| integrated_gradients | vs_grad_pearson | 0.924 | [0.911, 0.935] | 105 |
| integrated_gradients | vs_grad_spearman | 0.915 | [0.902, 0.927] | 105 |
| smoothgrad | ERD_L_pearson | 0.016 | [-0.010, 0.042] | 105 |
| smoothgrad | ERD_L_spearman | 0.009 | [-0.016, 0.034] | 105 |
| smoothgrad | ERD_R_pearson | 0.010 | [-0.014, 0.033] | 105 |
| smoothgrad | ERD_R_spearman | 0.018 | [-0.005, 0.041] | 105 |
| smoothgrad | ERD_LR_pearson | 0.011 | [-0.012, 0.033] | 105 |
| smoothgrad | ERD_LR_spearman | 0.002 | [-0.019, 0.024] | 105 |
| smoothgrad | ERD_comb_pearson | 0.014 | [-0.013, 0.040] | 105 |
| smoothgrad | ERD_comb_spearman | 0.017 | [-0.010, 0.044] | 105 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.046 | [-0.083, -0.008] | -0.026 | -0.037 | -0.010 | 105 |
| ERD_theta_R | -0.003 | [-0.037, 0.032] | -0.035 | 0.006 | -0.029 | 105 |
| ERD_theta_mag | -0.045 | [-0.079, -0.011] | -0.045 | -0.028 | -0.028 | 105 |
| ERD_mu_L | -0.006 | [-0.037, 0.025] | -0.020 | 0.000 | -0.026 | 105 |
| ERD_mu_R | 0.001 | [-0.029, 0.031] | 0.021 | 0.001 | 0.019 | 105 |
| ERD_mu_mag | 0.002 | [-0.029, 0.033] | 0.002 | -0.001 | -0.001 | 105 |
| ERD_beta_L | 0.009 | [-0.022, 0.041] | -0.011 | 0.011 | -0.013 | 105 |
| ERD_beta_R | -0.005 | [-0.033, 0.023] | -0.021 | 0.006 | -0.013 | 105 |
| ERD_beta_mag | -0.018 | [-0.049, 0.012] | -0.018 | -0.012 | -0.012 | 105 |
| ERS_beta_rebound_L | 0.010 | [-0.018, 0.038] | 0.019 | 0.013 | 0.018 | 105 |
| ERS_beta_rebound_R | -0.013 | [-0.041, 0.015] | -0.010 | -0.010 | -0.007 | 105 |
| ERS_beta_rebound_mag | 0.008 | [-0.022, 0.038] | 0.008 | 0.013 | 0.013 | 105 |
| ERD_lowgamma_L | -0.012 | [-0.040, 0.016] | -0.021 | -0.007 | -0.012 | 105 |
| ERD_lowgamma_R | 0.011 | [-0.020, 0.041] | 0.016 | 0.015 | 0.012 | 105 |
| ERD_lowgamma_mag | -0.008 | [-0.038, 0.023] | -0.008 | -0.005 | -0.005 | 105 |
| logpow_8_30 | 0.010 | [-0.018, 0.037] | 0.009 | 0.015 | 0.015 | 105 |
| r2_signed | -0.009 | [-0.033, 0.016] | 0.013 | 0.004 | 0.006 | 105 |
| r2_abs | 0.013 | [-0.012, 0.038] | 0.013 | 0.006 | 0.006 | 105 |
| TFR_ERD_dB_L | -0.001 | [-0.033, 0.031] | -0.022 | -0.005 | -0.022 | 105 |
| TFR_ERD_dB_R | -0.008 | [-0.038, 0.022] | -0.001 | -0.003 | -0.005 | 105 |
| TFR_ERD_dB_mag | -0.013 | [-0.045, 0.018] | -0.013 | -0.017 | -0.017 | 105 |
| ITC_4_30 | -0.018 | [-0.047, 0.012] | -0.018 | -0.015 | -0.015 | 105 |
| ERP_abs_0.2_0.8 | -0.055 | [-0.086, -0.023] | -0.055 | -0.048 | -0.048 | 105 |
| SlowPot_0.5_3 | -0.023 | [-0.055, 0.010] | -0.014 | -0.017 | -0.004 | 105 |
| ERD_beta_classspecific | — | — | -0.010 [-0.031, 0.011] | — | -0.009 | 105 |
| ERD_lowgamma_classspecific | — | — | 0.007 [-0.015, 0.029] | — | 0.010 | 105 |
| ERD_mu_classspecific | — | — | 0.002 [-0.021, 0.025] | — | -0.002 | 105 |
| ERD_theta_classspecific | — | — | -0.032 [-0.058, -0.005] | — | -0.023 | 105 |
| ERS_beta_rebound_classspecific | — | — | 0.014 [-0.008, 0.035] | — | 0.015 | 105 |
| TFR_ERD_dB_classspecific | — | — | -0.009 [-0.031, 0.013] | — | -0.012 | 105 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +1.06 ± 4.01 | +0.20 ± 2.05 | +0.53 ± 2.73 | 0.0133 | 0.2543 |
| 4 | +1.40 ± 5.23 | +0.44 ± 2.59 | +0.70 ± 3.39 | 0.0184 | 0.2008 |
| 6 | +1.59 ± 5.51 | +0.53 ± 2.82 | +0.57 ± 3.71 | 0.0190 | 0.0646 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.006 | [-0.020, 0.008] | 105 |
| grad_x_input | 0.077 | [0.058, 0.095] | 105 |
| integrated_gradients | 0.076 | [0.057, 0.094] | 105 |
| smoothgrad | -0.006 | [-0.020, 0.008] | 105 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=zscore, S02=zscore, S03=zscore, S04=zscore, S05=zscore, S06=zscore, S07=zscore, S08=zscore, S09=zscore, S10=zscore, S11=zscore, S12=zscore, S13=zscore, S14=zscore, S15=zscore, S16=zscore, S17=zscore, S18=zscore, S19=zscore, S20=zscore, S21=zscore, S22=zscore, S23=zscore, S24=zscore, S25=zscore, S26=zscore, S27=zscore, S28=zscore, S29=zscore, S30=zscore, S31=zscore, S32=zscore, S33=zscore, S34=zscore, S35=zscore, S36=zscore, S37=zscore, S38=zscore, S39=zscore, S40=zscore, S41=zscore, S42=zscore, S43=zscore, S44=zscore, S45=zscore, S46=zscore, S47=zscore, S48=zscore, S49=zscore, S50=zscore, S51=zscore, S52=zscore, S53=zscore, S54=zscore, S55=zscore, S56=zscore, S57=zscore, S58=zscore, S59=zscore, S60=zscore, S61=zscore, S62=zscore, S63=zscore, S64=zscore, S65=zscore, S66=zscore, S67=zscore, S68=zscore, S69=zscore, S70=zscore, S71=zscore, S72=zscore, S73=zscore, S74=zscore, S75=zscore, S76=zscore, S77=zscore, S78=zscore, S79=zscore, S80=zscore, S81=zscore, S82=zscore, S83=zscore, S84=zscore, S85=zscore, S86=zscore, S87=zscore, S89=zscore, S90=zscore, S91=zscore, S93=zscore, S94=zscore, S95=zscore, S96=zscore, S97=zscore, S98=zscore, S99=zscore, S101=zscore, S102=zscore, S103=zscore, S105=zscore, S106=zscore, S107=zscore, S108=zscore, S109=zscore

| Profile | Mean validation accuracy (%) |
|---|---|
| zscore | 66.31 |
| ems | 65.95 |
| original | 54.88 |

```
model: "EEGNetv4"
dataset: "PhysionetMI"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"zscore": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": null, "align": "zscore", "kernel_length": 80, "dropout": 0.25, "lr": 0.001, "weight_decay": 0.001, "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "ems": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": "ems", "align": null, "kernel_length": 80, "dropout": 0.25, "lr": 0.001, "weight_decay": 0.001, "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "original": {"sfreq": 160, "band_hz": [8.0, 32.0], "window_s": [0.5, 2.5], "norm": null, "align": null, "car": false, "kernel_length": 64, "dropout": 0.25, "lr": 0.01, "weight_decay": 1e-05, "scheduler": "multistep", "milestones": [20, 50], "gamma": 0.1, "max_epochs": 100, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "augment": false, "select_on": "val_loss", "min_select_epoch": 1}}
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
| S01 | 36.67 | 30.00 | 33.33 | 26.67 | 23.33 | 50.00 | 26.67 | 33.33 |
| S02 | 26.67 | 36.67 | 36.67 | 33.33 | 33.33 | 16.67 | 23.33 | 39.33 |
| S03 | 30.00 | 40.00 | 26.67 | 43.33 | 36.67 | 26.67 | 20.00 | 32.67 |
| S04 | 30.00 | 23.33 | 30.00 | 23.33 | 20.00 | 30.00 | 30.00 | 28.00 |
| S05 | 23.33 | 26.67 | 13.33 | 26.67 | 23.33 | 16.67 | 23.33 | 25.33 |
| S06 | 23.33 | 26.67 | 20.00 | 10.00 | 10.00 | 30.00 | 43.33 | 31.33 |
| S07 | 53.33 | 26.67 | 40.00 | 40.00 | 43.33 | 26.67 | 50.00 | 32.00 |
| S08 | 26.67 | 26.67 | 50.00 | 16.67 | 23.33 | 33.33 | 26.67 | 26.00 |
| S09 | 23.33 | 0.00 | 26.67 | 6.67 | 16.67 | 23.33 | 26.67 | 18.67 |
| S10 | 30.00 | 16.67 | 36.67 | 23.33 | 43.33 | 23.33 | 30.00 | 28.67 |
| S11 | 33.33 | 30.00 | 33.33 | 23.33 | 36.67 | 20.00 | 26.67 | 24.67 |
| S12 | 36.67 | 20.00 | 30.00 | 36.67 | 33.33 | 26.67 | 30.00 | 29.33 |
| S13 | 26.67 | 26.67 | 20.00 | 23.33 | 20.00 | 30.00 | 33.33 | 20.00 |
| S14 | 26.67 | 26.67 | 30.00 | 26.67 | 30.00 | 26.67 | 30.00 | 26.67 |
| S15 | 46.67 | 33.33 | 53.33 | 40.00 | 46.67 | 36.67 | 53.33 | 36.00 |
| S16 | 16.67 | 30.00 | 26.67 | 20.00 | 23.33 | 20.00 | 26.67 | 38.00 |
| S17 | 26.67 | 36.67 | 16.67 | 20.00 | 23.33 | 13.33 | 23.33 | 23.33 |
| S18 | 56.67 | 60.00 | 50.00 | 66.67 | 63.33 | 33.33 | 20.00 | 46.00 |
| S19 | 26.67 | 26.67 | 23.33 | 13.33 | 23.33 | 26.67 | 33.33 | 24.00 |
| S20 | 20.00 | 33.33 | 36.67 | 33.33 | 33.33 | 20.00 | 23.33 | 29.33 |
| S21 | 26.67 | 23.33 | 26.67 | 26.67 | 36.67 | 26.67 | 26.67 | 31.33 |
| S22 | 30.00 | 43.33 | 43.33 | 30.00 | 26.67 | 30.00 | 36.67 | 32.67 |
| S23 | 26.67 | 23.33 | 16.67 | 20.00 | 16.67 | 23.33 | 20.00 | 24.67 |
| S24 | 20.00 | 30.00 | 30.00 | 13.33 | 10.00 | 30.00 | 13.33 | 17.33 |
| S25 | 33.33 | 26.67 | 26.67 | 33.33 | 26.67 | 20.00 | 30.00 | 25.33 |
| S26 | 33.33 | 30.00 | 26.67 | 23.33 | 26.67 | 23.33 | 26.67 | 30.00 |
| S27 | 30.00 | 16.67 | 20.00 | 10.00 | 16.67 | 26.67 | 16.67 | 25.33 |
| S28 | 20.00 | 36.67 | 33.33 | 60.00 | 43.33 | 23.33 | 20.00 | 32.00 |
| S29 | 40.00 | 46.67 | 23.33 | 60.00 | 40.00 | 63.33 | 46.67 | 36.00 |
| S30 | 26.67 | 36.67 | 26.67 | 33.33 | 33.33 | 40.00 | 36.67 | 33.33 |
| S31 | 50.00 | 33.33 | 20.00 | 26.67 | 16.67 | 46.67 | 30.00 | 35.33 |
| S32 | 10.00 | 10.00 | 26.67 | 16.67 | 3.33 | 30.00 | 36.67 | 27.33 |
| S33 | 36.67 | 53.33 | 43.33 | 33.33 | 40.00 | 30.00 | 30.00 | 35.33 |
| S34 | 30.00 | 26.67 | 26.67 | 26.67 | 23.33 | 46.67 | 16.67 | 27.33 |
| S35 | 83.33 | 80.00 | 66.67 | 90.00 | 83.33 | 76.67 | 56.67 | 74.67 |
| S36 | 43.33 | 26.67 | 36.67 | 36.67 | 43.33 | 30.00 | 36.67 | 32.00 |
| S37 | 20.00 | 33.33 | 23.33 | 23.33 | 23.33 | 23.33 | 23.33 | 20.67 |
| S38 | 6.67 | 16.67 | 20.00 | 6.67 | 6.67 | 30.00 | 30.00 | 22.00 |
| S39 | 20.00 | 20.00 | 36.67 | 40.00 | 36.67 | 30.00 | 26.67 | 27.33 |
| S40 | 56.67 | 30.00 | 43.33 | 36.67 | 43.33 | 26.67 | 40.00 | 39.33 |
| S41 | 23.33 | 30.00 | 20.00 | 36.67 | 30.00 | 40.00 | 30.00 | 30.00 |
| S42 | 50.00 | 26.67 | 33.33 | 26.67 | 16.67 | 66.67 | 43.33 | 44.67 |
| S43 | 23.33 | 20.00 | 16.67 | 23.33 | 26.67 | 13.33 | 13.33 | 19.33 |
| S44 | 16.67 | 23.33 | 20.00 | 20.00 | 16.67 | 23.33 | 30.00 | 26.00 |
| S45 | 50.00 | 53.33 | 36.67 | 40.00 | 46.67 | 40.00 | 33.33 | 39.33 |
| S46 | 26.67 | 40.00 | 26.67 | 26.67 | 26.67 | 23.33 | 33.33 | 27.33 |
| S47 | 26.67 | 23.33 | 33.33 | 16.67 | 26.67 | 26.67 | 23.33 | 25.33 |
| S48 | 43.33 | 30.00 | 23.33 | 23.33 | 30.00 | 33.33 | 26.67 | 44.00 |
| S49 | 30.00 | 40.00 | 16.67 | 26.67 | 16.67 | 30.00 | 23.33 | 29.33 |
| S50 | 33.33 | 26.67 | 23.33 | 13.33 | 13.33 | 33.33 | 26.67 | 31.33 |
| S51 | 36.67 | 36.67 | 40.00 | 30.00 | 36.67 | 36.67 | 33.33 | 42.00 |
| S52 | 46.67 | 43.33 | 20.00 | 20.00 | 33.33 | 40.00 | 46.67 | 37.33 |
| S53 | 40.00 | 26.67 | 23.33 | 40.00 | 33.33 | 16.67 | 43.33 | 21.33 |
| S54 | 43.33 | 43.33 | 33.33 | 26.67 | 20.00 | 43.33 | 56.67 | 30.67 |
| S55 | 46.67 | 43.33 | 30.00 | 40.00 | 23.33 | 26.67 | 40.00 | 29.33 |
| S56 | 33.33 | 30.00 | 30.00 | 23.33 | 20.00 | 33.33 | 50.00 | 40.00 |
| S57 | 50.00 | 40.00 | 46.67 | 43.33 | 40.00 | 30.00 | 33.33 | 32.67 |
| S58 | 53.33 | 26.67 | 40.00 | 40.00 | 36.67 | 30.00 | 43.33 | 34.67 |
| S59 | 30.00 | 23.33 | 26.67 | 33.33 | 30.00 | 23.33 | 23.33 | 26.67 |
| S60 | 40.00 | 23.33 | 36.67 | 30.00 | 26.67 | 53.33 | 36.67 | 26.00 |
| S61 | 33.33 | 30.00 | 36.67 | 23.33 | 36.67 | 36.67 | 36.67 | 26.67 |
| S62 | 56.67 | 40.00 | 43.33 | 23.33 | 33.33 | 56.67 | 30.00 | 38.00 |
| S63 | 23.33 | 33.33 | 13.33 | 26.67 | 16.67 | 33.33 | 20.00 | 31.33 |
| S64 | 33.33 | 26.67 | 20.00 | 20.00 | 23.33 | 23.33 | 26.67 | 22.67 |
| S65 | 26.67 | 20.00 | 26.67 | 26.67 | 36.67 | 26.67 | 30.00 | 28.00 |
| S66 | 23.33 | 33.33 | 33.33 | 30.00 | 33.33 | 26.67 | 30.00 | 32.67 |
| S67 | 26.67 | 16.67 | 30.00 | 46.67 | 40.00 | 33.33 | 30.00 | 25.33 |
| S68 | 20.00 | 13.33 | 20.00 | 30.00 | 30.00 | 36.67 | 23.33 | 18.67 |
| S69 | 36.67 | 26.67 | 33.33 | 33.33 | 30.00 | 26.67 | 36.67 | 30.67 |
| S70 | 30.00 | 23.33 | 26.67 | 33.33 | 26.67 | 53.33 | 43.33 | 35.33 |
| S71 | 46.67 | 36.67 | 23.33 | 23.33 | 26.67 | 40.00 | 40.00 | 35.33 |
| S72 | 56.67 | 46.67 | 43.33 | 50.00 | 63.33 | 63.33 | 43.33 | 50.67 |
| S73 | 16.67 | 26.67 | 10.00 | 16.67 | 16.67 | 26.67 | 20.00 | 18.67 |
| S74 | 36.67 | 40.00 | 23.33 | 30.00 | 33.33 | 20.00 | 20.00 | 29.33 |
| S75 | 23.33 | 23.33 | 33.33 | 33.33 | 30.00 | 26.67 | 30.00 | 31.33 |
| S76 | 13.33 | 23.33 | 10.00 | 30.00 | 10.00 | 30.00 | 13.33 | 20.67 |
| S77 | 43.33 | 16.67 | 26.67 | 30.00 | 26.67 | 30.00 | 33.33 | 22.00 |
| S78 | 30.00 | 36.67 | 36.67 | 23.33 | 36.67 | 26.67 | 23.33 | 27.33 |
| S79 | 23.33 | 30.00 | 16.67 | 16.67 | 10.00 | 6.67 | 13.33 | 25.33 |
| S80 | 36.67 | 30.00 | 46.67 | 26.67 | 16.67 | 20.00 | 30.00 | 32.67 |
| S81 | 23.33 | 46.67 | 26.67 | 26.67 | 33.33 | 26.67 | 20.00 | 25.33 |
| S82 | 26.67 | 33.33 | 23.33 | 30.00 | 30.00 | 23.33 | 33.33 | 34.67 |
| S83 | 13.33 | 30.00 | 13.33 | 26.67 | 13.33 | 23.33 | 16.67 | 18.00 |
| S84 | 23.33 | 26.67 | 46.67 | 30.00 | 16.67 | 36.67 | 26.67 | 32.00 |
| S85 | 46.67 | 56.67 | 36.67 | 46.67 | 46.67 | 56.67 | 30.00 | 52.00 |
| S86 | 33.33 | 26.67 | 26.67 | 23.33 | 20.00 | 20.00 | 23.33 | 27.33 |
| S87 | 30.00 | 33.33 | 33.33 | 26.67 | 26.67 | 33.33 | 33.33 | 30.67 |
| S89 | 40.00 | 30.00 | 40.00 | 40.00 | 40.00 | 30.00 | 40.00 | 42.00 |
| S90 | 30.00 | 36.67 | 23.33 | 30.00 | 43.33 | 23.33 | 16.67 | 24.67 |
| S91 | 43.33 | 43.33 | 46.67 | 50.00 | 40.00 | 36.67 | 46.67 | 40.00 |
| S93 | 30.00 | 26.67 | 26.67 | 30.00 | 20.00 | 26.67 | 23.33 | 27.33 |
| S94 | 30.00 | 30.00 | 26.67 | 26.67 | 16.67 | 30.00 | 23.33 | 24.67 |
| S95 | 46.67 | 33.33 | 26.67 | 26.67 | 16.67 | 13.33 | 30.00 | 23.33 |
| S96 | 23.33 | 36.67 | 30.00 | 33.33 | 33.33 | 36.67 | 30.00 | 28.67 |
| S97 | 40.00 | 30.00 | 40.00 | 23.33 | 36.67 | 16.67 | 26.67 | 27.33 |
| S98 | 20.00 | 23.33 | 40.00 | 30.00 | 30.00 | 26.67 | 26.67 | 29.33 |
| S99 | 23.33 | 13.33 | 23.33 | 30.00 | 30.00 | 23.33 | 23.33 | 16.00 |
| S101 | 40.00 | 6.67 | 70.00 | 56.67 | 23.33 | 26.67 | 33.33 | 38.67 |
| S102 | 33.33 | 43.33 | 30.00 | 26.67 | 33.33 | 30.00 | 26.67 | 32.00 |
| S103 | 40.00 | 43.33 | 26.67 | 30.00 | 40.00 | 26.67 | 26.67 | 25.33 |
| S105 | 36.67 | 36.67 | 40.00 | 56.67 | 30.00 | 30.00 | 33.33 | 26.67 |
| S106 | 43.33 | 36.67 | 26.67 | 30.00 | 30.00 | 23.33 | 36.67 | 27.33 |
| S107 | 23.33 | 23.33 | 33.33 | 30.00 | 20.00 | 20.00 | 20.00 | 19.33 |
| S108 | 26.67 | 23.33 | 20.00 | 23.33 | 16.67 | 30.00 | 43.33 | 28.00 |
| S109 | 13.33 | 23.33 | 33.33 | 10.00 | 20.00 | 23.33 | 30.00 | 20.67 |
| Mean | 32.60 | 30.83 | 30.38 | 29.97 | 28.92 | 30.54 | 30.32 | 30.13 |
| SD | 12.15 | 11.15 | 10.70 | 12.60 | 12.18 | 11.64 | 9.42 | 8.33 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 92 | CP6 CP5 FT8 FT7 P8 F7 C6 T10 T8 TP8 PO8 F5 |
| CSP | 15 | Fp1 AF7 Fpz Fp2 AF3 AF8 AF4 F5 F7 F8 FC6 AFz |
| RLF | 82 | T8 T10 F7 AF7 FT8 C6 T7 FT7 F8 AF8 TP8 Iz |
| MI | 67 | T8 F8 FC5 AF8 F7 FT8 AF4 C6 T10 F6 C4 AF7 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 8 | PO8 O2 O1 PO4 Oz POz PO7 PO3 Fpz P4 P2 Fp2 |

### T3. Equivalence of each 12-channel montage with all 64 channels (paired TOST)

Difference = montage − 64ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 105 | -1.78 | 11.14 | [-3.58, +0.03] | <0.0001 | yes | -33.33 | 38 | 4,5,6,8,10 |
| CSP | 105 | -2.22 | 11.15 | [-4.03, -0.42] | 0.0004 | yes | -30.00 | 42 | 5,6,8,10 |
| RLF | 105 | -2.63 | 11.70 | [-4.53, -0.74] | 0.0020 | yes | -33.33 | 47 | 5,6,8,10 |
| MI | 105 | -3.68 | 10.70 | [-5.42, -1.95] | 0.0143 | yes | -33.33 | 44 | 6,8,10 |
| Fixed sensorimotor | 105 | -2.06 | 11.95 | [-4.00, -0.13] | 0.0005 | yes | -33.33 | 41 | 4,5,6,8,10 |
| ERD-only | 105 | -2.29 | 10.44 | [-3.98, -0.59] | 0.0002 | yes | -36.67 | 45 | 4,5,6,8,10 |
| Random (mean of draws) | 105 | -2.48 | 8.88 | [-3.92, -1.04] | <0.0001 | yes | -23.33 | 32 | 4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 105 | +0.44 | [-2.08, +2.97] | 0.7278 | 1.0000 | 0.4870 | 1.0000 | 49 / 11 / 45 |
| RLF | 105 | +0.86 | [-1.41, +3.12] | 0.4550 | 1.0000 | 0.2440 | 1.0000 | 57 / 7 / 41 |
| MI | 105 | +1.90 | [-0.19, +4.00] | 0.0741 | 0.4448 | 0.0589 | 0.3536 | 61 / 9 / 35 |
| Fixed sensorimotor | 105 | +0.29 | [-2.16, +2.73] | 0.8170 | 1.0000 | 0.5346 | 1.0000 | 43 / 19 / 43 |
| ERD-only | 105 | +0.51 | [-1.97, +2.99] | 0.6851 | 1.0000 | 0.8041 | 1.0000 | 50 / 14 / 41 |
| Random (mean of draws) | 105 | +0.70 | [-1.02, +2.42] | 0.4217 | 1.0000 | 0.3198 | 1.0000 | 54 / 3 / 48 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 64ch | 32.60 | left_hand 31.1, right_hand 29.9, hands 28.9, feet 40.4 | left_hand 23.0, right_hand 22.6, hands 23.2, feet 31.2 | 26.67 | 1 / 105 |
| Ours | 30.83 | left_hand 28.5, right_hand 30.2, hands 29.6, feet 35.1 | left_hand 21.3, right_hand 25.6, hands 24.8, feet 28.3 | 26.67 | 8 / 105 |
| CSP | 30.38 | left_hand 29.8, right_hand 29.8, hands 26.1, feet 35.8 | left_hand 23.9, right_hand 23.6, hands 22.3, feet 30.3 | 26.67 | 10 / 105 |
| RLF | 29.97 | left_hand 30.8, right_hand 31.5, hands 24.9, feet 32.8 | left_hand 24.4, right_hand 25.7, hands 22.4, feet 27.4 | 26.67 | 6 / 105 |
| MI | 28.92 | left_hand 30.0, right_hand 25.7, hands 26.6, feet 33.3 | left_hand 24.6, right_hand 21.7, hands 24.9, feet 28.8 | 26.67 | 5 / 105 |
| Fixed sensorimotor | 30.54 | left_hand 27.6, right_hand 27.4, hands 32.7, feet 34.4 | left_hand 23.9, right_hand 25.1, hands 26.9, feet 24.0 | 26.67 | 5 / 105 |
| ERD-only | 30.32 | left_hand 25.6, right_hand 28.4, hands 30.3, feet 37.0 | left_hand 22.6, right_hand 23.9, hands 24.6, feet 28.9 | 26.67 | 4 / 105 |
| Random (mean of draws) | 30.13 | left_hand 29.4, right_hand 28.9, hands 25.9, feet 36.3 | left_hand 24.1, right_hand 25.1, hands 22.6, feet 28.2 | 26.67 | 24 / 525 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 64ch | 3 | 42: 29.11, 43: 30.35, 44: 29.49 | 29.65 | 0.63 |
| Ours | 3 | 42: 29.27, 43: 28.79, 44: 29.40 | 29.15 | 0.32 |
| CSP | 3 | 42: 28.32, 43: 28.16, 44: 27.97 | 28.15 | 0.17 |
| RLF | 3 | 42: 29.46, 43: 27.65, 44: 29.21 | 28.77 | 0.98 |
| MI | 3 | 42: 28.29, 43: 27.49, 44: 28.63 | 28.14 | 0.59 |
| Fixed sensorimotor | 3 | 42: 30.19, 43: 29.37, 44: 29.05 | 29.53 | 0.59 |
| ERD-only | 3 | 42: 30.19, 43: 29.14, 44: 27.30 | 28.88 | 1.46 |
| Random (mean of draws) | 3 | 42: 28.88, 43: 28.17, 44: 28.34 | 28.46 | 0.37 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full64/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | -0.036 | [-0.063, -0.009] | 0.136 | 105 |
| grad | class_specific_pearson | 0.007 | [-0.013, 0.026] | 0.100 | 105 |
| grad | lateralisation_pearson | 0.011 | [-0.016, 0.038] | 0.137 | 105 |
| grad | magnitude_spearman | -0.039 | [-0.066, -0.011] | 0.139 | 105 |
| grad | class_specific_spearman | -0.000 | [-0.020, 0.019] | 0.099 | 105 |
| grad | lateralisation_spearman | 0.012 | [-0.015, 0.038] | 0.135 | 105 |
| grad | legacy_signed_LR_pearson | -0.007 | [-0.034, 0.021] | 0.139 | 105 |
| grad_x_input | magnitude_pearson | -0.050 | [-0.084, -0.015] | 0.173 | 105 |
| grad_x_input | class_specific_pearson | 0.002 | [-0.026, 0.031] | 0.144 | 105 |
| grad_x_input | lateralisation_pearson | -0.085 | [-0.128, -0.043] | 0.210 | 105 |
| grad_x_input | magnitude_spearman | -0.048 | [-0.083, -0.014] | 0.173 | 105 |
| grad_x_input | class_specific_spearman | -0.005 | [-0.032, 0.022] | 0.139 | 105 |
| grad_x_input | lateralisation_spearman | -0.070 | [-0.112, -0.028] | 0.208 | 105 |
| grad_x_input | legacy_signed_LR_pearson | -0.009 | [-0.047, 0.029] | 0.186 | 105 |
| integrated_gradients | magnitude_pearson | -0.050 | [-0.084, -0.016] | 0.169 | 105 |
| integrated_gradients | class_specific_pearson | 0.002 | [-0.026, 0.030] | 0.142 | 105 |
| integrated_gradients | lateralisation_pearson | -0.082 | [-0.124, -0.040] | 0.208 | 105 |
| integrated_gradients | magnitude_spearman | -0.048 | [-0.081, -0.014] | 0.169 | 105 |
| integrated_gradients | class_specific_spearman | -0.005 | [-0.032, 0.023] | 0.138 | 105 |
| integrated_gradients | lateralisation_spearman | -0.068 | [-0.109, -0.027] | 0.204 | 105 |
| integrated_gradients | legacy_signed_LR_pearson | -0.007 | [-0.045, 0.030] | 0.185 | 105 |
| smoothgrad | magnitude_pearson | -0.036 | [-0.062, -0.009] | 0.136 | 105 |
| smoothgrad | class_specific_pearson | 0.006 | [-0.013, 0.026] | 0.100 | 105 |
| smoothgrad | lateralisation_pearson | 0.011 | [-0.016, 0.038] | 0.137 | 105 |
| smoothgrad | magnitude_spearman | -0.039 | [-0.066, -0.011] | 0.139 | 105 |
| smoothgrad | class_specific_spearman | -0.001 | [-0.020, 0.019] | 0.100 | 105 |
| smoothgrad | lateralisation_spearman | 0.011 | [-0.015, 0.038] | 0.135 | 105 |
| smoothgrad | legacy_signed_LR_pearson | -0.007 | [-0.034, 0.021] | 0.139 | 105 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.004 | [-0.034, 0.025] | 105 |
| grad | ERD_L_spearman | 0.003 | [-0.024, 0.030] | 105 |
| grad | ERD_R_pearson | -0.007 | [-0.031, 0.017] | 105 |
| grad | ERD_R_spearman | -0.004 | [-0.029, 0.022] | 105 |
| grad | ERD_LR_pearson | -0.007 | [-0.034, 0.021] | 105 |
| grad | ERD_LR_spearman | 0.007 | [-0.020, 0.034] | 105 |
| grad | ERD_comb_pearson | -0.010 | [-0.037, 0.017] | 105 |
| grad | ERD_comb_spearman | -0.003 | [-0.030, 0.024] | 105 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |
| grad_x_input | ERD_L_pearson | -0.037 | [-0.076, 0.003] | 105 |
| grad_x_input | ERD_L_spearman | -0.023 | [-0.061, 0.015] | 105 |
| grad_x_input | ERD_R_pearson | -0.033 | [-0.071, 0.005] | 105 |
| grad_x_input | ERD_R_spearman | -0.026 | [-0.064, 0.012] | 105 |
| grad_x_input | ERD_LR_pearson | -0.009 | [-0.047, 0.029] | 105 |
| grad_x_input | ERD_LR_spearman | 0.009 | [-0.026, 0.045] | 105 |
| grad_x_input | ERD_comb_pearson | -0.049 | [-0.089, -0.010] | 105 |
| grad_x_input | ERD_comb_spearman | -0.034 | [-0.073, 0.006] | 105 |
| grad_x_input | vs_grad_pearson | 0.942 | [0.931, 0.950] | 105 |
| grad_x_input | vs_grad_spearman | 0.932 | [0.922, 0.941] | 105 |
| integrated_gradients | ERD_L_pearson | -0.036 | [-0.074, 0.003] | 105 |
| integrated_gradients | ERD_L_spearman | -0.022 | [-0.059, 0.015] | 105 |
| integrated_gradients | ERD_R_pearson | -0.034 | [-0.071, 0.004] | 105 |
| integrated_gradients | ERD_R_spearman | -0.026 | [-0.064, 0.011] | 105 |
| integrated_gradients | ERD_LR_pearson | -0.007 | [-0.045, 0.030] | 105 |
| integrated_gradients | ERD_LR_spearman | 0.011 | [-0.024, 0.046] | 105 |
| integrated_gradients | ERD_comb_pearson | -0.049 | [-0.088, -0.010] | 105 |
| integrated_gradients | ERD_comb_spearman | -0.033 | [-0.072, 0.006] | 105 |
| integrated_gradients | vs_grad_pearson | 0.940 | [0.931, 0.949] | 105 |
| integrated_gradients | vs_grad_spearman | 0.931 | [0.921, 0.940] | 105 |
| smoothgrad | ERD_L_pearson | -0.004 | [-0.034, 0.025] | 105 |
| smoothgrad | ERD_L_spearman | 0.003 | [-0.024, 0.031] | 105 |
| smoothgrad | ERD_R_pearson | -0.007 | [-0.031, 0.017] | 105 |
| smoothgrad | ERD_R_spearman | -0.004 | [-0.029, 0.021] | 105 |
| smoothgrad | ERD_LR_pearson | -0.007 | [-0.034, 0.021] | 105 |
| smoothgrad | ERD_LR_spearman | 0.007 | [-0.019, 0.034] | 105 |
| smoothgrad | ERD_comb_pearson | -0.010 | [-0.037, 0.017] | 105 |
| smoothgrad | ERD_comb_spearman | -0.003 | [-0.031, 0.024] | 105 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | 0.000 | [-0.034, 0.035] | -0.052 | 0.000 | -0.036 | 105 |
| ERD_theta_R | -0.037 | [-0.068, -0.005] | -0.068 | -0.017 | -0.054 | 105 |
| ERD_theta_mag | -0.080 | [-0.111, -0.048] | -0.080 | -0.057 | -0.057 | 105 |
| ERD_mu_L | -0.020 | [-0.051, 0.010] | -0.058 | -0.011 | -0.054 | 105 |
| ERD_mu_R | -0.025 | [-0.051, 0.002] | -0.004 | -0.019 | -0.007 | 105 |
| ERD_mu_mag | -0.044 | [-0.070, -0.017] | -0.044 | -0.048 | -0.048 | 105 |
| ERD_beta_L | -0.018 | [-0.051, 0.015] | -0.021 | -0.005 | -0.023 | 105 |
| ERD_beta_R | 0.017 | [-0.014, 0.047] | -0.020 | 0.016 | -0.011 | 105 |
| ERD_beta_mag | -0.029 | [-0.059, 0.001] | -0.029 | -0.025 | -0.025 | 105 |
| ERS_beta_rebound_L | -0.033 | [-0.065, 0.000] | -0.014 | -0.024 | -0.015 | 105 |
| ERS_beta_rebound_R | -0.006 | [-0.035, 0.022] | -0.007 | -0.000 | 0.000 | 105 |
| ERS_beta_rebound_mag | -0.016 | [-0.048, 0.017] | -0.016 | -0.017 | -0.017 | 105 |
| ERD_lowgamma_L | -0.004 | [-0.036, 0.028] | -0.010 | 0.003 | -0.003 | 105 |
| ERD_lowgamma_R | 0.019 | [-0.008, 0.047] | 0.003 | 0.017 | -0.014 | 105 |
| ERD_lowgamma_mag | -0.007 | [-0.035, 0.021] | -0.007 | -0.012 | -0.012 | 105 |
| logpow_8_30 | -0.032 | [-0.066, 0.002] | -0.031 | -0.021 | -0.021 | 105 |
| r2_signed | 0.018 | [-0.012, 0.048] | 0.004 | 0.013 | 0.001 | 105 |
| r2_abs | 0.004 | [-0.026, 0.034] | 0.004 | 0.001 | 0.001 | 105 |
| TFR_ERD_dB_L | -0.020 | [-0.054, 0.014] | -0.036 | -0.014 | -0.032 | 105 |
| TFR_ERD_dB_R | -0.006 | [-0.034, 0.023] | -0.024 | -0.007 | -0.018 | 105 |
| TFR_ERD_dB_mag | -0.041 | [-0.069, -0.012] | -0.041 | -0.037 | -0.037 | 105 |
| ITC_4_30 | -0.021 | [-0.052, 0.010] | -0.021 | -0.011 | -0.011 | 105 |
| ERP_abs_0.2_0.8 | -0.059 | [-0.090, -0.027] | -0.059 | -0.058 | -0.058 | 105 |
| SlowPot_0.5_3 | -0.022 | [-0.049, 0.005] | -0.018 | -0.025 | -0.005 | 105 |
| ERD_beta_classspecific | — | — | -0.013 [-0.033, 0.007] | — | -0.011 | 105 |
| ERD_lowgamma_classspecific | — | — | 0.007 [-0.011, 0.024] | — | 0.004 | 105 |
| ERD_mu_classspecific | — | — | -0.037 [-0.054, -0.019] | — | -0.035 | 105 |
| ERD_theta_classspecific | — | — | -0.055 [-0.078, -0.031] | — | -0.042 | 105 |
| ERS_beta_rebound_classspecific | — | — | -0.002 [-0.026, 0.022] | — | -0.003 | 105 |
| TFR_ERD_dB_classspecific | — | — | -0.028 [-0.049, -0.006] | — | -0.024 | 105 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +0.58 ± 3.44 | +0.26 ± 1.85 | +0.21 ± 2.45 | 0.2492 | 0.2006 |
| 4 | +0.85 ± 4.12 | +0.36 ± 2.52 | -0.13 ± 3.12 | 0.1055 | 0.0143 |
| 6 | +1.16 ± 4.67 | +0.73 ± 2.85 | +0.44 ± 3.50 | 0.2358 | 0.1214 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.010 | [-0.004, 0.024] | 105 |
| grad_x_input | 0.074 | [0.058, 0.089] | 105 |
| integrated_gradients | 0.072 | [0.056, 0.088] | 105 |
| smoothgrad | 0.010 | [-0.004, 0.024] | 105 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=zscore, S02=zscore, S03=zscore, S04=zscore, S05=zscore, S06=zscore, S07=zscore, S08=zscore, S09=zscore, S10=zscore, S11=zscore, S12=zscore, S13=zscore, S14=zscore, S15=zscore, S16=zscore, S17=zscore, S18=zscore, S19=zscore, S20=zscore, S21=zscore, S22=zscore, S23=zscore, S24=zscore, S25=zscore, S26=zscore, S27=zscore, S28=zscore, S29=zscore, S30=zscore, S31=zscore, S32=zscore, S33=zscore, S34=zscore, S35=zscore, S36=zscore, S37=zscore, S38=zscore, S39=zscore, S40=zscore, S41=zscore, S42=zscore, S43=zscore, S44=zscore, S45=zscore, S46=zscore, S47=zscore, S48=zscore, S49=zscore, S50=zscore, S51=zscore, S52=zscore, S53=zscore, S54=zscore, S55=zscore, S56=zscore, S57=zscore, S58=zscore, S59=zscore, S60=zscore, S61=zscore, S62=zscore, S63=zscore, S64=zscore, S65=zscore, S66=zscore, S67=zscore, S68=zscore, S69=zscore, S70=zscore, S71=zscore, S72=zscore, S73=zscore, S74=zscore, S75=zscore, S76=zscore, S77=zscore, S78=zscore, S79=zscore, S80=zscore, S81=zscore, S82=zscore, S83=zscore, S84=zscore, S85=zscore, S86=zscore, S87=zscore, S89=zscore, S90=zscore, S91=zscore, S93=zscore, S94=zscore, S95=zscore, S96=zscore, S97=zscore, S98=zscore, S99=zscore, S101=zscore, S102=zscore, S103=zscore, S105=zscore, S106=zscore, S107=zscore, S108=zscore, S109=zscore

| Profile | Mean validation accuracy (%) |
|---|---|
| zscore | 42.86 |
| ems | 41.90 |
| original | 28.81 |

```
model: "EEGNetv4"
dataset: "PhysionetMI"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"zscore": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": null, "align": "zscore", "kernel_length": 80, "dropout": 0.25, "lr": 0.001, "weight_decay": 0.001, "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "ems": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": "ems", "align": null, "kernel_length": 80, "dropout": 0.25, "lr": 0.001, "weight_decay": 0.001, "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "original": {"sfreq": 160, "band_hz": [8.0, 32.0], "window_s": [0.5, 2.5], "norm": null, "align": null, "car": false, "kernel_length": 64, "dropout": 0.25, "lr": 0.01, "weight_decay": 1e-05, "scheduler": "multistep", "milestones": [20, 50], "gamma": 0.1, "max_epochs": 100, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "augment": false, "select_on": "val_loss", "min_select_epoch": 1}}
checkpoint_rule: "highest validation accuracy (loss breaks ties), not before epoch 20 ('original': lowest validation loss); refit keeps the final weights"
folds: "run-wise chronological split per subject: test = the last imagery run of each type (run 12 left/right, run 14 hands/feet); training = runs 4, 8 (and 6, 10); validation = the chronologically last 25% of the training trials of EACH class; refit on all training runs"
seeds: [42, 43, 44]
```

Figures (530 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `4class\figures`.
