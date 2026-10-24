# Results report — EEGConformer\PhysioNet

Generated 2026-10-23 21:25 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Protocol: run-wise chronological split per subject: test = the last imagery run of each type (run 12 left/right, run 14 hands/feet); training = runs 4, 8 (and 6, 10); validation = the chronologically last 25% of the training trials of EACH class; refit on all training runs. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects (n = 105). Random = mean of the random-montage draws.

| Subject | 64ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|
| S01 | 66.67 | 73.33 | 53.33 | 73.33 | 60.00 | 46.67 | 53.33 | 62.67 |
| S02 | 93.33 | 53.33 | 46.67 | 93.33 | 80.00 | 93.33 | 73.33 | 81.33 |
| S03 | 53.33 | 53.33 | 53.33 | 40.00 | 46.67 | 46.67 | 46.67 | 50.67 |
| S04 | 73.33 | 66.67 | 73.33 | 86.67 | 66.67 | 53.33 | 66.67 | 73.33 |
| S05 | 53.33 | 60.00 | 53.33 | 66.67 | 60.00 | 33.33 | 53.33 | 57.33 |
| S06 | 46.67 | 53.33 | 60.00 | 80.00 | 73.33 | 60.00 | 60.00 | 50.67 |
| S07 | 93.33 | 66.67 | 73.33 | 86.67 | 73.33 | 93.33 | 86.67 | 73.33 |
| S08 | 46.67 | 53.33 | 53.33 | 40.00 | 40.00 | 40.00 | 60.00 | 41.33 |
| S09 | 73.33 | 53.33 | 60.00 | 60.00 | 46.67 | 53.33 | 66.67 | 57.33 |
| S10 | 46.67 | 53.33 | 40.00 | 66.67 | 66.67 | 66.67 | 33.33 | 49.33 |
| S11 | 66.67 | 60.00 | 60.00 | 53.33 | 53.33 | 40.00 | 73.33 | 66.67 |
| S12 | 60.00 | 46.67 | 53.33 | 53.33 | 46.67 | 46.67 | 60.00 | 56.00 |
| S13 | 73.33 | 66.67 | 33.33 | 53.33 | 46.67 | 73.33 | 60.00 | 54.67 |
| S14 | 60.00 | 33.33 | 46.67 | 60.00 | 46.67 | 46.67 | 66.67 | 54.67 |
| S15 | 100.00 | 73.33 | 33.33 | 66.67 | 80.00 | 100.00 | 93.33 | 72.00 |
| S16 | 66.67 | 60.00 | 60.00 | 53.33 | 66.67 | 73.33 | 60.00 | 65.33 |
| S17 | 66.67 | 53.33 | 60.00 | 46.67 | 46.67 | 60.00 | 60.00 | 50.67 |
| S18 | 66.67 | 46.67 | 80.00 | 53.33 | 66.67 | 46.67 | 40.00 | 52.00 |
| S19 | 40.00 | 33.33 | 33.33 | 46.67 | 46.67 | 40.00 | 46.67 | 42.67 |
| S20 | 46.67 | 46.67 | 46.67 | 60.00 | 46.67 | 40.00 | 53.33 | 54.67 |
| S21 | 60.00 | 60.00 | 73.33 | 66.67 | 40.00 | 60.00 | 60.00 | 53.33 |
| S22 | 73.33 | 66.67 | 93.33 | 73.33 | 66.67 | 53.33 | 60.00 | 66.67 |
| S23 | 60.00 | 53.33 | 60.00 | 40.00 | 53.33 | 73.33 | 86.67 | 60.00 |
| S24 | 60.00 | 53.33 | 53.33 | 60.00 | 66.67 | 53.33 | 53.33 | 44.00 |
| S25 | 53.33 | 46.67 | 46.67 | 46.67 | 46.67 | 53.33 | 86.67 | 49.33 |
| S26 | 86.67 | 46.67 | 40.00 | 53.33 | 53.33 | 73.33 | 86.67 | 70.67 |
| S27 | 60.00 | 66.67 | 66.67 | 66.67 | 60.00 | 60.00 | 53.33 | 60.00 |
| S28 | 60.00 | 73.33 | 60.00 | 60.00 | 60.00 | 66.67 | 53.33 | 57.33 |
| S29 | 93.33 | 80.00 | 53.33 | 86.67 | 80.00 | 100.00 | 66.67 | 82.67 |
| S30 | 66.67 | 60.00 | 46.67 | 60.00 | 46.67 | 46.67 | 53.33 | 53.33 |
| S31 | 93.33 | 86.67 | 60.00 | 86.67 | 86.67 | 86.67 | 60.00 | 85.33 |
| S32 | 73.33 | 66.67 | 73.33 | 66.67 | 80.00 | 60.00 | 66.67 | 61.33 |
| S33 | 93.33 | 80.00 | 40.00 | 80.00 | 86.67 | 80.00 | 86.67 | 84.00 |
| S34 | 80.00 | 80.00 | 80.00 | 80.00 | 73.33 | 73.33 | 93.33 | 76.00 |
| S35 | 80.00 | 60.00 | 40.00 | 53.33 | 66.67 | 60.00 | 80.00 | 66.67 |
| S36 | 73.33 | 66.67 | 93.33 | 66.67 | 73.33 | 86.67 | 80.00 | 80.00 |
| S37 | 46.67 | 46.67 | 66.67 | 46.67 | 60.00 | 33.33 | 46.67 | 49.33 |
| S38 | 73.33 | 66.67 | 66.67 | 66.67 | 60.00 | 60.00 | 66.67 | 68.00 |
| S39 | 66.67 | 53.33 | 53.33 | 53.33 | 53.33 | 66.67 | 46.67 | 54.67 |
| S40 | 80.00 | 66.67 | 86.67 | 73.33 | 60.00 | 46.67 | 66.67 | 64.00 |
| S41 | 80.00 | 53.33 | 60.00 | 66.67 | 60.00 | 53.33 | 60.00 | 60.00 |
| S42 | 73.33 | 66.67 | 60.00 | 66.67 | 73.33 | 86.67 | 100.00 | 78.67 |
| S43 | 53.33 | 53.33 | 40.00 | 40.00 | 46.67 | 66.67 | 66.67 | 61.33 |
| S44 | 66.67 | 73.33 | 53.33 | 66.67 | 60.00 | 66.67 | 80.00 | 60.00 |
| S45 | 40.00 | 46.67 | 40.00 | 46.67 | 66.67 | 46.67 | 46.67 | 46.67 |
| S46 | 60.00 | 46.67 | 60.00 | 40.00 | 60.00 | 53.33 | 46.67 | 54.67 |
| S47 | 60.00 | 73.33 | 60.00 | 53.33 | 73.33 | 60.00 | 53.33 | 64.00 |
| S48 | 53.33 | 33.33 | 46.67 | 46.67 | 40.00 | 53.33 | 73.33 | 62.67 |
| S49 | 86.67 | 46.67 | 73.33 | 53.33 | 66.67 | 93.33 | 46.67 | 61.33 |
| S50 | 60.00 | 60.00 | 53.33 | 66.67 | 46.67 | 60.00 | 60.00 | 54.67 |
| S51 | 66.67 | 60.00 | 86.67 | 60.00 | 66.67 | 33.33 | 73.33 | 65.33 |
| S52 | 66.67 | 40.00 | 60.00 | 53.33 | 53.33 | 66.67 | 53.33 | 56.00 |
| S53 | 93.33 | 86.67 | 86.67 | 100.00 | 93.33 | 80.00 | 86.67 | 85.33 |
| S54 | 93.33 | 93.33 | 86.67 | 86.67 | 93.33 | 80.00 | 93.33 | 86.67 |
| S55 | 40.00 | 46.67 | 40.00 | 40.00 | 40.00 | 40.00 | 60.00 | 57.33 |
| S56 | 66.67 | 60.00 | 53.33 | 60.00 | 53.33 | 73.33 | 66.67 | 60.00 |
| S57 | 60.00 | 53.33 | 46.67 | 66.67 | 53.33 | 66.67 | 60.00 | 56.00 |
| S58 | 80.00 | 73.33 | 66.67 | 80.00 | 46.67 | 66.67 | 60.00 | 76.00 |
| S59 | 33.33 | 53.33 | 33.33 | 33.33 | 33.33 | 33.33 | 46.67 | 38.67 |
| S60 | 93.33 | 73.33 | 73.33 | 80.00 | 86.67 | 86.67 | 73.33 | 73.33 |
| S61 | 53.33 | 66.67 | 60.00 | 80.00 | 60.00 | 60.00 | 66.67 | 54.67 |
| S62 | 86.67 | 73.33 | 46.67 | 66.67 | 73.33 | 73.33 | 73.33 | 82.67 |
| S63 | 40.00 | 40.00 | 60.00 | 53.33 | 60.00 | 60.00 | 53.33 | 57.33 |
| S64 | 66.67 | 53.33 | 40.00 | 46.67 | 73.33 | 60.00 | 80.00 | 69.33 |
| S65 | 66.67 | 66.67 | 60.00 | 60.00 | 66.67 | 60.00 | 73.33 | 65.33 |
| S66 | 60.00 | 53.33 | 60.00 | 53.33 | 53.33 | 53.33 | 60.00 | 54.67 |
| S67 | 66.67 | 66.67 | 60.00 | 73.33 | 40.00 | 66.67 | 53.33 | 58.67 |
| S68 | 53.33 | 46.67 | 73.33 | 46.67 | 60.00 | 46.67 | 33.33 | 53.33 |
| S69 | 60.00 | 86.67 | 73.33 | 86.67 | 86.67 | 60.00 | 60.00 | 66.67 |
| S70 | 53.33 | 60.00 | 66.67 | 66.67 | 53.33 | 60.00 | 73.33 | 60.00 |
| S71 | 66.67 | 40.00 | 33.33 | 46.67 | 66.67 | 73.33 | 46.67 | 60.00 |
| S72 | 80.00 | 80.00 | 60.00 | 80.00 | 80.00 | 80.00 | 80.00 | 82.67 |
| S73 | 53.33 | 46.67 | 66.67 | 66.67 | 80.00 | 66.67 | 66.67 | 58.67 |
| S74 | 33.33 | 26.67 | 26.67 | 26.67 | 33.33 | 20.00 | 33.33 | 37.33 |
| S75 | 60.00 | 66.67 | 53.33 | 66.67 | 66.67 | 66.67 | 53.33 | 64.00 |
| S76 | 40.00 | 46.67 | 40.00 | 33.33 | 40.00 | 40.00 | 60.00 | 41.33 |
| S77 | 46.67 | 53.33 | 53.33 | 46.67 | 53.33 | 53.33 | 46.67 | 58.67 |
| S78 | 66.67 | 86.67 | 73.33 | 73.33 | 73.33 | 53.33 | 73.33 | 66.67 |
| S79 | 46.67 | 66.67 | 40.00 | 53.33 | 46.67 | 66.67 | 60.00 | 41.33 |
| S80 | 46.67 | 40.00 | 53.33 | 46.67 | 46.67 | 40.00 | 66.67 | 50.67 |
| S81 | 40.00 | 46.67 | 40.00 | 46.67 | 40.00 | 40.00 | 46.67 | 34.67 |
| S82 | 60.00 | 46.67 | 73.33 | 53.33 | 46.67 | 46.67 | 60.00 | 54.67 |
| S83 | 46.67 | 53.33 | 33.33 | 26.67 | 53.33 | 40.00 | 53.33 | 48.00 |
| S84 | 60.00 | 66.67 | 40.00 | 66.67 | 60.00 | 60.00 | 33.33 | 60.00 |
| S85 | 80.00 | 80.00 | 60.00 | 80.00 | 66.67 | 93.33 | 80.00 | 82.67 |
| S86 | 46.67 | 46.67 | 86.67 | 46.67 | 46.67 | 53.33 | 46.67 | 50.67 |
| S87 | 66.67 | 46.67 | 40.00 | 40.00 | 33.33 | 66.67 | 46.67 | 40.00 |
| S89 | 73.33 | 80.00 | 80.00 | 80.00 | 86.67 | 40.00 | 53.33 | 74.67 |
| S90 | 53.33 | 53.33 | 60.00 | 53.33 | 66.67 | 60.00 | 60.00 | 57.33 |
| S91 | 73.33 | 66.67 | 66.67 | 66.67 | 86.67 | 66.67 | 60.00 | 68.00 |
| S93 | 60.00 | 46.67 | 46.67 | 66.67 | 53.33 | 60.00 | 80.00 | 53.33 |
| S94 | 86.67 | 60.00 | 66.67 | 66.67 | 66.67 | 93.33 | 66.67 | 86.67 |
| S95 | 53.33 | 53.33 | 40.00 | 73.33 | 53.33 | 46.67 | 33.33 | 53.33 |
| S96 | 86.67 | 66.67 | 80.00 | 73.33 | 86.67 | 66.67 | 66.67 | 84.00 |
| S97 | 80.00 | 100.00 | 73.33 | 93.33 | 66.67 | 73.33 | 53.33 | 65.33 |
| S98 | 66.67 | 66.67 | 46.67 | 60.00 | 60.00 | 60.00 | 53.33 | 58.67 |
| S99 | 60.00 | 53.33 | 60.00 | 66.67 | 46.67 | 60.00 | 60.00 | 56.00 |
| S101 | 40.00 | 40.00 | 46.67 | 46.67 | 46.67 | 46.67 | 40.00 | 54.67 |
| S102 | 40.00 | 60.00 | 60.00 | 40.00 | 40.00 | 53.33 | 53.33 | 48.00 |
| S103 | 60.00 | 53.33 | 66.67 | 53.33 | 60.00 | 60.00 | 66.67 | 61.33 |
| S105 | 53.33 | 46.67 | 66.67 | 60.00 | 80.00 | 80.00 | 53.33 | 54.67 |
| S106 | 53.33 | 60.00 | 60.00 | 53.33 | 53.33 | 46.67 | 60.00 | 57.33 |
| S107 | 60.00 | 60.00 | 60.00 | 46.67 | 46.67 | 53.33 | 73.33 | 54.67 |
| S108 | 66.67 | 60.00 | 66.67 | 60.00 | 66.67 | 53.33 | 66.67 | 66.67 |
| S109 | 53.33 | 33.33 | 46.67 | 46.67 | 46.67 | 40.00 | 46.67 | 48.00 |
| Mean | 64.00 | 58.98 | 57.84 | 60.70 | 60.19 | 60.32 | 61.90 | 60.66 |
| SD | 15.57 | 14.01 | 14.89 | 15.30 | 14.57 | 16.31 | 14.46 | 12.01 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 41 | T10 T9 TP8 TP7 T8 FT8 P8 F8 PO8 T7 C6 F7 |
| CSP | 15 | Fpz Fp1 AF7 Fp2 AFz AF3 AF8 AF4 F8 F7 F5 F1 |
| RLF | 70 | F7 CP6 C4 CP4 FT7 F8 P6 C6 P4 T9 FC4 PO4 |
| MI | 94 | F7 F8 F5 P6 F2 F3 FC4 CP4 CP6 CP2 P4 F4 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 8 | PO8 O2 O1 PO4 Oz POz PO7 PO3 Fpz P4 P2 Fp2 |

### T3. Equivalence of each 12-channel montage with all 64 channels (paired TOST)

Difference = montage − 64ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 105 | -5.02 | 13.03 | [-7.13, -2.90] | 0.2204 | no | -40.00 | 59 | 8,10 |
| CSP | 105 | -6.16 | 17.31 | [-8.96, -3.35] | 0.5373 | no | -66.67 | 58 | 10 |
| RLF | 105 | -3.30 | 12.48 | [-5.32, -1.28] | 0.0144 | yes | -33.33 | 55 | 6,8,10 |
| MI | 105 | -3.81 | 12.91 | [-5.90, -1.72] | 0.0425 | yes | -33.33 | 51 | 6,8,10 |
| Fixed sensorimotor | 105 | -3.68 | 11.91 | [-5.61, -1.75] | 0.0244 | yes | -33.33 | 51 | 6,8,10 |
| ERD-only | 105 | -2.10 | 14.09 | [-4.38, +0.19] | 0.0027 | yes | -40.00 | 47 | 5,6,8,10 |
| Random (mean of draws) | 105 | -3.34 | 8.82 | [-4.77, -1.91] | 0.0013 | yes | -28.00 | 33 | 5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 105 | +1.14 | [-1.90, +4.19] | 0.4582 | 1.0000 | 0.4445 | 0.8903 | 45 / 22 / 38 |
| RLF | 105 | -1.71 | [-3.83, +0.40] | 0.1114 | 0.5569 | 0.1108 | 0.4430 | 31 / 31 / 43 |
| MI | 105 | -1.21 | [-3.67, +1.26] | 0.3340 | 1.0000 | 0.2968 | 0.8903 | 37 / 24 / 44 |
| Fixed sensorimotor | 105 | -1.33 | [-4.36, +1.69] | 0.3837 | 1.0000 | 0.3606 | 0.8903 | 41 / 24 / 40 |
| ERD-only | 105 | -2.92 | [-5.92, +0.08] | 0.0562 | 0.3369 | 0.0560 | 0.3359 | 30 / 26 / 49 |
| Random (mean of draws) | 105 | -1.68 | [-3.78, +0.43] | 0.1173 | 0.5569 | 0.0845 | 0.4227 | 42 / 6 / 57 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 64ch | 64.00 | left_hand 63.1, right_hand 65.0 | left_hand 49.1, right_hand 50.9 | 53.33 | 8 / 105 |
| Ours | 58.98 | left_hand 56.7, right_hand 61.3 | left_hand 47.8, right_hand 52.2 | 53.33 | 16 / 105 |
| CSP | 57.84 | left_hand 58.3, right_hand 57.4 | left_hand 50.5, right_hand 49.5 | 53.33 | 17 / 105 |
| RLF | 60.70 | left_hand 61.8, right_hand 59.6 | left_hand 51.2, right_hand 48.8 | 53.33 | 11 / 105 |
| MI | 60.19 | left_hand 59.6, right_hand 60.7 | left_hand 49.5, right_hand 50.5 | 53.33 | 13 / 105 |
| Fixed sensorimotor | 60.32 | left_hand 62.9, right_hand 57.7 | left_hand 52.7, right_hand 47.3 | 53.33 | 10 / 105 |
| ERD-only | 61.90 | left_hand 63.9, right_hand 59.8 | left_hand 52.1, right_hand 47.9 | 53.33 | 7 / 105 |
| Random (mean of draws) | 60.66 | left_hand 61.0, right_hand 60.3 | left_hand 50.4, right_hand 49.6 | 53.33 | 48 / 525 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 64ch | 3 | 42: 62.22, 43: 62.48, 44: 60.63 | 61.78 | 1.00 |
| Ours | 3 | 42: 57.65, 43: 59.24, 44: 58.41 | 58.43 | 0.79 |
| CSP | 3 | 42: 57.14, 43: 55.94, 44: 57.08 | 56.72 | 0.68 |
| RLF | 3 | 42: 59.37, 43: 59.62, 44: 59.81 | 59.60 | 0.22 |
| MI | 3 | 42: 59.24, 43: 58.16, 44: 57.90 | 58.43 | 0.71 |
| Fixed sensorimotor | 3 | 42: 60.63, 43: 60.25, 44: 58.60 | 59.83 | 1.08 |
| ERD-only | 3 | 42: 60.25, 43: 59.87, 44: 60.00 | 60.04 | 0.19 |
| Random (mean of draws) | 3 | 42: 59.24, 43: 60.00, 44: 58.63 | 59.29 | 0.69 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full64/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | -0.018 | [-0.059, 0.023] | 0.204 | 105 |
| grad | class_specific_pearson | -0.036 | [-0.070, -0.001] | 0.174 | 105 |
| grad | lateralisation_pearson | -0.001 | [-0.037, 0.034] | 0.178 | 105 |
| grad | magnitude_spearman | -0.029 | [-0.072, 0.013] | 0.211 | 105 |
| grad | class_specific_spearman | -0.033 | [-0.068, 0.003] | 0.177 | 105 |
| grad | lateralisation_spearman | -0.008 | [-0.041, 0.026] | 0.169 | 105 |
| grad | legacy_signed_LR_pearson | 0.034 | [-0.005, 0.072] | 0.193 | 105 |
| grad_x_input | magnitude_pearson | -0.035 | [-0.080, 0.010] | 0.224 | 105 |
| grad_x_input | class_specific_pearson | -0.027 | [-0.064, 0.011] | 0.187 | 105 |
| grad_x_input | lateralisation_pearson | -0.096 | [-0.157, -0.034] | 0.294 | 105 |
| grad_x_input | magnitude_spearman | -0.050 | [-0.098, -0.002] | 0.234 | 105 |
| grad_x_input | class_specific_spearman | -0.024 | [-0.062, 0.015] | 0.190 | 105 |
| grad_x_input | lateralisation_spearman | -0.083 | [-0.142, -0.023] | 0.282 | 105 |
| grad_x_input | legacy_signed_LR_pearson | 0.030 | [-0.012, 0.072] | 0.210 | 105 |
| integrated_gradients | magnitude_pearson | -0.029 | [-0.073, 0.015] | 0.219 | 105 |
| integrated_gradients | class_specific_pearson | -0.027 | [-0.064, 0.010] | 0.185 | 105 |
| integrated_gradients | lateralisation_pearson | -0.138 | [-0.197, -0.078] | 0.284 | 105 |
| integrated_gradients | magnitude_spearman | -0.043 | [-0.089, 0.002] | 0.224 | 105 |
| integrated_gradients | class_specific_spearman | -0.025 | [-0.062, 0.013] | 0.186 | 105 |
| integrated_gradients | lateralisation_spearman | -0.113 | [-0.171, -0.053] | 0.280 | 105 |
| integrated_gradients | legacy_signed_LR_pearson | 0.025 | [-0.017, 0.066] | 0.205 | 105 |
| smoothgrad | magnitude_pearson | -0.018 | [-0.059, 0.023] | 0.204 | 105 |
| smoothgrad | class_specific_pearson | -0.035 | [-0.070, -0.001] | 0.174 | 105 |
| smoothgrad | lateralisation_pearson | -0.003 | [-0.039, 0.032] | 0.178 | 105 |
| smoothgrad | magnitude_spearman | -0.029 | [-0.072, 0.013] | 0.211 | 105 |
| smoothgrad | class_specific_spearman | -0.032 | [-0.067, 0.003] | 0.177 | 105 |
| smoothgrad | lateralisation_spearman | -0.013 | [-0.046, 0.020] | 0.167 | 105 |
| smoothgrad | legacy_signed_LR_pearson | 0.034 | [-0.005, 0.073] | 0.193 | 105 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | 0.052 | [0.007, 0.096] | 105 |
| grad | ERD_L_spearman | 0.044 | [-0.002, 0.090] | 105 |
| grad | ERD_R_pearson | 0.021 | [-0.018, 0.060] | 105 |
| grad | ERD_R_spearman | 0.020 | [-0.020, 0.060] | 105 |
| grad | ERD_LR_pearson | 0.034 | [-0.005, 0.072] | 105 |
| grad | ERD_LR_spearman | 0.036 | [-0.005, 0.076] | 105 |
| grad | ERD_comb_pearson | 0.042 | [-0.002, 0.085] | 105 |
| grad | ERD_comb_spearman | 0.039 | [-0.008, 0.086] | 105 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |
| grad_x_input | ERD_L_pearson | 0.032 | [-0.015, 0.079] | 105 |
| grad_x_input | ERD_L_spearman | 0.027 | [-0.022, 0.076] | 105 |
| grad_x_input | ERD_R_pearson | 0.007 | [-0.037, 0.050] | 105 |
| grad_x_input | ERD_R_spearman | 0.003 | [-0.042, 0.047] | 105 |
| grad_x_input | ERD_LR_pearson | 0.030 | [-0.012, 0.072] | 105 |
| grad_x_input | ERD_LR_spearman | 0.037 | [-0.010, 0.083] | 105 |
| grad_x_input | ERD_comb_pearson | 0.015 | [-0.032, 0.061] | 105 |
| grad_x_input | ERD_comb_spearman | 0.013 | [-0.037, 0.064] | 105 |
| grad_x_input | vs_grad_pearson | 0.978 | [0.974, 0.981] | 105 |
| grad_x_input | vs_grad_spearman | 0.972 | [0.967, 0.976] | 105 |
| integrated_gradients | ERD_L_pearson | 0.029 | [-0.016, 0.074] | 105 |
| integrated_gradients | ERD_L_spearman | 0.024 | [-0.023, 0.072] | 105 |
| integrated_gradients | ERD_R_pearson | 0.011 | [-0.032, 0.053] | 105 |
| integrated_gradients | ERD_R_spearman | 0.008 | [-0.034, 0.050] | 105 |
| integrated_gradients | ERD_LR_pearson | 0.025 | [-0.017, 0.066] | 105 |
| integrated_gradients | ERD_LR_spearman | 0.028 | [-0.016, 0.072] | 105 |
| integrated_gradients | ERD_comb_pearson | 0.017 | [-0.029, 0.062] | 105 |
| integrated_gradients | ERD_comb_spearman | 0.018 | [-0.031, 0.067] | 105 |
| integrated_gradients | vs_grad_pearson | 0.973 | [0.969, 0.976] | 105 |
| integrated_gradients | vs_grad_spearman | 0.965 | [0.960, 0.969] | 105 |
| smoothgrad | ERD_L_pearson | 0.051 | [0.007, 0.096] | 105 |
| smoothgrad | ERD_L_spearman | 0.043 | [-0.003, 0.089] | 105 |
| smoothgrad | ERD_R_pearson | 0.020 | [-0.019, 0.059] | 105 |
| smoothgrad | ERD_R_spearman | 0.020 | [-0.020, 0.060] | 105 |
| smoothgrad | ERD_LR_pearson | 0.034 | [-0.005, 0.073] | 105 |
| smoothgrad | ERD_LR_spearman | 0.035 | [-0.005, 0.075] | 105 |
| smoothgrad | ERD_comb_pearson | 0.041 | [-0.003, 0.085] | 105 |
| smoothgrad | ERD_comb_spearman | 0.038 | [-0.009, 0.085] | 105 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.026 | [-0.074, 0.022] | -0.034 | -0.030 | -0.044 | 105 |
| ERD_theta_R | -0.047 | [-0.093, -0.000] | -0.054 | -0.053 | -0.061 | 105 |
| ERD_theta_mag | -0.058 | [-0.100, -0.016] | -0.058 | -0.063 | -0.063 | 105 |
| ERD_mu_L | 0.032 | [-0.012, 0.077] | -0.028 | 0.030 | -0.038 | 105 |
| ERD_mu_R | 0.009 | [-0.034, 0.052] | -0.034 | 0.003 | -0.026 | 105 |
| ERD_mu_mag | -0.043 | [-0.082, -0.003] | -0.043 | -0.053 | -0.053 | 105 |
| ERD_beta_L | 0.024 | [-0.016, 0.065] | 0.005 | 0.028 | -0.005 | 105 |
| ERD_beta_R | 0.035 | [-0.004, 0.074] | 0.019 | 0.034 | 0.017 | 105 |
| ERD_beta_mag | 0.011 | [-0.030, 0.051] | 0.011 | 0.009 | 0.009 | 105 |
| ERS_beta_rebound_L | 0.016 | [-0.022, 0.053] | 0.021 | 0.010 | 0.014 | 105 |
| ERS_beta_rebound_R | -0.007 | [-0.047, 0.033] | 0.002 | -0.009 | -0.007 | 105 |
| ERS_beta_rebound_mag | 0.011 | [-0.030, 0.051] | 0.011 | 0.011 | 0.011 | 105 |
| ERD_lowgamma_L | -0.006 | [-0.045, 0.033] | 0.014 | -0.010 | 0.006 | 105 |
| ERD_lowgamma_R | 0.045 | [0.005, 0.084] | 0.049 | 0.039 | 0.032 | 105 |
| ERD_lowgamma_mag | 0.040 | [0.003, 0.076] | 0.040 | 0.021 | 0.021 | 105 |
| logpow_8_30 | -0.139 | [-0.186, -0.091] | -0.138 | -0.066 | -0.066 | 105 |
| r2_signed | 0.026 | [-0.016, 0.069] | 0.039 | 0.011 | 0.008 | 105 |
| r2_abs | 0.039 | [-0.004, 0.081] | 0.039 | 0.008 | 0.008 | 105 |
| TFR_ERD_dB_L | 0.022 | [-0.021, 0.065] | -0.001 | 0.023 | -0.011 | 105 |
| TFR_ERD_dB_R | 0.019 | [-0.022, 0.059] | -0.009 | 0.021 | -0.014 | 105 |
| TFR_ERD_dB_mag | -0.004 | [-0.047, 0.039] | -0.004 | -0.012 | -0.012 | 105 |
| ITC_4_30 | -0.074 | [-0.119, -0.029] | -0.074 | -0.056 | -0.056 | 105 |
| ERP_abs_0.2_0.8 | -0.158 | [-0.195, -0.121] | -0.158 | -0.153 | -0.153 | 105 |
| SlowPot_0.5_3 | -0.012 | [-0.055, 0.032] | -0.024 | -0.026 | -0.051 | 105 |
| ERD_beta_classspecific | — | — | 0.012 [-0.019, 0.043] | — | 0.006 | 105 |
| ERD_lowgamma_classspecific | — | — | 0.031 [0.003, 0.058] | — | 0.018 | 105 |
| ERD_mu_classspecific | — | — | -0.030 [-0.061, 0.001] | — | -0.030 | 105 |
| ERD_theta_classspecific | — | — | -0.045 [-0.080, -0.010] | — | -0.055 | 105 |
| ERS_beta_rebound_classspecific | — | — | 0.012 [-0.018, 0.042] | — | 0.005 | 105 |
| TFR_ERD_dB_classspecific | — | — | -0.005 [-0.037, 0.028] | — | -0.010 | 105 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +0.93 ± 5.26 | +0.71 ± 2.57 | +0.25 ± 3.42 | 0.5971 | 0.2152 |
| 4 | +1.50 ± 5.42 | +0.70 ± 3.35 | +0.11 ± 4.44 | 0.0860 | 0.0144 |
| 6 | +2.86 ± 5.95 | +1.20 ± 4.23 | +1.10 ± 4.88 | 0.0008 | 0.0080 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.011 | [-0.025, 0.003] | 105 |
| grad_x_input | 0.004 | [-0.012, 0.019] | 105 |
| integrated_gradients | 0.002 | [-0.014, 0.018] | 105 |
| smoothgrad | -0.011 | [-0.025, 0.003] | 105 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=zscore, S02=zscore, S03=zscore, S04=zscore, S05=zscore, S06=zscore, S07=zscore, S08=zscore, S09=zscore, S10=zscore, S11=zscore, S12=zscore, S13=zscore, S14=zscore, S15=zscore, S16=zscore, S17=zscore, S18=zscore, S19=zscore, S20=zscore, S21=zscore, S22=zscore, S23=zscore, S24=zscore, S25=zscore, S26=zscore, S27=zscore, S28=zscore, S29=zscore, S30=zscore, S31=zscore, S32=zscore, S33=zscore, S34=zscore, S35=zscore, S36=zscore, S37=zscore, S38=zscore, S39=zscore, S40=zscore, S41=zscore, S42=zscore, S43=zscore, S44=zscore, S45=zscore, S46=zscore, S47=zscore, S48=zscore, S49=zscore, S50=zscore, S51=zscore, S52=zscore, S53=zscore, S54=zscore, S55=zscore, S56=zscore, S57=zscore, S58=zscore, S59=zscore, S60=zscore, S61=zscore, S62=zscore, S63=zscore, S64=zscore, S65=zscore, S66=zscore, S67=zscore, S68=zscore, S69=zscore, S70=zscore, S71=zscore, S72=zscore, S73=zscore, S74=zscore, S75=zscore, S76=zscore, S77=zscore, S78=zscore, S79=zscore, S80=zscore, S81=zscore, S82=zscore, S83=zscore, S84=zscore, S85=zscore, S86=zscore, S87=zscore, S89=zscore, S90=zscore, S91=zscore, S93=zscore, S94=zscore, S95=zscore, S96=zscore, S97=zscore, S98=zscore, S99=zscore, S101=zscore, S102=zscore, S103=zscore, S105=zscore, S106=zscore, S107=zscore, S108=zscore, S109=zscore

| Profile | Mean validation accuracy (%) |
|---|---|
| zscore | 79.88 |
| ems | 79.52 |

```
model: "EEGConformer"
dataset: "PhysionetMI"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"zscore": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": null, "align": "zscore", "dropout": 0.5, "filter_time_length": 16, "pool_time_length": 48, "pool_time_stride": 10, "lr": 0.0005, "weight_decay": 0.0001, "betas": [0.5, 0.999], "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "ems": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": "ems", "align": null, "dropout": 0.5, "filter_time_length": 16, "pool_time_length": 48, "pool_time_stride": 10, "lr": 0.0005, "weight_decay": 0.0001, "betas": [0.5, 0.999], "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}}
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
| S01 | 43.33 | 20.00 | 26.67 | 23.33 | 30.00 | 60.00 | 46.67 | 40.67 |
| S02 | 63.33 | 46.67 | 43.33 | 40.00 | 40.00 | 70.00 | 53.33 | 55.33 |
| S03 | 36.67 | 36.67 | 33.33 | 40.00 | 30.00 | 46.67 | 23.33 | 31.33 |
| S04 | 50.00 | 36.67 | 36.67 | 33.33 | 43.33 | 50.00 | 20.00 | 37.33 |
| S05 | 36.67 | 46.67 | 23.33 | 23.33 | 30.00 | 20.00 | 20.00 | 34.00 |
| S06 | 26.67 | 60.00 | 20.00 | 56.67 | 13.33 | 6.67 | 40.00 | 27.33 |
| S07 | 46.67 | 36.67 | 43.33 | 50.00 | 33.33 | 70.00 | 53.33 | 41.33 |
| S08 | 16.67 | 16.67 | 10.00 | 23.33 | 26.67 | 26.67 | 16.67 | 16.00 |
| S09 | 6.67 | 6.67 | 13.33 | 16.67 | 3.33 | 20.00 | 26.67 | 18.67 |
| S10 | 36.67 | 30.00 | 23.33 | 50.00 | 53.33 | 26.67 | 26.67 | 38.00 |
| S11 | 23.33 | 23.33 | 20.00 | 26.67 | 13.33 | 13.33 | 36.67 | 28.67 |
| S12 | 30.00 | 20.00 | 36.67 | 23.33 | 36.67 | 20.00 | 26.67 | 28.67 |
| S13 | 40.00 | 36.67 | 20.00 | 46.67 | 46.67 | 33.33 | 26.67 | 40.00 |
| S14 | 40.00 | 40.00 | 33.33 | 30.00 | 36.67 | 33.33 | 33.33 | 31.33 |
| S15 | 50.00 | 53.33 | 46.67 | 53.33 | 36.67 | 43.33 | 50.00 | 48.67 |
| S16 | 30.00 | 40.00 | 20.00 | 30.00 | 26.67 | 26.67 | 26.67 | 23.33 |
| S17 | 43.33 | 26.67 | 33.33 | 26.67 | 26.67 | 13.33 | 20.00 | 30.67 |
| S18 | 36.67 | 43.33 | 36.67 | 40.00 | 33.33 | 26.67 | 26.67 | 40.67 |
| S19 | 40.00 | 33.33 | 33.33 | 30.00 | 40.00 | 33.33 | 33.33 | 32.67 |
| S20 | 43.33 | 20.00 | 36.67 | 20.00 | 26.67 | 33.33 | 36.67 | 28.00 |
| S21 | 40.00 | 33.33 | 36.67 | 46.67 | 36.67 | 26.67 | 30.00 | 33.33 |
| S22 | 40.00 | 26.67 | 33.33 | 20.00 | 33.33 | 33.33 | 26.67 | 34.00 |
| S23 | 26.67 | 26.67 | 23.33 | 30.00 | 23.33 | 26.67 | 33.33 | 32.67 |
| S24 | 20.00 | 26.67 | 26.67 | 26.67 | 20.00 | 13.33 | 13.33 | 17.33 |
| S25 | 30.00 | 26.67 | 13.33 | 16.67 | 23.33 | 40.00 | 36.67 | 30.00 |
| S26 | 40.00 | 20.00 | 30.00 | 26.67 | 26.67 | 33.33 | 20.00 | 35.33 |
| S27 | 33.33 | 26.67 | 33.33 | 36.67 | 40.00 | 36.67 | 26.67 | 34.67 |
| S28 | 50.00 | 66.67 | 16.67 | 73.33 | 63.33 | 20.00 | 23.33 | 34.67 |
| S29 | 66.67 | 83.33 | 16.67 | 56.67 | 63.33 | 80.00 | 36.67 | 61.33 |
| S30 | 26.67 | 33.33 | 16.67 | 26.67 | 33.33 | 36.67 | 30.00 | 29.33 |
| S31 | 36.67 | 46.67 | 26.67 | 26.67 | 26.67 | 43.33 | 40.00 | 36.00 |
| S32 | 16.67 | 13.33 | 20.00 | 10.00 | 10.00 | 23.33 | 16.67 | 12.67 |
| S33 | 70.00 | 66.67 | 23.33 | 73.33 | 66.67 | 56.67 | 50.00 | 54.00 |
| S34 | 56.67 | 43.33 | 53.33 | 50.00 | 40.00 | 33.33 | 43.33 | 36.67 |
| S35 | 80.00 | 56.67 | 56.67 | 56.67 | 60.00 | 66.67 | 83.33 | 66.00 |
| S36 | 26.67 | 20.00 | 33.33 | 23.33 | 36.67 | 16.67 | 30.00 | 18.00 |
| S37 | 20.00 | 10.00 | 16.67 | 23.33 | 26.67 | 23.33 | 20.00 | 17.33 |
| S38 | 30.00 | 20.00 | 36.67 | 20.00 | 3.33 | 33.33 | 36.67 | 22.00 |
| S39 | 20.00 | 30.00 | 30.00 | 20.00 | 23.33 | 30.00 | 40.00 | 29.33 |
| S40 | 40.00 | 40.00 | 46.67 | 36.67 | 46.67 | 46.67 | 43.33 | 42.00 |
| S41 | 36.67 | 40.00 | 36.67 | 53.33 | 50.00 | 23.33 | 26.67 | 38.00 |
| S42 | 63.33 | 66.67 | 30.00 | 56.67 | 60.00 | 76.67 | 50.00 | 73.33 |
| S43 | 36.67 | 33.33 | 30.00 | 26.67 | 26.67 | 56.67 | 30.00 | 27.33 |
| S44 | 46.67 | 43.33 | 43.33 | 50.00 | 36.67 | 43.33 | 56.67 | 48.00 |
| S45 | 40.00 | 53.33 | 26.67 | 63.33 | 56.67 | 60.00 | 36.67 | 44.67 |
| S46 | 33.33 | 26.67 | 20.00 | 30.00 | 23.33 | 26.67 | 23.33 | 24.67 |
| S47 | 26.67 | 23.33 | 33.33 | 30.00 | 23.33 | 20.00 | 30.00 | 23.33 |
| S48 | 40.00 | 26.67 | 30.00 | 30.00 | 30.00 | 43.33 | 43.33 | 46.00 |
| S49 | 30.00 | 23.33 | 13.33 | 30.00 | 30.00 | 40.00 | 26.67 | 34.67 |
| S50 | 33.33 | 26.67 | 33.33 | 26.67 | 20.00 | 33.33 | 43.33 | 40.67 |
| S51 | 46.67 | 33.33 | 60.00 | 63.33 | 56.67 | 26.67 | 30.00 | 42.00 |
| S52 | 50.00 | 26.67 | 16.67 | 43.33 | 23.33 | 46.67 | 63.33 | 44.67 |
| S53 | 53.33 | 50.00 | 50.00 | 46.67 | 53.33 | 26.67 | 46.67 | 33.33 |
| S54 | 60.00 | 23.33 | 36.67 | 13.33 | 46.67 | 43.33 | 53.33 | 49.33 |
| S55 | 50.00 | 36.67 | 40.00 | 46.67 | 33.33 | 40.00 | 50.00 | 47.33 |
| S56 | 56.67 | 40.00 | 50.00 | 46.67 | 50.00 | 46.67 | 63.33 | 53.33 |
| S57 | 40.00 | 40.00 | 36.67 | 26.67 | 26.67 | 30.00 | 30.00 | 35.33 |
| S58 | 33.33 | 36.67 | 43.33 | 36.67 | 13.33 | 26.67 | 50.00 | 40.00 |
| S59 | 40.00 | 36.67 | 23.33 | 33.33 | 23.33 | 30.00 | 33.33 | 34.00 |
| S60 | 40.00 | 33.33 | 36.67 | 53.33 | 33.33 | 46.67 | 40.00 | 37.33 |
| S61 | 30.00 | 26.67 | 33.33 | 36.67 | 26.67 | 20.00 | 30.00 | 28.00 |
| S62 | 63.33 | 26.67 | 46.67 | 30.00 | 43.33 | 56.67 | 46.67 | 47.33 |
| S63 | 50.00 | 40.00 | 36.67 | 46.67 | 46.67 | 30.00 | 43.33 | 43.33 |
| S64 | 23.33 | 33.33 | 23.33 | 30.00 | 23.33 | 23.33 | 30.00 | 25.33 |
| S65 | 33.33 | 26.67 | 56.67 | 50.00 | 40.00 | 16.67 | 46.67 | 30.00 |
| S66 | 33.33 | 23.33 | 40.00 | 26.67 | 33.33 | 23.33 | 43.33 | 35.33 |
| S67 | 23.33 | 16.67 | 33.33 | 26.67 | 20.00 | 10.00 | 30.00 | 20.67 |
| S68 | 26.67 | 20.00 | 40.00 | 30.00 | 20.00 | 20.00 | 26.67 | 26.00 |
| S69 | 20.00 | 33.33 | 40.00 | 26.67 | 40.00 | 26.67 | 36.67 | 26.67 |
| S70 | 36.67 | 30.00 | 30.00 | 33.33 | 33.33 | 26.67 | 40.00 | 34.00 |
| S71 | 23.33 | 20.00 | 26.67 | 20.00 | 20.00 | 33.33 | 30.00 | 40.67 |
| S72 | 73.33 | 60.00 | 50.00 | 60.00 | 63.33 | 63.33 | 60.00 | 62.00 |
| S73 | 26.67 | 36.67 | 6.67 | 30.00 | 30.00 | 33.33 | 13.33 | 34.67 |
| S74 | 16.67 | 16.67 | 26.67 | 26.67 | 10.00 | 26.67 | 13.33 | 22.00 |
| S75 | 36.67 | 46.67 | 26.67 | 43.33 | 36.67 | 13.33 | 30.00 | 28.67 |
| S76 | 23.33 | 30.00 | 16.67 | 20.00 | 26.67 | 30.00 | 20.00 | 18.00 |
| S77 | 26.67 | 26.67 | 26.67 | 23.33 | 26.67 | 33.33 | 26.67 | 33.33 |
| S78 | 40.00 | 40.00 | 26.67 | 36.67 | 46.67 | 16.67 | 40.00 | 33.33 |
| S79 | 30.00 | 36.67 | 46.67 | 43.33 | 40.00 | 20.00 | 20.00 | 31.33 |
| S80 | 33.33 | 40.00 | 43.33 | 33.33 | 43.33 | 33.33 | 30.00 | 34.00 |
| S81 | 43.33 | 20.00 | 26.67 | 23.33 | 26.67 | 23.33 | 23.33 | 26.00 |
| S82 | 33.33 | 40.00 | 36.67 | 30.00 | 36.67 | 26.67 | 30.00 | 28.67 |
| S83 | 30.00 | 33.33 | 23.33 | 26.67 | 16.67 | 43.33 | 46.67 | 31.33 |
| S84 | 33.33 | 36.67 | 33.33 | 30.00 | 33.33 | 46.67 | 30.00 | 32.67 |
| S85 | 70.00 | 50.00 | 43.33 | 56.67 | 60.00 | 66.67 | 53.33 | 60.00 |
| S86 | 43.33 | 50.00 | 40.00 | 46.67 | 43.33 | 26.67 | 43.33 | 48.00 |
| S87 | 23.33 | 13.33 | 26.67 | 10.00 | 13.33 | 20.00 | 36.67 | 23.33 |
| S89 | 56.67 | 40.00 | 43.33 | 43.33 | 53.33 | 30.00 | 33.33 | 34.00 |
| S90 | 26.67 | 40.00 | 20.00 | 53.33 | 26.67 | 30.00 | 16.67 | 34.00 |
| S91 | 33.33 | 30.00 | 36.67 | 30.00 | 30.00 | 26.67 | 36.67 | 37.33 |
| S93 | 50.00 | 33.33 | 20.00 | 40.00 | 43.33 | 40.00 | 23.33 | 38.67 |
| S94 | 43.33 | 16.67 | 33.33 | 13.33 | 33.33 | 56.67 | 30.00 | 37.33 |
| S95 | 33.33 | 33.33 | 26.67 | 36.67 | 33.33 | 53.33 | 40.00 | 32.67 |
| S96 | 43.33 | 33.33 | 40.00 | 23.33 | 20.00 | 33.33 | 33.33 | 36.67 |
| S97 | 46.67 | 30.00 | 33.33 | 26.67 | 33.33 | 33.33 | 16.67 | 21.33 |
| S98 | 23.33 | 26.67 | 30.00 | 30.00 | 33.33 | 23.33 | 33.33 | 33.33 |
| S99 | 6.67 | 16.67 | 13.33 | 13.33 | 13.33 | 10.00 | 16.67 | 14.00 |
| S101 | 26.67 | 30.00 | 50.00 | 26.67 | 26.67 | 16.67 | 43.33 | 38.00 |
| S102 | 30.00 | 36.67 | 36.67 | 26.67 | 33.33 | 30.00 | 20.00 | 26.00 |
| S103 | 26.67 | 33.33 | 23.33 | 30.00 | 33.33 | 20.00 | 30.00 | 38.00 |
| S105 | 33.33 | 30.00 | 40.00 | 30.00 | 53.33 | 33.33 | 40.00 | 31.33 |
| S106 | 33.33 | 30.00 | 33.33 | 20.00 | 26.67 | 40.00 | 33.33 | 32.00 |
| S107 | 33.33 | 50.00 | 26.67 | 43.33 | 46.67 | 23.33 | 36.67 | 34.67 |
| S108 | 40.00 | 30.00 | 23.33 | 30.00 | 23.33 | 23.33 | 30.00 | 37.33 |
| S109 | 40.00 | 26.67 | 33.33 | 26.67 | 20.00 | 33.33 | 40.00 | 39.33 |
| Mean | 37.52 | 33.81 | 31.81 | 34.57 | 33.62 | 33.78 | 34.51 | 34.93 |
| SD | 13.71 | 13.05 | 10.94 | 13.61 | 13.60 | 15.20 | 12.34 | 10.95 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 19 | T10 T9 T8 TP8 F8 Iz FT8 F7 FT7 C6 P8 O2 |
| CSP | 15 | Fp1 AF7 Fpz Fp2 AF3 AF8 AF4 F5 F7 F8 FC6 AFz |
| RLF | 82 | T8 T10 F7 AF7 FT8 C6 T7 FT7 F8 AF8 TP8 Iz |
| MI | 67 | T8 F8 FC5 AF8 F7 FT8 AF4 C6 T10 F6 C4 AF7 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 8 | PO8 O2 O1 PO4 Oz POz PO7 PO3 Fpz P4 P2 Fp2 |

### T3. Equivalence of each 12-channel montage with all 64 channels (paired TOST)

Difference = montage − 64ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 105 | -3.71 | 11.61 | [-5.59, -1.83] | 0.0231 | yes | -36.67 | 46 | 6,8,10 |
| CSP | 105 | -5.71 | 12.98 | [-7.82, -3.61] | 0.4110 | no | -50.00 | 53 | 8,10 |
| RLF | 105 | -2.95 | 12.34 | [-4.95, -0.95] | 0.0064 | yes | -46.67 | 42 | 5,6,8,10 |
| MI | 105 | -3.90 | 10.26 | [-5.57, -2.24] | 0.0194 | yes | -26.67 | 47 | 6,8,10 |
| Fixed sensorimotor | 105 | -3.75 | 11.81 | [-5.66, -1.83] | 0.0266 | yes | -30.00 | 57 | 6,8,10 |
| ERD-only | 105 | -3.02 | 11.46 | [-4.87, -1.16] | 0.0044 | yes | -30.00 | 45 | 5,6,8,10 |
| Random (mean of draws) | 105 | -2.60 | 7.84 | [-3.87, -1.33] | <0.0001 | yes | -25.33 | 32 | 4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 105 | +2.00 | [-1.00, +5.00] | 0.1885 | 1.0000 | 0.6131 | 1.0000 | 47 / 10 / 48 |
| RLF | 105 | -0.76 | [-2.52, +1.00] | 0.3921 | 1.0000 | 0.3886 | 1.0000 | 44 / 14 / 47 |
| MI | 105 | +0.19 | [-1.91, +2.29] | 0.8577 | 1.0000 | 0.9532 | 1.0000 | 49 / 13 / 43 |
| Fixed sensorimotor | 105 | +0.03 | [-2.92, +2.98] | 0.9830 | 1.0000 | 0.7919 | 1.0000 | 49 / 10 / 46 |
| ERD-only | 105 | -0.70 | [-3.45, +2.05] | 0.6159 | 1.0000 | 0.4599 | 1.0000 | 44 / 12 / 49 |
| Random (mean of draws) | 105 | -1.12 | [-3.12, +0.88] | 0.2707 | 1.0000 | 0.1303 | 0.7818 | 45 / 2 / 58 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 64ch | 37.52 | left_hand 32.0, right_hand 29.7, hands 37.6, feet 50.8 | left_hand 18.6, right_hand 20.2, hands 28.0, feet 33.1 | 26.67 | 3 / 105 |
| Ours | 33.81 | left_hand 25.9, right_hand 27.9, hands 38.2, feet 43.3 | left_hand 18.2, right_hand 21.0, hands 30.1, feet 30.8 | 26.67 | 3 / 105 |
| CSP | 31.81 | left_hand 27.0, right_hand 21.2, hands 37.6, feet 41.4 | left_hand 20.4, right_hand 17.6, hands 29.9, feet 32.2 | 26.67 | 4 / 105 |
| RLF | 34.57 | left_hand 26.9, right_hand 26.6, hands 39.5, feet 45.4 | left_hand 18.4, right_hand 20.8, hands 30.2, feet 30.6 | 26.67 | 3 / 105 |
| MI | 33.62 | left_hand 30.4, right_hand 26.0, hands 37.1, feet 41.0 | left_hand 20.2, right_hand 21.5, hands 29.2, feet 29.1 | 26.67 | 4 / 105 |
| Fixed sensorimotor | 33.78 | left_hand 28.9, right_hand 29.0, hands 34.8, feet 42.4 | left_hand 21.2, right_hand 22.3, hands 28.4, feet 28.2 | 26.67 | 1 / 105 |
| ERD-only | 34.51 | left_hand 28.1, right_hand 25.3, hands 37.2, feet 47.4 | left_hand 20.7, right_hand 20.3, hands 28.5, feet 30.4 | 26.67 | 1 / 105 |
| Random (mean of draws) | 34.93 | left_hand 29.3, right_hand 26.7, hands 36.6, feet 47.1 | left_hand 20.5, right_hand 20.0, hands 28.4, feet 31.0 | 26.67 | 3 / 525 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 64ch | 3 | 42: 36.00, 43: 35.71, 44: 34.54 | 35.42 | 0.77 |
| Ours | 3 | 42: 32.51, 43: 32.16, 44: 31.90 | 32.19 | 0.30 |
| CSP | 3 | 42: 30.13, 43: 30.86, 44: 31.40 | 30.79 | 0.64 |
| RLF | 3 | 42: 34.00, 43: 33.52, 44: 32.63 | 33.39 | 0.69 |
| MI | 3 | 42: 32.19, 43: 32.16, 44: 31.90 | 32.08 | 0.16 |
| Fixed sensorimotor | 3 | 42: 32.57, 43: 31.97, 44: 32.38 | 32.31 | 0.31 |
| ERD-only | 3 | 42: 33.40, 43: 33.05, 44: 32.73 | 33.06 | 0.33 |
| Random (mean of draws) | 3 | 42: 33.25, 43: 33.29, 44: 32.98 | 33.17 | 0.17 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full64/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | -0.069 | [-0.109, -0.029] | 0.199 | 105 |
| grad | class_specific_pearson | -0.054 | [-0.087, -0.021] | 0.166 | 105 |
| grad | lateralisation_pearson | 0.020 | [-0.016, 0.056] | 0.181 | 105 |
| grad | magnitude_spearman | -0.087 | [-0.130, -0.044] | 0.214 | 105 |
| grad | class_specific_spearman | -0.047 | [-0.081, -0.012] | 0.174 | 105 |
| grad | lateralisation_spearman | 0.020 | [-0.017, 0.056] | 0.181 | 105 |
| grad | legacy_signed_LR_pearson | 0.027 | [-0.021, 0.074] | 0.230 | 105 |
| grad_x_input | magnitude_pearson | -0.085 | [-0.128, -0.041] | 0.215 | 105 |
| grad_x_input | class_specific_pearson | -0.056 | [-0.093, -0.020] | 0.181 | 105 |
| grad_x_input | lateralisation_pearson | -0.044 | [-0.086, -0.002] | 0.211 | 105 |
| grad_x_input | magnitude_spearman | -0.102 | [-0.150, -0.053] | 0.235 | 105 |
| grad_x_input | class_specific_spearman | -0.053 | [-0.092, -0.014] | 0.192 | 105 |
| grad_x_input | lateralisation_spearman | -0.038 | [-0.080, 0.004] | 0.210 | 105 |
| grad_x_input | legacy_signed_LR_pearson | 0.021 | [-0.028, 0.071] | 0.243 | 105 |
| integrated_gradients | magnitude_pearson | -0.075 | [-0.115, -0.034] | 0.202 | 105 |
| integrated_gradients | class_specific_pearson | -0.042 | [-0.073, -0.010] | 0.158 | 105 |
| integrated_gradients | lateralisation_pearson | -0.032 | [-0.073, 0.010] | 0.204 | 105 |
| integrated_gradients | magnitude_spearman | -0.086 | [-0.130, -0.042] | 0.217 | 105 |
| integrated_gradients | class_specific_spearman | -0.040 | [-0.073, -0.006] | 0.169 | 105 |
| integrated_gradients | lateralisation_spearman | -0.037 | [-0.077, 0.002] | 0.194 | 105 |
| integrated_gradients | legacy_signed_LR_pearson | 0.018 | [-0.026, 0.063] | 0.221 | 105 |
| smoothgrad | magnitude_pearson | -0.069 | [-0.109, -0.029] | 0.199 | 105 |
| smoothgrad | class_specific_pearson | -0.054 | [-0.087, -0.020] | 0.168 | 105 |
| smoothgrad | lateralisation_pearson | 0.022 | [-0.014, 0.058] | 0.179 | 105 |
| smoothgrad | magnitude_spearman | -0.086 | [-0.128, -0.042] | 0.214 | 105 |
| smoothgrad | class_specific_spearman | -0.046 | [-0.081, -0.012] | 0.174 | 105 |
| smoothgrad | lateralisation_spearman | 0.020 | [-0.016, 0.056] | 0.180 | 105 |
| smoothgrad | legacy_signed_LR_pearson | 0.027 | [-0.021, 0.074] | 0.230 | 105 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | 0.066 | [0.022, 0.110] | 105 |
| grad | ERD_L_spearman | 0.048 | [0.001, 0.094] | 105 |
| grad | ERD_R_pearson | 0.044 | [0.000, 0.087] | 105 |
| grad | ERD_R_spearman | 0.049 | [0.004, 0.094] | 105 |
| grad | ERD_LR_pearson | 0.027 | [-0.021, 0.074] | 105 |
| grad | ERD_LR_spearman | 0.024 | [-0.024, 0.072] | 105 |
| grad | ERD_comb_pearson | 0.062 | [0.020, 0.104] | 105 |
| grad | ERD_comb_spearman | 0.058 | [0.012, 0.103] | 105 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 105 |
| grad_x_input | ERD_L_pearson | 0.053 | [0.006, 0.100] | 105 |
| grad_x_input | ERD_L_spearman | 0.043 | [-0.008, 0.095] | 105 |
| grad_x_input | ERD_R_pearson | 0.038 | [-0.011, 0.087] | 105 |
| grad_x_input | ERD_R_spearman | 0.048 | [-0.004, 0.099] | 105 |
| grad_x_input | ERD_LR_pearson | 0.021 | [-0.028, 0.071] | 105 |
| grad_x_input | ERD_LR_spearman | 0.024 | [-0.029, 0.076] | 105 |
| grad_x_input | ERD_comb_pearson | 0.046 | [0.001, 0.092] | 105 |
| grad_x_input | ERD_comb_spearman | 0.053 | [0.002, 0.103] | 105 |
| grad_x_input | vs_grad_pearson | 0.981 | [0.977, 0.984] | 105 |
| grad_x_input | vs_grad_spearman | 0.976 | [0.972, 0.980] | 105 |
| integrated_gradients | ERD_L_pearson | 0.048 | [0.004, 0.092] | 105 |
| integrated_gradients | ERD_L_spearman | 0.033 | [-0.015, 0.082] | 105 |
| integrated_gradients | ERD_R_pearson | 0.031 | [-0.015, 0.077] | 105 |
| integrated_gradients | ERD_R_spearman | 0.041 | [-0.006, 0.087] | 105 |
| integrated_gradients | ERD_LR_pearson | 0.018 | [-0.026, 0.063] | 105 |
| integrated_gradients | ERD_LR_spearman | 0.014 | [-0.032, 0.061] | 105 |
| integrated_gradients | ERD_comb_pearson | 0.040 | [-0.003, 0.083] | 105 |
| integrated_gradients | ERD_comb_spearman | 0.043 | [-0.004, 0.089] | 105 |
| integrated_gradients | vs_grad_pearson | 0.918 | [0.906, 0.929] | 105 |
| integrated_gradients | vs_grad_spearman | 0.904 | [0.890, 0.915] | 105 |
| smoothgrad | ERD_L_pearson | 0.067 | [0.023, 0.111] | 105 |
| smoothgrad | ERD_L_spearman | 0.049 | [0.002, 0.095] | 105 |
| smoothgrad | ERD_R_pearson | 0.044 | [0.000, 0.087] | 105 |
| smoothgrad | ERD_R_spearman | 0.050 | [0.005, 0.095] | 105 |
| smoothgrad | ERD_LR_pearson | 0.027 | [-0.021, 0.074] | 105 |
| smoothgrad | ERD_LR_spearman | 0.023 | [-0.025, 0.070] | 105 |
| smoothgrad | ERD_comb_pearson | 0.063 | [0.020, 0.105] | 105 |
| smoothgrad | ERD_comb_spearman | 0.058 | [0.012, 0.104] | 105 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 105 |
| smoothgrad | vs_grad_spearman | 0.999 | [0.999, 0.999] | 105 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | 0.019 | [-0.029, 0.066] | -0.022 | -0.003 | -0.035 | 105 |
| ERD_theta_R | -0.033 | [-0.075, 0.009] | -0.047 | -0.032 | -0.049 | 105 |
| ERD_theta_mag | -0.050 | [-0.091, -0.008] | -0.050 | -0.047 | -0.047 | 105 |
| ERD_mu_L | 0.034 | [-0.014, 0.081] | -0.044 | 0.014 | -0.056 | 105 |
| ERD_mu_R | 0.033 | [-0.013, 0.079] | -0.075 | 0.031 | -0.079 | 105 |
| ERD_mu_mag | -0.082 | [-0.125, -0.039] | -0.082 | -0.094 | -0.094 | 105 |
| ERD_beta_L | 0.066 | [0.019, 0.114] | -0.021 | 0.061 | -0.032 | 105 |
| ERD_beta_R | 0.040 | [-0.006, 0.085] | -0.023 | 0.040 | -0.032 | 105 |
| ERD_beta_mag | -0.032 | [-0.071, 0.008] | -0.032 | -0.042 | -0.042 | 105 |
| ERS_beta_rebound_L | 0.032 | [-0.015, 0.079] | -0.009 | 0.032 | -0.022 | 105 |
| ERS_beta_rebound_R | 0.015 | [-0.030, 0.060] | -0.016 | 0.017 | -0.038 | 105 |
| ERS_beta_rebound_mag | -0.013 | [-0.058, 0.033] | -0.013 | -0.033 | -0.033 | 105 |
| ERD_lowgamma_L | 0.010 | [-0.031, 0.051] | 0.033 | 0.004 | 0.017 | 105 |
| ERD_lowgamma_R | 0.020 | [-0.016, 0.057] | 0.019 | 0.015 | 0.001 | 105 |
| ERD_lowgamma_mag | 0.031 | [-0.002, 0.063] | 0.031 | 0.009 | 0.009 | 105 |
| logpow_8_30 | -0.157 | [-0.208, -0.106] | -0.158 | -0.082 | -0.082 | 105 |
| r2_signed | 0.005 | [-0.038, 0.048] | -0.029 | -0.003 | -0.030 | 105 |
| r2_abs | -0.029 | [-0.066, 0.009] | -0.029 | -0.030 | -0.030 | 105 |
| TFR_ERD_dB_L | 0.052 | [0.006, 0.097] | -0.023 | 0.044 | -0.034 | 105 |
| TFR_ERD_dB_R | 0.032 | [-0.014, 0.079] | -0.057 | 0.041 | -0.066 | 105 |
| TFR_ERD_dB_mag | -0.051 | [-0.091, -0.011] | -0.051 | -0.062 | -0.062 | 105 |
| ITC_4_30 | -0.116 | [-0.160, -0.071] | -0.116 | -0.088 | -0.088 | 105 |
| ERP_abs_0.2_0.8 | -0.176 | [-0.213, -0.139] | -0.176 | -0.185 | -0.185 | 105 |
| SlowPot_0.5_3 | -0.020 | [-0.067, 0.028] | -0.021 | -0.033 | -0.049 | 105 |
| ERD_beta_classspecific | — | — | -0.024 [-0.054, 0.007] | — | -0.032 | 105 |
| ERD_lowgamma_classspecific | — | — | 0.019 [-0.008, 0.046] | — | 0.002 | 105 |
| ERD_mu_classspecific | — | — | -0.056 [-0.087, -0.025] | — | -0.067 | 105 |
| ERD_theta_classspecific | — | — | -0.036 [-0.068, -0.003] | — | -0.041 | 105 |
| ERS_beta_rebound_classspecific | — | — | -0.014 [-0.046, 0.019] | — | -0.030 | 105 |
| TFR_ERD_dB_classspecific | — | — | -0.045 [-0.075, -0.014] | — | -0.051 | 105 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +0.76 ± 4.19 | +0.38 ± 1.70 | -0.05 ± 2.86 | 0.2854 | 0.0827 |
| 4 | +1.68 ± 4.97 | +0.73 ± 2.60 | +0.01 ± 3.56 | 0.0284 | 0.0022 |
| 6 | +2.21 ± 5.40 | +1.28 ± 3.33 | +0.68 ± 3.50 | 0.0339 | 0.0059 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.012 | [-0.005, 0.028] | 105 |
| grad_x_input | 0.032 | [0.013, 0.050] | 105 |
| integrated_gradients | 0.026 | [0.008, 0.043] | 105 |
| smoothgrad | 0.011 | [-0.005, 0.027] | 105 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=zscore, S02=zscore, S03=zscore, S04=zscore, S05=zscore, S06=zscore, S07=zscore, S08=zscore, S09=zscore, S10=zscore, S11=zscore, S12=zscore, S13=zscore, S14=zscore, S15=zscore, S16=zscore, S17=zscore, S18=zscore, S19=zscore, S20=zscore, S21=zscore, S22=zscore, S23=zscore, S24=zscore, S25=zscore, S26=zscore, S27=zscore, S28=zscore, S29=zscore, S30=zscore, S31=zscore, S32=zscore, S33=zscore, S34=zscore, S35=zscore, S36=zscore, S37=zscore, S38=zscore, S39=zscore, S40=zscore, S41=zscore, S42=zscore, S43=zscore, S44=zscore, S45=zscore, S46=zscore, S47=zscore, S48=zscore, S49=zscore, S50=zscore, S51=zscore, S52=zscore, S53=zscore, S54=zscore, S55=zscore, S56=zscore, S57=zscore, S58=zscore, S59=zscore, S60=zscore, S61=zscore, S62=zscore, S63=zscore, S64=zscore, S65=zscore, S66=zscore, S67=zscore, S68=zscore, S69=zscore, S70=zscore, S71=zscore, S72=zscore, S73=zscore, S74=zscore, S75=zscore, S76=zscore, S77=zscore, S78=zscore, S79=zscore, S80=zscore, S81=zscore, S82=zscore, S83=zscore, S84=zscore, S85=zscore, S86=zscore, S87=zscore, S89=zscore, S90=zscore, S91=zscore, S93=zscore, S94=zscore, S95=zscore, S96=zscore, S97=zscore, S98=zscore, S99=zscore, S101=zscore, S102=zscore, S103=zscore, S105=zscore, S106=zscore, S107=zscore, S108=zscore, S109=zscore

| Profile | Mean validation accuracy (%) |
|---|---|
| zscore | 59.05 |
| ems | 57.74 |

```
model: "EEGConformer"
dataset: "PhysionetMI"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"zscore": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": null, "align": "zscore", "dropout": 0.5, "filter_time_length": 16, "pool_time_length": 48, "pool_time_stride": 10, "lr": 0.0005, "weight_decay": 0.0001, "betas": [0.5, 0.999], "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "ems": {"sfreq": 160, "band_hz": [4.0, 40.0], "window_s": [0.0, 3.0], "car": false, "norm": "ems", "align": null, "dropout": 0.5, "filter_time_length": 16, "pool_time_length": 48, "pool_time_stride": 10, "lr": 0.0005, "weight_decay": 0.0001, "betas": [0.5, 0.999], "scheduler": "cosine", "max_epochs": 300, "patience": 80, "batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}}
checkpoint_rule: "highest validation accuracy (loss breaks ties), not before epoch 20 ('original': lowest validation loss); refit keeps the final weights"
folds: "run-wise chronological split per subject: test = the last imagery run of each type (run 12 left/right, run 14 hands/feet); training = runs 4, 8 (and 6, 10); validation = the chronologically last 25% of the training trials of EACH class; refit on all training runs"
seeds: [42, 43, 44]
```

Figures (530 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `4class\figures`.
