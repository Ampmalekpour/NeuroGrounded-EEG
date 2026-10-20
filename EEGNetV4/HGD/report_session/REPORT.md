# Results report — EEGNetV4\HGD

Generated 2026-10-20 20:46 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Protocol: train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects (n = 14). Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 | S13 | S14 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 50.00 | 88.75 | 97.50 | 100.00 | 72.50 | 87.50 | 62.50 | 83.75 | 80.00 | 98.75 | 66.25 | 98.75 | 81.25 | 95.00 | 83.04 | 15.46 |
| Ours | 58.75 | 87.50 | 97.50 | 98.75 | 73.75 | 81.25 | 67.50 | 65.00 | 72.50 | 98.75 | 62.50 | 95.00 | 82.50 | 93.75 | 81.07 | 14.43 |
| CSP | 55.00 | 88.75 | 97.50 | 97.50 | 82.50 | 88.75 | 62.50 | 78.75 | 75.00 | 98.75 | 70.00 | 100.00 | 86.25 | 93.75 | 83.93 | 14.12 |
| RLF | 55.00 | 87.50 | 98.75 | 96.25 | 81.25 | 86.25 | 72.50 | 75.00 | 78.75 | 100.00 | 72.50 | 98.75 | 81.25 | 92.50 | 84.02 | 12.89 |
| MI | 57.50 | 87.50 | 96.25 | 98.75 | 78.75 | 87.50 | 58.75 | 53.75 | 67.50 | 100.00 | 63.75 | 100.00 | 81.25 | 93.75 | 80.36 | 17.06 |
| Fixed sensorimotor | 51.25 | 85.00 | 93.75 | 96.25 | 78.75 | 92.50 | 63.75 | 62.50 | 75.00 | 98.75 | 65.00 | 100.00 | 72.50 | 95.00 | 80.71 | 15.95 |
| ERD-only | 52.50 | 92.50 | 98.75 | 97.50 | 85.00 | 91.25 | 60.00 | 77.50 | 81.25 | 98.75 | 65.00 | 98.75 | 80.00 | 95.00 | 83.84 | 15.38 |
| Random (mean of draws) | 58.00 | 87.25 | 97.25 | 97.25 | 73.25 | 85.50 | 62.75 | 56.75 | 65.25 | 99.50 | 62.25 | 98.75 | 76.75 | 93.75 | 79.59 | 16.43 |

### T0. Data used per subject

Trials after the class selection of this experiment (training trials include the validation part, which is also used to fit the final model; the test trials are never used for any choice).

| Subject | Training trials (incl. validation) | Validation trials | Test trials | Test trials per class | Majority-class test accuracy (%) | Chance (%) |
|---|---|---|---|---|---|---|
| S01 | 160 | 40 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S02 | 406 | 102 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S03 | 440 | 110 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S04 | 448 | 112 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S05 | 360 | 90 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S06 | 440 | 110 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S07 | 440 | 110 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S08 | 328 | 82 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S09 | 440 | 110 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S10 | 440 | 110 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S11 | 440 | 110 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S12 | 440 | 110 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S13 | 400 | 100 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |
| S14 | 440 | 110 | 80 | left_hand 40, right_hand 40 | 50.00 | 50.00 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 14 | C2 C1 CP3 FC4 Pz C4 FCz CP1 CP4 CP2 CPz C3 |
| CSP | 14 | C4 C2 CP2 C1 CPz Pz CP1 C3 CP3 FC1 CP6 FC4 |
| RLF | 14 | CP4 C2 CP2 CP1 CP3 CP6 C4 FC1 CP5 C6 FC2 C3 |
| MI | 14 | CP4 C4 C6 FC6 CP3 CP1 CP6 Pz FC1 FCz C1 CPz |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 13 | C4 C2 C1 C3 CP3 CP1 CP2 Pz CP6 CP4 CP5 CPz |

### T3. Equivalence of each 12-channel montage with all 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 14 | -1.96 | 6.39 | [-4.99, +1.06] | 0.0172 | yes | -18.75 | 3 | 5,6,8,10 |
| CSP | 14 | +0.89 | 4.06 | [-1.03, +2.81] | 0.0002 | yes | -5.00 | 0 | 3,4,5,6,8,10 |
| RLF | 14 | +0.98 | 5.05 | [-1.41, +3.37] | 0.0013 | yes | -8.75 | 1 | 4,5,6,8,10 |
| MI | 14 | -2.68 | 9.12 | [-6.99, +1.64] | 0.0980 | no | -30.00 | 2 | 8,10 |
| Fixed sensorimotor | 14 | -2.32 | 6.74 | [-5.51, +0.87] | 0.0309 | yes | -21.25 | 2 | 6,8,10 |
| ERD-only | 14 | +0.80 | 4.29 | [-1.23, +2.83] | 0.0003 | yes | -6.25 | 1 | 3,4,5,6,8,10 |
| Random (mean of draws) | 14 | -3.45 | 8.31 | [-7.38, +0.49] | 0.1354 | no | -27.00 | 2 | 8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 14 | -2.86 | [-5.86, +0.14] | 0.0604 | 0.3020 | 0.0681 | 0.3404 | 3 / 3 / 8 |
| RLF | 14 | -2.95 | [-5.55, -0.34] | 0.0295 | 0.1771 | 0.0328 | 0.1967 | 4 / 1 / 9 |
| MI | 14 | +0.71 | [-2.14, +3.57] | 0.5980 | 1.0000 | 0.6875 | 1.0000 | 6 / 3 / 5 |
| Fixed sensorimotor | 14 | +0.36 | [-2.81, +3.53] | 0.8115 | 1.0000 | 0.7517 | 1.0000 | 7 / 1 / 6 |
| ERD-only | 14 | -2.77 | [-6.36, +0.83] | 0.1201 | 0.4802 | 0.1241 | 0.4004 | 4 / 1 / 9 |
| Random (mean of draws) | 14 | +1.48 | [-0.67, +3.64] | 0.1615 | 0.4846 | 0.1001 | 0.4004 | 10 / 1 / 3 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`, per subject in `T5_confusion_per_subject.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 83.04 | left_hand 87.0, right_hand 79.1 | left_hand 53.9, right_hand 46.1 | 50.00 | 1 / 14 |
| Ours | 81.07 | left_hand 82.0, right_hand 80.2 | left_hand 50.9, right_hand 49.1 | 50.00 | 0 / 14 |
| CSP | 83.93 | left_hand 83.4, right_hand 84.5 | left_hand 49.5, right_hand 50.5 | 50.00 | 0 / 14 |
| RLF | 84.02 | left_hand 86.8, right_hand 81.2 | left_hand 52.8, right_hand 47.2 | 50.00 | 0 / 14 |
| MI | 80.36 | left_hand 81.6, right_hand 79.1 | left_hand 51.2, right_hand 48.8 | 50.00 | 0 / 14 |
| Fixed sensorimotor | 80.71 | left_hand 83.4, right_hand 78.0 | left_hand 52.7, right_hand 47.3 | 50.00 | 0 / 14 |
| ERD-only | 83.84 | left_hand 86.4, right_hand 81.2 | left_hand 52.6, right_hand 47.4 | 50.00 | 0 / 14 |
| Random (mean of draws) | 79.59 | left_hand 78.4, right_hand 80.8 | left_hand 48.8, right_hand 51.2 | 50.00 | 1 / 70 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 79.64, 43: 81.96, 44: 78.30 | 79.97 | 1.85 |
| Ours | 3 | 42: 78.48, 43: 78.66, 44: 78.21 | 78.45 | 0.22 |
| CSP | 3 | 42: 80.09, 43: 80.89, 44: 80.89 | 80.62 | 0.46 |
| RLF | 3 | 42: 79.91, 43: 82.95, 44: 81.43 | 81.43 | 1.52 |
| MI | 3 | 42: 75.89, 43: 78.21, 44: 77.77 | 77.29 | 1.23 |
| Fixed sensorimotor | 3 | 42: 78.48, 43: 79.82, 44: 78.13 | 78.81 | 0.89 |
| ERD-only | 3 | 42: 82.14, 43: 80.27, 44: 82.50 | 81.64 | 1.20 |
| Random (mean of draws) | 3 | 42: 77.64, 43: 77.75, 44: 76.43 | 77.27 | 0.73 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.293 | [0.093, 0.469] | 0.327 | 14 |
| grad | class_specific_pearson | 0.277 | [0.100, 0.437] | 0.288 | 14 |
| grad | lateralisation_pearson | -0.192 | [-0.307, -0.072] | 0.197 | 14 |
| grad | magnitude_spearman | 0.277 | [0.077, 0.455] | 0.316 | 14 |
| grad | class_specific_spearman | 0.210 | [0.031, 0.376] | 0.286 | 14 |
| grad | lateralisation_spearman | -0.203 | [-0.330, -0.068] | 0.211 | 14 |
| grad | legacy_signed_LR_pearson | 0.183 | [0.031, 0.327] | 0.250 | 14 |
| grad_x_input | magnitude_pearson | 0.136 | [-0.082, 0.342] | 0.348 | 14 |
| grad_x_input | class_specific_pearson | 0.076 | [-0.122, 0.268] | 0.309 | 14 |
| grad_x_input | lateralisation_pearson | -0.368 | [-0.525, -0.186] | 0.271 | 14 |
| grad_x_input | magnitude_spearman | 0.131 | [-0.095, 0.344] | 0.350 | 14 |
| grad_x_input | class_specific_spearman | 0.021 | [-0.175, 0.214] | 0.301 | 14 |
| grad_x_input | lateralisation_spearman | -0.396 | [-0.571, -0.186] | 0.299 | 14 |
| grad_x_input | legacy_signed_LR_pearson | 0.129 | [-0.041, 0.292] | 0.278 | 14 |
| integrated_gradients | magnitude_pearson | 0.129 | [-0.095, 0.341] | 0.357 | 14 |
| integrated_gradients | class_specific_pearson | 0.050 | [-0.150, 0.245] | 0.313 | 14 |
| integrated_gradients | lateralisation_pearson | -0.335 | [-0.499, -0.149] | 0.285 | 14 |
| integrated_gradients | magnitude_spearman | 0.120 | [-0.110, 0.339] | 0.359 | 14 |
| integrated_gradients | class_specific_spearman | -0.017 | [-0.210, 0.176] | 0.299 | 14 |
| integrated_gradients | lateralisation_spearman | -0.375 | [-0.551, -0.168] | 0.301 | 14 |
| integrated_gradients | legacy_signed_LR_pearson | 0.116 | [-0.049, 0.274] | 0.269 | 14 |
| smoothgrad | magnitude_pearson | 0.293 | [0.093, 0.470] | 0.327 | 14 |
| smoothgrad | class_specific_pearson | 0.277 | [0.100, 0.437] | 0.289 | 14 |
| smoothgrad | lateralisation_pearson | -0.192 | [-0.307, -0.071] | 0.198 | 14 |
| smoothgrad | magnitude_spearman | 0.275 | [0.079, 0.450] | 0.313 | 14 |
| smoothgrad | class_specific_spearman | 0.210 | [0.031, 0.377] | 0.286 | 14 |
| smoothgrad | lateralisation_spearman | -0.202 | [-0.329, -0.068] | 0.209 | 14 |
| smoothgrad | legacy_signed_LR_pearson | 0.183 | [0.031, 0.327] | 0.250 | 14 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.260 | [-0.419, -0.085] | 14 |
| grad | ERD_L_spearman | -0.199 | [-0.374, -0.011] | 14 |
| grad | ERD_R_pearson | -0.313 | [-0.484, -0.119] | 14 |
| grad | ERD_R_spearman | -0.211 | [-0.402, -0.002] | 14 |
| grad | ERD_LR_pearson | 0.183 | [0.031, 0.327] | 14 |
| grad | ERD_LR_spearman | 0.154 | [0.037, 0.266] | 14 |
| grad | ERD_comb_pearson | -0.296 | [-0.465, -0.106] | 14 |
| grad | ERD_comb_spearman | -0.217 | [-0.406, -0.011] | 14 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |
| grad_x_input | ERD_L_pearson | -0.058 | [-0.249, 0.138] | 14 |
| grad_x_input | ERD_L_spearman | -0.015 | [-0.211, 0.183] | 14 |
| grad_x_input | ERD_R_pearson | -0.114 | [-0.330, 0.113] | 14 |
| grad_x_input | ERD_R_spearman | -0.029 | [-0.252, 0.198] | 14 |
| grad_x_input | ERD_LR_pearson | 0.129 | [-0.041, 0.292] | 14 |
| grad_x_input | ERD_LR_spearman | 0.119 | [-0.001, 0.236] | 14 |
| grad_x_input | ERD_comb_pearson | -0.092 | [-0.309, 0.133] | 14 |
| grad_x_input | ERD_comb_spearman | -0.039 | [-0.255, 0.181] | 14 |
| grad_x_input | vs_grad_pearson | 0.979 | [0.960, 0.989] | 14 |
| grad_x_input | vs_grad_spearman | 0.969 | [0.939, 0.984] | 14 |
| integrated_gradients | ERD_L_pearson | -0.031 | [-0.229, 0.170] | 14 |
| integrated_gradients | ERD_L_spearman | 0.020 | [-0.177, 0.216] | 14 |
| integrated_gradients | ERD_R_pearson | -0.092 | [-0.315, 0.140] | 14 |
| integrated_gradients | ERD_R_spearman | 0.001 | [-0.224, 0.227] | 14 |
| integrated_gradients | ERD_LR_pearson | 0.116 | [-0.049, 0.274] | 14 |
| integrated_gradients | ERD_LR_spearman | 0.113 | [-0.003, 0.226] | 14 |
| integrated_gradients | ERD_comb_pearson | -0.070 | [-0.295, 0.162] | 14 |
| integrated_gradients | ERD_comb_spearman | -0.006 | [-0.221, 0.211] | 14 |
| integrated_gradients | vs_grad_pearson | 0.962 | [0.935, 0.978] | 14 |
| integrated_gradients | vs_grad_spearman | 0.945 | [0.909, 0.967] | 14 |
| smoothgrad | ERD_L_pearson | -0.260 | [-0.419, -0.085] | 14 |
| smoothgrad | ERD_L_spearman | -0.198 | [-0.371, -0.011] | 14 |
| smoothgrad | ERD_R_pearson | -0.313 | [-0.484, -0.119] | 14 |
| smoothgrad | ERD_R_spearman | -0.209 | [-0.398, -0.004] | 14 |
| smoothgrad | ERD_LR_pearson | 0.183 | [0.031, 0.327] | 14 |
| smoothgrad | ERD_LR_spearman | 0.155 | [0.038, 0.268] | 14 |
| smoothgrad | ERD_comb_pearson | -0.296 | [-0.466, -0.106] | 14 |
| smoothgrad | ERD_comb_spearman | -0.215 | [-0.401, -0.012] | 14 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | 0.008 | [-0.187, 0.203] | 0.172 | 0.026 | 0.126 | 14 |
| ERD_theta_R | -0.011 | [-0.205, 0.184] | 0.063 | 0.029 | 0.045 | 14 |
| ERD_theta_mag | 0.109 | [-0.081, 0.292] | 0.109 | 0.096 | 0.096 | 14 |
| ERD_mu_L | -0.096 | [-0.273, 0.087] | 0.111 | -0.032 | 0.102 | 14 |
| ERD_mu_R | -0.169 | [-0.353, 0.028] | 0.142 | -0.053 | 0.089 | 14 |
| ERD_mu_mag | 0.122 | [-0.048, 0.285] | 0.122 | 0.080 | 0.080 | 14 |
| ERD_beta_L | -0.282 | [-0.459, -0.082] | 0.294 | -0.176 | 0.185 | 14 |
| ERD_beta_R | -0.318 | [-0.503, -0.105] | 0.299 | -0.216 | 0.235 | 14 |
| ERD_beta_mag | 0.276 | [0.048, 0.477] | 0.276 | 0.210 | 0.210 | 14 |
| ERS_beta_rebound_L | -0.036 | [-0.236, 0.167] | 0.189 | -0.053 | 0.148 | 14 |
| ERS_beta_rebound_R | -0.165 | [-0.341, 0.022] | 0.195 | -0.155 | 0.179 | 14 |
| ERS_beta_rebound_mag | 0.228 | [0.058, 0.385] | 0.228 | 0.212 | 0.212 | 14 |
| ERD_lowgamma_L | -0.284 | [-0.424, -0.131] | 0.227 | -0.220 | 0.196 | 14 |
| ERD_lowgamma_R | -0.341 | [-0.479, -0.187] | 0.223 | -0.249 | 0.196 | 14 |
| ERD_lowgamma_mag | 0.208 | [-0.007, 0.404] | 0.208 | 0.198 | 0.198 | 14 |
| ERD_highgamma_L | -0.044 | [-0.236, 0.152] | 0.006 | 0.000 | 0.020 | 14 |
| ERD_highgamma_R | 0.076 | [-0.102, 0.251] | -0.066 | 0.102 | -0.021 | 14 |
| ERD_highgamma_mag | -0.045 | [-0.190, 0.102] | -0.045 | -0.003 | -0.003 | 14 |
| logpow_8_30 | 0.157 | [0.047, 0.262] | -0.181 | 0.030 | -0.154 | 14 |
| r2_signed | 0.020 | [-0.135, 0.173] | 0.220 | -0.001 | 0.165 | 14 |
| r2_abs | 0.220 | [0.103, 0.332] | 0.220 | 0.165 | 0.165 | 14 |
| TFR_ERD_dB_L | -0.238 | [-0.421, -0.037] | 0.254 | -0.176 | 0.193 | 14 |
| TFR_ERD_dB_R | -0.277 | [-0.472, -0.057] | 0.252 | -0.178 | 0.202 | 14 |
| TFR_ERD_dB_mag | 0.239 | [0.009, 0.444] | 0.239 | 0.205 | 0.205 | 14 |
| ITC_4_30 | 0.129 | [-0.030, 0.281] | 0.129 | 0.073 | 0.073 | 14 |
| ERP_abs_0.2_0.8 | -0.008 | [-0.120, 0.105] | -0.008 | 0.012 | 0.012 | 14 |
| SlowPot_0.5_3 | -0.084 | [-0.216, 0.050] | 0.092 | -0.090 | 0.093 | 14 |
| ERD_beta_classspecific | — | — | 0.297 [0.118, 0.458] | — | 0.219 | 14 |
| ERD_highgamma_classspecific | — | — | -0.040 [-0.157, 0.078] | — | -0.013 | 14 |
| ERD_lowgamma_classspecific | — | — | 0.221 [0.041, 0.387] | — | 0.194 | 14 |
| ERD_mu_classspecific | — | — | 0.127 [-0.019, 0.267] | — | 0.091 | 14 |
| ERD_theta_classspecific | — | — | 0.113 [-0.048, 0.269] | — | 0.086 | 14 |
| ERS_beta_rebound_classspecific | — | — | 0.192 [0.048, 0.327] | — | 0.163 | 14 |
| TFR_ERD_dB_classspecific | — | — | 0.249 [0.062, 0.418] | — | 0.203 | 14 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +17.77 ± 10.01 | +9.43 ± 6.53 | +3.39 ± 3.62 | 0.0017 | <0.0001 |
| 4 | +23.04 ± 14.48 | +14.49 ± 8.32 | +7.98 ± 7.06 | 0.0084 | 0.0006 |
| 6 | +23.12 ± 13.81 | +18.68 ± 10.89 | +13.42 ± 8.83 | 0.0262 | 0.0047 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.010 | [-0.084, 0.064] | 14 |
| grad_x_input | -0.018 | [-0.085, 0.050] | 14 |
| integrated_gradients | -0.013 | [-0.087, 0.060] | 14 |
| smoothgrad | -0.011 | [-0.085, 0.064] | 14 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=lawhern128, S02=lawhern128, S03=lawhern128, S04=lawhern128, S05=lawhern128, S06=lawhern128, S07=lawhern128, S08=lawhern128, S09=lawhern128, S10=lawhern128, S11=lawhern128, S12=lawhern128, S13=lawhern128, S14=lawhern128

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 77.92 |
| ems | 77.95 |
| lawhern128 | 81.13 |

```
model: "EEGNetv4"
dataset: "Schirrmeister2017"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"original": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": null, "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "ems": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "lawhern128": {"sfreq": 128, "band_hz": [4.0, 38.0], "window_s": [0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "wide250": {"sfreq": 250, "band_hz": [4.0, 100.0], "window_s": [0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 125}, "wide250_full": {"sfreq": 250, "band_hz": [4.0, 100.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 125}}
checkpoint_rule: "lowest validation loss, early stopping with patience; refit keeps final weights"
folds: "train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs"
seeds: [42, 43, 44]
```

Figures (75 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `2class\figures`.

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Protocol: train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects (n = 14). Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 | S13 | S14 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 75.00 | 84.38 | 97.50 | 98.12 | 85.00 | 87.50 | 79.38 | 88.75 | 85.62 | 88.12 | 78.12 | 88.75 | 87.50 | 95.00 | 87.05 | 6.79 |
| Ours | 75.62 | 81.88 | 96.25 | 96.88 | 88.12 | 88.75 | 80.62 | 91.88 | 78.75 | 87.50 | 75.62 | 80.62 | 82.50 | 90.00 | 85.36 | 7.00 |
| CSP | 76.88 | 85.00 | 95.62 | 96.88 | 86.25 | 92.50 | 73.12 | 90.62 | 86.88 | 90.00 | 71.88 | 81.88 | 89.38 | 92.50 | 86.38 | 7.89 |
| RLF | 78.75 | 82.50 | 93.75 | 98.12 | 90.00 | 90.00 | 79.38 | 90.62 | 81.25 | 87.50 | 80.62 | 76.25 | 78.12 | 91.88 | 85.62 | 6.86 |
| MI | 73.12 | 81.25 | 96.88 | 97.50 | 87.50 | 88.75 | 78.75 | 81.88 | 83.12 | 91.25 | 78.75 | 90.62 | 85.62 | 93.12 | 86.29 | 7.22 |
| Fixed sensorimotor | 73.12 | 78.12 | 95.62 | 96.25 | 88.75 | 89.38 | 78.75 | 81.88 | 78.12 | 86.88 | 71.88 | 91.88 | 81.25 | 91.25 | 84.51 | 7.99 |
| ERD-only | 75.00 | 76.88 | 95.62 | 93.12 | 87.50 | 90.62 | 81.25 | 88.75 | 86.25 | 90.00 | 80.62 | 86.25 | 83.75 | 88.75 | 86.03 | 5.94 |
| Random (mean of draws) | 74.75 | 81.50 | 94.38 | 95.62 | 87.62 | 87.50 | 76.62 | 81.63 | 75.88 | 91.25 | 73.62 | 87.38 | 79.62 | 90.38 | 84.13 | 7.43 |

### T0. Data used per subject

Trials after the class selection of this experiment (training trials include the validation part, which is also used to fit the final model; the test trials are never used for any choice).

| Subject | Training trials (incl. validation) | Validation trials | Test trials | Test trials per class | Majority-class test accuracy (%) | Chance (%) |
|---|---|---|---|---|---|---|
| S01 | 320 | 80 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S02 | 813 | 203 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S03 | 880 | 220 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S04 | 897 | 224 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S05 | 720 | 180 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S06 | 880 | 220 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S07 | 880 | 220 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S08 | 654 | 164 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S09 | 880 | 220 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S10 | 880 | 220 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S11 | 880 | 220 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S12 | 880 | 220 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S13 | 800 | 200 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |
| S14 | 880 | 220 | 160 | feet 40, left_hand 40, rest 40, right_hand 40 | 25.00 | 25.00 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 14 | C1 C3 C2 C4 CP2 CPz FCz CP1 FC4 FC2 CP3 Pz |
| CSP | 14 | CP2 CP1 C2 CP3 C1 C4 CPz C3 Pz FCz CP5 FC1 |
| RLF | 6 | C1 C2 C4 C3 CP3 CP1 CP2 CP4 Pz CPz FC1 CP5 |
| MI | 11 | C2 C1 C4 C3 CP3 CP1 CP2 CP4 C6 FC4 CPz CP6 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 13 | C4 C2 C1 C3 CP3 CP1 CP2 Pz CP6 CP4 CP5 CPz |

### T3. Equivalence of each 12-channel montage with all 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 14 | -1.70 | 3.53 | [-3.37, -0.03] | 0.0003 | yes | -8.12 | 2 | 4,5,6,8,10 |
| CSP | 14 | -0.67 | 3.65 | [-2.40, +1.06] | <0.0001 | yes | -6.87 | 3 | 3,4,5,6,8,10 |
| RLF | 14 | -1.43 | 4.94 | [-3.77, +0.91] | 0.0021 | yes | -12.50 | 2 | 4,5,6,8,10 |
| MI | 14 | -0.76 | 2.61 | [-1.99, +0.48] | <0.0001 | yes | -6.87 | 1 | 2,3,4,5,6,8,10 |
| Fixed sensorimotor | 14 | -2.54 | 3.76 | [-4.33, -0.76] | 0.0022 | yes | -7.50 | 5 | 5,6,8,10 |
| ERD-only | 14 | -1.03 | 3.50 | [-2.68, +0.63] | <0.0001 | yes | -7.50 | 2 | 3,4,5,6,8,10 |
| Random (mean of draws) | 14 | -2.93 | 3.72 | [-4.69, -1.17] | 0.0043 | yes | -9.75 | 3 | 5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 14 | -1.03 | [-3.37, +1.31] | 0.3607 | 1.0000 | 0.2632 | 0.9306 | 5 / 1 / 8 |
| RLF | 14 | -0.27 | [-1.85, +1.32] | 0.7213 | 1.0000 | 0.6244 | 1.0000 | 5 / 1 / 8 |
| MI | 14 | -0.94 | [-3.53, +1.65] | 0.4482 | 1.0000 | 0.2327 | 0.9306 | 5 / 1 / 8 |
| Fixed sensorimotor | 14 | +0.85 | [-1.73, +3.43] | 0.4903 | 1.0000 | 0.1649 | 0.8247 | 10 / 0 / 4 |
| ERD-only | 14 | -0.67 | [-2.76, +1.42] | 0.5004 | 1.0000 | 0.5498 | 1.0000 | 7 / 0 / 7 |
| Random (mean of draws) | 14 | +1.23 | [-0.96, +3.43] | 0.2468 | 1.0000 | 0.0899 | 0.5395 | 11 / 0 / 3 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`, per subject in `T5_confusion_per_subject.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 87.05 | feet 89.6, left_hand 84.6, rest 89.6, right_hand 84.3 | feet 24.6, left_hand 25.6, rest 24.3, right_hand 25.6 | 25.00 | 0 / 14 |
| Ours | 85.36 | feet 88.4, left_hand 80.7, rest 89.8, right_hand 82.5 | feet 24.3, left_hand 25.1, rest 25.3, right_hand 25.3 | 25.00 | 0 / 14 |
| CSP | 86.38 | feet 89.8, left_hand 82.7, rest 90.9, right_hand 82.1 | feet 24.2, left_hand 25.7, rest 25.1, right_hand 24.9 | 25.00 | 0 / 14 |
| RLF | 85.62 | feet 88.6, left_hand 81.6, rest 88.2, right_hand 84.1 | feet 25.0, left_hand 25.1, rest 24.6, right_hand 25.3 | 25.00 | 0 / 14 |
| MI | 86.29 | feet 91.6, left_hand 85.4, rest 88.2, right_hand 80.0 | feet 25.3, left_hand 26.9, rest 23.5, right_hand 24.4 | 25.00 | 0 / 14 |
| Fixed sensorimotor | 84.51 | feet 90.0, left_hand 81.2, rest 88.8, right_hand 78.0 | feet 24.6, left_hand 26.3, rest 24.3, right_hand 24.8 | 25.00 | 0 / 14 |
| ERD-only | 86.03 | feet 89.6, left_hand 83.0, rest 88.0, right_hand 83.4 | feet 24.9, left_hand 25.5, rest 23.9, right_hand 25.7 | 25.00 | 0 / 14 |
| Random (mean of draws) | 84.12 | feet 89.8, left_hand 80.4, rest 88.2, right_hand 78.2 | feet 24.8, left_hand 26.2, rest 23.9, right_hand 25.1 | 25.00 | 0 / 70 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 85.54, 43: 84.42, 44: 84.55 | 84.84 | 0.61 |
| Ours | 3 | 42: 80.98, 43: 82.32, 44: 84.02 | 82.44 | 1.52 |
| CSP | 3 | 42: 84.15, 43: 82.68, 44: 83.21 | 83.35 | 0.75 |
| RLF | 3 | 42: 81.34, 43: 82.41, 44: 85.13 | 82.96 | 1.96 |
| MI | 3 | 42: 85.18, 43: 84.69, 44: 84.02 | 84.63 | 0.58 |
| Fixed sensorimotor | 3 | 42: 82.59, 43: 80.98, 44: 82.50 | 82.02 | 0.90 |
| ERD-only | 3 | 42: 84.46, 43: 83.44, 44: 83.75 | 83.88 | 0.53 |
| Random (mean of draws) | 3 | 42: 81.86, 43: 81.28, 44: 81.71 | 81.61 | 0.30 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.451 | [0.302, 0.578] | 0.259 | 14 |
| grad | class_specific_pearson | 0.426 | [0.303, 0.535] | 0.207 | 14 |
| grad | lateralisation_pearson | -0.030 | [-0.162, 0.103] | 0.224 | 14 |
| grad | magnitude_spearman | 0.403 | [0.235, 0.548] | 0.285 | 14 |
| grad | class_specific_spearman | 0.419 | [0.304, 0.522] | 0.187 | 14 |
| grad | lateralisation_spearman | -0.073 | [-0.222, 0.079] | 0.252 | 14 |
| grad | legacy_signed_LR_pearson | 0.196 | [0.089, 0.299] | 0.177 | 14 |
| grad_x_input | magnitude_pearson | 0.419 | [0.269, 0.549] | 0.255 | 14 |
| grad_x_input | class_specific_pearson | 0.297 | [0.150, 0.431] | 0.243 | 14 |
| grad_x_input | lateralisation_pearson | -0.122 | [-0.260, 0.021] | 0.232 | 14 |
| grad_x_input | magnitude_spearman | 0.363 | [0.190, 0.514] | 0.284 | 14 |
| grad_x_input | class_specific_spearman | 0.248 | [0.100, 0.386] | 0.242 | 14 |
| grad_x_input | lateralisation_spearman | -0.168 | [-0.320, -0.007] | 0.258 | 14 |
| grad_x_input | legacy_signed_LR_pearson | 0.182 | [0.071, 0.287] | 0.184 | 14 |
| integrated_gradients | magnitude_pearson | 0.389 | [0.238, 0.522] | 0.255 | 14 |
| integrated_gradients | class_specific_pearson | 0.263 | [0.122, 0.394] | 0.232 | 14 |
| integrated_gradients | lateralisation_pearson | -0.109 | [-0.241, 0.026] | 0.216 | 14 |
| integrated_gradients | magnitude_spearman | 0.345 | [0.171, 0.498] | 0.286 | 14 |
| integrated_gradients | class_specific_spearman | 0.228 | [0.070, 0.375] | 0.256 | 14 |
| integrated_gradients | lateralisation_spearman | -0.170 | [-0.314, -0.019] | 0.241 | 14 |
| integrated_gradients | legacy_signed_LR_pearson | 0.182 | [0.076, 0.285] | 0.178 | 14 |
| smoothgrad | magnitude_pearson | 0.452 | [0.303, 0.579] | 0.259 | 14 |
| smoothgrad | class_specific_pearson | 0.428 | [0.304, 0.537] | 0.208 | 14 |
| smoothgrad | lateralisation_pearson | -0.025 | [-0.157, 0.107] | 0.224 | 14 |
| smoothgrad | magnitude_spearman | 0.402 | [0.231, 0.549] | 0.289 | 14 |
| smoothgrad | class_specific_spearman | 0.419 | [0.304, 0.522] | 0.188 | 14 |
| smoothgrad | lateralisation_spearman | -0.065 | [-0.210, 0.084] | 0.247 | 14 |
| smoothgrad | legacy_signed_LR_pearson | 0.196 | [0.089, 0.299] | 0.178 | 14 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.439 | [-0.548, -0.315] | 14 |
| grad | ERD_L_spearman | -0.392 | [-0.512, -0.258] | 14 |
| grad | ERD_R_pearson | -0.475 | [-0.582, -0.352] | 14 |
| grad | ERD_R_spearman | -0.466 | [-0.576, -0.339] | 14 |
| grad | ERD_LR_pearson | 0.196 | [0.089, 0.299] | 14 |
| grad | ERD_LR_spearman | 0.197 | [0.073, 0.315] | 14 |
| grad | ERD_comb_pearson | -0.471 | [-0.577, -0.349] | 14 |
| grad | ERD_comb_spearman | -0.444 | [-0.548, -0.325] | 14 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |
| grad_x_input | ERD_L_pearson | -0.390 | [-0.507, -0.260] | 14 |
| grad_x_input | ERD_L_spearman | -0.327 | [-0.455, -0.186] | 14 |
| grad_x_input | ERD_R_pearson | -0.427 | [-0.544, -0.295] | 14 |
| grad_x_input | ERD_R_spearman | -0.404 | [-0.531, -0.259] | 14 |
| grad_x_input | ERD_LR_pearson | 0.182 | [0.071, 0.287] | 14 |
| grad_x_input | ERD_LR_spearman | 0.178 | [0.048, 0.303] | 14 |
| grad_x_input | ERD_comb_pearson | -0.423 | [-0.538, -0.292] | 14 |
| grad_x_input | ERD_comb_spearman | -0.383 | [-0.505, -0.247] | 14 |
| grad_x_input | vs_grad_pearson | 0.997 | [0.995, 0.999] | 14 |
| grad_x_input | vs_grad_spearman | 0.992 | [0.984, 0.996] | 14 |
| integrated_gradients | ERD_L_pearson | -0.368 | [-0.483, -0.241] | 14 |
| integrated_gradients | ERD_L_spearman | -0.318 | [-0.441, -0.183] | 14 |
| integrated_gradients | ERD_R_pearson | -0.407 | [-0.516, -0.285] | 14 |
| integrated_gradients | ERD_R_spearman | -0.397 | [-0.515, -0.264] | 14 |
| integrated_gradients | ERD_LR_pearson | 0.182 | [0.076, 0.285] | 14 |
| integrated_gradients | ERD_LR_spearman | 0.185 | [0.059, 0.305] | 14 |
| integrated_gradients | ERD_comb_pearson | -0.402 | [-0.512, -0.280] | 14 |
| integrated_gradients | ERD_comb_spearman | -0.376 | [-0.491, -0.249] | 14 |
| integrated_gradients | vs_grad_pearson | 0.988 | [0.980, 0.993] | 14 |
| integrated_gradients | vs_grad_spearman | 0.976 | [0.963, 0.984] | 14 |
| smoothgrad | ERD_L_pearson | -0.440 | [-0.549, -0.316] | 14 |
| smoothgrad | ERD_L_spearman | -0.390 | [-0.509, -0.255] | 14 |
| smoothgrad | ERD_R_pearson | -0.476 | [-0.583, -0.352] | 14 |
| smoothgrad | ERD_R_spearman | -0.467 | [-0.579, -0.337] | 14 |
| smoothgrad | ERD_LR_pearson | 0.196 | [0.089, 0.299] | 14 |
| smoothgrad | ERD_LR_spearman | 0.201 | [0.077, 0.319] | 14 |
| smoothgrad | ERD_comb_pearson | -0.472 | [-0.578, -0.350] | 14 |
| smoothgrad | ERD_comb_spearman | -0.442 | [-0.548, -0.323] | 14 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| smoothgrad | vs_grad_spearman | 1.000 | [0.999, 1.000] | 14 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.095 | [-0.262, 0.078] | 0.227 | -0.027 | 0.249 | 14 |
| ERD_theta_R | -0.112 | [-0.256, 0.038] | 0.108 | -0.116 | 0.141 | 14 |
| ERD_theta_mag | 0.165 | [-0.001, 0.322] | 0.165 | 0.198 | 0.198 | 14 |
| ERD_mu_L | -0.334 | [-0.469, -0.184] | 0.338 | -0.295 | 0.309 | 14 |
| ERD_mu_R | -0.369 | [-0.492, -0.232] | 0.273 | -0.374 | 0.244 | 14 |
| ERD_mu_mag | 0.292 | [0.107, 0.458] | 0.292 | 0.258 | 0.258 | 14 |
| ERD_beta_L | -0.515 | [-0.602, -0.417] | 0.542 | -0.466 | 0.486 | 14 |
| ERD_beta_R | -0.524 | [-0.620, -0.412] | 0.501 | -0.509 | 0.450 | 14 |
| ERD_beta_mag | 0.516 | [0.368, 0.638] | 0.516 | 0.456 | 0.456 | 14 |
| ERS_beta_rebound_L | -0.290 | [-0.417, -0.152] | 0.294 | -0.266 | 0.316 | 14 |
| ERS_beta_rebound_R | -0.291 | [-0.454, -0.109] | 0.331 | -0.337 | 0.351 | 14 |
| ERS_beta_rebound_mag | 0.352 | [0.207, 0.483] | 0.352 | 0.370 | 0.370 | 14 |
| ERD_lowgamma_L | -0.495 | [-0.627, -0.334] | 0.436 | -0.491 | 0.415 | 14 |
| ERD_lowgamma_R | -0.498 | [-0.639, -0.325] | 0.403 | -0.511 | 0.404 | 14 |
| ERD_lowgamma_mag | 0.406 | [0.141, 0.617] | 0.406 | 0.421 | 0.421 | 14 |
| ERD_highgamma_L | 0.001 | [-0.185, 0.186] | 0.188 | 0.047 | 0.199 | 14 |
| ERD_highgamma_R | 0.131 | [-0.035, 0.290] | 0.108 | 0.145 | 0.107 | 14 |
| ERD_highgamma_mag | 0.124 | [-0.025, 0.268] | 0.124 | 0.176 | 0.176 | 14 |
| logpow_8_30 | 0.140 | [0.025, 0.252] | -0.382 | -0.109 | -0.360 | 14 |
| r2_signed | -0.071 | [-0.164, 0.023] | 0.048 | -0.039 | 0.117 | 14 |
| r2_abs | 0.048 | [-0.089, 0.184] | 0.048 | 0.117 | 0.117 | 14 |
| TFR_ERD_dB_L | -0.517 | [-0.606, -0.414] | 0.535 | -0.457 | 0.464 | 14 |
| TFR_ERD_dB_R | -0.533 | [-0.625, -0.425] | 0.491 | -0.499 | 0.436 | 14 |
| TFR_ERD_dB_mag | 0.506 | [0.337, 0.644] | 0.506 | 0.463 | 0.463 | 14 |
| ITC_4_30 | 0.219 | [0.093, 0.338] | 0.219 | 0.128 | 0.128 | 14 |
| ERP_abs_0.2_0.8 | -0.027 | [-0.112, 0.058] | -0.027 | -0.003 | -0.003 | 14 |
| SlowPot_0.5_3 | -0.002 | [-0.088, 0.084] | 0.016 | 0.021 | 0.070 | 14 |
| ERD_beta_classspecific | — | — | 0.459 [0.349, 0.557] | — | 0.392 | 14 |
| ERD_highgamma_classspecific | — | — | 0.158 [0.058, 0.254] | — | 0.178 | 14 |
| ERD_lowgamma_classspecific | — | — | 0.350 [0.144, 0.528] | — | 0.344 | 14 |
| ERD_mu_classspecific | — | — | 0.261 [0.108, 0.402] | — | 0.216 | 14 |
| ERD_theta_classspecific | — | — | 0.199 [0.083, 0.310] | — | 0.196 | 14 |
| ERS_beta_rebound_classspecific | — | — | 0.265 [0.141, 0.381] | — | 0.280 | 14 |
| TFR_ERD_dB_classspecific | — | — | 0.442 [0.311, 0.556] | — | 0.374 | 14 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +33.05 ± 12.97 | +15.27 ± 5.46 | +2.81 ± 2.04 | <0.0001 | <0.0001 |
| 4 | +40.28 ± 11.55 | +26.69 ± 7.52 | +11.25 ± 7.03 | 0.0001 | <0.0001 |
| 6 | +44.08 ± 8.53 | +33.66 ± 8.32 | +18.90 ± 6.76 | 0.0002 | <0.0001 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.046 | [-0.029, 0.121] | 14 |
| grad_x_input | -0.008 | [-0.086, 0.069] | 14 |
| integrated_gradients | 0.004 | [-0.070, 0.077] | 14 |
| smoothgrad | 0.044 | [-0.032, 0.120] | 14 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=lawhern128, S02=lawhern128, S03=lawhern128, S04=lawhern128, S05=lawhern128, S06=lawhern128, S07=lawhern128, S08=lawhern128, S09=lawhern128, S10=lawhern128, S11=lawhern128, S12=lawhern128, S13=lawhern128, S14=lawhern128

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 83.57 |
| ems | 81.63 |
| lawhern128 | 86.51 |

```
model: "EEGNetv4"
dataset: "Schirrmeister2017"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"original": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": null, "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "ems": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "lawhern128": {"sfreq": 128, "band_hz": [4.0, 38.0], "window_s": [0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "wide250": {"sfreq": 250, "band_hz": [4.0, 100.0], "window_s": [0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 125}, "wide250_full": {"sfreq": 250, "band_hz": [4.0, 100.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 125}}
checkpoint_rule: "lowest validation loss, early stopping with patience; refit keeps final weights"
folds: "train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs"
seeds: [42, 43, 44]
```

Figures (75 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `4class\figures`.
