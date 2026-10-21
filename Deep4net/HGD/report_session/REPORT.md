# Results report — Deep4net\HGD

Generated 2026-10-21 20:28 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Protocol: train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects (n = 14). Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 | S13 | S14 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 46.25 | 81.25 | 98.75 | 97.50 | 56.25 | 78.75 | 61.25 | 63.75 | 46.25 | 100.00 | 56.25 | 98.75 | 77.50 | 96.25 | 75.63 | 20.45 |
| Ours | 56.25 | 80.00 | 91.25 | 97.50 | 57.50 | 77.50 | 65.00 | 65.00 | 51.25 | 100.00 | 60.00 | 97.50 | 73.75 | 98.75 | 76.52 | 17.79 |
| CSP | 60.00 | 85.00 | 96.25 | 98.75 | 56.25 | 68.75 | 52.50 | 66.25 | 41.25 | 100.00 | 61.25 | 96.25 | 78.75 | 97.50 | 75.62 | 20.06 |
| RLF | 56.25 | 83.75 | 97.50 | 98.75 | 43.75 | 81.25 | 55.00 | 62.50 | 53.75 | 100.00 | 60.00 | 97.50 | 76.25 | 95.00 | 75.80 | 20.14 |
| MI | 56.25 | 87.50 | 98.75 | 92.50 | 53.75 | 78.75 | 57.50 | 63.75 | 52.50 | 100.00 | 60.00 | 95.00 | 68.75 | 95.00 | 75.71 | 18.53 |
| Fixed sensorimotor | 57.50 | 90.00 | 93.75 | 98.75 | 62.50 | 85.00 | 63.75 | 62.50 | 48.75 | 100.00 | 58.75 | 98.75 | 83.75 | 93.75 | 78.39 | 18.39 |
| ERD-only | 56.25 | 88.75 | 98.75 | 92.50 | 48.75 | 82.50 | 55.00 | 62.50 | 48.75 | 98.75 | 67.50 | 97.50 | 77.50 | 96.25 | 76.52 | 19.52 |
| Random (mean of draws) | 52.25 | 84.75 | 93.25 | 96.25 | 54.00 | 73.75 | 56.25 | 59.25 | 46.50 | 96.50 | 58.50 | 98.00 | 72.50 | 95.25 | 74.07 | 19.45 |

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
| Ours | 13 | C2 C4 CP2 FC4 C1 CP1 FC6 CP4 C6 FC2 CP3 Pz |
| CSP | 14 | CP2 C2 Pz FC1 FCz CP1 FC5 CPz C3 FC3 C4 C1 |
| RLF | 14 | CP4 CP3 CP2 CP5 C3 CP1 FC2 C4 C2 C1 CP6 FC1 |
| MI | 14 | CP1 C6 CP2 FC5 C2 C4 CP4 FCz CP5 CP6 FC6 FC3 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 13 | C4 C2 C1 C3 CP3 CP1 CP2 Pz CP6 CP4 CP5 CPz |

### T3. Equivalence of each 12-channel montage with all 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 14 | +0.89 | 4.17 | [-1.08, +2.87] | 0.0003 | yes | -7.50 | 1 | 3,4,5,6,8,10 |
| CSP | 14 | -0.00 | 5.90 | [-2.79, +2.79] | 0.0011 | yes | -10.00 | 2 | 3,4,5,6,8,10 |
| RLF | 14 | +0.18 | 5.46 | [-2.40, +2.76] | 0.0008 | yes | -12.50 | 2 | 3,4,5,6,8,10 |
| MI | 14 | +0.09 | 5.04 | [-2.29, +2.47] | 0.0004 | yes | -8.75 | 1 | 3,4,5,6,8,10 |
| Fixed sensorimotor | 14 | +2.77 | 4.52 | [+0.63, +4.91] | 0.0096 | yes | -5.00 | 0 | 5,6,8,10 |
| ERD-only | 14 | +0.89 | 5.66 | [-1.79, +3.57] | 0.0025 | yes | -7.50 | 2 | 4,5,6,8,10 |
| Random (mean of draws) | 14 | -1.55 | 3.57 | [-3.24, +0.14] | 0.0002 | yes | -5.50 | 0 | 4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 14 | +0.89 | [-2.39, +4.18] | 0.5668 | 1.0000 | 0.9720 | 1.0000 | 6 / 1 / 7 |
| RLF | 14 | +0.71 | [-2.42, +3.85] | 0.6309 | 1.0000 | 0.8383 | 1.0000 | 4 / 4 / 6 |
| MI | 14 | +0.80 | [-1.72, +3.33] | 0.5040 | 1.0000 | 0.3735 | 1.0000 | 7 / 3 / 4 |
| Fixed sensorimotor | 14 | -1.88 | [-4.57, +0.82] | 0.1570 | 0.7849 | 0.2337 | 1.0000 | 5 / 1 / 8 |
| ERD-only | 14 | -0.00 | [-3.40, +3.40] | 1.0000 | 1.0000 | 0.9687 | 1.0000 | 7 / 2 / 5 |
| Random (mean of draws) | 14 | +2.45 | [+0.50, +4.39] | 0.0176 | 0.1054 | 0.0258 | 0.1548 | 11 / 0 / 3 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`, per subject in `T5_confusion_per_subject.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 75.62 | left_hand 73.0, right_hand 78.2 | left_hand 47.4, right_hand 52.6 | 50.00 | 0 / 14 |
| Ours | 76.52 | left_hand 74.6, right_hand 78.4 | left_hand 48.1, right_hand 51.9 | 50.00 | 0 / 14 |
| CSP | 75.62 | left_hand 75.9, right_hand 75.4 | left_hand 50.3, right_hand 49.7 | 50.00 | 0 / 14 |
| RLF | 75.80 | left_hand 75.0, right_hand 76.6 | left_hand 49.2, right_hand 50.8 | 50.00 | 0 / 14 |
| MI | 75.71 | left_hand 74.1, right_hand 77.3 | left_hand 48.4, right_hand 51.6 | 50.00 | 0 / 14 |
| Fixed sensorimotor | 78.39 | left_hand 75.9, right_hand 80.9 | left_hand 47.5, right_hand 52.5 | 50.00 | 0 / 14 |
| ERD-only | 76.52 | left_hand 75.2, right_hand 77.9 | left_hand 48.7, right_hand 51.3 | 50.00 | 0 / 14 |
| Random (mean of draws) | 74.07 | left_hand 72.1, right_hand 76.0 | left_hand 48.0, right_hand 52.0 | 50.00 | 0 / 70 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 74.46, 43: 76.43, 44: 75.71 | 75.54 | 0.99 |
| Ours | 3 | 42: 75.54, 43: 74.29, 44: 76.07 | 75.30 | 0.92 |
| CSP | 3 | 42: 75.36, 43: 73.66, 44: 72.14 | 73.72 | 1.61 |
| RLF | 3 | 42: 73.66, 43: 74.64, 44: 75.27 | 74.52 | 0.81 |
| MI | 3 | 42: 74.91, 43: 74.55, 44: 74.64 | 74.70 | 0.19 |
| Fixed sensorimotor | 3 | 42: 75.45, 43: 74.64, 44: 74.29 | 74.79 | 0.59 |
| ERD-only | 3 | 42: 75.18, 43: 74.55, 44: 76.61 | 75.45 | 1.05 |
| Random (mean of draws) | 3 | 42: 74.18, 43: 72.80, 44: 72.68 | 73.22 | 0.83 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.307 | [0.166, 0.436] | 0.242 | 14 |
| grad | class_specific_pearson | 0.351 | [0.256, 0.439] | 0.162 | 14 |
| grad | lateralisation_pearson | -0.081 | [-0.275, 0.120] | 0.315 | 14 |
| grad | magnitude_spearman | 0.295 | [0.121, 0.452] | 0.283 | 14 |
| grad | class_specific_spearman | 0.323 | [0.193, 0.442] | 0.217 | 14 |
| grad | lateralisation_spearman | -0.068 | [-0.242, 0.110] | 0.283 | 14 |
| grad | legacy_signed_LR_pearson | 0.100 | [-0.132, 0.322] | 0.357 | 14 |
| grad_x_input | magnitude_pearson | 0.222 | [0.089, 0.348] | 0.225 | 14 |
| grad_x_input | class_specific_pearson | 0.243 | [0.122, 0.356] | 0.204 | 14 |
| grad_x_input | lateralisation_pearson | -0.277 | [-0.486, -0.037] | 0.352 | 14 |
| grad_x_input | magnitude_spearman | 0.195 | [0.021, 0.357] | 0.282 | 14 |
| grad_x_input | class_specific_spearman | 0.215 | [0.074, 0.347] | 0.235 | 14 |
| grad_x_input | lateralisation_spearman | -0.247 | [-0.465, -0.000] | 0.362 | 14 |
| grad_x_input | legacy_signed_LR_pearson | 0.055 | [-0.186, 0.289] | 0.372 | 14 |
| integrated_gradients | magnitude_pearson | 0.224 | [0.093, 0.348] | 0.222 | 14 |
| integrated_gradients | class_specific_pearson | 0.242 | [0.124, 0.353] | 0.201 | 14 |
| integrated_gradients | lateralisation_pearson | -0.375 | [-0.556, -0.161] | 0.322 | 14 |
| integrated_gradients | magnitude_spearman | 0.192 | [0.022, 0.351] | 0.278 | 14 |
| integrated_gradients | class_specific_spearman | 0.212 | [0.072, 0.344] | 0.234 | 14 |
| integrated_gradients | lateralisation_spearman | -0.345 | [-0.537, -0.120] | 0.328 | 14 |
| integrated_gradients | legacy_signed_LR_pearson | 0.034 | [-0.216, 0.280] | 0.388 | 14 |
| smoothgrad | magnitude_pearson | 0.307 | [0.166, 0.436] | 0.242 | 14 |
| smoothgrad | class_specific_pearson | 0.351 | [0.256, 0.439] | 0.162 | 14 |
| smoothgrad | lateralisation_pearson | -0.083 | [-0.274, 0.115] | 0.310 | 14 |
| smoothgrad | magnitude_spearman | 0.295 | [0.121, 0.451] | 0.283 | 14 |
| smoothgrad | class_specific_spearman | 0.323 | [0.193, 0.442] | 0.217 | 14 |
| smoothgrad | lateralisation_spearman | -0.066 | [-0.242, 0.113] | 0.285 | 14 |
| smoothgrad | legacy_signed_LR_pearson | 0.100 | [-0.132, 0.322] | 0.357 | 14 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.361 | [-0.452, -0.262] | 14 |
| grad | ERD_L_spearman | -0.332 | [-0.463, -0.187] | 14 |
| grad | ERD_R_pearson | -0.342 | [-0.441, -0.236] | 14 |
| grad | ERD_R_spearman | -0.315 | [-0.439, -0.180] | 14 |
| grad | ERD_LR_pearson | 0.100 | [-0.132, 0.322] | 14 |
| grad | ERD_LR_spearman | 0.095 | [-0.130, 0.311] | 14 |
| grad | ERD_comb_pearson | -0.360 | [-0.453, -0.258] | 14 |
| grad | ERD_comb_spearman | -0.339 | [-0.472, -0.190] | 14 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |
| grad_x_input | ERD_L_pearson | -0.256 | [-0.370, -0.135] | 14 |
| grad_x_input | ERD_L_spearman | -0.227 | [-0.378, -0.064] | 14 |
| grad_x_input | ERD_R_pearson | -0.235 | [-0.360, -0.100] | 14 |
| grad_x_input | ERD_R_spearman | -0.215 | [-0.350, -0.071] | 14 |
| grad_x_input | ERD_LR_pearson | 0.055 | [-0.186, 0.289] | 14 |
| grad_x_input | ERD_LR_spearman | 0.070 | [-0.157, 0.289] | 14 |
| grad_x_input | ERD_comb_pearson | -0.252 | [-0.373, -0.124] | 14 |
| grad_x_input | ERD_comb_spearman | -0.236 | [-0.388, -0.073] | 14 |
| grad_x_input | vs_grad_pearson | 0.992 | [0.984, 0.996] | 14 |
| grad_x_input | vs_grad_spearman | 0.986 | [0.977, 0.991] | 14 |
| integrated_gradients | ERD_L_pearson | -0.264 | [-0.378, -0.142] | 14 |
| integrated_gradients | ERD_L_spearman | -0.227 | [-0.374, -0.069] | 14 |
| integrated_gradients | ERD_R_pearson | -0.231 | [-0.356, -0.099] | 14 |
| integrated_gradients | ERD_R_spearman | -0.209 | [-0.341, -0.069] | 14 |
| integrated_gradients | ERD_LR_pearson | 0.034 | [-0.216, 0.280] | 14 |
| integrated_gradients | ERD_LR_spearman | 0.051 | [-0.181, 0.277] | 14 |
| integrated_gradients | ERD_comb_pearson | -0.254 | [-0.373, -0.127] | 14 |
| integrated_gradients | ERD_comb_spearman | -0.234 | [-0.381, -0.076] | 14 |
| integrated_gradients | vs_grad_pearson | 0.991 | [0.980, 0.996] | 14 |
| integrated_gradients | vs_grad_spearman | 0.983 | [0.972, 0.990] | 14 |
| smoothgrad | ERD_L_pearson | -0.361 | [-0.452, -0.262] | 14 |
| smoothgrad | ERD_L_spearman | -0.332 | [-0.462, -0.187] | 14 |
| smoothgrad | ERD_R_pearson | -0.343 | [-0.441, -0.236] | 14 |
| smoothgrad | ERD_R_spearman | -0.316 | [-0.440, -0.181] | 14 |
| smoothgrad | ERD_LR_pearson | 0.100 | [-0.132, 0.322] | 14 |
| smoothgrad | ERD_LR_spearman | 0.096 | [-0.130, 0.313] | 14 |
| smoothgrad | ERD_comb_pearson | -0.360 | [-0.453, -0.258] | 14 |
| smoothgrad | ERD_comb_spearman | -0.338 | [-0.471, -0.190] | 14 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.081 | [-0.291, 0.136] | 0.187 | -0.101 | 0.135 | 14 |
| ERD_theta_R | -0.168 | [-0.357, 0.035] | 0.123 | -0.174 | 0.134 | 14 |
| ERD_theta_mag | 0.178 | [-0.005, 0.350] | 0.178 | 0.159 | 0.159 | 14 |
| ERD_mu_L | -0.257 | [-0.366, -0.140] | 0.129 | -0.226 | 0.126 | 14 |
| ERD_mu_R | -0.250 | [-0.373, -0.118] | 0.185 | -0.279 | 0.211 | 14 |
| ERD_mu_mag | 0.155 | [-0.021, 0.322] | 0.155 | 0.169 | 0.169 | 14 |
| ERD_beta_L | -0.313 | [-0.394, -0.227] | 0.310 | -0.268 | 0.268 | 14 |
| ERD_beta_R | -0.310 | [-0.392, -0.223] | 0.303 | -0.284 | 0.271 | 14 |
| ERD_beta_mag | 0.302 | [0.185, 0.411] | 0.302 | 0.266 | 0.266 | 14 |
| ERS_beta_rebound_L | 0.000 | [-0.191, 0.191] | 0.186 | -0.018 | 0.161 | 14 |
| ERS_beta_rebound_R | -0.329 | [-0.485, -0.153] | 0.286 | -0.313 | 0.269 | 14 |
| ERS_beta_rebound_mag | 0.272 | [0.128, 0.404] | 0.272 | 0.231 | 0.231 | 14 |
| ERD_lowgamma_L | -0.210 | [-0.342, -0.070] | 0.215 | -0.172 | 0.167 | 14 |
| ERD_lowgamma_R | -0.283 | [-0.405, -0.150] | 0.259 | -0.218 | 0.225 | 14 |
| ERD_lowgamma_mag | 0.230 | [0.099, 0.354] | 0.230 | 0.201 | 0.201 | 14 |
| ERD_highgamma_L | 0.087 | [-0.092, 0.260] | 0.190 | 0.019 | 0.165 | 14 |
| ERD_highgamma_R | 0.133 | [-0.063, 0.320] | 0.059 | 0.112 | 0.056 | 14 |
| ERD_highgamma_mag | 0.139 | [-0.016, 0.287] | 0.139 | 0.111 | 0.111 | 14 |
| logpow_8_30 | 0.111 | [0.011, 0.209] | -0.182 | 0.008 | -0.176 | 14 |
| r2_signed | 0.218 | [-0.019, 0.432] | 0.336 | 0.186 | 0.307 | 14 |
| r2_abs | 0.336 | [0.143, 0.504] | 0.336 | 0.307 | 0.307 | 14 |
| TFR_ERD_dB_L | -0.336 | [-0.429, -0.237] | 0.328 | -0.310 | 0.304 | 14 |
| TFR_ERD_dB_R | -0.322 | [-0.410, -0.228] | 0.299 | -0.285 | 0.254 | 14 |
| TFR_ERD_dB_mag | 0.306 | [0.178, 0.424] | 0.306 | 0.286 | 0.286 | 14 |
| ITC_4_30 | 0.098 | [-0.103, 0.292] | 0.098 | 0.079 | 0.079 | 14 |
| ERP_abs_0.2_0.8 | 0.020 | [-0.126, 0.164] | 0.020 | 0.054 | 0.054 | 14 |
| SlowPot_0.5_3 | -0.051 | [-0.223, 0.123] | 0.129 | -0.046 | 0.159 | 14 |
| ERD_beta_classspecific | — | — | 0.306 [0.223, 0.384] | — | 0.270 | 14 |
| ERD_highgamma_classspecific | — | — | 0.115 [-0.019, 0.246] | — | 0.100 | 14 |
| ERD_lowgamma_classspecific | — | — | 0.232 [0.125, 0.333] | — | 0.193 | 14 |
| ERD_mu_classspecific | — | — | 0.156 [0.007, 0.298] | — | 0.164 | 14 |
| ERD_theta_classspecific | — | — | 0.152 [0.007, 0.291] | — | 0.135 | 14 |
| ERS_beta_rebound_classspecific | — | — | 0.232 [0.111, 0.345] | — | 0.211 | 14 |
| TFR_ERD_dB_classspecific | — | — | 0.312 [0.216, 0.401] | — | 0.280 | 14 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +16.46 ± 17.38 | +3.83 ± 5.04 | +0.39 ± 2.56 | 0.0034 | 0.0058 |
| 4 | +17.65 ± 17.91 | +6.46 ± 7.41 | +1.37 ± 2.75 | 0.0029 | 0.0033 |
| 6 | +18.51 ± 17.16 | +9.15 ± 9.08 | +2.02 ± 3.96 | 0.0018 | 0.0022 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.051 | [-0.133, 0.032] | 14 |
| grad_x_input | -0.199 | [-0.291, -0.104] | 14 |
| integrated_gradients | -0.205 | [-0.300, -0.106] | 14 |
| smoothgrad | -0.053 | [-0.131, 0.026] | 14 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=schirrmeister, S02=schirrmeister, S03=schirrmeister, S04=schirrmeister, S05=schirrmeister, S06=schirrmeister, S07=schirrmeister, S08=schirrmeister, S09=schirrmeister, S10=schirrmeister, S11=schirrmeister, S12=schirrmeister, S13=schirrmeister, S14=schirrmeister

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 70.98 |
| schirrmeister | 78.92 |
| cosine100 | 75.44 |

```
model: "Deep4Net"
dataset: "Schirrmeister2017"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"original": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "schirrmeister": {"sfreq": 250, "band_hz": [4.0, 38.0], "window_s": [-0.5, 4.0], "norm": "ems", "car": false, "lr": 0.01, "weight_decay": 0.0005, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 300, "patience": 100, "batch_size": 64, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.5, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "cosine100": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.01, "weight_decay": 0.0005, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 300, "patience": 100, "batch_size": 64, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.5, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "wide250": {"sfreq": 250, "band_hz": [4.0, 100.0], "window_s": [-0.5, 4.0], "norm": "ems", "car": false, "lr": 0.01, "weight_decay": 0.0005, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 300, "patience": 100, "batch_size": 64, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.5, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "wide250_lowlr": {"sfreq": 250, "band_hz": [4.0, 100.0], "window_s": [-0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 300, "patience": 100, "batch_size": 64, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.5, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}}
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
| 22ch | 65.62 | 83.12 | 95.62 | 90.62 | 80.00 | 87.50 | 73.12 | 81.25 | 55.00 | 89.38 | 71.88 | 90.00 | 74.38 | 91.88 | 80.67 | 11.52 |
| Ours | 66.25 | 83.12 | 95.62 | 91.25 | 76.25 | 83.12 | 71.88 | 85.00 | 54.37 | 85.62 | 73.12 | 88.75 | 80.00 | 90.62 | 80.36 | 11.12 |
| CSP | 64.38 | 83.12 | 91.88 | 86.25 | 76.25 | 80.62 | 72.50 | 84.38 | 52.50 | 82.50 | 72.50 | 90.00 | 79.38 | 93.12 | 79.24 | 11.12 |
| RLF | 68.75 | 82.50 | 93.12 | 91.25 | 79.38 | 85.62 | 70.62 | 83.12 | 55.00 | 84.38 | 71.88 | 89.38 | 75.00 | 92.50 | 80.18 | 10.88 |
| MI | 65.00 | 84.38 | 96.25 | 90.62 | 78.12 | 81.88 | 73.75 | 82.50 | 56.88 | 88.12 | 70.00 | 91.25 | 70.62 | 91.25 | 80.04 | 11.49 |
| Fixed sensorimotor | 66.25 | 78.75 | 92.50 | 93.75 | 76.25 | 81.88 | 71.25 | 80.00 | 50.62 | 85.00 | 70.62 | 90.00 | 71.25 | 89.38 | 78.39 | 11.88 |
| ERD-only | 65.62 | 79.38 | 95.00 | 84.38 | 75.00 | 76.25 | 70.62 | 83.12 | 56.25 | 87.50 | 66.88 | 88.12 | 78.12 | 90.00 | 78.30 | 10.78 |
| Random (mean of draws) | 66.25 | 80.87 | 94.38 | 89.75 | 77.88 | 81.50 | 71.75 | 80.62 | 54.00 | 88.12 | 72.00 | 88.00 | 72.88 | 91.38 | 79.24 | 11.12 |

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
| Ours | 14 | C3 C1 C2 C4 Pz FC2 FCz CP1 FC1 CP6 FC4 CP2 |
| CSP | 14 | CP2 C2 CPz CP1 C1 C4 C3 CP3 Pz FCz FC1 FC2 |
| RLF | 10 | C1 C2 C4 C3 CP3 CP1 CP2 CP4 Pz CPz CP5 FC1 |
| MI | 13 | C2 C1 C4 C3 CP1 CP3 CP2 CP4 C6 FC4 FCz CPz |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 13 | C4 C2 C1 C3 CP3 CP1 CP2 Pz CP6 CP4 CP5 CPz |

### T3. Equivalence of each 12-channel montage with all 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 14 | -0.31 | 2.76 | [-1.62, +0.99] | <0.0001 | yes | -4.37 | 0 | 2,3,4,5,6,8,10 |
| CSP | 14 | -1.43 | 3.49 | [-3.08, +0.22] | 0.0001 | yes | -6.88 | 2 | 4,5,6,8,10 |
| RLF | 14 | -0.49 | 2.02 | [-1.45, +0.47] | <0.0001 | yes | -5.00 | 0 | 2,3,4,5,6,8,10 |
| MI | 14 | -0.63 | 2.12 | [-1.63, +0.38] | <0.0001 | yes | -5.63 | 0 | 2,3,4,5,6,8,10 |
| Fixed sensorimotor | 14 | -2.28 | 2.37 | [-3.40, -1.15] | <0.0001 | yes | -5.63 | 0 | 4,5,6,8,10 |
| ERD-only | 14 | -2.37 | 3.80 | [-4.16, -0.57] | 0.0017 | yes | -11.25 | 2 | 5,6,8,10 |
| Random (mean of draws) | 14 | -1.43 | 1.55 | [-2.16, -0.70] | <0.0001 | yes | -6.00 | 0 | 3,4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 14 | +1.12 | [-0.06, +2.29] | 0.0613 | 0.2452 | 0.0646 | 0.2585 | 9 / 2 / 3 |
| RLF | 14 | +0.18 | [-1.12, +1.48] | 0.7714 | 1.0000 | 0.8885 | 1.0000 | 7 / 1 / 6 |
| MI | 14 | +0.31 | [-1.54, +2.17] | 0.7217 | 1.0000 | 0.8748 | 1.0000 | 6 / 0 / 8 |
| Fixed sensorimotor | 14 | +1.96 | [+0.30, +3.63] | 0.0245 | 0.1223 | 0.0309 | 0.1798 | 10 / 2 / 2 |
| ERD-only | 14 | +2.05 | [+0.39, +3.71] | 0.0191 | 0.1148 | 0.0300 | 0.1798 | 12 / 0 / 2 |
| Random (mean of draws) | 14 | +1.12 | [-0.27, +2.51] | 0.1066 | 0.3198 | 0.1079 | 0.3236 | 10 / 1 / 3 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`, per subject in `T5_confusion_per_subject.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 80.67 | feet 85.4, left_hand 81.1, rest 78.2, right_hand 78.0 | feet 24.6, left_hand 28.0, rest 21.0, right_hand 26.4 | 25.00 | 0 / 14 |
| Ours | 80.36 | feet 84.1, left_hand 80.4, rest 78.4, right_hand 78.6 | feet 24.7, left_hand 27.0, rest 21.6, right_hand 26.7 | 25.00 | 0 / 14 |
| CSP | 79.24 | feet 84.5, left_hand 81.6, rest 76.1, right_hand 74.8 | feet 24.5, left_hand 29.8, rest 20.4, right_hand 25.3 | 25.00 | 0 / 14 |
| RLF | 80.18 | feet 82.7, left_hand 80.4, rest 79.3, right_hand 78.4 | feet 23.8, left_hand 27.9, rest 21.7, right_hand 26.7 | 25.00 | 0 / 14 |
| MI | 80.04 | feet 83.2, left_hand 81.8, rest 78.0, right_hand 77.1 | feet 23.8, left_hand 28.8, rest 21.3, right_hand 26.1 | 25.00 | 0 / 14 |
| Fixed sensorimotor | 78.39 | feet 83.2, left_hand 80.2, rest 76.4, right_hand 73.8 | feet 24.4, left_hand 29.0, rest 20.4, right_hand 26.2 | 25.00 | 0 / 14 |
| ERD-only | 78.30 | feet 80.4, left_hand 77.7, rest 77.3, right_hand 77.9 | feet 23.7, left_hand 27.6, rest 21.6, right_hand 27.1 | 25.00 | 0 / 14 |
| Random (mean of draws) | 79.24 | feet 83.1, left_hand 78.9, rest 78.7, right_hand 76.2 | feet 23.9, left_hand 28.0, rest 21.5, right_hand 26.6 | 25.00 | 0 / 70 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 78.57, 43: 79.38, 44: 80.18 | 79.38 | 0.80 |
| Ours | 3 | 42: 79.38, 43: 78.75, 44: 79.55 | 79.23 | 0.42 |
| CSP | 3 | 42: 76.79, 43: 78.26, 44: 78.13 | 77.72 | 0.81 |
| RLF | 3 | 42: 78.88, 43: 77.68, 44: 78.39 | 78.32 | 0.61 |
| MI | 3 | 42: 78.97, 43: 78.79, 44: 79.29 | 79.02 | 0.25 |
| Fixed sensorimotor | 3 | 42: 77.50, 43: 77.72, 44: 77.68 | 77.63 | 0.12 |
| ERD-only | 3 | 42: 76.83, 43: 77.41, 44: 77.59 | 77.28 | 0.40 |
| Random (mean of draws) | 3 | 42: 77.71, 43: 77.37, 44: 77.50 | 77.52 | 0.17 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.412 | [0.256, 0.547] | 0.256 | 14 |
| grad | class_specific_pearson | 0.416 | [0.310, 0.512] | 0.173 | 14 |
| grad | lateralisation_pearson | 0.002 | [-0.224, 0.228] | 0.361 | 14 |
| grad | magnitude_spearman | 0.388 | [0.242, 0.517] | 0.242 | 14 |
| grad | class_specific_spearman | 0.389 | [0.290, 0.480] | 0.160 | 14 |
| grad | lateralisation_spearman | 0.000 | [-0.237, 0.237] | 0.379 | 14 |
| grad | legacy_signed_LR_pearson | 0.111 | [-0.004, 0.222] | 0.193 | 14 |
| grad_x_input | magnitude_pearson | 0.390 | [0.229, 0.530] | 0.259 | 14 |
| grad_x_input | class_specific_pearson | 0.350 | [0.234, 0.457] | 0.186 | 14 |
| grad_x_input | lateralisation_pearson | -0.203 | [-0.403, 0.014] | 0.334 | 14 |
| grad_x_input | magnitude_spearman | 0.372 | [0.212, 0.512] | 0.260 | 14 |
| grad_x_input | class_specific_spearman | 0.322 | [0.210, 0.425] | 0.179 | 14 |
| grad_x_input | lateralisation_spearman | -0.251 | [-0.447, -0.033] | 0.337 | 14 |
| grad_x_input | legacy_signed_LR_pearson | 0.096 | [-0.021, 0.212] | 0.199 | 14 |
| integrated_gradients | magnitude_pearson | 0.411 | [0.251, 0.548] | 0.260 | 14 |
| integrated_gradients | class_specific_pearson | 0.345 | [0.229, 0.451] | 0.184 | 14 |
| integrated_gradients | lateralisation_pearson | -0.316 | [-0.472, -0.141] | 0.285 | 14 |
| integrated_gradients | magnitude_spearman | 0.391 | [0.234, 0.528] | 0.253 | 14 |
| integrated_gradients | class_specific_spearman | 0.315 | [0.205, 0.418] | 0.176 | 14 |
| integrated_gradients | lateralisation_spearman | -0.286 | [-0.458, -0.093] | 0.313 | 14 |
| integrated_gradients | legacy_signed_LR_pearson | 0.083 | [-0.033, 0.197] | 0.195 | 14 |
| smoothgrad | magnitude_pearson | 0.412 | [0.256, 0.547] | 0.256 | 14 |
| smoothgrad | class_specific_pearson | 0.416 | [0.309, 0.512] | 0.173 | 14 |
| smoothgrad | lateralisation_pearson | 0.005 | [-0.218, 0.227] | 0.355 | 14 |
| smoothgrad | magnitude_spearman | 0.386 | [0.241, 0.515] | 0.241 | 14 |
| smoothgrad | class_specific_spearman | 0.387 | [0.287, 0.480] | 0.162 | 14 |
| smoothgrad | lateralisation_spearman | -0.012 | [-0.255, 0.232] | 0.384 | 14 |
| smoothgrad | legacy_signed_LR_pearson | 0.110 | [-0.004, 0.222] | 0.193 | 14 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.406 | [-0.494, -0.309] | 14 |
| grad | ERD_L_spearman | -0.390 | [-0.484, -0.288] | 14 |
| grad | ERD_R_pearson | -0.423 | [-0.540, -0.291] | 14 |
| grad | ERD_R_spearman | -0.366 | [-0.474, -0.248] | 14 |
| grad | ERD_LR_pearson | 0.111 | [-0.004, 0.222] | 14 |
| grad | ERD_LR_spearman | 0.056 | [-0.054, 0.165] | 14 |
| grad | ERD_comb_pearson | -0.422 | [-0.526, -0.305] | 14 |
| grad | ERD_comb_spearman | -0.402 | [-0.498, -0.296] | 14 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |
| grad_x_input | ERD_L_pearson | -0.378 | [-0.468, -0.280] | 14 |
| grad_x_input | ERD_L_spearman | -0.368 | [-0.470, -0.257] | 14 |
| grad_x_input | ERD_R_pearson | -0.393 | [-0.514, -0.256] | 14 |
| grad_x_input | ERD_R_spearman | -0.345 | [-0.462, -0.217] | 14 |
| grad_x_input | ERD_LR_pearson | 0.096 | [-0.021, 0.212] | 14 |
| grad_x_input | ERD_LR_spearman | 0.042 | [-0.069, 0.152] | 14 |
| grad_x_input | ERD_comb_pearson | -0.393 | [-0.500, -0.273] | 14 |
| grad_x_input | ERD_comb_spearman | -0.382 | [-0.488, -0.264] | 14 |
| grad_x_input | vs_grad_pearson | 0.998 | [0.997, 0.999] | 14 |
| grad_x_input | vs_grad_spearman | 0.994 | [0.990, 0.996] | 14 |
| integrated_gradients | ERD_L_pearson | -0.388 | [-0.474, -0.295] | 14 |
| integrated_gradients | ERD_L_spearman | -0.367 | [-0.457, -0.269] | 14 |
| integrated_gradients | ERD_R_pearson | -0.399 | [-0.517, -0.265] | 14 |
| integrated_gradients | ERD_R_spearman | -0.345 | [-0.453, -0.227] | 14 |
| integrated_gradients | ERD_LR_pearson | 0.083 | [-0.033, 0.197] | 14 |
| integrated_gradients | ERD_LR_spearman | 0.060 | [-0.048, 0.167] | 14 |
| integrated_gradients | ERD_comb_pearson | -0.400 | [-0.503, -0.285] | 14 |
| integrated_gradients | ERD_comb_spearman | -0.377 | [-0.471, -0.273] | 14 |
| integrated_gradients | vs_grad_pearson | 0.995 | [0.991, 0.997] | 14 |
| integrated_gradients | vs_grad_spearman | 0.990 | [0.984, 0.994] | 14 |
| smoothgrad | ERD_L_pearson | -0.406 | [-0.494, -0.309] | 14 |
| smoothgrad | ERD_L_spearman | -0.387 | [-0.480, -0.287] | 14 |
| smoothgrad | ERD_R_pearson | -0.423 | [-0.540, -0.290] | 14 |
| smoothgrad | ERD_R_spearman | -0.366 | [-0.474, -0.247] | 14 |
| smoothgrad | ERD_LR_pearson | 0.110 | [-0.004, 0.222] | 14 |
| smoothgrad | ERD_LR_spearman | 0.058 | [-0.050, 0.166] | 14 |
| smoothgrad | ERD_comb_pearson | -0.422 | [-0.526, -0.305] | 14 |
| smoothgrad | ERD_comb_spearman | -0.400 | [-0.496, -0.295] | 14 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.092 | [-0.255, 0.076] | 0.103 | -0.061 | 0.162 | 14 |
| ERD_theta_R | -0.099 | [-0.274, 0.084] | 0.007 | -0.007 | 0.048 | 14 |
| ERD_theta_mag | 0.038 | [-0.154, 0.227] | 0.038 | 0.105 | 0.105 | 14 |
| ERD_mu_L | -0.368 | [-0.485, -0.237] | 0.325 | -0.311 | 0.289 | 14 |
| ERD_mu_R | -0.372 | [-0.498, -0.230] | 0.257 | -0.325 | 0.218 | 14 |
| ERD_mu_mag | 0.276 | [0.082, 0.450] | 0.276 | 0.238 | 0.238 | 14 |
| ERD_beta_L | -0.445 | [-0.538, -0.342] | 0.483 | -0.484 | 0.506 | 14 |
| ERD_beta_R | -0.436 | [-0.541, -0.318] | 0.427 | -0.449 | 0.423 | 14 |
| ERD_beta_mag | 0.445 | [0.297, 0.572] | 0.445 | 0.445 | 0.445 | 14 |
| ERS_beta_rebound_L | -0.158 | [-0.319, 0.012] | 0.251 | -0.162 | 0.240 | 14 |
| ERS_beta_rebound_R | -0.194 | [-0.361, -0.016] | 0.262 | -0.192 | 0.258 | 14 |
| ERS_beta_rebound_mag | 0.298 | [0.131, 0.448] | 0.298 | 0.281 | 0.281 | 14 |
| ERD_lowgamma_L | -0.320 | [-0.451, -0.175] | 0.298 | -0.365 | 0.313 | 14 |
| ERD_lowgamma_R | -0.372 | [-0.510, -0.216] | 0.296 | -0.390 | 0.310 | 14 |
| ERD_lowgamma_mag | 0.271 | [0.081, 0.443] | 0.271 | 0.319 | 0.319 | 14 |
| ERD_highgamma_L | 0.058 | [-0.090, 0.204] | 0.151 | 0.108 | 0.199 | 14 |
| ERD_highgamma_R | 0.191 | [0.029, 0.343] | 0.052 | 0.197 | 0.089 | 14 |
| ERD_highgamma_mag | 0.108 | [-0.030, 0.242] | 0.108 | 0.184 | 0.184 | 14 |
| logpow_8_30 | 0.056 | [-0.016, 0.127] | -0.324 | -0.123 | -0.313 | 14 |
| r2_signed | -0.057 | [-0.200, 0.089] | 0.138 | -0.034 | 0.178 | 14 |
| r2_abs | 0.138 | [-0.027, 0.295] | 0.138 | 0.178 | 0.178 | 14 |
| TFR_ERD_dB_L | -0.494 | [-0.589, -0.385] | 0.518 | -0.468 | 0.470 | 14 |
| TFR_ERD_dB_R | -0.493 | [-0.615, -0.348] | 0.480 | -0.440 | 0.402 | 14 |
| TFR_ERD_dB_mag | 0.495 | [0.325, 0.634] | 0.495 | 0.435 | 0.435 | 14 |
| ITC_4_30 | 0.064 | [-0.047, 0.174] | 0.064 | 0.034 | 0.034 | 14 |
| ERP_abs_0.2_0.8 | -0.121 | [-0.203, -0.038] | -0.121 | -0.108 | -0.108 | 14 |
| SlowPot_0.5_3 | -0.081 | [-0.196, 0.036] | 0.017 | -0.027 | 0.063 | 14 |
| ERD_beta_classspecific | — | — | 0.441 [0.329, 0.540] | — | 0.451 | 14 |
| ERD_highgamma_classspecific | — | — | 0.109 [-0.001, 0.217] | — | 0.145 | 14 |
| ERD_lowgamma_classspecific | — | — | 0.274 [0.116, 0.418] | — | 0.284 | 14 |
| ERD_mu_classspecific | — | — | 0.282 [0.119, 0.431] | — | 0.238 | 14 |
| ERD_theta_classspecific | — | — | 0.060 [-0.097, 0.215] | — | 0.104 | 14 |
| ERS_beta_rebound_classspecific | — | — | 0.244 [0.111, 0.368] | — | 0.234 | 14 |
| TFR_ERD_dB_classspecific | — | — | 0.477 [0.344, 0.592] | — | 0.417 | 14 |

### T8a. Faithfulness: deletion test on the test trials

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +31.92 ± 13.11 | +7.81 ± 3.87 | +0.36 ± 1.33 | <0.0001 | <0.0001 |
| 4 | +41.26 ± 13.26 | +16.18 ± 5.05 | +1.67 ± 3.04 | <0.0001 | <0.0001 |
| 6 | +45.97 ± 10.97 | +22.13 ± 5.76 | +4.35 ± 4.78 | <0.0001 | <0.0001 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.026 | [-0.102, 0.051] | 14 |
| grad_x_input | -0.141 | [-0.214, -0.066] | 14 |
| integrated_gradients | -0.156 | [-0.225, -0.085] | 14 |
| smoothgrad | -0.027 | [-0.100, 0.047] | 14 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=schirrmeister, S02=schirrmeister, S03=schirrmeister, S04=schirrmeister, S05=schirrmeister, S06=schirrmeister, S07=schirrmeister, S08=schirrmeister, S09=schirrmeister, S10=schirrmeister, S11=schirrmeister, S12=schirrmeister, S13=schirrmeister, S14=schirrmeister

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 72.98 |
| schirrmeister | 81.94 |
| cosine100 | 79.48 |

```
model: "Deep4Net"
dataset: "Schirrmeister2017"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"original": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "schirrmeister": {"sfreq": 250, "band_hz": [4.0, 38.0], "window_s": [-0.5, 4.0], "norm": "ems", "car": false, "lr": 0.01, "weight_decay": 0.0005, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 300, "patience": 100, "batch_size": 64, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.5, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "cosine100": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.01, "weight_decay": 0.0005, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 300, "patience": 100, "batch_size": 64, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.5, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "wide250": {"sfreq": 250, "band_hz": [4.0, 100.0], "window_s": [-0.5, 4.0], "norm": "ems", "car": false, "lr": 0.01, "weight_decay": 0.0005, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 300, "patience": 100, "batch_size": 64, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.5, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}, "wide250_lowlr": {"sfreq": 250, "band_hz": [4.0, 100.0], "window_s": [-0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 300, "patience": 100, "batch_size": 64, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.5, "augment": false, "select_on": "val_acc", "min_select_epoch": 20}}
checkpoint_rule: "lowest validation loss, early stopping with patience; refit keeps final weights"
folds: "train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs"
seeds: [42, 43, 44]
```

Figures (75 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `4class\figures`.
