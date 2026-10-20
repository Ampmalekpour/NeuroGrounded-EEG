# Results report — EEGNetV4\HGD

Generated 2026-10-19 19:33 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 | S13 | S14 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 50.00 | 88.75 | 97.50 | 100.00 | 72.50 | 86.25 | 58.75 | 71.21 | 75.00 | 97.50 | 62.50 | 100.00 | 83.75 | 91.25 | 81.07 | 16.34 |
| Ours | 72.50 | 82.50 | 93.75 | 98.75 | 77.50 | 91.25 | 62.50 | 68.18 | 66.25 | 98.75 | 50.00 | 100.00 | 85.00 | 91.25 | 81.30 | 15.57 |
| CSP | 50.00 | 90.00 | 98.75 | 97.50 | 80.00 | 90.00 | 57.50 | 69.70 | 75.00 | 97.50 | 43.75 | 100.00 | 82.50 | 93.75 | 80.42 | 18.83 |
| RLF | 55.00 | 82.50 | 93.75 | 96.25 | 75.00 | 86.25 | 61.25 | 46.97 | 67.50 | 93.75 | 56.25 | 100.00 | 66.25 | 92.50 | 76.66 | 17.74 |
| MI | 58.75 | 82.50 | 92.50 | 95.00 | 77.50 | 83.75 | 47.50 | 43.94 | 78.75 | 100.00 | 50.00 | 100.00 | 62.50 | 92.50 | 76.08 | 19.93 |
| Fixed sensorimotor | 51.25 | 85.00 | 93.75 | 96.25 | 78.75 | 83.75 | 52.50 | 50.00 | 76.25 | 100.00 | 50.00 | 100.00 | 78.75 | 88.75 | 77.50 | 18.98 |
| ERD-only | 48.75 | 92.50 | 97.50 | 97.50 | 77.50 | 87.50 | 63.75 | 60.61 | 81.25 | 97.50 | 43.75 | 100.00 | 70.00 | 96.25 | 79.60 | 19.28 |
| Random (mean of draws) | 58.00 | 87.25 | 97.25 | 97.25 | 73.25 | 85.75 | 55.25 | 49.70 | 64.00 | 97.50 | 51.25 | 100.00 | 72.00 | 91.25 | 77.12 | 18.82 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 14 | C2 C1 CP1 FC4 C4 Pz CP2 CPz C3 CP3 CP4 C5 |
| CSP | 14 | C4 C2 CP2 Pz CP1 CPz C1 C3 C6 CP6 FC6 CP3 |
| RLF | 14 | CP2 C2 CP1 CP3 CP6 CP4 C4 CP5 Cz C6 FC3 FC1 |
| MI | 14 | CP4 C4 FC6 Cz CP1 FCz C1 CP3 FC2 C2 FC3 CP5 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 12 | C4 C2 C1 C3 CP3 CP1 CP2 Pz CP6 CP4 CP5 CPz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 14 | +0.23 | 8.18 | [-3.64, +4.10] | 0.0102 | yes | -12.50 | 3 | 5,6,8,10 |
| CSP | 14 | -0.64 | 5.80 | [-3.39, +2.10] | 0.0021 | yes | -18.75 | 1 | 4,5,6,8,10 |
| RLF | 14 | -4.41 | 8.03 | [-8.21, -0.61] | 0.2359 | no | -24.24 | 5 | 10 |
| MI | 14 | -4.98 | 10.19 | [-9.81, -0.16] | 0.3575 | no | -27.27 | 5 | 10 |
| Fixed sensorimotor | 14 | -3.57 | 6.77 | [-6.77, -0.37] | 0.1009 | no | -21.21 | 3 | 8,10 |
| ERD-only | 14 | -1.47 | 7.65 | [-5.09, +2.15] | 0.0226 | yes | -18.75 | 3 | 6,8,10 |
| Random (mean of draws) | 14 | -3.95 | 7.43 | [-7.46, -0.43] | 0.1600 | no | -21.52 | 4 | 8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 14 | +0.87 | [-3.49, +5.24] | 0.6725 | 1.0000 | 0.8886 | 1.0000 | 7 / 1 / 6 |
| RLF | 14 | +4.64 | [-0.20, +9.48] | 0.0587 | 0.2935 | 0.0751 | 0.3754 | 8 / 3 / 3 |
| MI | 14 | +5.21 | [-0.69, +11.12] | 0.0787 | 0.3148 | 0.0829 | 0.3754 | 7 / 4 / 3 |
| Fixed sensorimotor | 14 | +3.80 | [-1.00, +8.60] | 0.1109 | 0.3326 | 0.1186 | 0.3754 | 7 / 3 / 4 |
| ERD-only | 14 | +1.70 | [-3.91, +7.31] | 0.5237 | 1.0000 | 0.5298 | 1.0000 | 7 / 2 / 5 |
| Random (mean of draws) | 14 | +4.18 | [+0.18, +8.17] | 0.0418 | 0.2505 | 0.0414 | 0.2483 | 9 / 2 / 3 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 82.34 | left_hand 84.5, right_hand 80.2 | left_hand 52.1, right_hand 47.9 | 50.00 | 1 / 14 |
| Ours | 83.40 | left_hand 84.3, right_hand 82.5 | left_hand 50.9, right_hand 49.1 | 50.00 | 1 / 14 |
| CSP | 82.82 | left_hand 87.1, right_hand 78.5 | left_hand 54.3, right_hand 45.7 | 50.00 | 1 / 14 |
| RLF | 78.31 | left_hand 77.7, right_hand 78.9 | left_hand 49.4, right_hand 50.6 | 50.00 | 0 / 14 |
| MI | 78.12 | left_hand 77.2, right_hand 79.1 | left_hand 49.0, right_hand 51.0 | 50.00 | 0 / 14 |
| Fixed sensorimotor | 79.56 | left_hand 83.3, right_hand 75.8 | left_hand 53.7, right_hand 46.3 | 50.00 | 0 / 14 |
| ERD-only | 82.05 | left_hand 82.0, right_hand 82.1 | left_hand 49.9, right_hand 50.1 | 50.00 | 0 / 14 |
| Random (mean of draws) | 79.08 | left_hand 79.8, right_hand 78.3 | left_hand 50.8, right_hand 49.2 | 50.00 | 1 / 70 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 77.41, 43: 79.28, 44: 77.37 | 78.02 | 1.09 |
| Ours | 3 | 42: 77.98, 43: 77.51, 44: 76.28 | 77.26 | 0.88 |
| CSP | 3 | 42: 80.57, 43: 78.15, 44: 78.02 | 78.92 | 1.43 |
| RLF | 3 | 42: 74.66, 43: 73.68, 44: 74.25 | 74.20 | 0.49 |
| MI | 3 | 42: 77.01, 43: 73.03, 44: 73.46 | 74.50 | 2.18 |
| Fixed sensorimotor | 3 | 42: 74.72, 43: 76.40, 44: 75.73 | 75.61 | 0.84 |
| ERD-only | 3 | 42: 78.94, 43: 79.04, 44: 75.93 | 77.97 | 1.77 |
| Random (mean of draws) | 3 | 42: 75.26, 43: 74.79, 44: 74.73 | 74.93 | 0.29 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.282 | [0.116, 0.432] | 0.276 | 14 |
| grad | class_specific_pearson | 0.235 | [0.083, 0.376] | 0.252 | 14 |
| grad | lateralisation_pearson | -0.009 | [-0.140, 0.123] | 0.216 | 14 |
| grad | magnitude_spearman | 0.289 | [0.134, 0.430] | 0.260 | 14 |
| grad | class_specific_spearman | 0.200 | [0.069, 0.325] | 0.219 | 14 |
| grad | lateralisation_spearman | -0.043 | [-0.175, 0.090] | 0.219 | 14 |
| grad | legacy_signed_LR_pearson | 0.097 | [-0.070, 0.258] | 0.269 | 14 |
| grad_x_input | magnitude_pearson | 0.095 | [-0.097, 0.281] | 0.308 | 14 |
| grad_x_input | class_specific_pearson | -0.000 | [-0.161, 0.161] | 0.263 | 14 |
| grad_x_input | lateralisation_pearson | -0.235 | [-0.412, -0.042] | 0.288 | 14 |
| grad_x_input | magnitude_spearman | 0.126 | [-0.076, 0.318] | 0.317 | 14 |
| grad_x_input | class_specific_spearman | -0.009 | [-0.136, 0.118] | 0.211 | 14 |
| grad_x_input | lateralisation_spearman | -0.282 | [-0.489, -0.044] | 0.323 | 14 |
| grad_x_input | legacy_signed_LR_pearson | 0.047 | [-0.127, 0.218] | 0.273 | 14 |
| integrated_gradients | magnitude_pearson | 0.087 | [-0.114, 0.281] | 0.320 | 14 |
| integrated_gradients | class_specific_pearson | -0.023 | [-0.189, 0.144] | 0.272 | 14 |
| integrated_gradients | lateralisation_pearson | -0.210 | [-0.380, -0.026] | 0.281 | 14 |
| integrated_gradients | magnitude_spearman | 0.110 | [-0.096, 0.307] | 0.325 | 14 |
| integrated_gradients | class_specific_spearman | -0.043 | [-0.170, 0.084] | 0.211 | 14 |
| integrated_gradients | lateralisation_spearman | -0.267 | [-0.460, -0.050] | 0.307 | 14 |
| integrated_gradients | legacy_signed_LR_pearson | 0.025 | [-0.142, 0.190] | 0.260 | 14 |
| smoothgrad | magnitude_pearson | 0.281 | [0.116, 0.432] | 0.276 | 14 |
| smoothgrad | class_specific_pearson | 0.235 | [0.084, 0.377] | 0.252 | 14 |
| smoothgrad | lateralisation_pearson | -0.008 | [-0.140, 0.124] | 0.217 | 14 |
| smoothgrad | magnitude_spearman | 0.289 | [0.134, 0.430] | 0.259 | 14 |
| smoothgrad | class_specific_spearman | 0.200 | [0.068, 0.326] | 0.220 | 14 |
| smoothgrad | lateralisation_spearman | -0.042 | [-0.174, 0.092] | 0.220 | 14 |
| smoothgrad | legacy_signed_LR_pearson | 0.097 | [-0.070, 0.258] | 0.269 | 14 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.225 | [-0.376, -0.062] | 14 |
| grad | ERD_L_spearman | -0.190 | [-0.322, -0.050] | 14 |
| grad | ERD_R_pearson | -0.254 | [-0.412, -0.081] | 14 |
| grad | ERD_R_spearman | -0.205 | [-0.340, -0.062] | 14 |
| grad | ERD_LR_pearson | 0.097 | [-0.070, 0.258] | 14 |
| grad | ERD_LR_spearman | 0.061 | [-0.088, 0.206] | 14 |
| grad | ERD_comb_pearson | -0.247 | [-0.404, -0.075] | 14 |
| grad | ERD_comb_spearman | -0.197 | [-0.337, -0.048] | 14 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |
| grad_x_input | ERD_L_pearson | 0.012 | [-0.157, 0.180] | 14 |
| grad_x_input | ERD_L_spearman | 0.021 | [-0.116, 0.157] | 14 |
| grad_x_input | ERD_R_pearson | -0.017 | [-0.218, 0.185] | 14 |
| grad_x_input | ERD_R_spearman | -0.006 | [-0.156, 0.143] | 14 |
| grad_x_input | ERD_LR_pearson | 0.047 | [-0.127, 0.218] | 14 |
| grad_x_input | ERD_LR_spearman | 0.039 | [-0.082, 0.158] | 14 |
| grad_x_input | ERD_comb_pearson | -0.005 | [-0.202, 0.192] | 14 |
| grad_x_input | ERD_comb_spearman | 0.008 | [-0.144, 0.160] | 14 |
| grad_x_input | vs_grad_pearson | 0.968 | [0.940, 0.983] | 14 |
| grad_x_input | vs_grad_spearman | 0.959 | [0.920, 0.979] | 14 |
| integrated_gradients | ERD_L_pearson | 0.035 | [-0.139, 0.207] | 14 |
| integrated_gradients | ERD_L_spearman | 0.042 | [-0.099, 0.181] | 14 |
| integrated_gradients | ERD_R_pearson | 0.005 | [-0.205, 0.216] | 14 |
| integrated_gradients | ERD_R_spearman | 0.017 | [-0.131, 0.164] | 14 |
| integrated_gradients | ERD_LR_pearson | 0.025 | [-0.142, 0.190] | 14 |
| integrated_gradients | ERD_LR_spearman | 0.025 | [-0.078, 0.127] | 14 |
| integrated_gradients | ERD_comb_pearson | 0.016 | [-0.192, 0.222] | 14 |
| integrated_gradients | ERD_comb_spearman | 0.030 | [-0.122, 0.181] | 14 |
| integrated_gradients | vs_grad_pearson | 0.946 | [0.915, 0.966] | 14 |
| integrated_gradients | vs_grad_spearman | 0.936 | [0.891, 0.962] | 14 |
| smoothgrad | ERD_L_pearson | -0.225 | [-0.376, -0.063] | 14 |
| smoothgrad | ERD_L_spearman | -0.190 | [-0.321, -0.051] | 14 |
| smoothgrad | ERD_R_pearson | -0.254 | [-0.412, -0.082] | 14 |
| smoothgrad | ERD_R_spearman | -0.206 | [-0.341, -0.064] | 14 |
| smoothgrad | ERD_LR_pearson | 0.097 | [-0.070, 0.258] | 14 |
| smoothgrad | ERD_LR_spearman | 0.063 | [-0.083, 0.207] | 14 |
| smoothgrad | ERD_comb_pearson | -0.247 | [-0.405, -0.076] | 14 |
| smoothgrad | ERD_comb_spearman | -0.197 | [-0.337, -0.049] | 14 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | 0.003 | [-0.149, 0.154] | 0.123 | -0.002 | 0.097 | 14 |
| ERD_theta_R | 0.028 | [-0.177, 0.230] | 0.015 | 0.017 | 0.030 | 14 |
| ERD_theta_mag | 0.084 | [-0.090, 0.253] | 0.084 | 0.069 | 0.069 | 14 |
| ERD_mu_L | -0.064 | [-0.230, 0.106] | 0.094 | -0.038 | 0.061 | 14 |
| ERD_mu_R | -0.092 | [-0.283, 0.106] | 0.107 | -0.043 | 0.068 | 14 |
| ERD_mu_mag | 0.105 | [-0.058, 0.263] | 0.105 | 0.060 | 0.060 | 14 |
| ERD_beta_L | -0.235 | [-0.415, -0.039] | 0.264 | -0.167 | 0.179 | 14 |
| ERD_beta_R | -0.251 | [-0.431, -0.052] | 0.255 | -0.186 | 0.206 | 14 |
| ERD_beta_mag | 0.240 | [0.037, 0.425] | 0.240 | 0.182 | 0.182 | 14 |
| ERS_beta_rebound_L | 0.013 | [-0.148, 0.173] | 0.157 | 0.005 | 0.148 | 14 |
| ERS_beta_rebound_R | -0.073 | [-0.214, 0.070] | 0.164 | -0.098 | 0.149 | 14 |
| ERS_beta_rebound_mag | 0.193 | [0.068, 0.312] | 0.193 | 0.177 | 0.177 | 14 |
| ERD_lowgamma_L | -0.250 | [-0.405, -0.082] | 0.197 | -0.187 | 0.151 | 14 |
| ERD_lowgamma_R | -0.298 | [-0.425, -0.159] | 0.159 | -0.242 | 0.150 | 14 |
| ERD_lowgamma_mag | 0.159 | [-0.052, 0.357] | 0.159 | 0.133 | 0.133 | 14 |
| ERD_highgamma_L | 0.003 | [-0.157, 0.163] | -0.017 | 0.053 | 0.016 | 14 |
| ERD_highgamma_R | 0.075 | [-0.105, 0.251] | -0.083 | 0.072 | -0.030 | 14 |
| ERD_highgamma_mag | -0.073 | [-0.203, 0.060] | -0.073 | -0.004 | -0.004 | 14 |
| logpow_8_30 | 0.130 | [0.001, 0.255] | -0.200 | -0.035 | -0.200 | 14 |
| r2_signed | 0.019 | [-0.127, 0.165] | 0.234 | 0.036 | 0.176 | 14 |
| r2_abs | 0.234 | [0.128, 0.336] | 0.234 | 0.176 | 0.176 | 14 |
| TFR_ERD_dB_L | -0.201 | [-0.381, -0.005] | 0.235 | -0.160 | 0.176 | 14 |
| TFR_ERD_dB_R | -0.219 | [-0.399, -0.022] | 0.214 | -0.154 | 0.171 | 14 |
| TFR_ERD_dB_mag | 0.210 | [0.009, 0.395] | 0.210 | 0.170 | 0.170 | 14 |
| ITC_4_30 | 0.081 | [-0.085, 0.243] | 0.081 | 0.012 | 0.012 | 14 |
| ERP_abs_0.2_0.8 | -0.024 | [-0.161, 0.114] | -0.024 | 0.012 | 0.012 | 14 |
| SlowPot_0.5_3 | -0.041 | [-0.215, 0.136] | 0.055 | -0.020 | 0.068 | 14 |
| ERD_beta_classspecific | — | — | 0.263 [0.109, 0.404] | — | 0.200 | 14 |
| ERD_highgamma_classspecific | — | — | -0.050 [-0.161, 0.062] | — | -0.006 | 14 |
| ERD_lowgamma_classspecific | — | — | 0.178 [0.007, 0.338] | — | 0.157 | 14 |
| ERD_mu_classspecific | — | — | 0.097 [-0.032, 0.223] | — | 0.066 | 14 |
| ERD_theta_classspecific | — | — | 0.061 [-0.070, 0.190] | — | 0.065 | 14 |
| ERS_beta_rebound_classspecific | — | — | 0.163 [0.059, 0.265] | — | 0.144 | 14 |
| TFR_ERD_dB_classspecific | — | — | 0.223 [0.067, 0.368] | — | 0.175 | 14 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +15.22 ± 10.96 | +8.74 ± 6.55 | +4.07 ± 4.52 | 0.0124 | 0.0022 |
| 4 | +20.27 ± 15.93 | +13.13 ± 8.62 | +7.07 ± 7.41 | 0.0132 | 0.0042 |
| 6 | +20.30 ± 13.52 | +17.20 ± 11.61 | +10.15 ± 9.49 | 0.0783 | 0.0058 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.010 | [-0.056, 0.076] | 14 |
| grad_x_input | 0.002 | [-0.058, 0.062] | 14 |
| integrated_gradients | 0.005 | [-0.054, 0.065] | 14 |
| smoothgrad | 0.010 | [-0.056, 0.077] | 14 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=lawhern128, S02=lawhern128, S03=lawhern128, S04=lawhern128, S05=lawhern128, S06=lawhern128, S07=lawhern128, S08=lawhern128, S09=lawhern128, S10=lawhern128, S11=lawhern128, S12=lawhern128, S13=lawhern128, S14=lawhern128

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 77.71 |
| ems | 75.84 |
| lawhern128 | 79.54 |

```
model: "EEGNetv4"
dataset: "Schirrmeister2017"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"original": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": null, "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "ems": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "lawhern128": {"sfreq": 128, "band_hz": [4.0, 38.0], "window_s": [0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}}
checkpoint_rule: "lowest validation loss, early stopping with patience; refit keeps final weights"
folds: "train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs"
seeds: [42, 43, 44]
```

Figures (75 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `2class\figures`.

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 | S13 | S14 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 75.00 | 84.38 | 97.50 | 98.12 | 85.00 | 90.00 | 72.50 | 83.97 | 68.75 | 85.62 | 56.25 | 96.25 | 85.00 | 96.88 | 83.94 | 12.26 |
| Ours | 71.88 | 79.38 | 95.62 | 95.00 | 88.12 | 87.50 | 76.88 | 82.44 | 74.38 | 87.50 | 56.25 | 97.50 | 88.12 | 93.12 | 83.84 | 11.31 |
| CSP | 80.62 | 81.25 | 93.12 | 96.25 | 93.12 | 89.38 | 75.00 | 88.55 | 71.88 | 88.75 | 62.50 | 97.50 | 85.62 | 94.38 | 85.57 | 10.21 |
| RLF | 81.25 | 80.62 | 94.38 | 98.12 | 85.62 | 89.38 | 78.12 | 83.97 | 74.38 | 87.50 | 56.25 | 98.12 | 74.38 | 93.12 | 83.94 | 11.24 |
| MI | 76.25 | 79.38 | 94.38 | 96.25 | 86.25 | 90.00 | 79.38 | 76.34 | 80.62 | 84.38 | 59.38 | 91.88 | 83.75 | 89.38 | 83.40 | 9.46 |
| Fixed sensorimotor | 73.12 | 78.12 | 95.62 | 96.25 | 88.75 | 86.88 | 73.12 | 79.39 | 71.25 | 88.12 | 37.50 | 96.88 | 77.50 | 91.88 | 81.03 | 15.43 |
| ERD-only | 78.12 | 76.88 | 95.00 | 96.25 | 86.88 | 86.25 | 77.50 | 82.44 | 82.50 | 88.12 | 56.25 | 94.38 | 90.00 | 91.88 | 84.46 | 10.40 |
| Random (mean of draws) | 74.75 | 81.50 | 94.38 | 95.62 | 87.62 | 87.25 | 71.25 | 76.03 | 70.50 | 86.00 | 59.38 | 95.38 | 83.25 | 91.62 | 82.47 | 10.86 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 14 | C1 C3 C4 C2 CPz FC2 CP2 FCz FC1 CP1 FC4 Pz |
| CSP | 14 | CP1 CP2 CP3 C3 C1 C2 CPz C4 Pz FCz CP5 C6 |
| RLF | 10 | C2 C1 C4 C3 CP3 CP1 CP2 CP4 Pz CPz CP5 FC1 |
| MI | 11 | C1 C2 C4 CP3 C3 CP1 CP2 CP4 FC4 C6 CPz CP5 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 12 | C4 C2 C1 C3 CP3 CP1 CP2 Pz CP6 CP4 CP5 CPz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 14 | -0.11 | 3.35 | [-1.69, +1.48] | <0.0001 | yes | -5.00 | 0 | 2,3,4,5,6,8,10 |
| CSP | 14 | +1.62 | 3.80 | [-0.18, +3.42] | 0.0004 | yes | -4.37 | 0 | 4,5,6,8,10 |
| RLF | 14 | +0.00 | 4.48 | [-2.12, +2.12] | 0.0001 | yes | -10.62 | 1 | 3,4,5,6,8,10 |
| MI | 14 | -0.55 | 5.35 | [-3.08, +1.99] | 0.0011 | yes | -7.63 | 2 | 4,5,6,8,10 |
| Fixed sensorimotor | 14 | -2.92 | 5.69 | [-5.61, -0.22] | 0.0319 | yes | -18.75 | 3 | 6,8,10 |
| ERD-only | 14 | +0.52 | 5.30 | [-1.99, +3.03] | 0.0010 | yes | -7.50 | 1 | 4,5,6,8,10 |
| Random (mean of draws) | 14 | -1.48 | 3.00 | [-2.90, -0.06] | <0.0001 | yes | -7.94 | 1 | 3,4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 14 | -1.73 | [-3.82, +0.36] | 0.0966 | 0.4829 | 0.2476 | 0.9902 | 4 / 1 / 9 |
| RLF | 14 | -0.11 | [-2.90, +2.68] | 0.9340 | 1.0000 | 0.4752 | 1.0000 | 3 / 4 / 7 |
| MI | 14 | +0.44 | [-1.81, +2.68] | 0.6814 | 1.0000 | 0.6747 | 1.0000 | 7 / 1 / 6 |
| Fixed sensorimotor | 14 | +2.81 | [-0.38, +5.99] | 0.0791 | 0.4745 | 0.0537 | 0.3224 | 9 / 1 / 4 |
| ERD-only | 14 | -0.63 | [-2.43, +1.18] | 0.4669 | 1.0000 | 0.9062 | 1.0000 | 6 / 2 / 6 |
| Random (mean of draws) | 14 | +1.37 | [-0.38, +3.12] | 0.1145 | 0.4829 | 0.1577 | 0.7885 | 10 / 0 / 4 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 85.65 | feet 89.6, left_hand 84.3, rest 88.1, right_hand 80.6 | feet 25.0, left_hand 26.1, rest 24.1, right_hand 24.9 | 25.01 | 0 / 14 |
| Ours | 85.55 | feet 90.8, left_hand 84.8, rest 88.5, right_hand 78.1 | feet 25.1, left_hand 27.0, rest 23.9, right_hand 24.0 | 25.01 | 0 / 14 |
| CSP | 86.94 | feet 91.7, left_hand 86.2, rest 87.7, right_hand 82.1 | feet 25.5, left_hand 26.6, rest 23.6, right_hand 24.3 | 25.01 | 0 / 14 |
| RLF | 85.65 | feet 91.0, left_hand 82.0, rest 87.9, right_hand 81.8 | feet 25.0, left_hand 25.5, rest 24.0, right_hand 25.5 | 25.01 | 0 / 14 |
| MI | 84.97 | feet 89.8, left_hand 84.1, rest 86.4, right_hand 79.7 | feet 25.1, left_hand 26.7, rest 23.5, right_hand 24.7 | 25.01 | 0 / 14 |
| Fixed sensorimotor | 83.73 | feet 89.2, left_hand 79.7, rest 87.5, right_hand 78.5 | feet 24.8, left_hand 25.7, rest 23.8, right_hand 25.8 | 25.01 | 0 / 14 |
| ERD-only | 86.22 | feet 90.2, left_hand 85.0, rest 87.1, right_hand 82.5 | feet 25.3, left_hand 26.3, rest 23.6, right_hand 24.8 | 25.01 | 0 / 14 |
| Random (mean of draws) | 83.98 | feet 89.2, left_hand 81.6, rest 87.9, right_hand 77.1 | feet 24.7, left_hand 26.4, rest 24.2, right_hand 24.7 | 25.01 | 0 / 70 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 82.17, 43: 81.51, 44: 79.17 | 80.95 | 1.58 |
| Ours | 3 | 42: 83.31, 43: 82.41, 44: 80.53 | 82.08 | 1.42 |
| CSP | 3 | 42: 79.44, 43: 84.20, 44: 79.92 | 81.19 | 2.62 |
| RLF | 3 | 42: 81.72, 43: 81.58, 44: 79.48 | 80.93 | 1.25 |
| MI | 3 | 42: 78.26, 43: 78.70, 44: 82.15 | 79.70 | 2.13 |
| Fixed sensorimotor | 3 | 42: 78.89, 43: 78.27, 44: 77.80 | 78.32 | 0.54 |
| ERD-only | 3 | 42: 80.72, 43: 82.58, 44: 80.39 | 81.23 | 1.18 |
| Random (mean of draws) | 3 | 42: 79.84, 43: 78.94, 44: 78.46 | 79.08 | 0.70 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.401 | [0.250, 0.534] | 0.252 | 14 |
| grad | class_specific_pearson | 0.380 | [0.257, 0.491] | 0.200 | 14 |
| grad | lateralisation_pearson | -0.083 | [-0.311, 0.154] | 0.370 | 14 |
| grad | magnitude_spearman | 0.330 | [0.160, 0.481] | 0.277 | 14 |
| grad | class_specific_spearman | 0.356 | [0.239, 0.462] | 0.186 | 14 |
| grad | lateralisation_spearman | -0.056 | [-0.282, 0.175] | 0.367 | 14 |
| grad | legacy_signed_LR_pearson | 0.150 | [0.044, 0.254] | 0.179 | 14 |
| grad_x_input | magnitude_pearson | 0.365 | [0.220, 0.494] | 0.238 | 14 |
| grad_x_input | class_specific_pearson | 0.241 | [0.110, 0.364] | 0.212 | 14 |
| grad_x_input | lateralisation_pearson | -0.189 | [-0.409, 0.051] | 0.374 | 14 |
| grad_x_input | magnitude_spearman | 0.293 | [0.121, 0.447] | 0.272 | 14 |
| grad_x_input | class_specific_spearman | 0.191 | [0.069, 0.307] | 0.199 | 14 |
| grad_x_input | lateralisation_spearman | -0.165 | [-0.383, 0.071] | 0.370 | 14 |
| grad_x_input | legacy_signed_LR_pearson | 0.138 | [0.032, 0.242] | 0.179 | 14 |
| integrated_gradients | magnitude_pearson | 0.344 | [0.202, 0.472] | 0.235 | 14 |
| integrated_gradients | class_specific_pearson | 0.220 | [0.096, 0.336] | 0.199 | 14 |
| integrated_gradients | lateralisation_pearson | -0.174 | [-0.371, 0.039] | 0.343 | 14 |
| integrated_gradients | magnitude_spearman | 0.282 | [0.118, 0.431] | 0.264 | 14 |
| integrated_gradients | class_specific_spearman | 0.176 | [0.046, 0.300] | 0.210 | 14 |
| integrated_gradients | lateralisation_spearman | -0.147 | [-0.347, 0.065] | 0.341 | 14 |
| integrated_gradients | legacy_signed_LR_pearson | 0.135 | [0.025, 0.243] | 0.185 | 14 |
| smoothgrad | magnitude_pearson | 0.402 | [0.251, 0.535] | 0.252 | 14 |
| smoothgrad | class_specific_pearson | 0.381 | [0.258, 0.492] | 0.200 | 14 |
| smoothgrad | lateralisation_pearson | -0.083 | [-0.315, 0.159] | 0.374 | 14 |
| smoothgrad | magnitude_spearman | 0.327 | [0.154, 0.481] | 0.281 | 14 |
| smoothgrad | class_specific_spearman | 0.358 | [0.242, 0.463] | 0.184 | 14 |
| smoothgrad | lateralisation_spearman | -0.054 | [-0.280, 0.178] | 0.367 | 14 |
| smoothgrad | legacy_signed_LR_pearson | 0.150 | [0.044, 0.253] | 0.178 | 14 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.383 | [-0.505, -0.246] | 14 |
| grad | ERD_L_spearman | -0.339 | [-0.454, -0.212] | 14 |
| grad | ERD_R_pearson | -0.427 | [-0.542, -0.297] | 14 |
| grad | ERD_R_spearman | -0.401 | [-0.506, -0.283] | 14 |
| grad | ERD_LR_pearson | 0.150 | [0.044, 0.254] | 14 |
| grad | ERD_LR_spearman | 0.123 | [-0.022, 0.263] | 14 |
| grad | ERD_comb_pearson | -0.417 | [-0.536, -0.283] | 14 |
| grad | ERD_comb_spearman | -0.380 | [-0.488, -0.262] | 14 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |
| grad_x_input | ERD_L_pearson | -0.327 | [-0.452, -0.190] | 14 |
| grad_x_input | ERD_L_spearman | -0.275 | [-0.393, -0.147] | 14 |
| grad_x_input | ERD_R_pearson | -0.372 | [-0.493, -0.236] | 14 |
| grad_x_input | ERD_R_spearman | -0.340 | [-0.454, -0.214] | 14 |
| grad_x_input | ERD_LR_pearson | 0.138 | [0.032, 0.242] | 14 |
| grad_x_input | ERD_LR_spearman | 0.114 | [-0.028, 0.252] | 14 |
| grad_x_input | ERD_comb_pearson | -0.361 | [-0.485, -0.223] | 14 |
| grad_x_input | ERD_comb_spearman | -0.320 | [-0.437, -0.193] | 14 |
| grad_x_input | vs_grad_pearson | 0.996 | [0.993, 0.998] | 14 |
| grad_x_input | vs_grad_spearman | 0.991 | [0.984, 0.995] | 14 |
| integrated_gradients | ERD_L_pearson | -0.309 | [-0.432, -0.175] | 14 |
| integrated_gradients | ERD_L_spearman | -0.255 | [-0.370, -0.134] | 14 |
| integrated_gradients | ERD_R_pearson | -0.358 | [-0.475, -0.228] | 14 |
| integrated_gradients | ERD_R_spearman | -0.328 | [-0.444, -0.201] | 14 |
| integrated_gradients | ERD_LR_pearson | 0.135 | [0.025, 0.243] | 14 |
| integrated_gradients | ERD_LR_spearman | 0.121 | [-0.020, 0.257] | 14 |
| integrated_gradients | ERD_comb_pearson | -0.346 | [-0.467, -0.213] | 14 |
| integrated_gradients | ERD_comb_spearman | -0.303 | [-0.418, -0.179] | 14 |
| integrated_gradients | vs_grad_pearson | 0.989 | [0.983, 0.992] | 14 |
| integrated_gradients | vs_grad_spearman | 0.978 | [0.965, 0.987] | 14 |
| smoothgrad | ERD_L_pearson | -0.383 | [-0.506, -0.246] | 14 |
| smoothgrad | ERD_L_spearman | -0.337 | [-0.453, -0.211] | 14 |
| smoothgrad | ERD_R_pearson | -0.427 | [-0.543, -0.297] | 14 |
| smoothgrad | ERD_R_spearman | -0.403 | [-0.510, -0.283] | 14 |
| smoothgrad | ERD_LR_pearson | 0.150 | [0.044, 0.253] | 14 |
| smoothgrad | ERD_LR_spearman | 0.128 | [-0.018, 0.268] | 14 |
| smoothgrad | ERD_comb_pearson | -0.418 | [-0.537, -0.282] | 14 |
| smoothgrad | ERD_comb_spearman | -0.381 | [-0.489, -0.261] | 14 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 14 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 14 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.100 | [-0.246, 0.051] | 0.200 | -0.053 | 0.209 | 14 |
| ERD_theta_R | -0.095 | [-0.243, 0.057] | 0.080 | -0.068 | 0.094 | 14 |
| ERD_theta_mag | 0.145 | [-0.011, 0.295] | 0.145 | 0.133 | 0.133 | 14 |
| ERD_mu_L | -0.300 | [-0.458, -0.123] | 0.326 | -0.241 | 0.284 | 14 |
| ERD_mu_R | -0.333 | [-0.482, -0.164] | 0.261 | -0.311 | 0.221 | 14 |
| ERD_mu_mag | 0.279 | [0.084, 0.454] | 0.279 | 0.233 | 0.233 | 14 |
| ERD_beta_L | -0.453 | [-0.555, -0.339] | 0.494 | -0.409 | 0.431 | 14 |
| ERD_beta_R | -0.469 | [-0.567, -0.357] | 0.440 | -0.446 | 0.379 | 14 |
| ERD_beta_mag | 0.460 | [0.308, 0.589] | 0.460 | 0.400 | 0.400 | 14 |
| ERS_beta_rebound_L | -0.274 | [-0.428, -0.105] | 0.311 | -0.262 | 0.323 | 14 |
| ERS_beta_rebound_R | -0.293 | [-0.438, -0.134] | 0.307 | -0.282 | 0.280 | 14 |
| ERS_beta_rebound_mag | 0.339 | [0.176, 0.484] | 0.339 | 0.325 | 0.325 | 14 |
| ERD_lowgamma_L | -0.408 | [-0.552, -0.239] | 0.371 | -0.419 | 0.376 | 14 |
| ERD_lowgamma_R | -0.429 | [-0.587, -0.239] | 0.344 | -0.422 | 0.341 | 14 |
| ERD_lowgamma_mag | 0.340 | [0.097, 0.545] | 0.340 | 0.368 | 0.368 | 14 |
| ERD_highgamma_L | 0.018 | [-0.177, 0.212] | 0.168 | 0.041 | 0.208 | 14 |
| ERD_highgamma_R | 0.153 | [-0.039, 0.335] | 0.132 | 0.135 | 0.180 | 14 |
| ERD_highgamma_mag | 0.128 | [-0.045, 0.293] | 0.128 | 0.223 | 0.223 | 14 |
| logpow_8_30 | 0.181 | [0.070, 0.288] | -0.385 | -0.084 | -0.345 | 14 |
| r2_signed | -0.028 | [-0.121, 0.065] | 0.042 | -0.027 | 0.129 | 14 |
| r2_abs | 0.042 | [-0.095, 0.177] | 0.042 | 0.129 | 0.129 | 14 |
| TFR_ERD_dB_L | -0.452 | [-0.554, -0.336] | 0.488 | -0.411 | 0.437 | 14 |
| TFR_ERD_dB_R | -0.477 | [-0.574, -0.367] | 0.428 | -0.428 | 0.367 | 14 |
| TFR_ERD_dB_mag | 0.449 | [0.283, 0.588] | 0.449 | 0.411 | 0.411 | 14 |
| ITC_4_30 | 0.216 | [0.093, 0.333] | 0.216 | 0.102 | 0.102 | 14 |
| ERP_abs_0.2_0.8 | -0.045 | [-0.120, 0.031] | -0.045 | -0.011 | -0.011 | 14 |
| SlowPot_0.5_3 | 0.031 | [-0.070, 0.131] | 0.001 | 0.041 | 0.029 | 14 |
| ERD_beta_classspecific | — | — | 0.400 [0.294, 0.497] | — | 0.329 | 14 |
| ERD_highgamma_classspecific | — | — | 0.170 [0.029, 0.305] | — | 0.213 | 14 |
| ERD_lowgamma_classspecific | — | — | 0.282 [0.095, 0.450] | — | 0.278 | 14 |
| ERD_mu_classspecific | — | — | 0.248 [0.094, 0.391] | — | 0.198 | 14 |
| ERD_theta_classspecific | — | — | 0.157 [0.049, 0.263] | — | 0.153 | 14 |
| ERS_beta_rebound_classspecific | — | — | 0.259 [0.129, 0.380] | — | 0.227 | 14 |
| TFR_ERD_dB_classspecific | — | — | 0.381 [0.263, 0.488] | — | 0.318 | 14 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +30.91 ± 12.93 | +13.47 ± 6.54 | +2.87 ± 3.65 | <0.0001 | <0.0001 |
| 4 | +35.98 ± 12.14 | +24.70 ± 10.91 | +9.55 ± 6.26 | <0.0001 | <0.0001 |
| 6 | +41.56 ± 13.32 | +30.73 ± 11.88 | +17.94 ± 11.70 | <0.0001 | <0.0001 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.036 | [-0.033, 0.105] | 14 |
| grad_x_input | -0.022 | [-0.095, 0.052] | 14 |
| integrated_gradients | -0.002 | [-0.067, 0.064] | 14 |
| smoothgrad | 0.034 | [-0.036, 0.104] | 14 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=lawhern128, S02=lawhern128, S03=lawhern128, S04=lawhern128, S05=lawhern128, S06=lawhern128, S07=lawhern128, S08=lawhern128, S09=lawhern128, S10=lawhern128, S11=lawhern128, S12=lawhern128, S13=lawhern128, S14=lawhern128

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 81.40 |
| ems | 79.69 |
| lawhern128 | 83.67 |

```
model: "EEGNetv4"
dataset: "Schirrmeister2017"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"original": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": null, "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "ems": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}, "lawhern128": {"sfreq": 128, "band_hz": [4.0, 38.0], "window_s": [0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "select_on": "val_acc", "min_select_epoch": 20, "kernel_length": 64}}
checkpoint_rule: "lowest validation loss, early stopping with patience; refit keeps final weights"
folds: "train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs"
seeds: [42, 43, 44]
```

Figures (75 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `4class\figures`.
