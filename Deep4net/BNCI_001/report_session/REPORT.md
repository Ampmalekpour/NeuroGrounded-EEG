# Results report — Deep4net/BNCI_001

Generated 2026-10-17 10:50 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 56.25 | 46.53 | 72.92 | 58.33 | 60.42 | 56.94 | 54.17 | 93.06 | 91.67 | 65.59 | 16.67 |
| Ours | 64.58 | 49.31 | 73.61 | 56.94 | 58.33 | 52.08 | 53.47 | 91.67 | 87.50 | 65.28 | 15.60 |
| CSP | 68.75 | 49.31 | 74.31 | 54.86 | 59.72 | 53.47 | 54.86 | 92.36 | 90.28 | 66.44 | 16.13 |
| RLF | 56.94 | 48.61 | 59.72 | 57.64 | 60.42 | 55.56 | 52.78 | 93.75 | 87.50 | 63.66 | 15.78 |
| MI | 55.56 | 52.78 | 59.03 | 55.56 | 60.42 | 55.56 | 54.17 | 88.89 | 87.50 | 63.27 | 14.32 |
| Fixed sensorimotor | 60.42 | 47.22 | 73.61 | 58.33 | 56.25 | 53.47 | 54.86 | 93.06 | 90.97 | 65.35 | 16.69 |
| ERD-only | 52.78 | 49.31 | 69.44 | 60.42 | 61.11 | 54.86 | 59.03 | 92.36 | 87.50 | 65.20 | 15.18 |
| Random (mean of draws) | 58.33 | 49.58 | 72.08 | 58.06 | 59.58 | 53.61 | 54.03 | 89.17 | 85.42 | 64.43 | 14.40 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | CP4 C4 CP3 C6 CP1 CP2 FC2 FC1 P1 Cz FCz C5 |
| CSP | 9 | C3 CP4 C4 CP3 CP2 C5 CP1 P2 FC3 C2 FCz FC1 |
| RLF | 9 | CP4 Fz CP2 CPz C2 P1 Cz FCz C6 CP1 FC1 P2 |
| MI | 9 | CP4 P2 C2 P1 Cz POz Fz C4 C1 CPz C5 Pz |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -0.31 | 3.98 | [-2.77, +2.16] | 0.0013 | yes | -4.86 | 0 | 3,4,5,6,8,10 |
| CSP | 9 | +0.85 | 4.83 | [-2.15, +3.84] | 0.0063 | yes | -3.47 | 0 | 4,5,6,8,10 |
| RLF | 9 | -1.93 | 4.58 | [-4.77, +0.91] | 0.0142 | yes | -13.19 | 1 | 5,6,8,10 |
| MI | 9 | -2.31 | 5.35 | [-5.63, +1.00] | 0.0362 | yes | -13.89 | 1 | 6,8,10 |
| Fixed sensorimotor | 9 | -0.23 | 2.46 | [-1.75, +1.29] | <0.0001 | yes | -4.17 | 0 | 2,3,4,5,6,8,10 |
| ERD-only | 9 | -0.39 | 3.18 | [-2.36, +1.59] | 0.0004 | yes | -4.17 | 0 | 3,4,5,6,8,10 |
| Random (mean of draws) | 9 | -1.16 | 2.92 | [-2.97, +0.66] | 0.0005 | yes | -6.25 | 1 | 3,4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | -1.16 | [-2.49, +0.18] | 0.0805 | 0.4831 | 0.1016 | 0.6094 | 1 / 1 / 7 |
| RLF | 9 | +1.62 | [-2.67, +5.92] | 0.4097 | 1.0000 | 0.8750 | 1.0000 | 4 / 1 / 4 |
| MI | 9 | +2.01 | [-2.67, +6.69] | 0.3518 | 1.0000 | 0.7109 | 1.0000 | 4 / 1 / 4 |
| Fixed sensorimotor | 9 | -0.08 | [-1.92, +1.76] | 0.9253 | 1.0000 | 0.9219 | 1.0000 | 3 / 1 / 5 |
| ERD-only | 9 | +0.08 | [-3.91, +4.07] | 0.9655 | 1.0000 | 0.7812 | 1.0000 | 2 / 2 / 5 |
| Random (mean of draws) | 9 | +0.85 | [-1.09, +2.79] | 0.3428 | 1.0000 | 0.4453 | 1.0000 | 4 / 0 / 5 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 65.59 | left_hand 62.8, right_hand 68.4 | left_hand 47.2, right_hand 52.8 | 50.00 | 0 / 9 |
| Ours | 65.28 | left_hand 62.8, right_hand 67.7 | left_hand 47.5, right_hand 52.5 | 50.00 | 0 / 9 |
| CSP | 66.44 | left_hand 62.0, right_hand 70.8 | left_hand 45.6, right_hand 54.4 | 50.00 | 0 / 9 |
| RLF | 63.66 | left_hand 62.0, right_hand 65.3 | left_hand 48.4, right_hand 51.6 | 50.00 | 0 / 9 |
| MI | 63.27 | left_hand 62.2, right_hand 64.4 | left_hand 48.9, right_hand 51.1 | 50.00 | 0 / 9 |
| Fixed sensorimotor | 65.35 | left_hand 62.5, right_hand 68.2 | left_hand 47.1, right_hand 52.9 | 50.00 | 0 / 9 |
| ERD-only | 65.20 | left_hand 63.1, right_hand 67.3 | left_hand 47.9, right_hand 52.1 | 50.00 | 0 / 9 |
| Random (mean of draws) | 64.43 | left_hand 62.0, right_hand 66.9 | left_hand 47.6, right_hand 52.4 | 50.00 | 0 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 63.66, 43: 65.51, 44: 65.97 | 65.05 | 1.22 |
| Ours | 3 | 42: 63.66, 43: 64.04, 44: 62.65 | 63.45 | 0.72 |
| CSP | 3 | 42: 65.05, 43: 64.97, 44: 65.90 | 65.30 | 0.51 |
| RLF | 3 | 42: 65.20, 43: 63.43, 44: 62.42 | 63.68 | 1.41 |
| MI | 3 | 42: 62.81, 43: 62.04, 44: 63.12 | 62.65 | 0.56 |
| Fixed sensorimotor | 3 | 42: 62.50, 43: 65.28, 44: 65.12 | 64.30 | 1.56 |
| ERD-only | 3 | 42: 65.66, 43: 64.12, 44: 64.12 | 64.63 | 0.89 |
| Random (mean of draws) | 3 | 42: 63.18, 43: 62.05, 44: 63.83 | 63.02 | 0.90 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.115 | [-0.075, 0.296] | 0.232 | 9 |
| grad | class_specific_pearson | 0.136 | [-0.060, 0.321] | 0.234 | 9 |
| grad | lateralisation_pearson | -0.156 | [-0.424, 0.136] | 0.333 | 9 |
| grad | magnitude_spearman | 0.027 | [-0.131, 0.183] | 0.199 | 9 |
| grad | class_specific_spearman | 0.108 | [-0.075, 0.284] | 0.218 | 9 |
| grad | lateralisation_spearman | -0.123 | [-0.355, 0.124] | 0.278 | 9 |
| grad | legacy_signed_LR_pearson | -0.005 | [-0.302, 0.292] | 0.370 | 9 |
| grad_x_input | magnitude_pearson | 0.082 | [-0.118, 0.277] | 0.250 | 9 |
| grad_x_input | class_specific_pearson | 0.114 | [-0.111, 0.327] | 0.267 | 9 |
| grad_x_input | lateralisation_pearson | -0.349 | [-0.548, -0.112] | 0.279 | 9 |
| grad_x_input | magnitude_spearman | 0.011 | [-0.147, 0.169] | 0.203 | 9 |
| grad_x_input | class_specific_spearman | 0.071 | [-0.134, 0.271] | 0.249 | 9 |
| grad_x_input | lateralisation_spearman | -0.312 | [-0.518, -0.071] | 0.289 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.014 | [-0.316, 0.290] | 0.375 | 9 |
| integrated_gradients | magnitude_pearson | 0.083 | [-0.114, 0.274] | 0.247 | 9 |
| integrated_gradients | class_specific_pearson | 0.116 | [-0.101, 0.322] | 0.260 | 9 |
| integrated_gradients | lateralisation_pearson | -0.420 | [-0.616, -0.176] | 0.294 | 9 |
| integrated_gradients | magnitude_spearman | 0.009 | [-0.148, 0.165] | 0.200 | 9 |
| integrated_gradients | class_specific_spearman | 0.082 | [-0.113, 0.271] | 0.238 | 9 |
| integrated_gradients | lateralisation_spearman | -0.384 | [-0.594, -0.126] | 0.313 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.016 | [-0.318, 0.290] | 0.376 | 9 |
| smoothgrad | magnitude_pearson | 0.114 | [-0.076, 0.296] | 0.233 | 9 |
| smoothgrad | class_specific_pearson | 0.136 | [-0.059, 0.322] | 0.235 | 9 |
| smoothgrad | lateralisation_pearson | -0.199 | [-0.466, 0.101] | 0.333 | 9 |
| smoothgrad | magnitude_spearman | 0.024 | [-0.132, 0.179] | 0.198 | 9 |
| smoothgrad | class_specific_spearman | 0.113 | [-0.068, 0.287] | 0.216 | 9 |
| smoothgrad | lateralisation_spearman | -0.184 | [-0.414, 0.067] | 0.276 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.005 | [-0.302, 0.293] | 0.371 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.130 | [-0.373, 0.131] | 9 |
| grad | ERD_L_spearman | -0.076 | [-0.344, 0.203] | 9 |
| grad | ERD_R_pearson | -0.144 | [-0.357, 0.083] | 9 |
| grad | ERD_R_spearman | -0.137 | [-0.338, 0.075] | 9 |
| grad | ERD_LR_pearson | -0.005 | [-0.302, 0.292] | 9 |
| grad | ERD_LR_spearman | 0.084 | [-0.165, 0.324] | 9 |
| grad | ERD_comb_pearson | -0.164 | [-0.371, 0.058] | 9 |
| grad | ERD_comb_spearman | -0.161 | [-0.335, 0.023] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.128 | [-0.396, 0.160] | 9 |
| grad_x_input | ERD_L_spearman | -0.071 | [-0.354, 0.224] | 9 |
| grad_x_input | ERD_R_pearson | -0.127 | [-0.368, 0.131] | 9 |
| grad_x_input | ERD_R_spearman | -0.113 | [-0.341, 0.129] | 9 |
| grad_x_input | ERD_LR_pearson | -0.014 | [-0.316, 0.290] | 9 |
| grad_x_input | ERD_LR_spearman | 0.071 | [-0.194, 0.326] | 9 |
| grad_x_input | ERD_comb_pearson | -0.151 | [-0.396, 0.114] | 9 |
| grad_x_input | ERD_comb_spearman | -0.142 | [-0.354, 0.083] | 9 |
| grad_x_input | vs_grad_pearson | 0.993 | [0.988, 0.996] | 9 |
| grad_x_input | vs_grad_spearman | 0.987 | [0.976, 0.993] | 9 |
| integrated_gradients | ERD_L_pearson | -0.131 | [-0.395, 0.153] | 9 |
| integrated_gradients | ERD_L_spearman | -0.075 | [-0.355, 0.216] | 9 |
| integrated_gradients | ERD_R_pearson | -0.127 | [-0.364, 0.124] | 9 |
| integrated_gradients | ERD_R_spearman | -0.123 | [-0.352, 0.119] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.016 | [-0.318, 0.290] | 9 |
| integrated_gradients | ERD_LR_spearman | 0.079 | [-0.184, 0.332] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.153 | [-0.391, 0.104] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.151 | [-0.359, 0.071] | 9 |
| integrated_gradients | vs_grad_pearson | 0.993 | [0.988, 0.996] | 9 |
| integrated_gradients | vs_grad_spearman | 0.987 | [0.976, 0.993] | 9 |
| smoothgrad | ERD_L_pearson | -0.131 | [-0.375, 0.130] | 9 |
| smoothgrad | ERD_L_spearman | -0.081 | [-0.351, 0.200] | 9 |
| smoothgrad | ERD_R_pearson | -0.145 | [-0.359, 0.082] | 9 |
| smoothgrad | ERD_R_spearman | -0.139 | [-0.341, 0.075] | 9 |
| smoothgrad | ERD_LR_pearson | -0.005 | [-0.302, 0.293] | 9 |
| smoothgrad | ERD_LR_spearman | 0.082 | [-0.167, 0.321] | 9 |
| smoothgrad | ERD_comb_pearson | -0.166 | [-0.372, 0.057] | 9 |
| smoothgrad | ERD_comb_spearman | -0.164 | [-0.339, 0.022] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [0.998, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.126 | [-0.431, 0.205] | -0.228 | -0.057 | -0.293 | 9 |
| ERD_theta_R | -0.170 | [-0.494, 0.195] | -0.023 | -0.195 | -0.066 | 9 |
| ERD_theta_mag | -0.088 | [-0.442, 0.289] | -0.088 | -0.157 | -0.157 | 9 |
| ERD_mu_L | -0.129 | [-0.439, 0.208] | -0.028 | -0.088 | -0.057 | 9 |
| ERD_mu_R | -0.087 | [-0.460, 0.313] | 0.327 | -0.059 | 0.243 | 9 |
| ERD_mu_mag | 0.187 | [-0.065, 0.416] | 0.187 | 0.127 | 0.127 | 9 |
| ERD_beta_L | -0.038 | [-0.278, 0.206] | 0.097 | -0.009 | 0.040 | 9 |
| ERD_beta_R | -0.022 | [-0.193, 0.150] | -0.126 | -0.032 | -0.116 | 9 |
| ERD_beta_mag | -0.027 | [-0.266, 0.216] | -0.027 | -0.003 | -0.003 | 9 |
| ERS_beta_rebound_L | -0.139 | [-0.443, 0.194] | -0.119 | -0.157 | -0.152 | 9 |
| ERS_beta_rebound_R | -0.107 | [-0.338, 0.135] | -0.050 | -0.176 | -0.115 | 9 |
| ERS_beta_rebound_mag | -0.150 | [-0.396, 0.115] | -0.150 | -0.191 | -0.191 | 9 |
| ERD_lowgamma_L | -0.202 | [-0.475, 0.106] | -0.132 | -0.149 | -0.189 | 9 |
| ERD_lowgamma_R | -0.079 | [-0.334, 0.187] | 0.027 | -0.019 | 0.140 | 9 |
| ERD_lowgamma_mag | -0.068 | [-0.360, 0.236] | -0.068 | -0.055 | -0.055 | 9 |
| logpow_8_30 | -0.163 | [-0.442, 0.146] | -0.163 | -0.225 | -0.225 | 9 |
| r2_signed | 0.040 | [-0.300, 0.371] | 0.139 | -0.110 | 0.033 | 9 |
| r2_abs | 0.139 | [-0.198, 0.447] | 0.139 | 0.033 | 0.033 | 9 |
| TFR_ERD_dB_L | -0.159 | [-0.422, 0.129] | 0.008 | -0.114 | -0.119 | 9 |
| TFR_ERD_dB_R | -0.147 | [-0.371, 0.093] | 0.096 | -0.166 | 0.034 | 9 |
| TFR_ERD_dB_mag | 0.058 | [-0.168, 0.279] | 0.058 | -0.060 | -0.060 | 9 |
| ITC_4_30 | -0.185 | [-0.447, 0.107] | -0.185 | -0.159 | -0.159 | 9 |
| ERP_abs_0.2_0.8 | -0.047 | [-0.320, 0.233] | -0.047 | -0.101 | -0.101 | 9 |
| SlowPot_0.5_3 | 0.071 | [-0.121, 0.259] | -0.052 | 0.106 | -0.060 | 9 |
| ERD_beta_classspecific | — | — | -0.016 [-0.177, 0.146] | — | -0.042 | 9 |
| ERD_lowgamma_classspecific | — | — | -0.053 [-0.271, 0.169] | — | -0.027 | 9 |
| ERD_mu_classspecific | — | — | 0.142 [-0.060, 0.332] | — | 0.084 | 9 |
| ERD_theta_classspecific | — | — | -0.126 [-0.420, 0.192] | — | -0.178 | 9 |
| ERS_beta_rebound_classspecific | — | — | -0.079 [-0.232, 0.079] | — | -0.118 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.050 [-0.118, 0.215] | — | -0.045 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +6.20 ± 10.61 | +0.99 ± 1.68 | +0.08 ± 1.10 | 0.1266 | 0.1118 |
| 4 | +8.46 ± 13.97 | +1.62 ± 3.27 | -0.21 ± 1.39 | 0.1041 | 0.0857 |
| 6 | +8.85 ± 14.64 | +3.17 ± 6.41 | +0.31 ± 2.37 | 0.1057 | 0.0836 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.006 | [-0.045, 0.056] | 9 |
| grad_x_input | 0.048 | [-0.017, 0.112] | 9 |
| integrated_gradients | 0.042 | [-0.030, 0.114] | 9 |
| smoothgrad | 0.011 | [-0.045, 0.067] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: False): S01=original, S02=original, S03=original, S04=original, S05=original, S06=original, S07=original, S08=original, S09=cue_0_4

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 57.41 |
| cue_0_4 | 56.17 |

```
model: "Deep4Net"
dataset: "BNCI2014_001"
sfreq_hz: 250
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "cue_0_4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [0.0, 4.0], "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "no_aug": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": false, "mixup_p": 0.0, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}}
checkpoint_rule: "SWA average from swa_start_frac*max_epochs + BN update (no early stopping); validation block only scored, never used for checkpointing"
folds: "train on session 1 (last 25% of it = validation, chronological), refit on all of session 1, test once on session 2 (official BCI IV-2a split)"
seeds: [42, 43, 44]
```

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 62.15 | 25.00 | 54.17 | 42.01 | 26.04 | 24.31 | 31.94 | 65.97 | 74.65 | 45.14 | 19.57 |
| Ours | 61.81 | 23.61 | 55.56 | 38.89 | 26.39 | 26.39 | 38.89 | 63.54 | 72.92 | 45.33 | 18.51 |
| CSP | 58.68 | 24.65 | 52.08 | 40.62 | 26.39 | 26.04 | 34.72 | 63.54 | 72.57 | 44.37 | 17.97 |
| RLF | 60.07 | 24.31 | 54.51 | 42.01 | 28.47 | 23.96 | 35.07 | 63.89 | 72.57 | 44.98 | 18.32 |
| MI | 56.60 | 22.92 | 53.47 | 38.54 | 30.56 | 23.26 | 37.85 | 66.67 | 72.92 | 44.75 | 18.43 |
| Fixed sensorimotor | 65.28 | 23.61 | 51.39 | 26.04 | 27.43 | 24.31 | 41.67 | 67.01 | 71.53 | 44.25 | 20.00 |
| ERD-only | 61.46 | 22.57 | 53.47 | 40.62 | 28.47 | 26.74 | 31.25 | 66.32 | 73.26 | 44.91 | 19.09 |
| Random (mean of draws) | 59.38 | 24.17 | 54.79 | 36.46 | 27.01 | 24.44 | 36.74 | 60.56 | 70.49 | 43.78 | 17.68 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | CP2 CP3 CPz CP4 P2 CP1 Pz C4 P1 C3 C1 FC2 |
| CSP | 9 | P1 P2 Pz CP3 CPz CP1 CP4 CP2 POz C3 C4 C2 |
| RLF | 9 | C3 CP3 CP1 P2 CPz P1 Fz CP4 POz CP2 C5 Pz |
| MI | 9 | P2 CP1 CP2 P1 CP4 CPz C3 Pz CP3 C6 POz C5 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | +0.19 | 3.06 | [-1.71, +2.09] | 0.0002 | yes | -3.12 | 0 | 3,4,5,6,8,10 |
| CSP | 9 | -0.77 | 2.07 | [-2.05, +0.51] | <0.0001 | yes | -3.47 | 0 | 3,4,5,6,8,10 |
| RLF | 9 | -0.15 | 1.91 | [-1.34, +1.03] | <0.0001 | yes | -2.08 | 0 | 2,3,4,5,6,8,10 |
| MI | 9 | -0.39 | 3.64 | [-2.64, +1.87] | 0.0009 | yes | -5.56 | 0 | 3,4,5,6,8,10 |
| Fixed sensorimotor | 9 | -0.89 | 6.84 | [-5.13, +3.35] | 0.0277 | yes | -15.97 | 1 | 6,8,10 |
| ERD-only | 9 | -0.23 | 1.68 | [-1.27, +0.81] | <0.0001 | yes | -2.43 | 0 | 2,3,4,5,6,8,10 |
| Random (mean of draws) | 9 | -1.36 | 3.42 | [-3.48, +0.76] | 0.0018 | yes | -5.56 | 0 | 4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | +0.96 | [-0.65, +2.58] | 0.2048 | 1.0000 | 0.2812 | 1.0000 | 5 / 2 / 2 |
| RLF | 9 | +0.35 | [-1.34, +2.03] | 0.6469 | 1.0000 | 0.6719 | 1.0000 | 5 / 0 / 4 |
| MI | 9 | +0.58 | [-1.65, +2.80] | 0.5653 | 1.0000 | 0.4844 | 1.0000 | 6 / 1 / 2 |
| Fixed sensorimotor | 9 | +1.08 | [-2.87, +5.03] | 0.5457 | 1.0000 | 0.8125 | 1.0000 | 4 / 1 / 4 |
| ERD-only | 9 | +0.42 | [-1.97, +2.82] | 0.6934 | 1.0000 | 1.0000 | 1.0000 | 4 / 0 / 5 |
| Random (mean of draws) | 9 | +1.55 | [+0.51, +2.59] | 0.0089 | 0.0534 | 0.0195 | 0.1172 | 7 / 0 / 2 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 45.14 | feet 36.1, left_hand 47.5, right_hand 56.6, tongue 40.3 | feet 22.1, left_hand 25.6, right_hand 31.4, tongue 20.9 | 25.00 | 0 / 9 |
| Ours | 45.33 | feet 35.8, left_hand 49.1, right_hand 56.6, tongue 39.8 | feet 21.5, left_hand 26.1, right_hand 32.2, tongue 20.2 | 25.00 | 0 / 9 |
| CSP | 44.37 | feet 36.1, left_hand 47.4, right_hand 55.9, tongue 38.1 | feet 22.5, left_hand 25.9, right_hand 31.9, tongue 19.8 | 25.00 | 0 / 9 |
| RLF | 44.98 | feet 36.3, left_hand 47.7, right_hand 55.1, tongue 40.9 | feet 22.0, left_hand 26.9, right_hand 30.6, tongue 20.5 | 25.00 | 0 / 9 |
| MI | 44.75 | feet 35.8, left_hand 48.1, right_hand 54.8, tongue 40.3 | feet 21.9, left_hand 26.5, right_hand 31.1, tongue 20.5 | 25.00 | 0 / 9 |
| Fixed sensorimotor | 44.25 | feet 36.7, left_hand 46.3, right_hand 56.3, tongue 37.7 | feet 23.5, left_hand 26.5, right_hand 30.4, tongue 19.6 | 25.00 | 0 / 9 |
| ERD-only | 44.91 | feet 36.4, left_hand 48.8, right_hand 56.2, tongue 38.3 | feet 22.0, left_hand 26.3, right_hand 31.9, tongue 19.8 | 25.00 | 0 / 9 |
| Random (mean of draws) | 43.78 | feet 36.0, left_hand 46.7, right_hand 53.6, tongue 38.8 | feet 22.6, left_hand 26.8, right_hand 30.7, tongue 19.9 | 25.00 | 0 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 44.21, 43: 44.64, 44: 44.17 | 44.34 | 0.26 |
| Ours | 3 | 42: 43.21, 43: 46.06, 44: 44.83 | 44.70 | 1.43 |
| CSP | 3 | 42: 42.40, 43: 45.56, 44: 43.71 | 43.89 | 1.59 |
| RLF | 3 | 42: 43.75, 43: 43.75, 44: 45.06 | 44.19 | 0.76 |
| MI | 3 | 42: 43.75, 43: 44.41, 44: 44.52 | 44.23 | 0.42 |
| Fixed sensorimotor | 3 | 42: 43.21, 43: 42.36, 44: 44.44 | 43.34 | 1.05 |
| ERD-only | 3 | 42: 43.06, 43: 45.87, 44: 43.67 | 44.20 | 1.48 |
| Random (mean of draws) | 3 | 42: 42.51, 43: 42.54, 44: 43.50 | 42.85 | 0.57 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.308 | [-0.044, 0.593] | 0.386 | 9 |
| grad | class_specific_pearson | 0.111 | [-0.201, 0.403] | 0.370 | 9 |
| grad | lateralisation_pearson | 0.447 | [0.113, 0.691] | 0.352 | 9 |
| grad | magnitude_spearman | 0.254 | [-0.092, 0.545] | 0.348 | 9 |
| grad | class_specific_spearman | 0.105 | [-0.232, 0.418] | 0.378 | 9 |
| grad | lateralisation_spearman | 0.432 | [0.105, 0.674] | 0.332 | 9 |
| grad | legacy_signed_LR_pearson | 0.053 | [-0.364, 0.453] | 0.481 | 9 |
| grad_x_input | magnitude_pearson | 0.293 | [-0.064, 0.584] | 0.392 | 9 |
| grad_x_input | class_specific_pearson | 0.108 | [-0.209, 0.404] | 0.376 | 9 |
| grad_x_input | lateralisation_pearson | 0.262 | [-0.107, 0.568] | 0.400 | 9 |
| grad_x_input | magnitude_spearman | 0.244 | [-0.110, 0.543] | 0.356 | 9 |
| grad_x_input | class_specific_spearman | 0.083 | [-0.242, 0.392] | 0.377 | 9 |
| grad_x_input | lateralisation_spearman | 0.261 | [-0.111, 0.570] | 0.390 | 9 |
| grad_x_input | legacy_signed_LR_pearson | 0.041 | [-0.384, 0.451] | 0.490 | 9 |
| integrated_gradients | magnitude_pearson | 0.292 | [-0.046, 0.570] | 0.380 | 9 |
| integrated_gradients | class_specific_pearson | 0.090 | [-0.210, 0.375] | 0.364 | 9 |
| integrated_gradients | lateralisation_pearson | 0.018 | [-0.292, 0.324] | 0.365 | 9 |
| integrated_gradients | magnitude_spearman | 0.239 | [-0.111, 0.536] | 0.350 | 9 |
| integrated_gradients | class_specific_spearman | 0.076 | [-0.235, 0.373] | 0.365 | 9 |
| integrated_gradients | lateralisation_spearman | 0.063 | [-0.275, 0.387] | 0.381 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | 0.052 | [-0.389, 0.474] | 0.503 | 9 |
| smoothgrad | magnitude_pearson | 0.308 | [-0.044, 0.592] | 0.386 | 9 |
| smoothgrad | class_specific_pearson | 0.110 | [-0.202, 0.402] | 0.370 | 9 |
| smoothgrad | lateralisation_pearson | 0.446 | [0.103, 0.694] | 0.359 | 9 |
| smoothgrad | magnitude_spearman | 0.252 | [-0.090, 0.540] | 0.346 | 9 |
| smoothgrad | class_specific_spearman | 0.104 | [-0.234, 0.420] | 0.379 | 9 |
| smoothgrad | lateralisation_spearman | 0.440 | [0.089, 0.693] | 0.356 | 9 |
| smoothgrad | legacy_signed_LR_pearson | 0.052 | [-0.364, 0.451] | 0.480 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.089 | [-0.423, 0.266] | 9 |
| grad | ERD_L_spearman | -0.062 | [-0.408, 0.300] | 9 |
| grad | ERD_R_pearson | -0.135 | [-0.488, 0.256] | 9 |
| grad | ERD_R_spearman | -0.094 | [-0.467, 0.307] | 9 |
| grad | ERD_LR_pearson | 0.053 | [-0.364, 0.453] | 9 |
| grad | ERD_LR_spearman | 0.067 | [-0.304, 0.420] | 9 |
| grad | ERD_comb_pearson | -0.112 | [-0.481, 0.290] | 9 |
| grad | ERD_comb_spearman | -0.092 | [-0.488, 0.334] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.108 | [-0.446, 0.257] | 9 |
| grad_x_input | ERD_L_spearman | -0.066 | [-0.418, 0.302] | 9 |
| grad_x_input | ERD_R_pearson | -0.136 | [-0.489, 0.255] | 9 |
| grad_x_input | ERD_R_spearman | -0.087 | [-0.470, 0.325] | 9 |
| grad_x_input | ERD_LR_pearson | 0.041 | [-0.384, 0.451] | 9 |
| grad_x_input | ERD_LR_spearman | 0.065 | [-0.317, 0.429] | 9 |
| grad_x_input | ERD_comb_pearson | -0.124 | [-0.492, 0.281] | 9 |
| grad_x_input | ERD_comb_spearman | -0.092 | [-0.491, 0.339] | 9 |
| grad_x_input | vs_grad_pearson | 0.999 | [0.997, 0.999] | 9 |
| grad_x_input | vs_grad_spearman | 0.997 | [0.995, 0.999] | 9 |
| integrated_gradients | ERD_L_pearson | -0.092 | [-0.424, 0.262] | 9 |
| integrated_gradients | ERD_L_spearman | -0.045 | [-0.406, 0.328] | 9 |
| integrated_gradients | ERD_R_pearson | -0.137 | [-0.494, 0.260] | 9 |
| integrated_gradients | ERD_R_spearman | -0.100 | [-0.481, 0.313] | 9 |
| integrated_gradients | ERD_LR_pearson | 0.052 | [-0.389, 0.474] | 9 |
| integrated_gradients | ERD_LR_spearman | 0.101 | [-0.301, 0.473] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.113 | [-0.474, 0.281] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.085 | [-0.484, 0.343] | 9 |
| integrated_gradients | vs_grad_pearson | 0.997 | [0.993, 0.999] | 9 |
| integrated_gradients | vs_grad_spearman | 0.993 | [0.981, 0.997] | 9 |
| smoothgrad | ERD_L_pearson | -0.089 | [-0.423, 0.266] | 9 |
| smoothgrad | ERD_L_spearman | -0.060 | [-0.405, 0.300] | 9 |
| smoothgrad | ERD_R_pearson | -0.134 | [-0.487, 0.256] | 9 |
| smoothgrad | ERD_R_spearman | -0.093 | [-0.465, 0.307] | 9 |
| smoothgrad | ERD_LR_pearson | 0.052 | [-0.364, 0.451] | 9 |
| smoothgrad | ERD_LR_spearman | 0.068 | [-0.303, 0.422] | 9 |
| smoothgrad | ERD_comb_pearson | -0.112 | [-0.480, 0.290] | 9 |
| smoothgrad | ERD_comb_spearman | -0.089 | [-0.481, 0.333] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.074 | [-0.268, 0.126] | 0.019 | -0.111 | 0.068 | 9 |
| ERD_theta_R | -0.333 | [-0.442, -0.214] | 0.111 | -0.334 | 0.127 | 9 |
| ERD_theta_mag | 0.105 | [-0.142, 0.340] | 0.105 | 0.123 | 0.123 | 9 |
| ERD_mu_L | -0.137 | [-0.419, 0.168] | 0.262 | -0.106 | 0.255 | 9 |
| ERD_mu_R | -0.247 | [-0.538, 0.097] | 0.360 | -0.183 | 0.347 | 9 |
| ERD_mu_mag | 0.427 | [0.199, 0.611] | 0.427 | 0.405 | 0.405 | 9 |
| ERD_beta_L | 0.027 | [-0.359, 0.405] | 0.043 | 0.094 | 0.009 | 9 |
| ERD_beta_R | 0.057 | [-0.310, 0.410] | 0.087 | 0.163 | 0.077 | 9 |
| ERD_beta_mag | 0.109 | [-0.375, 0.546] | 0.109 | 0.087 | 0.087 | 9 |
| ERS_beta_rebound_L | -0.171 | [-0.495, 0.195] | 0.069 | -0.185 | 0.054 | 9 |
| ERS_beta_rebound_R | 0.006 | [-0.193, 0.205] | -0.071 | 0.026 | -0.029 | 9 |
| ERS_beta_rebound_mag | -0.046 | [-0.375, 0.293] | -0.046 | -0.008 | -0.008 | 9 |
| ERD_lowgamma_L | -0.089 | [-0.355, 0.190] | -0.011 | -0.036 | -0.039 | 9 |
| ERD_lowgamma_R | -0.003 | [-0.371, 0.365] | -0.075 | 0.030 | -0.004 | 9 |
| ERD_lowgamma_mag | -0.065 | [-0.384, 0.267] | -0.065 | -0.019 | -0.019 | 9 |
| logpow_8_30 | 0.128 | [-0.091, 0.336] | 0.128 | 0.128 | 0.128 | 9 |
| r2_signed | 0.002 | [-0.430, 0.433] | 0.408 | 0.060 | 0.290 | 9 |
| r2_abs | 0.408 | [0.047, 0.675] | 0.408 | 0.290 | 0.290 | 9 |
| TFR_ERD_dB_L | -0.063 | [-0.416, 0.307] | 0.034 | 0.012 | -0.053 | 9 |
| TFR_ERD_dB_R | -0.149 | [-0.511, 0.258] | 0.241 | -0.059 | 0.190 | 9 |
| TFR_ERD_dB_mag | 0.233 | [-0.199, 0.589] | 0.233 | 0.131 | 0.131 | 9 |
| ITC_4_30 | 0.112 | [-0.158, 0.367] | 0.112 | 0.079 | 0.079 | 9 |
| ERP_abs_0.2_0.8 | 0.256 | [-0.078, 0.538] | 0.256 | 0.245 | 0.245 | 9 |
| SlowPot_0.5_3 | -0.218 | [-0.519, 0.132] | 0.032 | -0.290 | 0.129 | 9 |
| ERD_beta_classspecific | — | — | 0.045 [-0.337, 0.414] | — | 0.027 | 9 |
| ERD_lowgamma_classspecific | — | — | -0.041 [-0.237, 0.159] | — | -0.013 | 9 |
| ERD_mu_classspecific | — | — | 0.297 [0.123, 0.453] | — | 0.297 | 9 |
| ERD_theta_classspecific | — | — | 0.049 [-0.122, 0.217] | — | 0.088 | 9 |
| ERS_beta_rebound_classspecific | — | — | -0.004 [-0.260, 0.254] | — | 0.015 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.131 [-0.208, 0.442] | — | 0.072 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +8.53 ± 9.67 | +2.35 ± 2.29 | +0.23 ± 0.71 | 0.0433 | 0.0355 |
| 4 | +12.78 ± 13.12 | +4.37 ± 4.58 | +1.52 ± 1.62 | 0.0208 | 0.0208 |
| 6 | +15.26 ± 16.21 | +6.64 ± 7.30 | +2.70 ± 3.04 | 0.0217 | 0.0242 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.033 | [-0.139, 0.074] | 9 |
| grad_x_input | -0.062 | [-0.174, 0.052] | 9 |
| integrated_gradients | -0.056 | [-0.165, 0.054] | 9 |
| smoothgrad | -0.031 | [-0.130, 0.068] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: False): S01=original, S02=cue_0_4, S03=original, S04=cue_0_4, S05=cue_0_4, S06=cue_0_4, S07=cue_0_4, S08=cue_0_4, S09=original

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 40.90 |
| cue_0_4 | 41.20 |

```
model: "Deep4Net"
dataset: "BNCI2014_001"
sfreq_hz: 250
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "cue_0_4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [0.0, 4.0], "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "no_aug": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": false, "mixup_p": 0.0, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}}
checkpoint_rule: "SWA average from swa_start_frac*max_epochs + BN update (no early stopping); validation block only scored, never used for checkpointing"
folds: "train on session 1 (last 25% of it = validation, chronological), refit on all of session 1, test once on session 2 (official BCI IV-2a split)"
seeds: [42, 43, 44]
```
