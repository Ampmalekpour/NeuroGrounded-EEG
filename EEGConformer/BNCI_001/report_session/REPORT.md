# Results report — EEGConformer/BNCI_001

Generated 2026-10-17 10:50 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 86.11 | 59.03 | 93.75 | 75.00 | 69.44 | 60.42 | 79.86 | 95.83 | 88.89 | 78.70 | 13.70 |
| Ours | 82.64 | 48.61 | 91.67 | 76.39 | 67.36 | 63.89 | 75.00 | 95.83 | 86.11 | 76.39 | 14.82 |
| CSP | 86.81 | 52.08 | 93.75 | 70.83 | 59.72 | 66.67 | 77.78 | 95.14 | 89.58 | 76.93 | 15.53 |
| RLF | 83.33 | 54.17 | 88.89 | 77.08 | 60.42 | 62.50 | 75.69 | 94.44 | 86.11 | 75.85 | 13.99 |
| MI | 82.64 | 54.17 | 89.58 | 75.00 | 61.11 | 57.64 | 71.53 | 94.44 | 88.19 | 74.92 | 14.85 |
| Fixed sensorimotor | 89.58 | 52.78 | 91.67 | 70.14 | 57.64 | 63.19 | 83.33 | 95.83 | 88.19 | 76.93 | 16.17 |
| ERD-only | 88.19 | 50.69 | 92.36 | 77.78 | 62.50 | 66.67 | 77.08 | 91.67 | 85.42 | 76.93 | 14.39 |
| Random (mean of draws) | 85.00 | 56.11 | 92.22 | 73.61 | 59.58 | 60.97 | 73.89 | 94.58 | 88.47 | 76.05 | 14.77 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | CP4 C6 C5 CP3 C4 POz CP2 Fz P2 FCz FC3 P1 |
| CSP | 9 | CP4 C3 C4 CP3 CP2 C5 FC3 CP1 C2 C1 P2 FC1 |
| RLF | 9 | CP4 Fz CP2 P1 CPz C2 Cz FCz C6 P2 Pz CP1 |
| MI | 9 | CP4 C2 P2 P1 POz Fz Cz CPz C1 C3 C4 C5 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -2.31 | 3.97 | [-4.78, +0.15] | 0.0119 | yes | -10.42 | 1 | 5,6,8,10 |
| CSP | 9 | -1.77 | 4.70 | [-4.69, +1.14] | 0.0136 | yes | -9.72 | 2 | 5,6,8,10 |
| RLF | 9 | -2.85 | 3.51 | [-5.03, -0.68] | 0.0139 | yes | -9.03 | 1 | 6,8,10 |
| MI | 9 | -3.78 | 3.03 | [-5.66, -1.90] | 0.0296 | yes | -8.33 | 2 | 6,8,10 |
| Fixed sensorimotor | 9 | -1.77 | 5.13 | [-4.95, +1.40] | 0.0193 | yes | -11.81 | 2 | 5,6,8,10 |
| ERD-only | 9 | -1.77 | 4.74 | [-4.71, +1.16] | 0.0140 | yes | -8.33 | 2 | 5,6,8,10 |
| Random (mean of draws) | 9 | -2.65 | 3.27 | [-4.68, -0.63] | 0.0077 | yes | -9.86 | 1 | 5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | -0.54 | [-3.85, +2.77] | 0.7165 | 1.0000 | 0.6250 | 1.0000 | 3 / 0 / 6 |
| RLF | 9 | +0.54 | [-2.03, +3.11] | 0.6410 | 1.0000 | 0.5234 | 1.0000 | 4 / 1 / 4 |
| MI | 9 | +1.47 | [-1.44, +4.37] | 0.2780 | 1.0000 | 0.2734 | 1.0000 | 6 / 1 / 2 |
| Fixed sensorimotor | 9 | -0.54 | [-5.01, +3.93] | 0.7877 | 1.0000 | 0.8125 | 1.0000 | 3 / 2 / 4 |
| ERD-only | 9 | -0.54 | [-3.10, +2.02] | 0.6392 | 1.0000 | 0.5156 | 1.0000 | 3 / 0 / 6 |
| Random (mean of draws) | 9 | +0.34 | [-2.95, +3.63] | 0.8177 | 1.0000 | 0.6523 | 1.0000 | 5 / 0 / 4 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 78.70 | left_hand 79.3, right_hand 78.1 | left_hand 50.6, right_hand 49.4 | 50.00 | 0 / 9 |
| Ours | 76.39 | left_hand 77.5, right_hand 75.3 | left_hand 51.1, right_hand 48.9 | 50.00 | 0 / 9 |
| CSP | 76.93 | left_hand 76.9, right_hand 77.0 | left_hand 49.9, right_hand 50.1 | 50.00 | 0 / 9 |
| RLF | 75.85 | left_hand 76.5, right_hand 75.2 | left_hand 50.7, right_hand 49.3 | 50.00 | 0 / 9 |
| MI | 74.92 | left_hand 78.9, right_hand 71.0 | left_hand 53.9, right_hand 46.1 | 50.00 | 0 / 9 |
| Fixed sensorimotor | 76.93 | left_hand 77.3, right_hand 76.5 | left_hand 50.4, right_hand 49.6 | 50.00 | 0 / 9 |
| ERD-only | 76.93 | left_hand 75.8, right_hand 78.1 | left_hand 48.8, right_hand 51.2 | 50.00 | 0 / 9 |
| Random (mean of draws) | 76.05 | left_hand 77.6, right_hand 74.5 | left_hand 51.6, right_hand 48.4 | 50.00 | 0 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 78.09, 43: 76.08, 44: 77.55 | 77.24 | 1.04 |
| Ours | 3 | 42: 77.16, 43: 76.39, 44: 75.15 | 76.23 | 1.01 |
| CSP | 3 | 42: 76.47, 43: 75.62, 44: 75.93 | 76.00 | 0.43 |
| RLF | 3 | 42: 76.23, 43: 75.62, 44: 73.69 | 75.18 | 1.33 |
| MI | 3 | 42: 73.69, 43: 74.46, 44: 73.15 | 73.77 | 0.66 |
| Fixed sensorimotor | 3 | 42: 76.39, 43: 76.54, 44: 75.77 | 76.23 | 0.41 |
| ERD-only | 3 | 42: 76.93, 43: 76.62, 44: 75.23 | 76.26 | 0.90 |
| Random (mean of draws) | 3 | 42: 75.20, 43: 75.48, 44: 74.91 | 75.20 | 0.29 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.053 | [-0.191, 0.290] | 0.292 | 9 |
| grad | class_specific_pearson | 0.198 | [-0.026, 0.402] | 0.278 | 9 |
| grad | lateralisation_pearson | 0.015 | [-0.093, 0.122] | 0.139 | 9 |
| grad | magnitude_spearman | 0.031 | [-0.215, 0.273] | 0.292 | 9 |
| grad | class_specific_spearman | 0.177 | [0.000, 0.343] | 0.219 | 9 |
| grad | lateralisation_spearman | 0.059 | [-0.095, 0.210] | 0.191 | 9 |
| grad | legacy_signed_LR_pearson | -0.180 | [-0.401, 0.061] | 0.276 | 9 |
| grad_x_input | magnitude_pearson | 0.033 | [-0.214, 0.276] | 0.301 | 9 |
| grad_x_input | class_specific_pearson | 0.197 | [-0.032, 0.406] | 0.284 | 9 |
| grad_x_input | lateralisation_pearson | 0.060 | [-0.220, 0.331] | 0.342 | 9 |
| grad_x_input | magnitude_spearman | 0.012 | [-0.223, 0.245] | 0.285 | 9 |
| grad_x_input | class_specific_spearman | 0.165 | [-0.006, 0.327] | 0.215 | 9 |
| grad_x_input | lateralisation_spearman | 0.075 | [-0.198, 0.337] | 0.334 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.185 | [-0.402, 0.051] | 0.270 | 9 |
| integrated_gradients | magnitude_pearson | 0.024 | [-0.217, 0.262] | 0.287 | 9 |
| integrated_gradients | class_specific_pearson | 0.166 | [-0.052, 0.368] | 0.268 | 9 |
| integrated_gradients | lateralisation_pearson | 0.127 | [-0.027, 0.275] | 0.194 | 9 |
| integrated_gradients | magnitude_spearman | 0.023 | [-0.189, 0.232] | 0.252 | 9 |
| integrated_gradients | class_specific_spearman | 0.142 | [-0.036, 0.311] | 0.217 | 9 |
| integrated_gradients | lateralisation_spearman | 0.084 | [-0.043, 0.208] | 0.160 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.137 | [-0.372, 0.114] | 0.297 | 9 |
| smoothgrad | magnitude_pearson | 0.053 | [-0.191, 0.290] | 0.292 | 9 |
| smoothgrad | class_specific_pearson | 0.198 | [-0.026, 0.402] | 0.278 | 9 |
| smoothgrad | lateralisation_pearson | 0.003 | [-0.101, 0.108] | 0.135 | 9 |
| smoothgrad | magnitude_spearman | 0.031 | [-0.214, 0.273] | 0.292 | 9 |
| smoothgrad | class_specific_spearman | 0.178 | [0.001, 0.345] | 0.219 | 9 |
| smoothgrad | lateralisation_spearman | 0.037 | [-0.114, 0.185] | 0.191 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.180 | [-0.401, 0.061] | 0.275 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.281 | [-0.513, -0.011] | 9 |
| grad | ERD_L_spearman | -0.270 | [-0.487, -0.022] | 9 |
| grad | ERD_R_pearson | -0.120 | [-0.333, 0.104] | 9 |
| grad | ERD_R_spearman | -0.104 | [-0.290, 0.089] | 9 |
| grad | ERD_LR_pearson | -0.180 | [-0.401, 0.061] | 9 |
| grad | ERD_LR_spearman | -0.138 | [-0.343, 0.079] | 9 |
| grad | ERD_comb_pearson | -0.223 | [-0.458, 0.041] | 9 |
| grad | ERD_comb_spearman | -0.233 | [-0.416, -0.032] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.285 | [-0.517, -0.014] | 9 |
| grad_x_input | ERD_L_spearman | -0.261 | [-0.477, -0.015] | 9 |
| grad_x_input | ERD_R_pearson | -0.109 | [-0.331, 0.124] | 9 |
| grad_x_input | ERD_R_spearman | -0.088 | [-0.270, 0.100] | 9 |
| grad_x_input | ERD_LR_pearson | -0.185 | [-0.402, 0.051] | 9 |
| grad_x_input | ERD_LR_spearman | -0.136 | [-0.339, 0.080] | 9 |
| grad_x_input | ERD_comb_pearson | -0.219 | [-0.459, 0.052] | 9 |
| grad_x_input | ERD_comb_spearman | -0.219 | [-0.395, -0.027] | 9 |
| grad_x_input | vs_grad_pearson | 0.998 | [0.996, 0.999] | 9 |
| grad_x_input | vs_grad_spearman | 0.997 | [0.974, 1.000] | 9 |
| integrated_gradients | ERD_L_pearson | -0.217 | [-0.445, 0.037] | 9 |
| integrated_gradients | ERD_L_spearman | -0.194 | [-0.395, 0.025] | 9 |
| integrated_gradients | ERD_R_pearson | -0.101 | [-0.320, 0.129] | 9 |
| integrated_gradients | ERD_R_spearman | -0.074 | [-0.269, 0.127] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.137 | [-0.372, 0.114] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.106 | [-0.326, 0.125] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.180 | [-0.414, 0.077] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.172 | [-0.349, 0.016] | 9 |
| integrated_gradients | vs_grad_pearson | 0.971 | [0.925, 0.989] | 9 |
| integrated_gradients | vs_grad_spearman | 0.942 | [0.892, 0.969] | 9 |
| smoothgrad | ERD_L_pearson | -0.281 | [-0.513, -0.011] | 9 |
| smoothgrad | ERD_L_spearman | -0.268 | [-0.484, -0.021] | 9 |
| smoothgrad | ERD_R_pearson | -0.120 | [-0.333, 0.104] | 9 |
| smoothgrad | ERD_R_spearman | -0.103 | [-0.291, 0.093] | 9 |
| smoothgrad | ERD_LR_pearson | -0.180 | [-0.401, 0.061] | 9 |
| smoothgrad | ERD_LR_spearman | -0.135 | [-0.339, 0.081] | 9 |
| smoothgrad | ERD_comb_pearson | -0.223 | [-0.458, 0.041] | 9 |
| smoothgrad | ERD_comb_spearman | -0.231 | [-0.415, -0.030] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | 0.040 | [-0.162, 0.239] | -0.149 | 0.034 | -0.096 | 9 |
| ERD_theta_R | 0.012 | [-0.222, 0.244] | 0.087 | -0.025 | 0.088 | 9 |
| ERD_theta_mag | -0.016 | [-0.197, 0.165] | -0.016 | 0.012 | 0.012 | 9 |
| ERD_mu_L | -0.287 | [-0.501, -0.040] | 0.040 | -0.275 | 0.045 | 9 |
| ERD_mu_R | -0.166 | [-0.346, 0.026] | 0.047 | -0.141 | 0.042 | 9 |
| ERD_mu_mag | 0.032 | [-0.257, 0.316] | 0.032 | 0.024 | 0.024 | 9 |
| ERD_beta_L | -0.096 | [-0.336, 0.155] | 0.068 | -0.007 | 0.010 | 9 |
| ERD_beta_R | 0.089 | [-0.145, 0.312] | 0.002 | 0.082 | -0.058 | 9 |
| ERD_beta_mag | 0.048 | [-0.180, 0.272] | 0.048 | 0.010 | 0.010 | 9 |
| ERS_beta_rebound_L | 0.021 | [-0.197, 0.236] | -0.018 | 0.013 | -0.008 | 9 |
| ERS_beta_rebound_R | -0.142 | [-0.340, 0.069] | -0.039 | -0.179 | -0.021 | 9 |
| ERS_beta_rebound_mag | -0.038 | [-0.232, 0.159] | -0.038 | -0.007 | -0.007 | 9 |
| ERD_lowgamma_L | -0.149 | [-0.288, -0.003] | -0.010 | -0.072 | -0.035 | 9 |
| ERD_lowgamma_R | -0.088 | [-0.277, 0.108] | 0.077 | -0.011 | 0.106 | 9 |
| ERD_lowgamma_mag | 0.033 | [-0.257, 0.316] | 0.033 | -0.006 | -0.006 | 9 |
| logpow_8_30 | -0.107 | [-0.350, 0.149] | -0.107 | -0.044 | -0.044 | 9 |
| r2_signed | 0.237 | [-0.020, 0.464] | 0.282 | 0.119 | 0.121 | 9 |
| r2_abs | 0.282 | [0.000, 0.522] | 0.282 | 0.121 | 0.121 | 9 |
| TFR_ERD_dB_L | -0.237 | [-0.456, 0.010] | 0.056 | -0.176 | 0.011 | 9 |
| TFR_ERD_dB_R | -0.034 | [-0.254, 0.189] | 0.160 | 0.002 | 0.122 | 9 |
| TFR_ERD_dB_mag | 0.135 | [-0.156, 0.404] | 0.135 | 0.090 | 0.090 | 9 |
| ITC_4_30 | -0.141 | [-0.309, 0.036] | -0.141 | -0.087 | -0.087 | 9 |
| ERP_abs_0.2_0.8 | -0.121 | [-0.293, 0.059] | -0.121 | -0.098 | -0.098 | 9 |
| SlowPot_0.5_3 | 0.002 | [-0.132, 0.136] | 0.127 | -0.041 | 0.156 | 9 |
| ERD_beta_classspecific | — | — | 0.043 [-0.121, 0.206] | — | -0.013 | 9 |
| ERD_lowgamma_classspecific | — | — | 0.033 [-0.153, 0.216] | — | 0.043 | 9 |
| ERD_mu_classspecific | — | — | 0.045 [-0.196, 0.281] | — | 0.049 | 9 |
| ERD_theta_classspecific | — | — | -0.040 [-0.200, 0.122] | — | -0.014 | 9 |
| ERS_beta_rebound_classspecific | — | — | -0.031 [-0.189, 0.127] | — | -0.026 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.103 [-0.085, 0.284] | — | 0.065 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +9.47 ± 5.22 | +6.08 ± 3.22 | +4.40 ± 3.31 | 0.0875 | 0.0144 |
| 4 | +15.38 ± 10.57 | +10.30 ± 4.62 | +5.66 ± 4.25 | 0.1234 | 0.0167 |
| 6 | +20.91 ± 12.38 | +14.45 ± 7.26 | +13.91 ± 7.27 | 0.0563 | 0.0992 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.062 | [-0.045, 0.167] | 9 |
| grad_x_input | 0.069 | [-0.038, 0.175] | 9 |
| integrated_gradients | 0.064 | [-0.052, 0.178] | 9 |
| smoothgrad | 0.063 | [-0.044, 0.169] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=original, S02=original, S03=original, S04=original, S05=original, S06=original, S07=original, S08=original, S09=original

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 70.99 |
| lr_2e-4 | 69.44 |

```
model: "EEGConformer"
dataset: "BNCI2014_001"
sfreq_hz: 250
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.6, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "lr_2e-4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.6, "lr": 0.0002, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "cue_0_4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [0.0, 4.0], "dropout": 0.6, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}}
checkpoint_rule: "SWA average from swa_start_frac*max_epochs + BN update (no early stopping); validation block only scored, never used for checkpointing"
folds: "train on session 1 (last 25% of it = validation, chronological), refit on all of session 1, test once on session 2 (official BCI IV-2a split)"
seeds: [42, 43, 44]
```

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 75.00 | 42.71 | 77.08 | 71.18 | 43.06 | 46.18 | 78.82 | 74.65 | 75.00 | 64.85 | 15.81 |
| Ours | 68.40 | 37.50 | 72.22 | 66.67 | 38.89 | 42.71 | 72.57 | 70.49 | 74.65 | 60.46 | 15.80 |
| CSP | 70.49 | 33.68 | 73.26 | 62.85 | 40.97 | 39.24 | 67.01 | 68.75 | 73.26 | 58.83 | 16.08 |
| RLF | 69.79 | 34.38 | 78.82 | 61.46 | 34.72 | 40.62 | 65.62 | 66.67 | 73.61 | 58.41 | 17.18 |
| MI | 70.83 | 37.15 | 78.12 | 62.85 | 35.76 | 44.44 | 68.75 | 72.92 | 74.31 | 60.57 | 16.77 |
| Fixed sensorimotor | 70.49 | 38.89 | 73.26 | 53.47 | 33.33 | 40.62 | 75.00 | 75.69 | 69.79 | 58.95 | 17.39 |
| ERD-only | 69.79 | 38.89 | 70.83 | 60.42 | 38.19 | 41.32 | 69.10 | 70.14 | 75.00 | 59.30 | 15.37 |
| Random (mean of draws) | 68.19 | 35.56 | 71.74 | 60.69 | 34.44 | 40.14 | 67.85 | 70.21 | 73.68 | 58.06 | 16.47 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | C5 P2 POz C6 CP4 C3 CP3 CP2 FC3 Fz CPz P1 |
| CSP | 9 | P1 CP3 Pz P2 CP4 CPz CP1 POz CP2 C3 C4 C2 |
| RLF | 9 | P1 CP3 Pz P2 CPz POz CP1 CP4 C3 CP2 Fz C5 |
| MI | 9 | P1 P2 Pz CP4 CPz CP1 POz C3 CP3 FC1 C1 C6 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -4.40 | 1.82 | [-5.53, -3.27] | 0.0149 | yes | -6.60 | 2 | 6,8,10 |
| CSP | 9 | -6.02 | 3.35 | [-8.09, -3.94] | 0.5064 | no | -11.81 | 4 | 10 |
| RLF | 9 | -6.44 | 4.48 | [-9.22, -3.67] | 0.6128 | no | -13.19 | 5 | 10 |
| MI | 9 | -4.28 | 3.79 | [-6.63, -1.93] | 0.1054 | no | -10.07 | 3 | 8,10 |
| Fixed sensorimotor | 9 | -5.90 | 5.22 | [-9.14, -2.67] | 0.4784 | no | -17.71 | 2 | 10 |
| ERD-only | 9 | -5.56 | 3.18 | [-7.53, -3.58] | 0.3431 | no | -10.76 | 3 | 8,10 |
| Random (mean of draws) | 9 | -6.80 | 3.01 | [-8.67, -4.93] | 0.7750 | no | -10.97 | 6 | 10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | +1.62 | [-0.54, +3.78] | 0.1219 | 0.6096 | 0.1562 | 0.7812 | 6 / 0 / 3 |
| RLF | 9 | +2.04 | [-1.06, +5.15] | 0.1670 | 0.6681 | 0.1641 | 0.7812 | 7 / 0 / 2 |
| MI | 9 | -0.12 | [-2.68, +2.44] | 0.9195 | 0.9195 | 0.8906 | 1.0000 | 5 / 0 / 4 |
| Fixed sensorimotor | 9 | +1.50 | [-2.82, +5.83] | 0.4453 | 0.8906 | 0.7344 | 1.0000 | 4 / 0 / 5 |
| ERD-only | 9 | +1.16 | [-0.72, +3.03] | 0.1920 | 0.6681 | 0.2422 | 0.7812 | 6 / 0 / 3 |
| Random (mean of draws) | 9 | +2.40 | [+0.73, +4.07] | 0.0105 | 0.0631 | 0.0039 | 0.0234 | 9 / 0 / 0 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 64.85 | feet 58.0, left_hand 67.6, right_hand 65.0, tongue 68.8 | feet 20.9, left_hand 27.6, right_hand 25.5, tongue 26.0 | 25.00 | 0 / 9 |
| Ours | 60.46 | feet 48.6, left_hand 68.5, right_hand 62.0, tongue 62.7 | feet 20.2, left_hand 29.1, right_hand 26.2, tongue 24.5 | 25.00 | 0 / 9 |
| CSP | 58.83 | feet 50.3, left_hand 66.2, right_hand 63.3, tongue 55.6 | feet 20.2, left_hand 29.3, right_hand 28.2, tongue 22.3 | 25.00 | 0 / 9 |
| RLF | 58.41 | feet 49.4, left_hand 62.0, right_hand 65.6, tongue 56.6 | feet 19.3, left_hand 27.7, right_hand 30.0, tongue 23.0 | 25.00 | 0 / 9 |
| MI | 60.57 | feet 53.1, left_hand 65.7, right_hand 64.2, tongue 59.3 | feet 20.6, left_hand 29.0, right_hand 27.1, tongue 23.3 | 25.00 | 0 / 9 |
| Fixed sensorimotor | 58.95 | feet 52.2, left_hand 66.0, right_hand 62.0, tongue 55.6 | feet 21.5, left_hand 29.2, right_hand 26.9, tongue 22.5 | 25.00 | 0 / 9 |
| ERD-only | 59.30 | feet 54.6, left_hand 62.7, right_hand 64.7, tongue 55.2 | feet 22.1, left_hand 26.2, right_hand 29.1, tongue 22.6 | 25.00 | 0 / 9 |
| Random (mean of draws) | 58.06 | feet 48.7, left_hand 61.8, right_hand 61.5, tongue 60.2 | feet 20.2, left_hand 27.2, right_hand 28.1, tongue 24.6 | 25.00 | 0 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 62.27, 43: 63.66, 44: 62.04 | 62.65 | 0.88 |
| Ours | 3 | 42: 60.07, 43: 58.22, 44: 58.41 | 58.90 | 1.02 |
| CSP | 3 | 42: 55.75, 43: 58.02, 44: 58.10 | 57.29 | 1.34 |
| RLF | 3 | 42: 56.29, 43: 57.48, 44: 57.10 | 56.96 | 0.61 |
| MI | 3 | 42: 59.38, 43: 58.80, 44: 58.22 | 58.80 | 0.58 |
| Fixed sensorimotor | 3 | 42: 56.71, 43: 57.75, 44: 58.80 | 57.75 | 1.04 |
| ERD-only | 3 | 42: 56.37, 43: 59.30, 44: 57.29 | 57.65 | 1.50 |
| Random (mean of draws) | 3 | 42: 55.66, 43: 57.04, 44: 57.16 | 56.62 | 0.83 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.001 | [-0.209, 0.211] | 0.264 | 9 |
| grad | class_specific_pearson | 0.153 | [0.042, 0.262] | 0.140 | 9 |
| grad | lateralisation_pearson | 0.061 | [-0.063, 0.182] | 0.158 | 9 |
| grad | magnitude_spearman | -0.000 | [-0.242, 0.242] | 0.301 | 9 |
| grad | class_specific_spearman | 0.146 | [0.051, 0.239] | 0.122 | 9 |
| grad | lateralisation_spearman | 0.046 | [-0.084, 0.174] | 0.166 | 9 |
| grad | legacy_signed_LR_pearson | -0.144 | [-0.331, 0.054] | 0.242 | 9 |
| grad_x_input | magnitude_pearson | -0.031 | [-0.252, 0.194] | 0.277 | 9 |
| grad_x_input | class_specific_pearson | 0.154 | [0.029, 0.275] | 0.157 | 9 |
| grad_x_input | lateralisation_pearson | 0.056 | [-0.084, 0.195] | 0.179 | 9 |
| grad_x_input | magnitude_spearman | -0.044 | [-0.308, 0.226] | 0.322 | 9 |
| grad_x_input | class_specific_spearman | 0.158 | [0.053, 0.259] | 0.134 | 9 |
| grad_x_input | lateralisation_spearman | 0.044 | [-0.094, 0.181] | 0.177 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.140 | [-0.337, 0.069] | 0.257 | 9 |
| integrated_gradients | magnitude_pearson | -0.089 | [-0.245, 0.071] | 0.203 | 9 |
| integrated_gradients | class_specific_pearson | 0.127 | [-0.017, 0.266] | 0.179 | 9 |
| integrated_gradients | lateralisation_pearson | 0.061 | [-0.105, 0.223] | 0.210 | 9 |
| integrated_gradients | magnitude_spearman | -0.103 | [-0.324, 0.129] | 0.280 | 9 |
| integrated_gradients | class_specific_spearman | 0.137 | [0.015, 0.254] | 0.153 | 9 |
| integrated_gradients | lateralisation_spearman | 0.056 | [-0.081, 0.191] | 0.174 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.017 | [-0.208, 0.176] | 0.242 | 9 |
| smoothgrad | magnitude_pearson | -0.000 | [-0.209, 0.209] | 0.264 | 9 |
| smoothgrad | class_specific_pearson | 0.149 | [0.035, 0.260] | 0.143 | 9 |
| smoothgrad | lateralisation_pearson | 0.069 | [-0.058, 0.194] | 0.162 | 9 |
| smoothgrad | magnitude_spearman | -0.019 | [-0.256, 0.220] | 0.296 | 9 |
| smoothgrad | class_specific_spearman | 0.127 | [0.022, 0.228] | 0.134 | 9 |
| smoothgrad | lateralisation_spearman | 0.065 | [-0.070, 0.198] | 0.172 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.144 | [-0.330, 0.053] | 0.240 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.195 | [-0.361, -0.017] | 9 |
| grad | ERD_L_spearman | -0.247 | [-0.430, -0.044] | 9 |
| grad | ERD_R_pearson | -0.030 | [-0.233, 0.176] | 9 |
| grad | ERD_R_spearman | -0.048 | [-0.259, 0.168] | 9 |
| grad | ERD_LR_pearson | -0.144 | [-0.331, 0.054] | 9 |
| grad | ERD_LR_spearman | -0.156 | [-0.342, 0.041] | 9 |
| grad | ERD_comb_pearson | -0.131 | [-0.318, 0.065] | 9 |
| grad | ERD_comb_spearman | -0.184 | [-0.372, 0.018] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.210 | [-0.379, -0.027] | 9 |
| grad_x_input | ERD_L_spearman | -0.259 | [-0.465, -0.027] | 9 |
| grad_x_input | ERD_R_pearson | -0.035 | [-0.253, 0.186] | 9 |
| grad_x_input | ERD_R_spearman | -0.053 | [-0.289, 0.190] | 9 |
| grad_x_input | ERD_LR_pearson | -0.140 | [-0.337, 0.069] | 9 |
| grad_x_input | ERD_LR_spearman | -0.148 | [-0.356, 0.075] | 9 |
| grad_x_input | ERD_comb_pearson | -0.142 | [-0.334, 0.062] | 9 |
| grad_x_input | ERD_comb_spearman | -0.197 | [-0.408, 0.034] | 9 |
| grad_x_input | vs_grad_pearson | 0.998 | [0.995, 0.999] | 9 |
| grad_x_input | vs_grad_spearman | 0.992 | [0.986, 0.996] | 9 |
| integrated_gradients | ERD_L_pearson | -0.146 | [-0.330, 0.050] | 9 |
| integrated_gradients | ERD_L_spearman | -0.212 | [-0.414, 0.010] | 9 |
| integrated_gradients | ERD_R_pearson | -0.070 | [-0.246, 0.110] | 9 |
| integrated_gradients | ERD_R_spearman | -0.099 | [-0.345, 0.160] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.017 | [-0.208, 0.176] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.032 | [-0.217, 0.156] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.133 | [-0.328, 0.074] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.189 | [-0.420, 0.064] | 9 |
| integrated_gradients | vs_grad_pearson | 0.904 | [0.826, 0.948] | 9 |
| integrated_gradients | vs_grad_spearman | 0.885 | [0.750, 0.949] | 9 |
| smoothgrad | ERD_L_pearson | -0.190 | [-0.359, -0.009] | 9 |
| smoothgrad | ERD_L_spearman | -0.245 | [-0.431, -0.039] | 9 |
| smoothgrad | ERD_R_pearson | -0.026 | [-0.229, 0.180] | 9 |
| smoothgrad | ERD_R_spearman | -0.041 | [-0.259, 0.181] | 9 |
| smoothgrad | ERD_LR_pearson | -0.144 | [-0.330, 0.053] | 9 |
| smoothgrad | ERD_LR_spearman | -0.162 | [-0.351, 0.039] | 9 |
| smoothgrad | ERD_comb_pearson | -0.127 | [-0.316, 0.071] | 9 |
| smoothgrad | ERD_comb_spearman | -0.183 | [-0.380, 0.029] | 9 |
| smoothgrad | vs_grad_pearson | 0.999 | [0.998, 0.999] | 9 |
| smoothgrad | vs_grad_spearman | 0.995 | [0.990, 0.998] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.013 | [-0.218, 0.193] | -0.068 | -0.108 | -0.083 | 9 |
| ERD_theta_R | -0.015 | [-0.203, 0.175] | -0.035 | -0.035 | -0.042 | 9 |
| ERD_theta_mag | -0.075 | [-0.295, 0.154] | -0.075 | -0.067 | -0.067 | 9 |
| ERD_mu_L | -0.196 | [-0.366, -0.014] | -0.013 | -0.265 | -0.031 | 9 |
| ERD_mu_R | -0.106 | [-0.286, 0.081] | -0.045 | -0.112 | -0.085 | 9 |
| ERD_mu_mag | -0.078 | [-0.297, 0.148] | -0.078 | -0.109 | -0.109 | 9 |
| ERD_beta_L | -0.106 | [-0.259, 0.051] | 0.057 | -0.131 | 0.027 | 9 |
| ERD_beta_R | 0.012 | [-0.179, 0.203] | 0.008 | 0.029 | -0.056 | 9 |
| ERD_beta_mag | 0.015 | [-0.222, 0.250] | 0.015 | -0.032 | -0.032 | 9 |
| ERS_beta_rebound_L | -0.003 | [-0.173, 0.167] | 0.008 | -0.032 | -0.007 | 9 |
| ERS_beta_rebound_R | -0.095 | [-0.286, 0.103] | 0.020 | -0.095 | 0.052 | 9 |
| ERS_beta_rebound_mag | -0.005 | [-0.211, 0.201] | -0.005 | 0.036 | 0.036 | 9 |
| ERD_lowgamma_L | -0.042 | [-0.228, 0.146] | 0.074 | -0.062 | 0.045 | 9 |
| ERD_lowgamma_R | -0.070 | [-0.271, 0.135] | 0.118 | -0.004 | 0.079 | 9 |
| ERD_lowgamma_mag | 0.075 | [-0.110, 0.255] | 0.075 | -0.019 | -0.019 | 9 |
| logpow_8_30 | -0.152 | [-0.308, 0.012] | -0.152 | -0.052 | -0.052 | 9 |
| r2_signed | 0.145 | [-0.049, 0.329] | 0.146 | 0.175 | 0.144 | 9 |
| r2_abs | 0.146 | [-0.074, 0.352] | 0.146 | 0.144 | 0.144 | 9 |
| TFR_ERD_dB_L | -0.207 | [-0.339, -0.067] | 0.160 | -0.247 | 0.106 | 9 |
| TFR_ERD_dB_R | -0.042 | [-0.232, 0.151] | 0.081 | -0.032 | 0.048 | 9 |
| TFR_ERD_dB_mag | 0.119 | [-0.068, 0.298] | 0.119 | 0.074 | 0.074 | 9 |
| ITC_4_30 | -0.025 | [-0.189, 0.141] | -0.025 | 0.004 | 0.004 | 9 |
| ERP_abs_0.2_0.8 | -0.070 | [-0.239, 0.103] | -0.070 | -0.054 | -0.054 | 9 |
| SlowPot_0.5_3 | 0.021 | [-0.091, 0.132] | 0.002 | 0.018 | 0.005 | 9 |
| ERD_beta_classspecific | — | — | 0.075 [-0.123, 0.267] | — | 0.028 | 9 |
| ERD_lowgamma_classspecific | — | — | 0.088 [-0.039, 0.212] | — | 0.061 | 9 |
| ERD_mu_classspecific | — | — | -0.052 [-0.195, 0.092] | — | -0.045 | 9 |
| ERD_theta_classspecific | — | — | -0.031 [-0.199, 0.139] | — | -0.034 | 9 |
| ERS_beta_rebound_classspecific | — | — | -0.011 [-0.145, 0.124] | — | 0.009 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.139 [0.009, 0.264] | — | 0.135 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +17.09 ± 8.74 | +15.57 ± 9.09 | +12.27 ± 11.64 | 0.3495 | 0.0939 |
| 4 | +25.03 ± 13.47 | +23.29 ± 11.72 | +17.99 ± 9.03 | 0.2012 | 0.0057 |
| 6 | +30.72 ± 13.56 | +27.21 ± 13.21 | +22.69 ± 12.86 | 0.0238 | 0.0073 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.058 | [-0.179, 0.064] | 9 |
| grad_x_input | -0.056 | [-0.179, 0.070] | 9 |
| integrated_gradients | -0.062 | [-0.204, 0.082] | 9 |
| smoothgrad | -0.053 | [-0.169, 0.064] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=lr_2e-4, S02=lr_2e-4, S03=lr_2e-4, S04=lr_2e-4, S05=lr_2e-4, S06=lr_2e-4, S07=lr_2e-4, S08=lr_2e-4, S09=lr_2e-4

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 53.09 |
| lr_2e-4 | 59.57 |

```
model: "EEGConformer"
dataset: "BNCI2014_001"
sfreq_hz: 250
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "lr_2e-4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.5, "lr": 0.0002, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "cue_0_4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [0.0, 4.0], "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}}
checkpoint_rule: "SWA average from swa_start_frac*max_epochs + BN update (no early stopping); validation block only scored, never used for checkpointing"
folds: "train on session 1 (last 25% of it = validation, chronological), refit on all of session 1, test once on session 2 (official BCI IV-2a split)"
seeds: [42, 43, 44]
```
