# Results report — EEGNetV4\HGD

Generated 2026-10-18 01:23 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 51.25 | 88.75 | 98.75 | 100.00 | 78.75 | 99.55 | 87.05 | 62.20 | 97.73 | 84.89 | 17.69 |
| Ours | 50.00 | 91.25 | 93.75 | 95.00 | 57.50 | 97.27 | 89.09 | 75.30 | 84.09 | 81.47 | 17.13 |
| CSP | 50.00 | 70.00 | 97.50 | 96.25 | 70.00 | 99.32 | 88.64 | 67.38 | 84.77 | 80.43 | 16.95 |
| RLF | 50.00 | 85.00 | 97.50 | 95.00 | 68.75 | 99.55 | 100.00 | 88.41 | 83.41 | 85.29 | 16.57 |
| MI | 67.50 | 88.75 | 98.75 | 98.75 | 72.50 | 100.00 | 89.77 | 68.29 | 65.45 | 83.31 | 14.74 |
| Fixed sensorimotor | 48.75 | 87.50 | 96.25 | 100.00 | 71.25 | 99.77 | 54.32 | 68.60 | 94.77 | 80.13 | 19.91 |
| ERD-only | 55.00 | 86.25 | 97.50 | 97.50 | 86.25 | 98.86 | 70.45 | 91.46 | 77.73 | 84.56 | 14.63 |
| Random (mean of draws) | 57.50 | 88.50 | 98.00 | 98.50 | 67.75 | 99.64 | 86.55 | 81.89 | 81.77 | 84.45 | 14.37 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | CP1 CP5 CP3 C5 FC3 CP6 FC4 C6 FC2 C3 C4 CP2 |
| CSP | 9 | CP6 CP5 Pz C4 C6 FC6 FC5 CP2 C5 FC4 CP3 FC3 |
| RLF | 9 | CP3 CP2 CP4 CP1 CP5 FC2 CP6 C4 C3 C1 FC5 FCz |
| MI | 9 | C6 CP4 C2 CP2 CP6 FC3 C4 FC4 FC6 FC5 C3 CP3 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | C2 C4 C1 CP3 C3 CP1 CP2 Pz CP6 CP5 CPz CP4 |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -3.42 | 9.82 | [-9.50, +2.67] | 0.2264 | no | -21.25 | 2 | 10 |
| CSP | 9 | -4.46 | 7.61 | [-9.18, +0.26] | 0.2806 | no | -18.75 | 3 | 10 |
| RLF | 9 | +0.40 | 12.25 | [-7.19, +8.00] | 0.1038 | no | -14.32 | 2 | 8,10 |
| MI | 9 | -1.58 | 13.08 | [-9.69, +6.52] | 0.1703 | no | -32.27 | 2 | 10 |
| Fixed sensorimotor | 9 | -4.76 | 11.11 | [-11.64, +2.13] | 0.3728 | no | -32.73 | 2 | none |
| ERD-only | 9 | -0.33 | 14.21 | [-9.14, +8.48] | 0.1329 | no | -20.00 | 2 | 10 |
| Random (mean of draws) | 9 | -0.44 | 10.02 | [-6.65, +5.78] | 0.0672 | no | -15.95 | 2 | 8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | +1.04 | [-6.05, +8.14] | 0.7428 | 1.0000 | 0.7422 | 1.0000 | 3 / 1 / 5 |
| RLF | 9 | -3.82 | [-8.87, +1.23] | 0.1193 | 0.5966 | 0.1562 | 0.7812 | 2 / 2 / 5 |
| MI | 9 | -1.83 | [-10.21, +6.54] | 0.6273 | 1.0000 | 0.5703 | 1.0000 | 3 / 0 / 6 |
| Fixed sensorimotor | 9 | +1.34 | [-9.51, +12.19] | 0.7833 | 1.0000 | 0.8203 | 1.0000 | 4 / 0 / 5 |
| ERD-only | 9 | -3.08 | [-13.48, +7.31] | 0.5133 | 1.0000 | 0.6523 | 1.0000 | 3 / 0 / 6 |
| Random (mean of draws) | 9 | -2.98 | [-6.63, +0.66] | 0.0960 | 0.5762 | 0.0977 | 0.5859 | 3 / 0 / 6 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 87.35 | left_hand 93.8, right_hand 80.9 | left_hand 56.5, right_hand 43.5 | 50.00 | 1 / 9 |
| Ours | 85.30 | left_hand 85.1, right_hand 85.5 | left_hand 49.8, right_hand 50.2 | 50.00 | 1 / 9 |
| CSP | 84.38 | left_hand 86.9, right_hand 81.8 | left_hand 52.5, right_hand 47.5 | 50.00 | 1 / 9 |
| RLF | 90.43 | left_hand 90.7, right_hand 90.1 | left_hand 50.3, right_hand 49.7 | 50.00 | 1 / 9 |
| MI | 82.42 | left_hand 81.9, right_hand 82.9 | left_hand 49.5, right_hand 50.5 | 50.00 | 0 / 9 |
| Fixed sensorimotor | 80.22 | left_hand 75.6, right_hand 84.9 | left_hand 45.4, right_hand 54.6 | 50.00 | 1 / 9 |
| ERD-only | 84.23 | left_hand 77.2, right_hand 91.2 | left_hand 43.0, right_hand 57.0 | 50.00 | 0 / 9 |
| Random (mean of draws) | 86.71 | left_hand 85.9, right_hand 87.5 | left_hand 49.2, right_hand 50.8 | 50.00 | 1 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 77.62, 43: 77.23, 44: 78.84 | 77.90 | 0.84 |
| Ours | 3 | 42: 79.11, 43: 73.97, 44: 77.20 | 76.76 | 2.60 |
| CSP | 3 | 42: 75.82, 43: 74.59, 44: 74.34 | 74.92 | 0.79 |
| RLF | 3 | 42: 82.58, 43: 74.64, 44: 74.87 | 77.36 | 4.52 |
| MI | 3 | 42: 76.72, 43: 76.59, 44: 77.23 | 76.85 | 0.34 |
| Fixed sensorimotor | 3 | 42: 78.34, 43: 76.29, 44: 73.05 | 75.89 | 2.67 |
| ERD-only | 3 | 42: 80.08, 43: 78.69, 44: 76.56 | 78.44 | 1.77 |
| Random (mean of draws) | 3 | 42: 76.18, 43: 79.23, 44: 77.10 | 77.50 | 1.56 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.049 | [-0.067, 0.165] | 0.150 | 9 |
| grad | class_specific_pearson | 0.051 | [-0.086, 0.187] | 0.176 | 9 |
| grad | lateralisation_pearson | -0.073 | [-0.196, 0.052] | 0.159 | 9 |
| grad | magnitude_spearman | 0.109 | [0.011, 0.204] | 0.122 | 9 |
| grad | class_specific_spearman | 0.087 | [-0.022, 0.194] | 0.138 | 9 |
| grad | lateralisation_spearman | -0.031 | [-0.216, 0.157] | 0.239 | 9 |
| grad | legacy_signed_LR_pearson | 0.040 | [-0.185, 0.261] | 0.272 | 9 |
| grad_x_input | magnitude_pearson | 0.034 | [-0.174, 0.239] | 0.261 | 9 |
| grad_x_input | class_specific_pearson | 0.038 | [-0.199, 0.270] | 0.293 | 9 |
| grad_x_input | lateralisation_pearson | -0.249 | [-0.399, -0.086] | 0.198 | 9 |
| grad_x_input | magnitude_spearman | -0.108 | [-0.372, 0.172] | 0.309 | 9 |
| grad_x_input | class_specific_spearman | -0.026 | [-0.194, 0.144] | 0.217 | 9 |
| grad_x_input | lateralisation_spearman | -0.236 | [-0.423, -0.030] | 0.252 | 9 |
| grad_x_input | legacy_signed_LR_pearson | 0.136 | [-0.204, 0.446] | 0.377 | 9 |
| integrated_gradients | magnitude_pearson | 0.025 | [-0.176, 0.225] | 0.254 | 9 |
| integrated_gradients | class_specific_pearson | 0.035 | [-0.195, 0.261] | 0.285 | 9 |
| integrated_gradients | lateralisation_pearson | -0.208 | [-0.352, -0.054] | 0.189 | 9 |
| integrated_gradients | magnitude_spearman | -0.088 | [-0.333, 0.168] | 0.297 | 9 |
| integrated_gradients | class_specific_spearman | -0.027 | [-0.193, 0.139] | 0.212 | 9 |
| integrated_gradients | lateralisation_spearman | -0.170 | [-0.340, 0.010] | 0.225 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | 0.140 | [-0.195, 0.446] | 0.372 | 9 |
| smoothgrad | magnitude_pearson | 0.049 | [-0.067, 0.165] | 0.150 | 9 |
| smoothgrad | class_specific_pearson | 0.051 | [-0.087, 0.187] | 0.176 | 9 |
| smoothgrad | lateralisation_pearson | -0.072 | [-0.195, 0.054] | 0.160 | 9 |
| smoothgrad | magnitude_spearman | 0.108 | [0.009, 0.205] | 0.123 | 9 |
| smoothgrad | class_specific_spearman | 0.087 | [-0.022, 0.194] | 0.138 | 9 |
| smoothgrad | lateralisation_spearman | -0.030 | [-0.216, 0.157] | 0.238 | 9 |
| smoothgrad | legacy_signed_LR_pearson | 0.040 | [-0.184, 0.261] | 0.272 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.071 | [-0.212, 0.073] | 9 |
| grad | ERD_L_spearman | -0.118 | [-0.226, -0.008] | 9 |
| grad | ERD_R_pearson | -0.043 | [-0.201, 0.117] | 9 |
| grad | ERD_R_spearman | -0.081 | [-0.195, 0.035] | 9 |
| grad | ERD_LR_pearson | 0.040 | [-0.185, 0.261] | 9 |
| grad | ERD_LR_spearman | 0.041 | [-0.135, 0.215] | 9 |
| grad | ERD_comb_pearson | -0.049 | [-0.200, 0.104] | 9 |
| grad | ERD_comb_spearman | -0.106 | [-0.225, 0.015] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.015 | [-0.230, 0.201] | 9 |
| grad_x_input | ERD_L_spearman | 0.076 | [-0.074, 0.222] | 9 |
| grad_x_input | ERD_R_pearson | -0.062 | [-0.360, 0.248] | 9 |
| grad_x_input | ERD_R_spearman | -0.036 | [-0.325, 0.260] | 9 |
| grad_x_input | ERD_LR_pearson | 0.136 | [-0.204, 0.446] | 9 |
| grad_x_input | ERD_LR_spearman | 0.222 | [-0.075, 0.483] | 9 |
| grad_x_input | ERD_comb_pearson | -0.052 | [-0.337, 0.241] | 9 |
| grad_x_input | ERD_comb_spearman | -0.044 | [-0.324, 0.243] | 9 |
| grad_x_input | vs_grad_pearson | 0.719 | [0.133, 0.932] | 9 |
| grad_x_input | vs_grad_spearman | 0.670 | [0.163, 0.897] | 9 |
| integrated_gradients | ERD_L_pearson | -0.010 | [-0.216, 0.197] | 9 |
| integrated_gradients | ERD_L_spearman | 0.071 | [-0.078, 0.218] | 9 |
| integrated_gradients | ERD_R_pearson | -0.062 | [-0.355, 0.242] | 9 |
| integrated_gradients | ERD_R_spearman | -0.031 | [-0.301, 0.243] | 9 |
| integrated_gradients | ERD_LR_pearson | 0.140 | [-0.195, 0.446] | 9 |
| integrated_gradients | ERD_LR_spearman | 0.219 | [-0.057, 0.464] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.051 | [-0.330, 0.236] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.037 | [-0.300, 0.231] | 9 |
| integrated_gradients | vs_grad_pearson | 0.678 | [0.212, 0.893] | 9 |
| integrated_gradients | vs_grad_spearman | 0.656 | [0.214, 0.875] | 9 |
| smoothgrad | ERD_L_pearson | -0.071 | [-0.212, 0.073] | 9 |
| smoothgrad | ERD_L_spearman | -0.121 | [-0.231, -0.008] | 9 |
| smoothgrad | ERD_R_pearson | -0.043 | [-0.201, 0.117] | 9 |
| smoothgrad | ERD_R_spearman | -0.082 | [-0.197, 0.036] | 9 |
| smoothgrad | ERD_LR_pearson | 0.040 | [-0.184, 0.261] | 9 |
| smoothgrad | ERD_LR_spearman | 0.039 | [-0.138, 0.214] | 9 |
| smoothgrad | ERD_comb_pearson | -0.049 | [-0.200, 0.104] | 9 |
| smoothgrad | ERD_comb_spearman | -0.108 | [-0.228, 0.015] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.026 | [-0.272, 0.224] | 0.138 | 0.017 | 0.121 | 9 |
| ERD_theta_R | 0.093 | [-0.213, 0.382] | 0.005 | 0.051 | -0.091 | 9 |
| ERD_theta_mag | 0.070 | [-0.189, 0.319] | 0.070 | -0.046 | -0.046 | 9 |
| ERD_mu_L | -0.025 | [-0.175, 0.126] | -0.014 | 0.012 | -0.053 | 9 |
| ERD_mu_R | -0.061 | [-0.272, 0.157] | 0.068 | -0.073 | -0.047 | 9 |
| ERD_mu_mag | 0.039 | [-0.134, 0.209] | 0.039 | -0.087 | -0.087 | 9 |
| ERD_beta_L | -0.194 | [-0.351, -0.026] | 0.188 | -0.149 | 0.142 | 9 |
| ERD_beta_R | -0.181 | [-0.397, 0.055] | 0.170 | -0.158 | 0.050 | 9 |
| ERD_beta_mag | 0.159 | [-0.054, 0.359] | 0.159 | 0.058 | 0.058 | 9 |
| ERS_beta_rebound_L | 0.030 | [-0.209, 0.265] | 0.146 | 0.081 | 0.071 | 9 |
| ERS_beta_rebound_R | -0.022 | [-0.274, 0.233] | 0.070 | 0.060 | -0.036 | 9 |
| ERS_beta_rebound_mag | 0.091 | [-0.168, 0.338] | 0.091 | -0.019 | -0.019 | 9 |
| ERD_lowgamma_L | -0.159 | [-0.282, -0.032] | 0.164 | -0.199 | 0.190 | 9 |
| ERD_lowgamma_R | -0.174 | [-0.343, 0.007] | 0.090 | -0.124 | 0.017 | 9 |
| ERD_lowgamma_mag | 0.117 | [-0.096, 0.320] | 0.117 | 0.089 | 0.089 | 9 |
| ERD_highgamma_L | 0.069 | [-0.268, 0.391] | 0.106 | 0.133 | 0.164 | 9 |
| ERD_highgamma_R | 0.107 | [-0.184, 0.381] | -0.032 | 0.142 | 0.008 | 9 |
| ERD_highgamma_mag | 0.007 | [-0.254, 0.268] | 0.007 | 0.049 | 0.049 | 9 |
| logpow_8_30 | 0.343 | [0.150, 0.511] | 0.127 | 0.304 | 0.116 | 9 |
| r2_signed | -0.088 | [-0.298, 0.131] | 0.079 | -0.106 | 0.002 | 9 |
| r2_abs | 0.079 | [-0.095, 0.248] | 0.079 | 0.002 | 0.002 | 9 |
| TFR_ERD_dB_L | -0.136 | [-0.292, 0.027] | 0.110 | -0.130 | 0.103 | 9 |
| TFR_ERD_dB_R | -0.161 | [-0.388, 0.085] | 0.088 | -0.133 | 0.012 | 9 |
| TFR_ERD_dB_mag | 0.069 | [-0.160, 0.290] | 0.069 | 0.035 | 0.035 | 9 |
| ITC_4_30 | 0.229 | [-0.086, 0.503] | 0.229 | 0.151 | 0.151 | 9 |
| ERP_abs_0.2_0.8 | 0.083 | [-0.204, 0.356] | 0.083 | 0.077 | 0.077 | 9 |
| SlowPot_0.5_3 | -0.220 | [-0.504, 0.107] | 0.356 | -0.192 | 0.317 | 9 |
| ERD_beta_classspecific | — | — | 0.174 [0.009, 0.330] | — | 0.101 | 9 |
| ERD_highgamma_classspecific | — | — | 0.036 [-0.159, 0.228] | — | 0.096 | 9 |
| ERD_lowgamma_classspecific | — | — | 0.120 [-0.037, 0.271] | — | 0.092 | 9 |
| ERD_mu_classspecific | — | — | 0.009 [-0.097, 0.115] | — | -0.050 | 9 |
| ERD_theta_classspecific | — | — | 0.059 [-0.105, 0.219] | — | 0.010 | 9 |
| ERS_beta_rebound_classspecific | — | — | 0.108 [-0.129, 0.334] | — | 0.020 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.098 [-0.056, 0.246] | — | 0.058 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +16.71 ± 13.85 | +8.97 ± 7.42 | +2.03 ± 2.62 | 0.0218 | 0.0087 |
| 4 | +19.85 ± 14.87 | +14.56 ± 12.55 | +5.60 ± 6.03 | 0.0407 | 0.0064 |
| 6 | +22.39 ± 16.62 | +17.19 ± 14.30 | +8.82 ± 9.65 | 0.0092 | 0.0062 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.078 | [-0.187, 0.034] | 9 |
| grad_x_input | 0.419 | [0.081, 0.671] | 9 |
| integrated_gradients | 0.406 | [0.070, 0.659] | 9 |
| smoothgrad | -0.079 | [-0.189, 0.033] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: False): S01=ems, S02=original, S03=original, S04=original, S05=original, S06=ems, S07=ems, S08=original, S09=original

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 71.77 |
| ems | 71.57 |
| lawhern128 | 69.55 |

```
model: "EEGNetv4"
dataset: "Schirrmeister2017"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"original": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": null, "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "kernel_length": 64}, "ems": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "kernel_length": 64}, "lawhern128": {"sfreq": 128, "band_hz": [4.0, 38.0], "window_s": [0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "kernel_length": 64}}
checkpoint_rule: "lowest validation loss, early stopping with patience; refit keeps final weights"
folds: "train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs"
seeds: [42, 43, 44]
```

Figures (50 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `2class\figures`.

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 75.00 | 82.50 | 95.00 | 98.75 | 85.00 | 98.86 | 91.70 | 97.71 | 97.95 | 91.39 | 8.62 |
| Ours | 75.00 | 82.50 | 97.50 | 97.50 | 88.75 | 98.86 | 96.02 | 99.69 | 97.73 | 92.62 | 8.68 |
| CSP | 80.00 | 80.62 | 94.38 | 97.50 | 90.62 | 99.66 | 95.00 | 99.54 | 98.07 | 92.82 | 7.64 |
| RLF | 69.38 | 81.88 | 92.50 | 97.50 | 90.00 | 98.41 | 93.07 | 97.40 | 97.27 | 90.82 | 9.59 |
| MI | 84.38 | 80.00 | 93.12 | 96.88 | 88.12 | 98.75 | 92.27 | 94.19 | 98.75 | 91.83 | 6.50 |
| Fixed sensorimotor | 73.75 | 77.50 | 93.12 | 96.88 | 87.50 | 98.64 | 89.66 | 99.39 | 97.05 | 90.39 | 9.32 |
| ERD-only | 76.88 | 77.50 | 95.62 | 96.25 | 87.50 | 99.20 | 94.77 | 97.86 | 99.43 | 91.67 | 8.93 |
| Random (mean of draws) | 74.38 | 81.00 | 93.88 | 96.00 | 86.37 | 98.48 | 91.16 | 95.11 | 96.50 | 90.32 | 8.14 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | C2 C1 C3 FC4 C4 CP2 CP1 CPz FCz FC2 FC1 Pz |
| CSP | 8 | CP1 CP2 C2 CP3 CPz C1 C3 C4 Pz CP5 FCz FC2 |
| RLF | 9 | C2 C1 C4 C3 CP3 CP1 CP2 Pz CPz FC1 CP4 FC5 |
| MI | 9 | C2 C4 C1 C3 CP3 CP1 CP2 FC4 C6 CP4 CPz FC1 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | C2 C4 C1 CP3 C3 CP1 CP2 Pz CP6 CP5 CPz CP4 |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | +1.23 | 1.96 | [+0.01, +2.45] | <0.0001 | yes | -1.25 | 0 | 3,4,5,6,8,10 |
| CSP | 9 | +1.43 | 2.70 | [-0.24, +3.11] | 0.0005 | yes | -1.87 | 0 | 4,5,6,8,10 |
| RLF | 9 | -0.56 | 2.84 | [-2.33, +1.20] | 0.0002 | yes | -5.63 | 0 | 3,4,5,6,8,10 |
| MI | 9 | +0.44 | 3.91 | [-1.98, +2.87] | 0.0014 | yes | -3.52 | 0 | 3,4,5,6,8,10 |
| Fixed sensorimotor | 9 | -1.00 | 2.20 | [-2.36, +0.36] | <0.0001 | yes | -5.00 | 0 | 3,4,5,6,8,10 |
| ERD-only | 9 | +0.28 | 2.56 | [-1.30, +1.87] | <0.0001 | yes | -5.00 | 0 | 2,3,4,5,6,8,10 |
| Random (mean of draws) | 9 | -1.07 | 1.25 | [-1.84, -0.30] | <0.0001 | yes | -2.75 | 0 | 2,3,4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | -0.20 | [-1.99, +1.58] | 0.7990 | 1.0000 | 0.9453 | 0.9453 | 4 / 1 / 4 |
| RLF | 9 | +1.80 | [-0.01, +3.60] | 0.0507 | 0.2029 | 0.0469 | 0.1875 | 7 / 1 / 1 |
| MI | 9 | +0.79 | [-2.58, +4.16] | 0.6044 | 1.0000 | 0.2891 | 0.8672 | 7 / 0 / 2 |
| Fixed sensorimotor | 9 | +2.23 | [+0.43, +4.03] | 0.0213 | 0.1066 | 0.0039 | 0.0234 | 9 / 0 / 0 |
| ERD-only | 9 | +0.95 | [-0.66, +2.56] | 0.2107 | 0.6320 | 0.3555 | 0.8672 | 6 / 0 / 3 |
| Random (mean of draws) | 9 | +2.30 | [+1.01, +3.59] | 0.0034 | 0.0202 | 0.0039 | 0.0234 | 9 / 0 / 0 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 94.68 | feet 97.8, left_hand 94.2, rest 95.7, right_hand 91.0 | feet 25.3, left_hand 26.0, rest 24.3, right_hand 24.4 | 25.01 | 0 / 9 |
| Ours | 96.07 | feet 97.8, left_hand 95.0, rest 97.1, right_hand 94.4 | feet 25.0, left_hand 25.3, rest 24.6, right_hand 25.1 | 25.01 | 0 / 9 |
| CSP | 96.14 | feet 99.0, left_hand 96.2, rest 96.0, right_hand 93.4 | feet 25.5, left_hand 25.9, rest 24.2, right_hand 24.4 | 25.01 | 0 / 9 |
| RLF | 94.48 | feet 97.7, left_hand 93.3, rest 95.9, right_hand 91.1 | feet 25.1, left_hand 25.8, rest 24.3, right_hand 24.7 | 25.01 | 0 / 9 |
| MI | 94.63 | feet 98.0, left_hand 94.7, rest 95.7, right_hand 90.0 | feet 25.3, left_hand 26.3, rest 24.3, right_hand 24.1 | 25.01 | 0 / 9 |
| Fixed sensorimotor | 93.97 | feet 97.5, left_hand 91.5, rest 95.8, right_hand 91.1 | feet 25.1, left_hand 25.3, rest 24.5, right_hand 25.1 | 25.01 | 0 / 9 |
| ERD-only | 95.65 | feet 98.3, left_hand 95.8, rest 96.3, right_hand 92.2 | feet 25.4, left_hand 26.0, rest 24.4, right_hand 24.3 | 25.01 | 0 / 9 |
| Random (mean of draws) | 93.57 | feet 97.6, left_hand 92.1, rest 95.5, right_hand 89.2 | feet 25.3, left_hand 25.9, rest 24.3, right_hand 24.5 | 25.01 | 0 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 90.98, 43: 89.21, 44: 88.07 | 89.42 | 1.47 |
| Ours | 3 | 42: 89.35, 43: 90.57, 44: 89.77 | 89.90 | 0.62 |
| CSP | 3 | 42: 90.59, 43: 88.80, 44: 89.71 | 89.70 | 0.89 |
| RLF | 3 | 42: 86.70, 43: 88.42, 44: 87.51 | 87.54 | 0.86 |
| MI | 3 | 42: 88.91, 43: 89.58, 44: 87.07 | 88.52 | 1.30 |
| Fixed sensorimotor | 3 | 42: 86.88, 43: 86.55, 44: 88.03 | 87.15 | 0.78 |
| ERD-only | 3 | 42: 89.33, 43: 89.50, 44: 89.09 | 89.30 | 0.21 |
| Random (mean of draws) | 3 | 42: 87.70, 43: 86.81, 44: 87.24 | 87.25 | 0.44 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.390 | [0.147, 0.589] | 0.298 | 9 |
| grad | class_specific_pearson | 0.401 | [0.206, 0.566] | 0.243 | 9 |
| grad | lateralisation_pearson | -0.118 | [-0.327, 0.103] | 0.269 | 9 |
| grad | magnitude_spearman | 0.309 | [0.023, 0.547] | 0.338 | 9 |
| grad | class_specific_spearman | 0.408 | [0.252, 0.544] | 0.195 | 9 |
| grad | lateralisation_spearman | -0.076 | [-0.300, 0.156] | 0.289 | 9 |
| grad | legacy_signed_LR_pearson | 0.191 | [0.062, 0.313] | 0.162 | 9 |
| grad_x_input | magnitude_pearson | 0.354 | [0.120, 0.550] | 0.282 | 9 |
| grad_x_input | class_specific_pearson | 0.287 | [0.081, 0.469] | 0.251 | 9 |
| grad_x_input | lateralisation_pearson | -0.203 | [-0.418, 0.034] | 0.287 | 9 |
| grad_x_input | magnitude_spearman | 0.282 | [0.006, 0.517] | 0.324 | 9 |
| grad_x_input | class_specific_spearman | 0.272 | [0.084, 0.441] | 0.225 | 9 |
| grad_x_input | lateralisation_spearman | -0.171 | [-0.430, 0.114] | 0.339 | 9 |
| grad_x_input | legacy_signed_LR_pearson | 0.180 | [0.046, 0.308] | 0.169 | 9 |
| integrated_gradients | magnitude_pearson | 0.320 | [0.096, 0.512] | 0.274 | 9 |
| integrated_gradients | class_specific_pearson | 0.252 | [0.054, 0.431] | 0.240 | 9 |
| integrated_gradients | lateralisation_pearson | -0.199 | [-0.386, 0.003] | 0.245 | 9 |
| integrated_gradients | magnitude_spearman | 0.249 | [-0.025, 0.487] | 0.324 | 9 |
| integrated_gradients | class_specific_spearman | 0.224 | [0.022, 0.409] | 0.244 | 9 |
| integrated_gradients | lateralisation_spearman | -0.181 | [-0.417, 0.079] | 0.314 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | 0.179 | [0.038, 0.313] | 0.178 | 9 |
| smoothgrad | magnitude_pearson | 0.391 | [0.147, 0.591] | 0.298 | 9 |
| smoothgrad | class_specific_pearson | 0.403 | [0.206, 0.568] | 0.245 | 9 |
| smoothgrad | lateralisation_pearson | -0.111 | [-0.320, 0.109] | 0.268 | 9 |
| smoothgrad | magnitude_spearman | 0.307 | [0.021, 0.547] | 0.338 | 9 |
| smoothgrad | class_specific_spearman | 0.407 | [0.249, 0.545] | 0.198 | 9 |
| smoothgrad | lateralisation_spearman | -0.057 | [-0.279, 0.170] | 0.284 | 9 |
| smoothgrad | legacy_signed_LR_pearson | 0.190 | [0.061, 0.313] | 0.162 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.400 | [-0.568, -0.200] | 9 |
| grad | ERD_L_spearman | -0.353 | [-0.518, -0.162] | 9 |
| grad | ERD_R_pearson | -0.459 | [-0.627, -0.251] | 9 |
| grad | ERD_R_spearman | -0.456 | [-0.610, -0.268] | 9 |
| grad | ERD_LR_pearson | 0.191 | [0.062, 0.313] | 9 |
| grad | ERD_LR_spearman | 0.232 | [0.091, 0.364] | 9 |
| grad | ERD_comb_pearson | -0.445 | [-0.611, -0.240] | 9 |
| grad | ERD_comb_spearman | -0.430 | [-0.572, -0.263] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.357 | [-0.531, -0.155] | 9 |
| grad_x_input | ERD_L_spearman | -0.306 | [-0.490, -0.097] | 9 |
| grad_x_input | ERD_R_pearson | -0.418 | [-0.593, -0.205] | 9 |
| grad_x_input | ERD_R_spearman | -0.408 | [-0.585, -0.193] | 9 |
| grad_x_input | ERD_LR_pearson | 0.180 | [0.046, 0.308] | 9 |
| grad_x_input | ERD_LR_spearman | 0.218 | [0.073, 0.355] | 9 |
| grad_x_input | ERD_comb_pearson | -0.403 | [-0.576, -0.195] | 9 |
| grad_x_input | ERD_comb_spearman | -0.384 | [-0.549, -0.190] | 9 |
| grad_x_input | vs_grad_pearson | 0.998 | [0.995, 0.999] | 9 |
| grad_x_input | vs_grad_spearman | 0.997 | [0.973, 1.000] | 9 |
| integrated_gradients | ERD_L_pearson | -0.331 | [-0.497, -0.143] | 9 |
| integrated_gradients | ERD_L_spearman | -0.284 | [-0.446, -0.105] | 9 |
| integrated_gradients | ERD_R_pearson | -0.398 | [-0.563, -0.202] | 9 |
| integrated_gradients | ERD_R_spearman | -0.402 | [-0.560, -0.215] | 9 |
| integrated_gradients | ERD_LR_pearson | 0.179 | [0.038, 0.313] | 9 |
| integrated_gradients | ERD_LR_spearman | 0.240 | [0.096, 0.374] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.382 | [-0.546, -0.190] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.372 | [-0.515, -0.208] | 9 |
| integrated_gradients | vs_grad_pearson | 0.991 | [0.984, 0.995] | 9 |
| integrated_gradients | vs_grad_spearman | 0.980 | [0.966, 0.989] | 9 |
| smoothgrad | ERD_L_pearson | -0.401 | [-0.570, -0.200] | 9 |
| smoothgrad | ERD_L_spearman | -0.351 | [-0.518, -0.157] | 9 |
| smoothgrad | ERD_R_pearson | -0.460 | [-0.628, -0.251] | 9 |
| smoothgrad | ERD_R_spearman | -0.453 | [-0.611, -0.261] | 9 |
| smoothgrad | ERD_LR_pearson | 0.190 | [0.061, 0.313] | 9 |
| smoothgrad | ERD_LR_spearman | 0.230 | [0.087, 0.364] | 9 |
| smoothgrad | ERD_comb_pearson | -0.445 | [-0.613, -0.240] | 9 |
| smoothgrad | ERD_comb_spearman | -0.427 | [-0.572, -0.256] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [0.999, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.058 | [-0.281, 0.170] | 0.223 | 0.048 | 0.232 | 9 |
| ERD_theta_R | -0.095 | [-0.290, 0.109] | 0.096 | -0.012 | 0.120 | 9 |
| ERD_theta_mag | 0.152 | [-0.074, 0.362] | 0.152 | 0.139 | 0.139 | 9 |
| ERD_mu_L | -0.320 | [-0.520, -0.088] | 0.320 | -0.270 | 0.250 | 9 |
| ERD_mu_R | -0.372 | [-0.559, -0.149] | 0.224 | -0.371 | 0.158 | 9 |
| ERD_mu_mag | 0.237 | [-0.074, 0.506] | 0.237 | 0.159 | 0.159 | 9 |
| ERD_beta_L | -0.469 | [-0.597, -0.319] | 0.519 | -0.398 | 0.430 | 9 |
| ERD_beta_R | -0.487 | [-0.633, -0.306] | 0.441 | -0.472 | 0.370 | 9 |
| ERD_beta_mag | 0.463 | [0.222, 0.651] | 0.463 | 0.382 | 0.382 | 9 |
| ERS_beta_rebound_L | -0.251 | [-0.465, -0.010] | 0.270 | -0.207 | 0.275 | 9 |
| ERS_beta_rebound_R | -0.273 | [-0.497, -0.013] | 0.302 | -0.274 | 0.294 | 9 |
| ERS_beta_rebound_mag | 0.317 | [0.057, 0.537] | 0.317 | 0.315 | 0.315 | 9 |
| ERD_lowgamma_L | -0.416 | [-0.620, -0.159] | 0.363 | -0.444 | 0.378 | 9 |
| ERD_lowgamma_R | -0.451 | [-0.657, -0.182] | 0.328 | -0.448 | 0.323 | 9 |
| ERD_lowgamma_mag | 0.307 | [-0.071, 0.608] | 0.307 | 0.358 | 0.358 | 9 |
| ERD_highgamma_L | 0.030 | [-0.273, 0.328] | 0.231 | 0.156 | 0.300 | 9 |
| ERD_highgamma_R | 0.164 | [-0.134, 0.435] | 0.144 | 0.191 | 0.214 | 9 |
| ERD_highgamma_mag | 0.137 | [-0.137, 0.392] | 0.137 | 0.266 | 0.266 | 9 |
| logpow_8_30 | 0.171 | [0.015, 0.318] | -0.351 | -0.076 | -0.337 | 9 |
| r2_signed | -0.025 | [-0.156, 0.106] | -0.043 | -0.008 | 0.044 | 9 |
| r2_abs | -0.043 | [-0.181, 0.095] | -0.043 | 0.044 | 0.044 | 9 |
| TFR_ERD_dB_L | -0.486 | [-0.622, -0.321] | 0.521 | -0.425 | 0.443 | 9 |
| TFR_ERD_dB_R | -0.518 | [-0.659, -0.340] | 0.435 | -0.480 | 0.378 | 9 |
| TFR_ERD_dB_mag | 0.459 | [0.175, 0.673] | 0.459 | 0.410 | 0.410 | 9 |
| ITC_4_30 | 0.229 | [0.053, 0.391] | 0.229 | 0.128 | 0.128 | 9 |
| ERP_abs_0.2_0.8 | -0.026 | [-0.121, 0.070] | -0.026 | -0.001 | -0.001 | 9 |
| SlowPot_0.5_3 | -0.005 | [-0.121, 0.110] | -0.018 | 0.013 | 0.039 | 9 |
| ERD_beta_classspecific | — | — | 0.414 [0.267, 0.542] | — | 0.342 | 9 |
| ERD_highgamma_classspecific | — | — | 0.194 [0.010, 0.365] | — | 0.270 | 9 |
| ERD_lowgamma_classspecific | — | — | 0.271 [0.001, 0.504] | — | 0.291 | 9 |
| ERD_mu_classspecific | — | — | 0.228 [-0.016, 0.446] | — | 0.154 | 9 |
| ERD_theta_classspecific | — | — | 0.183 [0.037, 0.321] | — | 0.165 | 9 |
| ERS_beta_rebound_classspecific | — | — | 0.234 [0.032, 0.418] | — | 0.238 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.407 [0.223, 0.563] | — | 0.353 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +37.94 ± 15.41 | +16.63 ± 8.96 | +4.15 ± 3.44 | <0.0001 | <0.0001 |
| 4 | +42.23 ± 14.90 | +29.95 ± 10.68 | +14.02 ± 10.80 | 0.0142 | <0.0001 |
| 6 | +48.81 ± 11.32 | +37.27 ± 10.40 | +22.38 ± 9.13 | 0.0037 | <0.0001 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.061 | [-0.032, 0.152] | 9 |
| grad_x_input | 0.001 | [-0.092, 0.095] | 9 |
| integrated_gradients | 0.026 | [-0.064, 0.116] | 9 |
| smoothgrad | 0.060 | [-0.035, 0.155] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=lawhern128, S02=lawhern128, S03=lawhern128, S04=lawhern128, S05=lawhern128, S06=lawhern128, S07=lawhern128, S08=lawhern128, S09=lawhern128

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 79.02 |
| ems | 78.48 |
| lawhern128 | 83.91 |

```
model: "EEGNetv4"
dataset: "Schirrmeister2017"
sfreq_hz: "per profile (profiles[*].sfreq)"
profiles: {"original": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": null, "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "kernel_length": 64}, "ems": {"sfreq": 100, "band_hz": [4.0, 38.0], "window_s": [0.0, 4.0], "norm": "ems", "car": false, "lr": 0.0005, "weight_decay": 0.0001, "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 600, "patience": 150, "batch_size": 96, "label_smoothing": 0.1, "grad_clip": 10.0, "dropout": 0.3, "augment": false, "kernel_length": 64}, "lawhern128": {"sfreq": 128, "band_hz": [4.0, 38.0], "window_s": [0.5, 4.0], "norm": "ems", "car": false, "lr": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "plateau_factor": 0.5, "plateau_patience": 30, "max_epochs": 500, "patience": 100, "batch_size": 64, "label_smoothing": 0.0, "grad_clip": 10.0, "dropout": 0.25, "augment": false, "kernel_length": 64}}
checkpoint_rule: "lowest validation loss, early stopping with patience; refit keeps final weights"
folds: "train on the official HGD training runs (last 25% = validation, chronological), refit on all training runs, test once on the official test runs"
seeds: [42, 43, 44]
```

Figures (50 topomaps: saliency per method/subject, group maps, ERD/ERS) copied to `4class\figures`.
