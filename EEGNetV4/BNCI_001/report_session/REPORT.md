# Results report — EEGNetV4/BNCI_001

Generated 2026-10-17 10:50 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 50.69 | 45.14 | 93.06 | 63.89 | 52.08 | 59.03 | 57.64 | 92.36 | 90.28 | 67.13 | 19.34 |
| Ours | 50.69 | 50.00 | 88.89 | 63.19 | 55.56 | 55.56 | 51.39 | 93.06 | 88.19 | 66.28 | 18.29 |
| CSP | 80.56 | 50.00 | 94.44 | 63.89 | 50.69 | 51.39 | 55.56 | 92.36 | 90.97 | 69.98 | 19.41 |
| RLF | 50.00 | 50.00 | 90.97 | 59.03 | 51.39 | 62.50 | 55.56 | 92.36 | 89.58 | 66.82 | 18.59 |
| MI | 50.00 | 50.00 | 90.97 | 59.03 | 54.17 | 57.64 | 52.78 | 90.97 | 90.28 | 66.20 | 18.65 |
| Fixed sensorimotor | 83.33 | 50.00 | 93.75 | 68.06 | 53.47 | 55.56 | 56.25 | 93.75 | 88.89 | 71.45 | 18.43 |
| ERD-only | 51.39 | 50.00 | 88.19 | 49.31 | 52.08 | 53.47 | 54.17 | 92.36 | 90.28 | 64.58 | 19.36 |
| Random (mean of draws) | 58.19 | 50.00 | 90.28 | 62.22 | 52.78 | 58.19 | 53.47 | 90.28 | 89.03 | 67.16 | 17.40 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | CP4 C4 C6 Fz C3 CP3 FCz FC2 CP2 Cz C5 FC4 |
| CSP | 9 | CP4 C4 CP2 P2 CP3 C6 Fz Pz C3 POz P1 C2 |
| RLF | 9 | CP4 C4 FC3 P2 FC1 P1 FCz POz Fz C3 FC2 C2 |
| MI | 9 | FC1 C3 C4 Pz CP4 Fz POz C1 Cz CP1 CP3 C2 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -0.85 | 3.59 | [-3.07, +1.38] | 0.0013 | yes | -6.25 | 1 | 4,5,6,8,10 |
| CSP | 9 | +2.85 | 10.66 | [-3.75, +9.46] | 0.2009 | no | -7.64 | 1 | 10 |
| RLF | 9 | -0.31 | 2.93 | [-2.12, +1.51] | 0.0002 | yes | -4.86 | 0 | 3,4,5,6,8,10 |
| MI | 9 | -0.93 | 3.09 | [-2.84, +0.99] | 0.0006 | yes | -4.86 | 0 | 3,4,5,6,8,10 |
| Fixed sensorimotor | 9 | +4.32 | 10.95 | [-2.46, +11.11] | 0.3288 | no | -3.47 | 0 | none |
| ERD-only | 9 | -2.55 | 5.53 | [-5.98, +0.88] | 0.0490 | yes | -14.58 | 1 | 6,8,10 |
| Random (mean of draws) | 9 | +0.03 | 3.79 | [-2.32, +2.38] | 0.0007 | yes | -4.17 | 0 | 3,4,5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | -3.70 | [-11.70, +4.29] | 0.3167 | 1.0000 | 0.4453 | 1.0000 | 3 / 1 / 5 |
| RLF | 9 | -0.54 | [-3.32, +2.24] | 0.6664 | 1.0000 | 0.6719 | 1.0000 | 4 / 1 / 4 |
| MI | 9 | +0.08 | [-1.62, +1.77] | 0.9191 | 1.0000 | 0.9688 | 1.0000 | 4 / 1 / 4 |
| Fixed sensorimotor | 9 | -5.17 | [-13.33, +2.99] | 0.1820 | 1.0000 | 0.0781 | 0.4688 | 1 / 2 / 6 |
| ERD-only | 9 | +1.70 | [-2.12, +5.51] | 0.3346 | 1.0000 | 0.4844 | 1.0000 | 5 / 1 / 3 |
| Random (mean of draws) | 9 | -0.88 | [-3.30, +1.54] | 0.4271 | 1.0000 | 0.7422 | 1.0000 | 3 / 1 / 5 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 67.13 | left_hand 77.3, right_hand 56.9 | left_hand 60.2, right_hand 39.8 | 50.00 | 2 / 9 |
| Ours | 66.28 | left_hand 56.5, right_hand 76.1 | left_hand 40.2, right_hand 59.8 | 50.00 | 2 / 9 |
| CSP | 69.98 | left_hand 69.6, right_hand 70.4 | left_hand 49.6, right_hand 50.4 | 50.00 | 2 / 9 |
| RLF | 66.82 | left_hand 59.1, right_hand 74.5 | left_hand 42.3, right_hand 57.7 | 50.00 | 2 / 9 |
| MI | 66.20 | left_hand 58.3, right_hand 74.1 | left_hand 42.1, right_hand 57.9 | 50.00 | 3 / 9 |
| Fixed sensorimotor | 71.45 | left_hand 63.4, right_hand 79.5 | left_hand 42.0, right_hand 58.0 | 50.00 | 1 / 9 |
| ERD-only | 64.58 | left_hand 62.2, right_hand 67.0 | left_hand 47.6, right_hand 52.4 | 50.00 | 4 / 9 |
| Random (mean of draws) | 67.16 | left_hand 61.5, right_hand 72.8 | left_hand 44.4, right_hand 55.6 | 50.00 | 10 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 65.28, 43: 59.88, 44: 66.51 | 63.89 | 3.53 |
| Ours | 3 | 42: 66.05, 43: 64.20, 44: 61.88 | 64.04 | 2.09 |
| CSP | 3 | 42: 65.43, 43: 67.75, 44: 64.12 | 65.77 | 1.84 |
| RLF | 3 | 42: 66.98, 43: 60.80, 44: 64.35 | 64.04 | 3.10 |
| MI | 3 | 42: 61.73, 43: 65.35, 44: 64.04 | 63.71 | 1.84 |
| Fixed sensorimotor | 3 | 42: 68.52, 43: 64.74, 44: 68.36 | 67.21 | 2.14 |
| ERD-only | 3 | 42: 65.82, 43: 65.28, 44: 59.34 | 63.48 | 3.60 |
| Random (mean of draws) | 3 | 42: 65.76, 43: 65.48, 44: 63.50 | 64.91 | 1.23 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.118 | [-0.080, 0.307] | 0.248 | 9 |
| grad | class_specific_pearson | 0.224 | [0.089, 0.351] | 0.172 | 9 |
| grad | lateralisation_pearson | 0.011 | [-0.300, 0.320] | 0.378 | 9 |
| grad | magnitude_spearman | 0.086 | [-0.105, 0.271] | 0.236 | 9 |
| grad | class_specific_spearman | 0.196 | [0.054, 0.331] | 0.175 | 9 |
| grad | lateralisation_spearman | 0.008 | [-0.273, 0.287] | 0.345 | 9 |
| grad | legacy_signed_LR_pearson | -0.187 | [-0.432, 0.084] | 0.326 | 9 |
| grad_x_input | magnitude_pearson | 0.062 | [-0.146, 0.265] | 0.262 | 9 |
| grad_x_input | class_specific_pearson | 0.216 | [0.063, 0.358] | 0.194 | 9 |
| grad_x_input | lateralisation_pearson | -0.070 | [-0.398, 0.274] | 0.388 | 9 |
| grad_x_input | magnitude_spearman | 0.043 | [-0.166, 0.248] | 0.259 | 9 |
| grad_x_input | class_specific_spearman | 0.178 | [0.024, 0.324] | 0.192 | 9 |
| grad_x_input | lateralisation_spearman | -0.083 | [-0.387, 0.238] | 0.360 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.221 | [-0.420, -0.001] | 0.267 | 9 |
| integrated_gradients | magnitude_pearson | 0.072 | [-0.139, 0.276] | 0.264 | 9 |
| integrated_gradients | class_specific_pearson | 0.219 | [0.061, 0.366] | 0.199 | 9 |
| integrated_gradients | lateralisation_pearson | -0.078 | [-0.411, 0.273] | 0.393 | 9 |
| integrated_gradients | magnitude_spearman | 0.058 | [-0.157, 0.268] | 0.267 | 9 |
| integrated_gradients | class_specific_spearman | 0.185 | [0.011, 0.347] | 0.213 | 9 |
| integrated_gradients | lateralisation_spearman | -0.086 | [-0.383, 0.227] | 0.359 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.225 | [-0.427, -0.003] | 0.270 | 9 |
| smoothgrad | magnitude_pearson | 0.118 | [-0.080, 0.307] | 0.249 | 9 |
| smoothgrad | class_specific_pearson | 0.224 | [0.089, 0.352] | 0.172 | 9 |
| smoothgrad | lateralisation_pearson | 0.012 | [-0.298, 0.320] | 0.377 | 9 |
| smoothgrad | magnitude_spearman | 0.086 | [-0.105, 0.271] | 0.236 | 9 |
| smoothgrad | class_specific_spearman | 0.197 | [0.055, 0.331] | 0.175 | 9 |
| smoothgrad | lateralisation_spearman | 0.010 | [-0.270, 0.288] | 0.343 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.187 | [-0.432, 0.084] | 0.327 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.263 | [-0.434, -0.072] | 9 |
| grad | ERD_L_spearman | -0.210 | [-0.389, -0.016] | 9 |
| grad | ERD_R_pearson | -0.157 | [-0.359, 0.060] | 9 |
| grad | ERD_R_spearman | -0.153 | [-0.355, 0.062] | 9 |
| grad | ERD_LR_pearson | -0.187 | [-0.432, 0.084] | 9 |
| grad | ERD_LR_spearman | -0.102 | [-0.305, 0.111] | 9 |
| grad | ERD_comb_pearson | -0.218 | [-0.388, -0.034] | 9 |
| grad | ERD_comb_spearman | -0.185 | [-0.379, 0.024] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.283 | [-0.425, -0.127] | 9 |
| grad_x_input | ERD_L_spearman | -0.229 | [-0.364, -0.086] | 9 |
| grad_x_input | ERD_R_pearson | -0.128 | [-0.337, 0.093] | 9 |
| grad_x_input | ERD_R_spearman | -0.107 | [-0.311, 0.106] | 9 |
| grad_x_input | ERD_LR_pearson | -0.221 | [-0.420, -0.001] | 9 |
| grad_x_input | ERD_LR_spearman | -0.146 | [-0.299, 0.014] | 9 |
| grad_x_input | ERD_comb_pearson | -0.220 | [-0.394, -0.030] | 9 |
| grad_x_input | ERD_comb_spearman | -0.194 | [-0.386, 0.015] | 9 |
| grad_x_input | vs_grad_pearson | 0.994 | [0.985, 0.997] | 9 |
| grad_x_input | vs_grad_spearman | 0.980 | [0.962, 0.989] | 9 |
| integrated_gradients | ERD_L_pearson | -0.288 | [-0.438, -0.123] | 9 |
| integrated_gradients | ERD_L_spearman | -0.253 | [-0.408, -0.085] | 9 |
| integrated_gradients | ERD_R_pearson | -0.131 | [-0.347, 0.097] | 9 |
| integrated_gradients | ERD_R_spearman | -0.115 | [-0.319, 0.099] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.225 | [-0.427, -0.003] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.170 | [-0.329, -0.000] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.224 | [-0.406, -0.026] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.209 | [-0.410, 0.012] | 9 |
| integrated_gradients | vs_grad_pearson | 0.989 | [0.981, 0.994] | 9 |
| integrated_gradients | vs_grad_spearman | 0.972 | [0.947, 0.986] | 9 |
| smoothgrad | ERD_L_pearson | -0.263 | [-0.434, -0.072] | 9 |
| smoothgrad | ERD_L_spearman | -0.210 | [-0.389, -0.016] | 9 |
| smoothgrad | ERD_R_pearson | -0.157 | [-0.359, 0.060] | 9 |
| smoothgrad | ERD_R_spearman | -0.153 | [-0.355, 0.062] | 9 |
| smoothgrad | ERD_LR_pearson | -0.187 | [-0.432, 0.084] | 9 |
| smoothgrad | ERD_LR_spearman | -0.102 | [-0.305, 0.111] | 9 |
| smoothgrad | ERD_comb_pearson | -0.218 | [-0.388, -0.034] | 9 |
| smoothgrad | ERD_comb_spearman | -0.185 | [-0.379, 0.024] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.139 | [-0.274, 0.002] | 0.029 | -0.123 | 0.032 | 9 |
| ERD_theta_R | -0.041 | [-0.220, 0.142] | -0.011 | -0.059 | 0.015 | 9 |
| ERD_theta_mag | 0.028 | [-0.127, 0.182] | 0.028 | 0.024 | 0.024 | 9 |
| ERD_mu_L | -0.296 | [-0.497, -0.064] | 0.191 | -0.224 | 0.171 | 9 |
| ERD_mu_R | -0.143 | [-0.318, 0.042] | 0.086 | -0.093 | 0.056 | 9 |
| ERD_mu_mag | 0.188 | [-0.035, 0.394] | 0.188 | 0.167 | 0.167 | 9 |
| ERD_beta_L | -0.110 | [-0.275, 0.062] | 0.094 | -0.067 | 0.072 | 9 |
| ERD_beta_R | 0.010 | [-0.212, 0.231] | 0.098 | 0.028 | 0.105 | 9 |
| ERD_beta_mag | 0.110 | [-0.111, 0.320] | 0.110 | 0.127 | 0.127 | 9 |
| ERS_beta_rebound_L | -0.092 | [-0.310, 0.136] | 0.012 | -0.038 | 0.050 | 9 |
| ERS_beta_rebound_R | -0.068 | [-0.279, 0.149] | -0.139 | -0.022 | -0.119 | 9 |
| ERS_beta_rebound_mag | -0.065 | [-0.298, 0.175] | -0.065 | 0.021 | 0.021 | 9 |
| ERD_lowgamma_L | -0.081 | [-0.253, 0.095] | 0.004 | -0.010 | -0.003 | 9 |
| ERD_lowgamma_R | -0.063 | [-0.186, 0.062] | -0.122 | -0.038 | -0.058 | 9 |
| ERD_lowgamma_mag | -0.040 | [-0.253, 0.177] | -0.040 | 0.031 | 0.031 | 9 |
| logpow_8_30 | -0.009 | [-0.200, 0.184] | -0.009 | -0.075 | -0.075 | 9 |
| r2_signed | 0.284 | [0.009, 0.520] | 0.363 | 0.153 | 0.153 | 9 |
| r2_abs | 0.363 | [0.180, 0.521] | 0.363 | 0.153 | 0.153 | 9 |
| TFR_ERD_dB_L | -0.241 | [-0.408, -0.057] | 0.037 | -0.188 | -0.025 | 9 |
| TFR_ERD_dB_R | -0.110 | [-0.331, 0.123] | 0.048 | -0.100 | 0.018 | 9 |
| TFR_ERD_dB_mag | 0.055 | [-0.150, 0.255] | 0.055 | 0.001 | 0.001 | 9 |
| ITC_4_30 | -0.130 | [-0.323, 0.074] | -0.130 | -0.127 | -0.127 | 9 |
| ERP_abs_0.2_0.8 | -0.093 | [-0.271, 0.092] | -0.093 | -0.096 | -0.096 | 9 |
| SlowPot_0.5_3 | -0.056 | [-0.215, 0.107] | 0.013 | -0.072 | 0.031 | 9 |
| ERD_beta_classspecific | — | — | 0.055 [-0.089, 0.197] | — | 0.062 | 9 |
| ERD_lowgamma_classspecific | — | — | -0.081 [-0.162, 0.001] | — | -0.055 | 9 |
| ERD_mu_classspecific | — | — | 0.089 [-0.113, 0.285] | — | 0.055 | 9 |
| ERD_theta_classspecific | — | — | 0.015 [-0.101, 0.131] | — | 0.047 | 9 |
| ERS_beta_rebound_classspecific | — | — | -0.092 [-0.291, 0.115] | — | -0.054 | 9 |
| TFR_ERD_dB_classspecific | — | — | -0.024 [-0.217, 0.170] | — | -0.068 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +4.73 ± 6.58 | +0.70 ± 0.89 | +0.28 ± 0.90 | 0.0806 | 0.0708 |
| 4 | +8.20 ± 11.15 | +1.75 ± 1.99 | +1.21 ± 1.40 | 0.0825 | 0.0702 |
| 6 | +9.23 ± 12.92 | +3.07 ± 3.59 | +2.11 ± 1.65 | 0.0945 | 0.1033 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.021 | [-0.053, 0.095] | 9 |
| grad_x_input | 0.031 | [-0.044, 0.106] | 9 |
| integrated_gradients | 0.047 | [-0.017, 0.111] | 9 |
| smoothgrad | 0.021 | [-0.053, 0.094] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: False): S01=original, S02=k64_aug, S03=k64_aug, S04=original, S05=k64_aug, S06=k64_aug, S07=k64_aug, S08=k64_aug, S09=k64_aug

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 68.21 |
| k64 | 63.58 |
| k64_long | 63.27 |
| k64_aug | 68.83 |
| k64_long_aug | 63.27 |

```
model: "EEGNetv4"
dataset: "BNCI2014_001"
sfreq_hz: 128
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 32, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64_long": {"band_hz": [4.0, 40.0], "window_s": [0.5, 4.0], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64_aug": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": true}, "k64_long_aug": {"band_hz": [4.0, 40.0], "window_s": [0.5, 4.0], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": true}}
checkpoint_rule: "lowest validation loss, early stopping with patience"
folds: "train on session 1 (last 25% of it = validation, chronological), refit on all of session 1, test once on session 2 (official BCI IV-2a split)"
seeds: [42, 43, 44]
```

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 77.78 | 27.08 | 83.33 | 50.69 | 46.18 | 45.83 | 73.61 | 74.31 | 85.76 | 62.73 | 20.64 |
| Ours | 71.88 | 43.75 | 77.43 | 51.39 | 31.25 | 40.28 | 71.18 | 76.74 | 84.03 | 60.88 | 19.28 |
| CSP | 70.83 | 27.78 | 78.12 | 48.96 | 25.00 | 42.36 | 67.01 | 74.65 | 84.38 | 57.68 | 22.23 |
| RLF | 72.57 | 26.04 | 76.74 | 44.44 | 34.38 | 40.97 | 69.79 | 73.96 | 79.86 | 57.64 | 20.88 |
| MI | 71.53 | 37.15 | 80.56 | 43.40 | 37.15 | 42.71 | 70.49 | 76.74 | 78.82 | 59.84 | 19.10 |
| Fixed sensorimotor | 73.96 | 41.67 | 82.99 | 45.14 | 35.42 | 43.40 | 78.47 | 76.39 | 79.51 | 61.88 | 19.74 |
| ERD-only | 70.83 | 40.28 | 82.29 | 46.18 | 28.12 | 46.18 | 68.40 | 78.82 | 81.25 | 60.26 | 20.24 |
| Random (mean of draws) | 71.94 | 34.44 | 80.21 | 48.33 | 25.42 | 45.14 | 66.88 | 75.56 | 82.22 | 58.90 | 21.00 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | C3 CP3 CPz C4 CP2 POz Cz P1 FCz CP4 P2 C1 |
| CSP | 9 | P1 CP3 P2 Pz CP1 POz C3 CP4 CP2 CPz C1 C4 |
| RLF | 9 | C3 P1 CP3 Pz FC3 P2 POz CP4 FC1 CPz CP1 Fz |
| MI | 9 | CP3 C3 Pz C1 CP1 P2 CPz CP2 FC2 CP4 P1 FC1 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -1.85 | 8.56 | [-7.16, +3.45] | 0.0920 | no | -14.93 | 1 | 8,10 |
| CSP | 9 | -5.05 | 6.66 | [-9.18, -0.92] | 0.3407 | no | -21.18 | 3 | 10 |
| RLF | 9 | -5.09 | 3.35 | [-7.17, -3.01] | 0.2202 | no | -11.81 | 3 | 8,10 |
| MI | 9 | -2.89 | 5.93 | [-6.57, +0.78] | 0.0773 | no | -9.03 | 4 | 8,10 |
| Fixed sensorimotor | 9 | -0.85 | 7.42 | [-5.45, +3.75] | 0.0354 | yes | -10.76 | 2 | 6,8,10 |
| ERD-only | 9 | -2.47 | 8.54 | [-7.76, +2.82] | 0.1249 | no | -18.06 | 2 | 8,10 |
| Random (mean of draws) | 9 | -3.83 | 7.61 | [-8.54, +0.89] | 0.2082 | no | -20.76 | 2 | 10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | +3.20 | [-0.97, +7.37] | 0.1146 | 0.6876 | 0.0742 | 0.4453 | 6 / 0 / 3 |
| RLF | 9 | +3.24 | [-1.51, +7.99] | 0.1543 | 0.7715 | 0.1602 | 0.8008 | 6 / 0 / 3 |
| MI | 9 | +1.04 | [-2.56, +4.64] | 0.5237 | 1.0000 | 0.5469 | 1.0000 | 5 / 1 / 3 |
| Fixed sensorimotor | 9 | -1.00 | [-4.55, +2.54] | 0.5326 | 1.0000 | 0.5703 | 1.0000 | 4 / 0 / 5 |
| ERD-only | 9 | +0.62 | [-2.42, +3.65] | 0.6517 | 1.0000 | 0.6250 | 1.0000 | 6 / 0 / 3 |
| Random (mean of draws) | 9 | +1.98 | [-1.35, +5.30] | 0.2075 | 0.8300 | 0.2500 | 1.0000 | 6 / 0 / 3 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 62.73 | feet 48.9, left_hand 71.3, right_hand 69.0, tongue 61.7 | feet 18.0, left_hand 29.7, right_hand 29.4, tongue 22.8 | 25.00 | 0 / 9 |
| Ours | 60.88 | feet 55.2, left_hand 68.1, right_hand 59.0, tongue 61.3 | feet 19.8, left_hand 30.8, right_hand 25.2, tongue 24.3 | 25.00 | 0 / 9 |
| CSP | 57.68 | feet 46.0, left_hand 73.0, right_hand 58.3, tongue 53.4 | feet 16.2, left_hand 37.4, right_hand 26.8, tongue 19.6 | 25.00 | 1 / 9 |
| RLF | 57.64 | feet 38.6, left_hand 65.1, right_hand 69.3, tongue 57.6 | feet 15.4, left_hand 28.2, right_hand 34.0, tongue 22.4 | 25.00 | 0 / 9 |
| MI | 59.84 | feet 50.2, left_hand 69.8, right_hand 61.9, tongue 57.6 | feet 19.1, left_hand 31.4, right_hand 27.3, tongue 22.2 | 25.00 | 0 / 9 |
| Fixed sensorimotor | 61.88 | feet 49.7, left_hand 70.7, right_hand 65.6, tongue 61.6 | feet 18.5, left_hand 29.7, right_hand 27.3, tongue 24.5 | 25.00 | 0 / 9 |
| ERD-only | 60.26 | feet 56.0, left_hand 67.6, right_hand 59.7, tongue 57.7 | feet 21.2, left_hand 31.3, right_hand 26.3, tongue 21.2 | 25.00 | 0 / 9 |
| Random (mean of draws) | 58.90 | feet 46.0, left_hand 70.0, right_hand 59.5, tongue 60.1 | feet 17.0, left_hand 32.8, right_hand 25.5, tongue 24.7 | 25.00 | 2 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 59.99, 43: 59.57, 44: 58.18 | 59.25 | 0.95 |
| Ours | 3 | 42: 56.87, 43: 56.40, 44: 59.92 | 57.73 | 1.91 |
| CSP | 3 | 42: 54.94, 43: 55.86, 44: 56.13 | 55.65 | 0.63 |
| RLF | 3 | 42: 54.67, 43: 54.63, 44: 55.90 | 55.07 | 0.72 |
| MI | 3 | 42: 57.21, 43: 55.13, 44: 58.80 | 57.05 | 1.84 |
| Fixed sensorimotor | 3 | 42: 57.41, 43: 56.83, 44: 59.80 | 58.01 | 1.57 |
| ERD-only | 3 | 42: 58.26, 43: 55.56, 44: 58.80 | 57.54 | 1.74 |
| Random (mean of draws) | 3 | 42: 54.25, 43: 53.75, 44: 57.55 | 55.18 | 2.06 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.215 | [-0.037, 0.442] | 0.305 | 9 |
| grad | class_specific_pearson | 0.313 | [0.180, 0.435] | 0.173 | 9 |
| grad | lateralisation_pearson | 0.292 | [0.027, 0.519] | 0.286 | 9 |
| grad | magnitude_spearman | 0.116 | [-0.143, 0.359] | 0.315 | 9 |
| grad | class_specific_spearman | 0.264 | [0.153, 0.369] | 0.138 | 9 |
| grad | lateralisation_spearman | 0.290 | [-0.026, 0.554] | 0.321 | 9 |
| grad | legacy_signed_LR_pearson | -0.164 | [-0.318, -0.002] | 0.204 | 9 |
| grad_x_input | magnitude_pearson | 0.192 | [-0.065, 0.426] | 0.312 | 9 |
| grad_x_input | class_specific_pearson | 0.307 | [0.177, 0.427] | 0.169 | 9 |
| grad_x_input | lateralisation_pearson | 0.251 | [-0.035, 0.500] | 0.318 | 9 |
| grad_x_input | magnitude_spearman | 0.083 | [-0.163, 0.320] | 0.303 | 9 |
| grad_x_input | class_specific_spearman | 0.243 | [0.136, 0.344] | 0.135 | 9 |
| grad_x_input | lateralisation_spearman | 0.244 | [-0.058, 0.505] | 0.321 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.173 | [-0.333, -0.004] | 0.212 | 9 |
| integrated_gradients | magnitude_pearson | 0.201 | [-0.056, 0.432] | 0.309 | 9 |
| integrated_gradients | class_specific_pearson | 0.297 | [0.174, 0.412] | 0.161 | 9 |
| integrated_gradients | lateralisation_pearson | 0.238 | [-0.036, 0.480] | 0.314 | 9 |
| integrated_gradients | magnitude_spearman | 0.098 | [-0.150, 0.335] | 0.302 | 9 |
| integrated_gradients | class_specific_spearman | 0.244 | [0.134, 0.347] | 0.137 | 9 |
| integrated_gradients | lateralisation_spearman | 0.237 | [-0.063, 0.498] | 0.326 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.172 | [-0.342, 0.009] | 0.225 | 9 |
| smoothgrad | magnitude_pearson | 0.216 | [-0.037, 0.443] | 0.305 | 9 |
| smoothgrad | class_specific_pearson | 0.314 | [0.180, 0.437] | 0.174 | 9 |
| smoothgrad | lateralisation_pearson | 0.296 | [0.030, 0.522] | 0.286 | 9 |
| smoothgrad | magnitude_spearman | 0.118 | [-0.140, 0.361] | 0.314 | 9 |
| smoothgrad | class_specific_spearman | 0.263 | [0.149, 0.370] | 0.141 | 9 |
| smoothgrad | lateralisation_spearman | 0.287 | [-0.018, 0.544] | 0.313 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.164 | [-0.317, -0.003] | 0.203 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.342 | [-0.443, -0.233] | 9 |
| grad | ERD_L_spearman | -0.298 | [-0.410, -0.177] | 9 |
| grad | ERD_R_pearson | -0.217 | [-0.425, 0.012] | 9 |
| grad | ERD_R_spearman | -0.182 | [-0.412, 0.070] | 9 |
| grad | ERD_LR_pearson | -0.164 | [-0.318, -0.002] | 9 |
| grad | ERD_LR_spearman | -0.143 | [-0.312, 0.034] | 9 |
| grad | ERD_comb_pearson | -0.314 | [-0.454, -0.158] | 9 |
| grad | ERD_comb_spearman | -0.298 | [-0.431, -0.154] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.358 | [-0.453, -0.255] | 9 |
| grad_x_input | ERD_L_spearman | -0.299 | [-0.394, -0.199] | 9 |
| grad_x_input | ERD_R_pearson | -0.218 | [-0.425, 0.011] | 9 |
| grad_x_input | ERD_R_spearman | -0.170 | [-0.394, 0.073] | 9 |
| grad_x_input | ERD_LR_pearson | -0.173 | [-0.333, -0.004] | 9 |
| grad_x_input | ERD_LR_spearman | -0.151 | [-0.312, 0.018] | 9 |
| grad_x_input | ERD_comb_pearson | -0.323 | [-0.455, -0.177] | 9 |
| grad_x_input | ERD_comb_spearman | -0.293 | [-0.413, -0.164] | 9 |
| grad_x_input | vs_grad_pearson | 0.999 | [0.997, 0.999] | 9 |
| grad_x_input | vs_grad_spearman | 0.994 | [0.989, 0.997] | 9 |
| integrated_gradients | ERD_L_pearson | -0.343 | [-0.435, -0.244] | 9 |
| integrated_gradients | ERD_L_spearman | -0.293 | [-0.401, -0.177] | 9 |
| integrated_gradients | ERD_R_pearson | -0.200 | [-0.400, 0.017] | 9 |
| integrated_gradients | ERD_R_spearman | -0.152 | [-0.368, 0.080] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.172 | [-0.342, 0.009] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.145 | [-0.314, 0.034] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.308 | [-0.432, -0.173] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.283 | [-0.411, -0.144] | 9 |
| integrated_gradients | vs_grad_pearson | 0.993 | [0.985, 0.996] | 9 |
| integrated_gradients | vs_grad_spearman | 0.980 | [0.969, 0.987] | 9 |
| smoothgrad | ERD_L_pearson | -0.343 | [-0.445, -0.233] | 9 |
| smoothgrad | ERD_L_spearman | -0.301 | [-0.411, -0.182] | 9 |
| smoothgrad | ERD_R_pearson | -0.219 | [-0.426, 0.011] | 9 |
| smoothgrad | ERD_R_spearman | -0.184 | [-0.416, 0.069] | 9 |
| smoothgrad | ERD_LR_pearson | -0.164 | [-0.317, -0.003] | 9 |
| smoothgrad | ERD_LR_spearman | -0.147 | [-0.312, 0.027] | 9 |
| smoothgrad | ERD_comb_pearson | -0.315 | [-0.455, -0.159] | 9 |
| smoothgrad | ERD_comb_spearman | -0.300 | [-0.434, -0.154] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [0.999, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.203 | [-0.321, -0.078] | 0.079 | -0.224 | 0.083 | 9 |
| ERD_theta_R | -0.093 | [-0.308, 0.130] | 0.072 | -0.096 | 0.034 | 9 |
| ERD_theta_mag | 0.087 | [-0.134, 0.301] | 0.087 | 0.049 | 0.049 | 9 |
| ERD_mu_L | -0.285 | [-0.445, -0.107] | 0.112 | -0.213 | 0.067 | 9 |
| ERD_mu_R | -0.157 | [-0.365, 0.066] | 0.107 | -0.087 | 0.070 | 9 |
| ERD_mu_mag | 0.125 | [-0.123, 0.357] | 0.125 | 0.084 | 0.084 | 9 |
| ERD_beta_L | -0.273 | [-0.441, -0.086] | -0.014 | -0.219 | -0.048 | 9 |
| ERD_beta_R | -0.108 | [-0.346, 0.142] | 0.056 | -0.056 | 0.003 | 9 |
| ERD_beta_mag | -0.032 | [-0.315, 0.255] | -0.032 | -0.060 | -0.060 | 9 |
| ERS_beta_rebound_L | -0.153 | [-0.277, -0.024] | -0.146 | -0.078 | -0.106 | 9 |
| ERS_beta_rebound_R | -0.064 | [-0.213, 0.087] | -0.065 | -0.032 | -0.050 | 9 |
| ERS_beta_rebound_mag | -0.151 | [-0.270, -0.027] | -0.151 | -0.098 | -0.098 | 9 |
| ERD_lowgamma_L | -0.085 | [-0.314, 0.153] | -0.016 | -0.027 | 0.003 | 9 |
| ERD_lowgamma_R | -0.193 | [-0.429, 0.067] | -0.137 | -0.141 | -0.139 | 9 |
| ERD_lowgamma_mag | -0.096 | [-0.388, 0.213] | -0.096 | -0.085 | -0.085 | 9 |
| logpow_8_30 | 0.051 | [-0.141, 0.240] | 0.051 | 0.070 | 0.070 | 9 |
| r2_signed | 0.124 | [-0.141, 0.373] | 0.113 | 0.125 | 0.066 | 9 |
| r2_abs | 0.113 | [-0.209, 0.413] | 0.113 | 0.066 | 0.066 | 9 |
| TFR_ERD_dB_L | -0.332 | [-0.445, -0.208] | 0.156 | -0.272 | 0.111 | 9 |
| TFR_ERD_dB_R | -0.211 | [-0.452, 0.060] | 0.206 | -0.141 | 0.129 | 9 |
| TFR_ERD_dB_mag | 0.172 | [-0.097, 0.418] | 0.172 | 0.087 | 0.087 | 9 |
| ITC_4_30 | -0.056 | [-0.237, 0.130] | -0.056 | -0.077 | -0.077 | 9 |
| ERP_abs_0.2_0.8 | 0.044 | [-0.110, 0.196] | 0.044 | 0.056 | 0.056 | 9 |
| SlowPot_0.5_3 | 0.040 | [-0.146, 0.224] | -0.026 | 0.038 | -0.012 | 9 |
| ERD_beta_classspecific | — | — | 0.017 [-0.206, 0.239] | — | -0.036 | 9 |
| ERD_lowgamma_classspecific | — | — | -0.067 [-0.247, 0.117] | — | -0.071 | 9 |
| ERD_mu_classspecific | — | — | 0.113 [-0.118, 0.332] | — | 0.043 | 9 |
| ERD_theta_classspecific | — | — | 0.057 [-0.128, 0.238] | — | 0.048 | 9 |
| ERS_beta_rebound_classspecific | — | — | -0.099 [-0.166, -0.031] | — | -0.072 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.188 [-0.047, 0.403] | — | 0.095 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +21.06 ± 13.43 | +13.99 ± 10.30 | +6.84 ± 5.32 | 0.0052 | 0.0034 |
| 4 | +24.47 ± 16.85 | +20.33 ± 13.82 | +13.73 ± 9.42 | 0.0141 | 0.0043 |
| 6 | +26.09 ± 16.52 | +23.58 ± 15.50 | +16.91 ± 13.04 | 0.0308 | 0.0029 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.027 | [-0.121, 0.067] | 9 |
| grad_x_input | -0.032 | [-0.131, 0.067] | 9 |
| integrated_gradients | -0.037 | [-0.133, 0.060] | 9 |
| smoothgrad | -0.027 | [-0.120, 0.065] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=original, S02=original, S03=original, S04=original, S05=original, S06=original, S07=original, S08=original, S09=original

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 62.96 |
| k64 | 57.10 |
| k64_long | 57.41 |
| k64_aug | 50.62 |
| k64_long_aug | 49.38 |

```
model: "EEGNetv4"
dataset: "BNCI2014_001"
sfreq_hz: 128
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 32, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64_long": {"band_hz": [4.0, 40.0], "window_s": [0.5, 4.0], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64_aug": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": true}, "k64_long_aug": {"band_hz": [4.0, 40.0], "window_s": [0.5, 4.0], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": true}}
checkpoint_rule: "lowest validation loss, early stopping with patience"
folds: "train on session 1 (last 25% of it = validation, chronological), refit on all of session 1, test once on session 2 (official BCI IV-2a split)"
seeds: [42, 43, 44]
```
