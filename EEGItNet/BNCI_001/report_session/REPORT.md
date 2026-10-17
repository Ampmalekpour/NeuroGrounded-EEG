# Results report — EEGItNet/BNCI_001

Generated 2026-10-17 10:50 from the saved `results.json` files. TOST margin ±6 percentage points. Results folder: `<class>/results_session/`.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 72.92 | 55.56 | 81.25 | 70.83 | 63.19 | 61.11 | 69.44 | 84.72 | 80.56 | 71.06 | 9.92 |
| Ours | 72.22 | 56.94 | 76.39 | 67.36 | 64.58 | 65.28 | 66.67 | 88.89 | 86.11 | 71.60 | 10.48 |
| CSP | 70.14 | 51.39 | 79.86 | 65.28 | 65.97 | 59.03 | 63.19 | 85.42 | 81.25 | 69.06 | 11.21 |
| RLF | 52.08 | 53.47 | 76.39 | 64.58 | 61.81 | 63.89 | 69.44 | 85.42 | 85.42 | 68.06 | 12.30 |
| MI | 66.67 | 52.78 | 66.67 | 67.36 | 61.81 | 61.81 | 65.28 | 79.17 | 78.47 | 66.67 | 8.20 |
| Fixed sensorimotor | 65.28 | 56.25 | 78.47 | 60.42 | 59.72 | 59.03 | 64.58 | 87.50 | 79.17 | 67.82 | 11.06 |
| ERD-only | 76.39 | 56.25 | 72.22 | 65.28 | 63.89 | 65.28 | 67.36 | 86.11 | 85.42 | 70.91 | 10.08 |
| Random (mean of draws) | 62.78 | 54.03 | 73.06 | 61.39 | 60.28 | 58.19 | 65.83 | 82.64 | 75.14 | 65.93 | 9.21 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | C4 CP4 C5 Fz CPz P2 CP3 CP2 CP1 FC3 FC1 P1 |
| CSP | 9 | CP3 C3 CP4 C4 CP2 C5 CP1 P2 FC3 C2 CPz FC2 |
| RLF | 9 | CP4 Fz CP2 C6 C3 CPz C5 Cz P2 FCz CP1 POz |
| MI | 9 | P2 CPz FC3 CP4 Fz C3 P1 FC1 C1 C6 POz C2 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | +0.54 | 3.72 | [-1.77, +2.85] | 0.0011 | yes | -4.86 | 0 | 3,4,5,6,8,10 |
| CSP | 9 | -2.01 | 3.04 | [-3.89, -0.12] | 0.0021 | yes | -6.25 | 1 | 4,5,6,8,10 |
| RLF | 9 | -3.01 | 7.53 | [-7.68, +1.66] | 0.1337 | no | -20.83 | 2 | 8,10 |
| MI | 9 | -4.40 | 4.36 | [-7.10, -1.69] | 0.1514 | no | -14.58 | 2 | 8,10 |
| Fixed sensorimotor | 9 | -3.24 | 4.03 | [-5.74, -0.74] | 0.0371 | yes | -10.42 | 2 | 6,8,10 |
| ERD-only | 9 | -0.15 | 4.64 | [-3.03, +2.72] | 0.0027 | yes | -9.03 | 1 | 4,5,6,8,10 |
| Random (mean of draws) | 9 | -5.14 | 3.31 | [-7.19, -3.09] | 0.2287 | no | -10.14 | 3 | 8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | +2.55 | [+0.09, +5.01] | 0.0441 | 0.1323 | 0.0742 | 0.1641 | 7 / 0 / 2 |
| RLF | 9 | +3.55 | [-1.48, +8.57] | 0.1420 | 0.2840 | 0.0547 | 0.1641 | 7 / 1 / 1 |
| MI | 9 | +4.94 | [+2.25, +7.63] | 0.0029 | 0.0143 | 0.0078 | 0.0391 | 8 / 1 / 0 |
| Fixed sensorimotor | 9 | +3.78 | [+1.21, +6.36] | 0.0096 | 0.0383 | 0.0273 | 0.1094 | 8 / 0 / 1 |
| ERD-only | 9 | +0.69 | [-1.12, +2.50] | 0.4021 | 0.4021 | 0.3281 | 0.3281 | 6 / 1 / 2 |
| Random (mean of draws) | 9 | +5.68 | [+3.20, +8.16] | 0.0007 | 0.0044 | 0.0039 | 0.0234 | 9 / 0 / 0 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 71.06 | left_hand 68.2, right_hand 73.9 | left_hand 47.1, right_hand 52.9 | 50.00 | 0 / 9 |
| Ours | 71.60 | left_hand 70.5, right_hand 72.7 | left_hand 48.9, right_hand 51.1 | 50.00 | 0 / 9 |
| CSP | 69.06 | left_hand 66.2, right_hand 71.9 | left_hand 47.1, right_hand 52.9 | 50.00 | 0 / 9 |
| RLF | 68.06 | left_hand 68.1, right_hand 68.1 | left_hand 50.0, right_hand 50.0 | 50.00 | 0 / 9 |
| MI | 66.67 | left_hand 64.2, right_hand 69.1 | left_hand 47.5, right_hand 52.5 | 50.00 | 0 / 9 |
| Fixed sensorimotor | 67.82 | left_hand 66.4, right_hand 69.3 | left_hand 48.5, right_hand 51.5 | 50.00 | 0 / 9 |
| ERD-only | 70.91 | left_hand 69.3, right_hand 72.5 | left_hand 48.4, right_hand 51.6 | 50.00 | 0 / 9 |
| Random (mean of draws) | 65.93 | left_hand 64.0, right_hand 67.8 | left_hand 48.1, right_hand 51.9 | 50.00 | 0 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 70.68, 43: 66.98, 44: 69.14 | 68.93 | 1.86 |
| Ours | 3 | 42: 70.22, 43: 68.98, 44: 70.91 | 70.04 | 0.98 |
| CSP | 3 | 42: 67.59, 43: 68.60, 44: 67.44 | 67.88 | 0.63 |
| RLF | 3 | 42: 65.51, 43: 66.20, 44: 66.13 | 65.95 | 0.38 |
| MI | 3 | 42: 66.59, 43: 63.97, 44: 65.43 | 65.33 | 1.31 |
| Fixed sensorimotor | 3 | 42: 64.35, 43: 65.82, 44: 66.98 | 65.72 | 1.31 |
| ERD-only | 3 | 42: 68.98, 43: 68.90, 44: 69.98 | 69.29 | 0.60 |
| Random (mean of draws) | 3 | 42: 63.07, 43: 65.08, 44: 66.28 | 64.81 | 1.62 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.066 | [-0.108, 0.236] | 0.221 | 9 |
| grad | class_specific_pearson | 0.143 | [0.043, 0.239] | 0.129 | 9 |
| grad | lateralisation_pearson | 0.020 | [-0.204, 0.242] | 0.282 | 9 |
| grad | magnitude_spearman | 0.049 | [-0.109, 0.204] | 0.201 | 9 |
| grad | class_specific_spearman | 0.111 | [0.023, 0.198] | 0.114 | 9 |
| grad | lateralisation_spearman | 0.012 | [-0.172, 0.195] | 0.235 | 9 |
| grad | legacy_signed_LR_pearson | -0.130 | [-0.318, 0.068] | 0.245 | 9 |
| grad_x_input | magnitude_pearson | 0.015 | [-0.163, 0.193] | 0.228 | 9 |
| grad_x_input | class_specific_pearson | 0.136 | [0.036, 0.233] | 0.129 | 9 |
| grad_x_input | lateralisation_pearson | -0.066 | [-0.400, 0.282] | 0.411 | 9 |
| grad_x_input | magnitude_spearman | 0.003 | [-0.138, 0.144] | 0.181 | 9 |
| grad_x_input | class_specific_spearman | 0.092 | [0.021, 0.162] | 0.092 | 9 |
| grad_x_input | lateralisation_spearman | -0.082 | [-0.403, 0.258] | 0.400 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.136 | [-0.320, 0.057] | 0.240 | 9 |
| integrated_gradients | magnitude_pearson | 0.031 | [-0.144, 0.204] | 0.223 | 9 |
| integrated_gradients | class_specific_pearson | 0.142 | [0.046, 0.236] | 0.124 | 9 |
| integrated_gradients | lateralisation_pearson | -0.122 | [-0.437, 0.219] | 0.403 | 9 |
| integrated_gradients | magnitude_spearman | 0.037 | [-0.096, 0.168] | 0.170 | 9 |
| integrated_gradients | class_specific_spearman | 0.088 | [0.001, 0.174] | 0.112 | 9 |
| integrated_gradients | lateralisation_spearman | -0.103 | [-0.407, 0.222] | 0.388 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.124 | [-0.314, 0.076] | 0.251 | 9 |
| smoothgrad | magnitude_pearson | 0.065 | [-0.110, 0.236] | 0.222 | 9 |
| smoothgrad | class_specific_pearson | 0.141 | [0.041, 0.238] | 0.129 | 9 |
| smoothgrad | lateralisation_pearson | 0.016 | [-0.207, 0.237] | 0.281 | 9 |
| smoothgrad | magnitude_spearman | 0.047 | [-0.111, 0.202] | 0.201 | 9 |
| smoothgrad | class_specific_spearman | 0.110 | [0.022, 0.196] | 0.114 | 9 |
| smoothgrad | lateralisation_spearman | 0.001 | [-0.192, 0.193] | 0.246 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.131 | [-0.319, 0.066] | 0.244 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.187 | [-0.293, -0.076] | 9 |
| grad | ERD_L_spearman | -0.136 | [-0.236, -0.034] | 9 |
| grad | ERD_R_pearson | -0.103 | [-0.256, 0.055] | 9 |
| grad | ERD_R_spearman | -0.113 | [-0.260, 0.040] | 9 |
| grad | ERD_LR_pearson | -0.130 | [-0.318, 0.068] | 9 |
| grad | ERD_LR_spearman | -0.082 | [-0.268, 0.110] | 9 |
| grad | ERD_comb_pearson | -0.155 | [-0.279, -0.027] | 9 |
| grad | ERD_comb_spearman | -0.134 | [-0.221, -0.044] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.192 | [-0.293, -0.085] | 9 |
| grad_x_input | ERD_L_spearman | -0.129 | [-0.221, -0.035] | 9 |
| grad_x_input | ERD_R_pearson | -0.087 | [-0.246, 0.077] | 9 |
| grad_x_input | ERD_R_spearman | -0.072 | [-0.222, 0.082] | 9 |
| grad_x_input | ERD_LR_pearson | -0.136 | [-0.320, 0.057] | 9 |
| grad_x_input | ERD_LR_spearman | -0.088 | [-0.264, 0.093] | 9 |
| grad_x_input | ERD_comb_pearson | -0.151 | [-0.285, -0.010] | 9 |
| grad_x_input | ERD_comb_spearman | -0.114 | [-0.196, -0.031] | 9 |
| grad_x_input | vs_grad_pearson | 0.993 | [0.989, 0.996] | 9 |
| grad_x_input | vs_grad_spearman | 0.986 | [0.965, 0.994] | 9 |
| integrated_gradients | ERD_L_pearson | -0.196 | [-0.289, -0.099] | 9 |
| integrated_gradients | ERD_L_spearman | -0.121 | [-0.206, -0.033] | 9 |
| integrated_gradients | ERD_R_pearson | -0.102 | [-0.263, 0.064] | 9 |
| integrated_gradients | ERD_R_spearman | -0.091 | [-0.246, 0.069] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.124 | [-0.314, 0.076] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.077 | [-0.255, 0.106] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.163 | [-0.291, -0.030] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.121 | [-0.212, -0.027] | 9 |
| integrated_gradients | vs_grad_pearson | 0.979 | [0.964, 0.988] | 9 |
| integrated_gradients | vs_grad_spearman | 0.968 | [0.933, 0.985] | 9 |
| smoothgrad | ERD_L_pearson | -0.185 | [-0.292, -0.074] | 9 |
| smoothgrad | ERD_L_spearman | -0.140 | [-0.239, -0.038] | 9 |
| smoothgrad | ERD_R_pearson | -0.101 | [-0.253, 0.057] | 9 |
| smoothgrad | ERD_R_spearman | -0.107 | [-0.255, 0.045] | 9 |
| smoothgrad | ERD_LR_pearson | -0.131 | [-0.319, 0.066] | 9 |
| smoothgrad | ERD_LR_spearman | -0.088 | [-0.272, 0.102] | 9 |
| smoothgrad | ERD_comb_pearson | -0.154 | [-0.277, -0.025] | 9 |
| smoothgrad | ERD_comb_spearman | -0.133 | [-0.220, -0.045] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.174 | [-0.317, -0.023] | 0.078 | -0.077 | 0.026 | 9 |
| ERD_theta_R | -0.129 | [-0.226, -0.030] | 0.085 | -0.147 | 0.094 | 9 |
| ERD_theta_mag | 0.125 | [0.002, 0.244] | 0.125 | 0.107 | 0.107 | 9 |
| ERD_mu_L | -0.168 | [-0.330, 0.003] | 0.091 | -0.093 | 0.094 | 9 |
| ERD_mu_R | -0.086 | [-0.252, 0.085] | 0.047 | -0.107 | 0.027 | 9 |
| ERD_mu_mag | 0.116 | [-0.069, 0.293] | 0.116 | 0.102 | 0.102 | 9 |
| ERD_beta_L | -0.102 | [-0.246, 0.047] | 0.029 | -0.105 | 0.039 | 9 |
| ERD_beta_R | -0.028 | [-0.217, 0.163] | 0.080 | -0.076 | 0.047 | 9 |
| ERD_beta_mag | 0.055 | [-0.087, 0.193] | 0.055 | 0.039 | 0.039 | 9 |
| ERS_beta_rebound_L | -0.136 | [-0.230, -0.040] | -0.093 | -0.158 | -0.116 | 9 |
| ERS_beta_rebound_R | -0.030 | [-0.229, 0.172] | -0.056 | -0.107 | -0.128 | 9 |
| ERS_beta_rebound_mag | -0.108 | [-0.292, 0.084] | -0.108 | -0.153 | -0.153 | 9 |
| ERD_lowgamma_L | -0.033 | [-0.184, 0.120] | 0.087 | -0.003 | 0.077 | 9 |
| ERD_lowgamma_R | -0.053 | [-0.218, 0.115] | -0.078 | -0.099 | -0.109 | 9 |
| ERD_lowgamma_mag | 0.034 | [-0.130, 0.196] | 0.034 | 0.022 | 0.022 | 9 |
| logpow_8_30 | -0.101 | [-0.183, -0.017] | -0.101 | -0.083 | -0.083 | 9 |
| r2_signed | 0.118 | [-0.127, 0.349] | 0.187 | 0.053 | 0.146 | 9 |
| r2_abs | 0.187 | [-0.009, 0.369] | 0.187 | 0.146 | 0.146 | 9 |
| TFR_ERD_dB_L | -0.165 | [-0.288, -0.037] | 0.122 | -0.098 | 0.105 | 9 |
| TFR_ERD_dB_R | -0.070 | [-0.251, 0.116] | 0.011 | -0.113 | -0.057 | 9 |
| TFR_ERD_dB_mag | 0.066 | [-0.144, 0.269] | 0.066 | 0.047 | 0.047 | 9 |
| ITC_4_30 | -0.054 | [-0.248, 0.145] | -0.054 | -0.021 | -0.021 | 9 |
| ERP_abs_0.2_0.8 | -0.068 | [-0.228, 0.096] | -0.068 | -0.064 | -0.064 | 9 |
| SlowPot_0.5_3 | -0.166 | [-0.303, -0.021] | 0.082 | -0.118 | 0.030 | 9 |
| ERD_beta_classspecific | — | — | 0.056 [-0.038, 0.149] | — | 0.041 | 9 |
| ERD_lowgamma_classspecific | — | — | 0.007 [-0.090, 0.104] | — | -0.012 | 9 |
| ERD_mu_classspecific | — | — | 0.064 [-0.095, 0.219] | — | 0.058 | 9 |
| ERD_theta_classspecific | — | — | 0.081 [-0.013, 0.174] | — | 0.064 | 9 |
| ERS_beta_rebound_classspecific | — | — | -0.070 [-0.223, 0.087] | — | -0.116 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.059 [-0.089, 0.204] | — | 0.021 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +5.89 ± 4.24 | +2.45 ± 1.84 | +0.90 ± 1.71 | 0.0152 | 0.0087 |
| 4 | +10.21 ± 6.15 | +5.37 ± 3.63 | +2.96 ± 3.06 | 0.0080 | 0.0028 |
| 6 | +10.34 ± 6.70 | +6.97 ± 4.63 | +5.04 ± 3.12 | 0.0724 | 0.0155 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.072 | [-0.043, 0.185] | 9 |
| grad_x_input | 0.072 | [-0.041, 0.183] | 9 |
| integrated_gradients | 0.073 | [-0.036, 0.181] | 9 |
| smoothgrad | 0.071 | [-0.047, 0.188] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: False): S01=cue_0_4, S02=original, S03=original, S04=cue_0_4, S05=cue_0_4, S06=original, S07=original, S08=cue_0_4, S09=original

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 62.35 |
| cue_0_4 | 62.35 |

```
model: "EEGITNet"
dataset: "BNCI2014_001"
sfreq_hz: 250
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "cue_0_4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [0.0, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "no_aug": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": false, "mixup_p": 0.0, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "drop_0.25": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.25, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}}
checkpoint_rule: "SWA average from swa_start_frac*max_epochs + BN update (no early stopping); validation block only scored, never used for checkpointing"
folds: "train on session 1 (last 25% of it = validation, chronological), refit on all of session 1, test once on session 2 (official BCI IV-2a split)"
seeds: [42, 43, 44]
```

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Session protocol: trained on session 1, tested once on session 2. Subject accuracy = seed ensemble (mean softmax of 3 seeds); single seeds in T6. SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 59.03 | 36.46 | 61.81 | 50.35 | 37.50 | 35.42 | 52.43 | 60.42 | 65.28 | 50.96 | 11.79 |
| Ours | 60.07 | 38.19 | 55.90 | 50.00 | 36.11 | 36.81 | 57.64 | 57.64 | 62.85 | 50.58 | 10.73 |
| CSP | 60.42 | 36.11 | 59.72 | 51.39 | 37.85 | 36.46 | 52.08 | 56.60 | 59.72 | 50.04 | 10.44 |
| RLF | 60.42 | 35.42 | 60.42 | 47.57 | 37.15 | 31.94 | 49.65 | 60.07 | 61.11 | 49.31 | 11.98 |
| MI | 60.42 | 37.15 | 56.60 | 46.18 | 38.54 | 38.89 | 51.04 | 55.90 | 65.97 | 50.08 | 10.47 |
| Fixed sensorimotor | 58.68 | 35.42 | 59.38 | 40.97 | 35.42 | 31.94 | 54.17 | 56.25 | 61.46 | 48.19 | 12.01 |
| ERD-only | 62.15 | 35.07 | 58.33 | 47.22 | 37.15 | 37.85 | 51.39 | 57.64 | 61.46 | 49.81 | 10.91 |
| Random (mean of draws) | 56.60 | 36.25 | 57.36 | 45.35 | 35.90 | 35.28 | 49.65 | 55.56 | 60.62 | 48.06 | 10.21 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | POz Fz P2 Cz CP4 CPz C3 C5 CP3 FC3 CP1 CP2 |
| CSP | 9 | P1 P2 CP3 Pz CP4 CPz CP1 CP2 POz C3 C4 C2 |
| RLF | 9 | C3 CP3 CP1 P2 CPz CP4 P1 Fz CP2 POz Pz C5 |
| MI | 9 | CP1 P1 CP4 P2 CPz CP2 Pz CP3 C3 FC3 C6 C5 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -0.39 | 3.21 | [-2.37, +1.60] | 0.0004 | yes | -5.90 | 0 | 3,4,5,6,8,10 |
| CSP | 9 | -0.93 | 2.41 | [-2.42, +0.57] | 0.0001 | yes | -5.56 | 0 | 3,4,5,6,8,10 |
| RLF | 9 | -1.66 | 1.78 | [-2.76, -0.56] | <0.0001 | yes | -4.17 | 0 | 3,4,5,6,8,10 |
| MI | 9 | -0.89 | 3.08 | [-2.79, +1.02] | 0.0005 | yes | -5.21 | 0 | 3,4,5,6,8,10 |
| Fixed sensorimotor | 9 | -2.78 | 3.10 | [-4.70, -0.86] | 0.0071 | yes | -9.38 | 1 | 5,6,8,10 |
| ERD-only | 9 | -1.16 | 2.52 | [-2.72, +0.41] | 0.0002 | yes | -3.82 | 0 | 3,4,5,6,8,10 |
| Random (mean of draws) | 9 | -2.90 | 1.95 | [-4.11, -1.69] | 0.0007 | yes | -5.00 | 0 | 5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | +0.54 | [-1.62, +2.70] | 0.5801 | 1.0000 | 0.6523 | 1.0000 | 5 / 0 / 4 |
| RLF | 9 | +1.27 | [-1.67, +4.21] | 0.3471 | 1.0000 | 0.3789 | 1.0000 | 5 / 0 / 4 |
| MI | 9 | +0.50 | [-1.94, +2.94] | 0.6481 | 1.0000 | 0.9102 | 1.0000 | 4 / 0 / 5 |
| Fixed sensorimotor | 9 | +2.39 | [-0.21, +4.99] | 0.0669 | 0.3346 | 0.0586 | 0.2930 | 8 / 0 / 1 |
| ERD-only | 9 | +0.77 | [-1.43, +2.97] | 0.4420 | 1.0000 | 0.4609 | 1.0000 | 4 / 1 / 4 |
| Random (mean of draws) | 9 | +2.52 | [+0.44, +4.59] | 0.0232 | 0.1391 | 0.0117 | 0.0703 | 8 / 0 / 1 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 50.96 | feet 43.5, left_hand 56.6, right_hand 60.8, tongue 42.9 | feet 22.3, left_hand 28.6, right_hand 31.2, tongue 17.9 | 25.00 | 0 / 9 |
| Ours | 50.58 | feet 43.1, left_hand 56.9, right_hand 57.3, tongue 45.1 | feet 22.3, left_hand 29.0, right_hand 29.7, tongue 18.9 | 25.00 | 0 / 9 |
| CSP | 50.04 | feet 43.4, left_hand 57.3, right_hand 56.9, tongue 42.6 | feet 22.1, left_hand 29.5, right_hand 30.1, tongue 18.3 | 25.00 | 0 / 9 |
| RLF | 49.31 | feet 39.7, left_hand 54.8, right_hand 59.0, tongue 43.8 | feet 21.2, left_hand 28.1, right_hand 32.2, tongue 18.5 | 25.00 | 0 / 9 |
| MI | 50.08 | feet 42.9, left_hand 56.5, right_hand 57.6, tongue 43.4 | feet 21.9, left_hand 29.7, right_hand 29.9, tongue 18.5 | 25.00 | 0 / 9 |
| Fixed sensorimotor | 48.19 | feet 39.4, left_hand 55.1, right_hand 57.1, tongue 41.2 | feet 20.8, left_hand 28.6, right_hand 30.6, tongue 20.0 | 25.00 | 0 / 9 |
| ERD-only | 49.81 | feet 40.3, left_hand 57.6, right_hand 57.1, tongue 44.3 | feet 21.1, left_hand 29.2, right_hand 30.8, tongue 18.9 | 25.00 | 0 / 9 |
| Random (mean of draws) | 48.06 | feet 40.2, left_hand 54.1, right_hand 55.6, tongue 42.3 | feet 21.4, left_hand 28.6, right_hand 31.3, tongue 18.8 | 25.00 | 0 / 45 |

### T6. Seed stability

Grand mean of each single seed, before the seed ensemble used in T1.

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 3 | 42: 48.73, 43: 49.11, 44: 49.31 | 49.05 | 0.29 |
| Ours | 3 | 42: 46.99, 43: 47.22, 44: 51.08 | 48.43 | 2.30 |
| CSP | 3 | 42: 47.18, 43: 47.99, 44: 48.73 | 47.97 | 0.77 |
| RLF | 3 | 42: 47.84, 43: 48.73, 44: 48.96 | 48.51 | 0.59 |
| MI | 3 | 42: 48.57, 43: 47.53, 44: 48.50 | 48.20 | 0.58 |
| Fixed sensorimotor | 3 | 42: 46.22, 43: 46.99, 44: 47.03 | 46.75 | 0.46 |
| ERD-only | 3 | 42: 48.50, 43: 48.84, 44: 49.38 | 48.91 | 0.45 |
| Random (mean of draws) | 3 | 42: 46.41, 43: 46.51, 44: 47.35 | 46.76 | 0.52 |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.184 | [-0.083, 0.426] | 0.323 | 9 |
| grad | class_specific_pearson | 0.178 | [-0.042, 0.382] | 0.264 | 9 |
| grad | lateralisation_pearson | 0.093 | [-0.182, 0.355] | 0.318 | 9 |
| grad | magnitude_spearman | 0.133 | [-0.091, 0.343] | 0.279 | 9 |
| grad | class_specific_spearman | 0.176 | [-0.043, 0.378] | 0.251 | 9 |
| grad | lateralisation_spearman | 0.093 | [-0.134, 0.310] | 0.272 | 9 |
| grad | legacy_signed_LR_pearson | -0.055 | [-0.203, 0.096] | 0.193 | 9 |
| grad_x_input | magnitude_pearson | 0.172 | [-0.109, 0.428] | 0.340 | 9 |
| grad_x_input | class_specific_pearson | 0.164 | [-0.042, 0.357] | 0.253 | 9 |
| grad_x_input | lateralisation_pearson | 0.022 | [-0.260, 0.301] | 0.331 | 9 |
| grad_x_input | magnitude_spearman | 0.129 | [-0.108, 0.353] | 0.296 | 9 |
| grad_x_input | class_specific_spearman | 0.165 | [-0.025, 0.344] | 0.223 | 9 |
| grad_x_input | lateralisation_spearman | 0.019 | [-0.241, 0.276] | 0.312 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.068 | [-0.223, 0.090] | 0.201 | 9 |
| integrated_gradients | magnitude_pearson | 0.153 | [-0.140, 0.421] | 0.350 | 9 |
| integrated_gradients | class_specific_pearson | 0.140 | [-0.058, 0.328] | 0.242 | 9 |
| integrated_gradients | lateralisation_pearson | -0.077 | [-0.341, 0.199] | 0.339 | 9 |
| integrated_gradients | magnitude_spearman | 0.154 | [-0.075, 0.368] | 0.285 | 9 |
| integrated_gradients | class_specific_spearman | 0.145 | [-0.036, 0.316] | 0.214 | 9 |
| integrated_gradients | lateralisation_spearman | -0.098 | [-0.354, 0.171] | 0.334 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.075 | [-0.228, 0.082] | 0.201 | 9 |
| smoothgrad | magnitude_pearson | 0.183 | [-0.085, 0.426] | 0.323 | 9 |
| smoothgrad | class_specific_pearson | 0.178 | [-0.042, 0.382] | 0.265 | 9 |
| smoothgrad | lateralisation_pearson | 0.093 | [-0.183, 0.355] | 0.318 | 9 |
| smoothgrad | magnitude_spearman | 0.135 | [-0.092, 0.348] | 0.282 | 9 |
| smoothgrad | class_specific_spearman | 0.175 | [-0.042, 0.377] | 0.251 | 9 |
| smoothgrad | lateralisation_spearman | 0.092 | [-0.136, 0.310] | 0.273 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.055 | [-0.205, 0.097] | 0.194 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.157 | [-0.377, 0.079] | 9 |
| grad | ERD_L_spearman | -0.180 | [-0.348, -0.001] | 9 |
| grad | ERD_R_pearson | -0.138 | [-0.399, 0.144] | 9 |
| grad | ERD_R_spearman | -0.156 | [-0.427, 0.142] | 9 |
| grad | ERD_LR_pearson | -0.055 | [-0.203, 0.096] | 9 |
| grad | ERD_LR_spearman | -0.077 | [-0.261, 0.113] | 9 |
| grad | ERD_comb_pearson | -0.167 | [-0.418, 0.106] | 9 |
| grad | ERD_comb_spearman | -0.169 | [-0.387, 0.068] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.179 | [-0.401, 0.063] | 9 |
| grad_x_input | ERD_L_spearman | -0.200 | [-0.376, -0.010] | 9 |
| grad_x_input | ERD_R_pearson | -0.135 | [-0.394, 0.144] | 9 |
| grad_x_input | ERD_R_spearman | -0.152 | [-0.429, 0.152] | 9 |
| grad_x_input | ERD_LR_pearson | -0.068 | [-0.223, 0.090] | 9 |
| grad_x_input | ERD_LR_spearman | -0.087 | [-0.275, 0.107] | 9 |
| grad_x_input | ERD_comb_pearson | -0.176 | [-0.424, 0.097] | 9 |
| grad_x_input | ERD_comb_spearman | -0.176 | [-0.397, 0.065] | 9 |
| grad_x_input | vs_grad_pearson | 0.997 | [0.993, 0.999] | 9 |
| grad_x_input | vs_grad_spearman | 0.992 | [0.985, 0.996] | 9 |
| integrated_gradients | ERD_L_pearson | -0.164 | [-0.394, 0.086] | 9 |
| integrated_gradients | ERD_L_spearman | -0.191 | [-0.368, -0.000] | 9 |
| integrated_gradients | ERD_R_pearson | -0.109 | [-0.375, 0.174] | 9 |
| integrated_gradients | ERD_R_spearman | -0.150 | [-0.425, 0.149] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.075 | [-0.228, 0.082] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.078 | [-0.262, 0.111] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.155 | [-0.414, 0.127] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.175 | [-0.378, 0.044] | 9 |
| integrated_gradients | vs_grad_pearson | 0.988 | [0.979, 0.993] | 9 |
| integrated_gradients | vs_grad_spearman | 0.982 | [0.969, 0.989] | 9 |
| smoothgrad | ERD_L_pearson | -0.157 | [-0.378, 0.080] | 9 |
| smoothgrad | ERD_L_spearman | -0.180 | [-0.351, 0.002] | 9 |
| smoothgrad | ERD_R_pearson | -0.138 | [-0.400, 0.146] | 9 |
| smoothgrad | ERD_R_spearman | -0.158 | [-0.432, 0.143] | 9 |
| smoothgrad | ERD_LR_pearson | -0.055 | [-0.205, 0.097] | 9 |
| smoothgrad | ERD_LR_spearman | -0.074 | [-0.261, 0.118] | 9 |
| smoothgrad | ERD_comb_pearson | -0.167 | [-0.419, 0.108] | 9 |
| smoothgrad | ERD_comb_spearman | -0.169 | [-0.392, 0.072] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.036 | [-0.182, 0.111] | 0.022 | -0.033 | -0.011 | 9 |
| ERD_theta_R | -0.102 | [-0.314, 0.120] | 0.098 | -0.125 | 0.102 | 9 |
| ERD_theta_mag | 0.105 | [-0.070, 0.274] | 0.105 | 0.099 | 0.099 | 9 |
| ERD_mu_L | -0.175 | [-0.398, 0.069] | 0.176 | -0.190 | 0.171 | 9 |
| ERD_mu_R | -0.145 | [-0.376, 0.103] | 0.059 | -0.132 | 0.046 | 9 |
| ERD_mu_mag | 0.117 | [-0.153, 0.371] | 0.117 | 0.125 | 0.125 | 9 |
| ERD_beta_L | -0.028 | [-0.231, 0.177] | -0.003 | -0.098 | -0.055 | 9 |
| ERD_beta_R | 0.093 | [-0.063, 0.244] | 0.103 | 0.035 | 0.033 | 9 |
| ERD_beta_mag | 0.030 | [-0.168, 0.227] | 0.030 | -0.038 | -0.038 | 9 |
| ERS_beta_rebound_L | -0.005 | [-0.225, 0.216] | -0.031 | -0.097 | -0.110 | 9 |
| ERS_beta_rebound_R | 0.108 | [-0.097, 0.304] | 0.125 | 0.017 | 0.042 | 9 |
| ERS_beta_rebound_mag | 0.012 | [-0.229, 0.252] | 0.012 | -0.071 | -0.071 | 9 |
| ERD_lowgamma_L | -0.089 | [-0.270, 0.097] | 0.037 | -0.113 | 0.070 | 9 |
| ERD_lowgamma_R | 0.090 | [-0.078, 0.252] | 0.014 | 0.111 | 0.025 | 9 |
| ERD_lowgamma_mag | 0.056 | [-0.072, 0.183] | 0.056 | 0.093 | 0.093 | 9 |
| logpow_8_30 | 0.092 | [-0.129, 0.303] | 0.092 | 0.110 | 0.110 | 9 |
| r2_signed | 0.069 | [-0.142, 0.274] | 0.101 | 0.083 | 0.124 | 9 |
| r2_abs | 0.101 | [-0.153, 0.342] | 0.101 | 0.124 | 0.124 | 9 |
| TFR_ERD_dB_L | -0.103 | [-0.325, 0.130] | 0.106 | -0.103 | 0.047 | 9 |
| TFR_ERD_dB_R | -0.043 | [-0.297, 0.217] | 0.126 | -0.059 | 0.082 | 9 |
| TFR_ERD_dB_mag | 0.131 | [-0.046, 0.300] | 0.131 | 0.095 | 0.095 | 9 |
| ITC_4_30 | 0.040 | [-0.170, 0.248] | 0.040 | 0.027 | 0.027 | 9 |
| ERP_abs_0.2_0.8 | 0.034 | [-0.128, 0.195] | 0.034 | 0.050 | 0.050 | 9 |
| SlowPot_0.5_3 | -0.207 | [-0.363, -0.040] | 0.186 | -0.176 | 0.116 | 9 |
| ERD_beta_classspecific | — | — | 0.054 [-0.068, 0.175] | — | 0.011 | 9 |
| ERD_lowgamma_classspecific | — | — | 0.019 [-0.078, 0.114] | — | 0.045 | 9 |
| ERD_mu_classspecific | — | — | 0.108 [-0.090, 0.297] | — | 0.100 | 9 |
| ERD_theta_classspecific | — | — | 0.053 [-0.051, 0.156] | — | 0.034 | 9 |
| ERS_beta_rebound_classspecific | — | — | 0.020 [-0.163, 0.202] | — | -0.029 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.121 [-0.025, 0.261] | — | 0.089 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +5.80 ± 3.02 | +3.58 ± 2.12 | +0.98 ± 2.03 | 0.0158 | <0.0001 |
| 4 | +9.85 ± 5.10 | +6.82 ± 3.71 | +4.17 ± 4.98 | 0.0049 | 0.0018 |
| 6 | +15.20 ± 7.42 | +9.40 ± 5.53 | +5.63 ± 4.87 | 0.0002 | <0.0001 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.051 | [-0.080, 0.181] | 9 |
| grad_x_input | 0.042 | [-0.088, 0.171] | 9 |
| integrated_gradients | 0.034 | [-0.095, 0.161] | 9 |
| smoothgrad | 0.054 | [-0.077, 0.184] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=cue_0_4, S02=cue_0_4, S03=cue_0_4, S04=cue_0_4, S05=cue_0_4, S06=cue_0_4, S07=cue_0_4, S08=cue_0_4, S09=cue_0_4

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 38.27 |
| cue_0_4 | 41.51 |

```
model: "EEGITNet"
dataset: "BNCI2014_001"
sfreq_hz: 250
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "cue_0_4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [0.0, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "no_aug": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": false, "mixup_p": 0.0, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "drop_0.25": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.25, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}}
checkpoint_rule: "SWA average from swa_start_frac*max_epochs + BN update (no early stopping); validation block only scored, never used for checkpointing"
folds: "train on session 1 (last 25% of it = validation, chronological), refit on all of session 1, test once on session 2 (official BCI IV-2a split)"
seeds: [42, 43, 44]
```
