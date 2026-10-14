# Results report — EEGNetV4/BNCI_001

Generated 2026-10-14 04:37 from the saved `results.json` files. TOST margin ±6 percentage points.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Subject accuracy = mean over the 4 folds, then over seeds (1 seed(s)). SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 79.86 | 50.69 | 93.75 | 55.21 | 60.76 | 55.21 | 77.43 | 92.71 | 90.28 | 72.88 | 17.55 |
| Ours | 78.82 | 51.04 | 95.14 | 57.64 | 50.35 | 53.82 | 62.85 | 94.44 | 80.90 | 69.44 | 18.15 |
| CSP | 81.25 | 51.04 | 92.36 | 62.50 | 52.08 | 51.74 | 71.18 | 94.10 | 82.29 | 70.95 | 17.40 |
| RLF | 80.21 | 52.08 | 94.79 | 58.68 | 53.12 | 56.94 | 68.06 | 94.44 | 82.29 | 71.18 | 17.19 |
| MI | 80.56 | 50.00 | 93.06 | 63.89 | 50.35 | 57.29 | 61.81 | 93.06 | 80.21 | 70.02 | 17.07 |
| Fixed sensorimotor | 84.03 | 51.39 | 92.01 | 62.85 | 48.96 | 52.43 | 55.21 | 93.40 | 81.25 | 69.06 | 18.42 |
| ERD-only | 80.56 | 49.65 | 91.32 | 58.33 | 57.99 | 53.82 | 58.68 | 94.79 | 81.25 | 69.60 | 17.28 |
| Random (mean of draws) | 76.94 | 51.88 | 93.12 | 60.62 | 51.46 | 52.78 | 56.39 | 93.61 | 79.17 | 68.44 | 17.48 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | CP4 CP3 C4 P2 FC3 C3 FC4 Cz C6 FCz Fz C2 |
| CSP | 9 | CP4 C4 CP3 P2 C6 CP2 P1 POz Pz CP1 C3 C2 |
| RLF | 9 | CP4 P2 C4 CP3 C3 POz P1 C5 FC3 FC1 C1 CP1 |
| MI | 9 | CP4 C4 C6 CP1 FC2 C3 FCz C2 CP3 CPz P1 FC1 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 Pz CP3 POz P1 C3 CPz CP2 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -3.43 | 6.29 | [-7.34, +0.47] | 0.1280 | no | -14.58 | 3 | 8,10 |
| CSP | 9 | -1.93 | 5.19 | [-5.15, +1.29] | 0.0233 | yes | -8.68 | 3 | 6,8,10 |
| RLF | 9 | -1.70 | 5.07 | [-4.84, +1.44] | 0.0171 | yes | -9.38 | 3 | 5,6,8,10 |
| MI | 9 | -2.85 | 7.60 | [-7.57, +1.86] | 0.1248 | no | -15.62 | 3 | 8,10 |
| Fixed sensorimotor | 9 | -3.82 | 9.16 | [-9.50, +1.86] | 0.2477 | no | -22.22 | 3 | 10 |
| ERD-only | 9 | -3.28 | 6.78 | [-7.48, +0.92] | 0.1315 | no | -18.75 | 2 | 8,10 |
| Random (mean of draws) | 9 | -4.44 | 8.08 | [-9.44, +0.57] | 0.2888 | no | -21.04 | 3 | 10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | -1.50 | [-4.16, +1.15] | 0.2282 | 1.0000 | 0.3828 | 1.0000 | 3 / 1 / 5 |
| RLF | 9 | -1.74 | [-3.06, -0.42] | 0.0163 | 0.0978 | 0.0156 | 0.0938 | 1 / 1 / 7 |
| MI | 9 | -0.58 | [-2.68, +1.53] | 0.5439 | 1.0000 | 0.8203 | 1.0000 | 5 / 1 / 3 |
| Fixed sensorimotor | 9 | +0.39 | [-2.67, +3.44] | 0.7782 | 1.0000 | 0.6328 | 1.0000 | 5 / 0 / 4 |
| ERD-only | 9 | -0.15 | [-2.81, +2.51] | 0.8969 | 1.0000 | 0.9453 | 1.0000 | 3 / 1 / 5 |
| Random (mean of draws) | 9 | +1.00 | [-1.03, +3.03] | 0.2878 | 1.0000 | 0.3594 | 1.0000 | 6 / 0 / 3 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 72.88 | left_hand 80.8, right_hand 65.0 | left_hand 57.9, right_hand 42.1 | 50.00 | 4 / 36 |
| Ours | 69.44 | left_hand 71.5, right_hand 67.4 | left_hand 52.0, right_hand 48.0 | 50.00 | 8 / 36 |
| CSP | 70.95 | left_hand 69.8, right_hand 72.1 | left_hand 48.9, right_hand 51.1 | 50.00 | 6 / 36 |
| RLF | 71.18 | left_hand 71.1, right_hand 71.2 | left_hand 50.0, right_hand 50.0 | 50.00 | 8 / 36 |
| MI | 70.02 | left_hand 72.2, right_hand 67.8 | left_hand 52.2, right_hand 47.8 | 50.00 | 7 / 36 |
| Fixed sensorimotor | 69.06 | left_hand 70.6, right_hand 67.5 | left_hand 51.5, right_hand 48.5 | 50.00 | 9 / 36 |
| ERD-only | 69.60 | left_hand 70.7, right_hand 68.5 | left_hand 51.1, right_hand 48.9 | 50.00 | 6 / 36 |
| Random (mean of draws) | 68.44 | left_hand 71.0, right_hand 65.8 | left_hand 52.6, right_hand 47.4 | 50.00 | 43 / 180 |

### T6. Seed stability

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 1 | 42: 72.88 | 72.88 | — |
| Ours | 1 | 42: 69.44 | 69.44 | — |
| CSP | 1 | 42: 70.95 | 70.95 | — |
| RLF | 1 | 42: 71.18 | 71.18 | — |
| MI | 1 | 42: 70.02 | 70.02 | — |
| Fixed sensorimotor | 1 | 42: 69.06 | 69.06 | — |
| ERD-only | 1 | 42: 69.60 | 69.60 | — |
| Random (mean of draws) | 1 | 42: 68.44 | 68.44 | — |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.100 | [-0.121, 0.312] | 0.275 | 9 |
| grad | class_specific_pearson | 0.170 | [0.075, 0.262] | 0.121 | 9 |
| grad | lateralisation_pearson | -0.184 | [-0.469, 0.135] | 0.380 | 9 |
| grad | magnitude_spearman | 0.097 | [-0.093, 0.280] | 0.239 | 9 |
| grad | class_specific_spearman | 0.114 | [0.070, 0.158] | 0.058 | 9 |
| grad | lateralisation_spearman | -0.128 | [-0.393, 0.157] | 0.341 | 9 |
| grad | legacy_signed_LR_pearson | -0.079 | [-0.269, 0.117] | 0.245 | 9 |
| grad_x_input | magnitude_pearson | 0.075 | [-0.152, 0.295] | 0.281 | 9 |
| grad_x_input | class_specific_pearson | 0.156 | [0.055, 0.254] | 0.129 | 9 |
| grad_x_input | lateralisation_pearson | -0.382 | [-0.655, -0.020] | 0.415 | 9 |
| grad_x_input | magnitude_spearman | 0.053 | [-0.151, 0.253] | 0.259 | 9 |
| grad_x_input | class_specific_spearman | 0.100 | [0.042, 0.157] | 0.074 | 9 |
| grad_x_input | lateralisation_spearman | -0.324 | [-0.605, 0.029] | 0.400 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.090 | [-0.264, 0.090] | 0.226 | 9 |
| integrated_gradients | magnitude_pearson | 0.056 | [-0.182, 0.288] | 0.294 | 9 |
| integrated_gradients | class_specific_pearson | 0.136 | [0.016, 0.252] | 0.153 | 9 |
| integrated_gradients | lateralisation_pearson | -0.353 | [-0.634, 0.011] | 0.416 | 9 |
| integrated_gradients | magnitude_spearman | 0.061 | [-0.146, 0.263] | 0.259 | 9 |
| integrated_gradients | class_specific_spearman | 0.098 | [0.034, 0.161] | 0.082 | 9 |
| integrated_gradients | lateralisation_spearman | -0.281 | [-0.559, 0.054] | 0.388 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.083 | [-0.262, 0.101] | 0.232 | 9 |
| smoothgrad | magnitude_pearson | 0.100 | [-0.120, 0.310] | 0.273 | 9 |
| smoothgrad | class_specific_pearson | 0.172 | [0.078, 0.263] | 0.120 | 9 |
| smoothgrad | lateralisation_pearson | -0.181 | [-0.465, 0.136] | 0.378 | 9 |
| smoothgrad | magnitude_spearman | 0.100 | [-0.087, 0.279] | 0.236 | 9 |
| smoothgrad | class_specific_spearman | 0.115 | [0.070, 0.159] | 0.058 | 9 |
| smoothgrad | lateralisation_spearman | -0.133 | [-0.401, 0.156] | 0.345 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.078 | [-0.267, 0.116] | 0.244 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.193 | [-0.335, -0.043] | 9 |
| grad | ERD_L_spearman | -0.152 | [-0.223, -0.080] | 9 |
| grad | ERD_R_pearson | -0.164 | [-0.302, -0.019] | 9 |
| grad | ERD_R_spearman | -0.087 | [-0.200, 0.028] | 9 |
| grad | ERD_LR_pearson | -0.079 | [-0.269, 0.117] | 9 |
| grad | ERD_LR_spearman | -0.074 | [-0.210, 0.065] | 9 |
| grad | ERD_comb_pearson | -0.194 | [-0.305, -0.078] | 9 |
| grad | ERD_comb_spearman | -0.135 | [-0.213, -0.056] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.204 | [-0.355, -0.041] | 9 |
| grad_x_input | ERD_L_spearman | -0.170 | [-0.263, -0.074] | 9 |
| grad_x_input | ERD_R_pearson | -0.149 | [-0.270, -0.023] | 9 |
| grad_x_input | ERD_R_spearman | -0.072 | [-0.171, 0.027] | 9 |
| grad_x_input | ERD_LR_pearson | -0.090 | [-0.264, 0.090] | 9 |
| grad_x_input | ERD_LR_spearman | -0.088 | [-0.203, 0.029] | 9 |
| grad_x_input | ERD_comb_pearson | -0.194 | [-0.312, -0.069] | 9 |
| grad_x_input | ERD_comb_spearman | -0.139 | [-0.222, -0.053] | 9 |
| grad_x_input | vs_grad_pearson | 0.998 | [0.996, 0.999] | 9 |
| grad_x_input | vs_grad_spearman | 0.992 | [0.983, 0.996] | 9 |
| integrated_gradients | ERD_L_pearson | -0.182 | [-0.343, -0.011] | 9 |
| integrated_gradients | ERD_L_spearman | -0.158 | [-0.252, -0.060] | 9 |
| integrated_gradients | ERD_R_pearson | -0.130 | [-0.274, 0.021] | 9 |
| integrated_gradients | ERD_R_spearman | -0.090 | [-0.222, 0.046] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.083 | [-0.262, 0.101] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.061 | [-0.196, 0.075] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.169 | [-0.307, -0.023] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.139 | [-0.221, -0.056] | 9 |
| integrated_gradients | vs_grad_pearson | 0.990 | [0.984, 0.994] | 9 |
| integrated_gradients | vs_grad_spearman | 0.973 | [0.962, 0.981] | 9 |
| smoothgrad | ERD_L_pearson | -0.194 | [-0.335, -0.045] | 9 |
| smoothgrad | ERD_L_spearman | -0.155 | [-0.231, -0.078] | 9 |
| smoothgrad | ERD_R_pearson | -0.166 | [-0.302, -0.023] | 9 |
| smoothgrad | ERD_R_spearman | -0.089 | [-0.198, 0.022] | 9 |
| smoothgrad | ERD_LR_pearson | -0.078 | [-0.267, 0.116] | 9 |
| smoothgrad | ERD_LR_spearman | -0.073 | [-0.207, 0.063] | 9 |
| smoothgrad | ERD_comb_pearson | -0.196 | [-0.306, -0.082] | 9 |
| smoothgrad | ERD_comb_spearman | -0.138 | [-0.218, -0.057] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.059 | [-0.212, 0.098] | -0.063 | -0.093 | -0.046 | 9 |
| ERD_theta_R | -0.005 | [-0.155, 0.146] | 0.023 | -0.018 | -0.005 | 9 |
| ERD_theta_mag | -0.011 | [-0.158, 0.138] | -0.011 | -0.034 | -0.034 | 9 |
| ERD_mu_L | -0.171 | [-0.308, -0.028] | 0.075 | -0.136 | 0.052 | 9 |
| ERD_mu_R | -0.079 | [-0.239, 0.085] | 0.145 | 0.006 | 0.045 | 9 |
| ERD_mu_mag | 0.117 | [-0.053, 0.281] | 0.117 | 0.059 | 0.059 | 9 |
| ERD_beta_L | -0.127 | [-0.270, 0.021] | 0.063 | -0.101 | -0.015 | 9 |
| ERD_beta_R | -0.058 | [-0.193, 0.079] | -0.004 | -0.007 | -0.061 | 9 |
| ERD_beta_mag | 0.034 | [-0.104, 0.170] | 0.034 | -0.011 | -0.011 | 9 |
| ERS_beta_rebound_L | -0.010 | [-0.129, 0.109] | 0.058 | -0.023 | 0.051 | 9 |
| ERS_beta_rebound_R | -0.013 | [-0.124, 0.099] | -0.015 | -0.024 | 0.031 | 9 |
| ERS_beta_rebound_mag | 0.012 | [-0.108, 0.131] | 0.012 | 0.040 | 0.040 | 9 |
| ERD_lowgamma_L | -0.075 | [-0.263, 0.117] | -0.045 | -0.012 | -0.068 | 9 |
| ERD_lowgamma_R | -0.200 | [-0.349, -0.042] | -0.088 | -0.159 | -0.049 | 9 |
| ERD_lowgamma_mag | -0.093 | [-0.323, 0.148] | -0.093 | -0.123 | -0.123 | 9 |
| logpow_8_30 | -0.127 | [-0.297, 0.050] | -0.127 | -0.074 | -0.074 | 9 |
| r2_signed | 0.138 | [-0.117, 0.375] | 0.348 | 0.069 | 0.134 | 9 |
| r2_abs | 0.348 | [0.145, 0.522] | 0.348 | 0.134 | 0.134 | 9 |
| TFR_ERD_dB_L | -0.210 | [-0.351, -0.060] | 0.108 | -0.170 | 0.066 | 9 |
| TFR_ERD_dB_R | -0.138 | [-0.230, -0.044] | 0.055 | -0.064 | -0.015 | 9 |
| TFR_ERD_dB_mag | 0.106 | [-0.113, 0.316] | 0.106 | 0.036 | 0.036 | 9 |
| ITC_4_30 | -0.017 | [-0.213, 0.180] | -0.017 | -0.008 | -0.008 | 9 |
| ERP_abs_0.2_0.8 | -0.049 | [-0.180, 0.083] | -0.049 | -0.027 | -0.027 | 9 |
| SlowPot_0.5_3 | 0.055 | [-0.055, 0.164] | -0.090 | -0.045 | -0.008 | 9 |
| ERD_beta_classspecific | — | — | 0.021 [-0.094, 0.135] | — | -0.054 | 9 |
| ERD_lowgamma_classspecific | — | — | -0.085 [-0.232, 0.065] | — | -0.058 | 9 |
| ERD_mu_classspecific | — | — | 0.091 [-0.062, 0.240] | — | 0.048 | 9 |
| ERD_theta_classspecific | — | — | -0.033 [-0.147, 0.082] | — | -0.021 | 9 |
| ERS_beta_rebound_classspecific | — | — | 0.018 [-0.086, 0.122] | — | 0.030 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.062 [-0.098, 0.219] | — | 0.017 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +17.05 ± 15.11 | +5.48 ± 4.74 | +0.54 ± 1.61 | 0.0142 | 0.0088 |
| 4 | +16.44 ± 15.04 | +10.38 ± 8.35 | +2.85 ± 4.07 | 0.0660 | 0.0251 |
| 6 | +18.79 ± 16.68 | +13.22 ± 11.00 | +4.17 ± 5.46 | 0.0379 | 0.0173 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | 0.070 | [-0.025, 0.163] | 9 |
| grad_x_input | 0.070 | [-0.030, 0.169] | 9 |
| integrated_gradients | 0.074 | [-0.027, 0.173] | 9 |
| smoothgrad | 0.069 | [-0.025, 0.161] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: True): S01=original, S02=original, S03=original, S04=original, S05=original, S06=original, S07=original, S08=original, S09=original

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 74.50 |
| k64 | 72.49 |
| k64_long | 71.76 |
| k64_aug | 69.52 |
| k64_long_aug | 69.87 |

```
model: "EEGNetv4"
dataset: "BNCI2014_001"
sfreq_hz: 128
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 32, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64_long": {"band_hz": [4.0, 40.0], "window_s": [0.5, 4.0], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64_aug": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": true}, "k64_long_aug": {"band_hz": [4.0, 40.0], "window_s": [0.5, 4.0], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": true}}
checkpoint_rule: "lowest validation loss, early stopping with patience"
folds: "4 chronological blocks; test block i, validation block i-1, rest train"
seeds: [42]
```

## 4class

### T1. Test accuracy (%) per subject — chance 25.0%

Subject accuracy = mean over the 4 folds, then over seeds (1 seed(s)). SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 70.83 | 23.96 | 86.46 | 40.45 | 30.90 | 40.80 | 78.12 | 75.00 | 84.20 | 58.97 | 24.61 |
| Ours | 70.83 | 31.42 | 83.33 | 40.97 | 33.33 | 43.06 | 73.09 | 75.17 | 85.59 | 59.65 | 22.06 |
| CSP | 69.27 | 31.08 | 81.25 | 48.61 | 31.60 | 40.62 | 65.28 | 76.04 | 84.20 | 58.66 | 21.04 |
| RLF | 68.40 | 29.51 | 76.74 | 39.93 | 27.95 | 34.55 | 70.66 | 69.79 | 79.51 | 55.23 | 21.63 |
| MI | 67.36 | 31.42 | 82.99 | 37.50 | 32.99 | 37.85 | 71.35 | 76.39 | 81.08 | 57.66 | 22.13 |
| Fixed sensorimotor | 67.71 | 38.37 | 84.38 | 37.85 | 32.47 | 44.27 | 79.69 | 76.22 | 82.64 | 60.40 | 21.74 |
| ERD-only | 70.49 | 38.54 | 84.55 | 49.31 | 31.42 | 39.58 | 71.88 | 74.83 | 84.03 | 60.51 | 20.79 |
| Random (mean of draws) | 69.76 | 34.10 | 79.10 | 43.51 | 29.55 | 41.42 | 68.44 | 74.51 | 81.32 | 57.97 | 20.54 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | C3 CP3 CPz C4 Cz POz CP4 C5 CP2 C1 FCz Pz |
| CSP | 9 | P2 CP3 P1 Pz CP4 C3 POz CP1 CP2 C4 CPz C1 |
| RLF | 9 | C3 CP4 CP3 C4 Fz POz FC2 FC3 P2 P1 FCz FC1 |
| MI | 9 | P2 C3 CP3 FCz P1 CP2 CP4 FC1 C4 CP1 C1 FC3 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 Pz CP3 POz P1 C3 CPz CP2 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | +0.68 | 3.54 | [-1.52, +2.87] | 0.0010 | yes | -5.03 | 0 | 3,4,5,6,8,10 |
| CSP | 9 | -0.31 | 6.25 | [-4.18, +3.57] | 0.0129 | yes | -12.85 | 1 | 5,6,8,10 |
| RLF | 9 | -3.74 | 4.45 | [-6.50, -0.99] | 0.0831 | no | -9.72 | 3 | 8,10 |
| MI | 9 | -1.31 | 4.24 | [-3.94, +1.31] | 0.0053 | yes | -6.77 | 1 | 4,5,6,8,10 |
| Fixed sensorimotor | 9 | +1.43 | 5.37 | [-1.90, +4.76] | 0.0170 | yes | -3.13 | 0 | 5,6,8,10 |
| ERD-only | 9 | +1.54 | 6.27 | [-2.34, +5.43] | 0.0327 | yes | -6.25 | 1 | 6,8,10 |
| Random (mean of draws) | 9 | -1.00 | 5.73 | [-4.56, +2.55] | 0.0155 | yes | -9.69 | 2 | 5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | +0.98 | [-2.11, +4.07] | 0.4836 | 1.0000 | 0.1641 | 0.5156 | 7 / 0 / 2 |
| RLF | 9 | +4.42 | [+2.46, +6.37] | 0.0008 | 0.0048 | 0.0039 | 0.0234 | 9 / 0 / 0 |
| MI | 9 | +1.99 | [+0.25, +3.72] | 0.0298 | 0.1488 | 0.0391 | 0.1953 | 7 / 1 / 1 |
| Fixed sensorimotor | 9 | -0.75 | [-3.72, +2.22] | 0.5750 | 1.0000 | 0.7148 | 1.0000 | 4 / 0 / 5 |
| ERD-only | 9 | -0.87 | [-4.02, +2.29] | 0.5433 | 1.0000 | 0.9102 | 1.0000 | 6 / 0 / 3 |
| Random (mean of draws) | 9 | +1.68 | [-0.51, +3.86] | 0.1145 | 0.4579 | 0.1289 | 0.5156 | 7 / 0 / 2 |

### T5. Confusion, class balance and majority baseline (chance 25.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 58.97 | feet 55.4, left_hand 64.4, right_hand 57.6, tongue 58.6 | feet 24.8, left_hand 28.3, right_hand 22.3, tongue 24.6 | 25.00 | 2 / 36 |
| Ours | 59.65 | feet 53.7, left_hand 63.0, right_hand 59.6, tongue 62.2 | feet 21.6, left_hand 27.2, right_hand 24.0, tongue 27.2 | 25.00 | 1 / 36 |
| CSP | 58.66 | feet 52.5, left_hand 65.5, right_hand 57.6, tongue 59.0 | feet 21.5, left_hand 29.6, right_hand 23.2, tongue 25.7 | 25.00 | 1 / 36 |
| RLF | 55.23 | feet 45.5, left_hand 60.1, right_hand 58.3, tongue 57.0 | feet 22.4, left_hand 25.9, right_hand 25.3, tongue 26.5 | 25.00 | 2 / 36 |
| MI | 57.66 | feet 52.5, left_hand 58.3, right_hand 60.4, tongue 59.3 | feet 23.2, left_hand 24.2, right_hand 26.7, tongue 25.9 | 25.00 | 2 / 36 |
| Fixed sensorimotor | 60.40 | feet 53.5, left_hand 64.4, right_hand 62.7, tongue 61.1 | feet 21.4, left_hand 26.9, right_hand 25.4, tongue 26.3 | 25.00 | 2 / 36 |
| ERD-only | 60.51 | feet 54.4, left_hand 61.3, right_hand 60.9, tongue 65.5 | feet 20.7, left_hand 25.0, right_hand 25.4, tongue 29.0 | 25.00 | 0 / 36 |
| Random (mean of draws) | 57.97 | feet 49.8, left_hand 61.9, right_hand 59.5, tongue 60.7 | feet 20.4, left_hand 26.4, right_hand 25.8, tongue 27.4 | 25.00 | 3 / 180 |

### T6. Seed stability

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 1 | 42: 58.97 | 58.97 | — |
| Ours | 1 | 42: 59.65 | 59.65 | — |
| CSP | 1 | 42: 58.66 | 58.66 | — |
| RLF | 1 | 42: 55.23 | 55.23 | — |
| MI | 1 | 42: 57.66 | 57.66 | — |
| Fixed sensorimotor | 1 | 42: 60.40 | 60.40 | — |
| ERD-only | 1 | 42: 60.51 | 60.51 | — |
| Random (mean of draws) | 1 | 42: 57.97 | 57.97 | — |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.109 | [-0.261, 0.451] | 0.424 | 9 |
| grad | class_specific_pearson | 0.354 | [0.234, 0.463] | 0.153 | 9 |
| grad | lateralisation_pearson | 0.168 | [-0.073, 0.391] | 0.290 | 9 |
| grad | magnitude_spearman | 0.089 | [-0.267, 0.423] | 0.406 | 9 |
| grad | class_specific_spearman | 0.328 | [0.205, 0.440] | 0.154 | 9 |
| grad | lateralisation_spearman | 0.153 | [-0.084, 0.374] | 0.290 | 9 |
| grad | legacy_signed_LR_pearson | -0.068 | [-0.294, 0.165] | 0.280 | 9 |
| grad_x_input | magnitude_pearson | 0.086 | [-0.297, 0.446] | 0.442 | 9 |
| grad_x_input | class_specific_pearson | 0.350 | [0.234, 0.457] | 0.150 | 9 |
| grad_x_input | lateralisation_pearson | 0.112 | [-0.137, 0.347] | 0.303 | 9 |
| grad_x_input | magnitude_spearman | 0.066 | [-0.305, 0.420] | 0.427 | 9 |
| grad_x_input | class_specific_spearman | 0.328 | [0.212, 0.435] | 0.148 | 9 |
| grad_x_input | lateralisation_spearman | 0.092 | [-0.137, 0.312] | 0.286 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.084 | [-0.311, 0.152] | 0.284 | 9 |
| integrated_gradients | magnitude_pearson | 0.095 | [-0.280, 0.445] | 0.433 | 9 |
| integrated_gradients | class_specific_pearson | 0.328 | [0.203, 0.443] | 0.159 | 9 |
| integrated_gradients | lateralisation_pearson | 0.047 | [-0.159, 0.250] | 0.260 | 9 |
| integrated_gradients | magnitude_spearman | 0.071 | [-0.285, 0.410] | 0.412 | 9 |
| integrated_gradients | class_specific_spearman | 0.287 | [0.172, 0.395] | 0.144 | 9 |
| integrated_gradients | lateralisation_spearman | 0.053 | [-0.146, 0.247] | 0.251 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.072 | [-0.303, 0.166] | 0.288 | 9 |
| smoothgrad | magnitude_pearson | 0.108 | [-0.263, 0.451] | 0.424 | 9 |
| smoothgrad | class_specific_pearson | 0.355 | [0.237, 0.463] | 0.152 | 9 |
| smoothgrad | lateralisation_pearson | 0.169 | [-0.072, 0.391] | 0.290 | 9 |
| smoothgrad | magnitude_spearman | 0.088 | [-0.268, 0.422] | 0.407 | 9 |
| smoothgrad | class_specific_spearman | 0.332 | [0.210, 0.444] | 0.153 | 9 |
| smoothgrad | lateralisation_spearman | 0.147 | [-0.091, 0.369] | 0.290 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.066 | [-0.293, 0.167] | 0.281 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.381 | [-0.532, -0.206] | 9 |
| grad | ERD_L_spearman | -0.313 | [-0.489, -0.113] | 9 |
| grad | ERD_R_pearson | -0.358 | [-0.487, -0.213] | 9 |
| grad | ERD_R_spearman | -0.353 | [-0.479, -0.212] | 9 |
| grad | ERD_LR_pearson | -0.068 | [-0.294, 0.165] | 9 |
| grad | ERD_LR_spearman | -0.000 | [-0.192, 0.191] | 9 |
| grad | ERD_comb_pearson | -0.422 | [-0.556, -0.267] | 9 |
| grad | ERD_comb_spearman | -0.426 | [-0.576, -0.247] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.407 | [-0.555, -0.234] | 9 |
| grad_x_input | ERD_L_spearman | -0.340 | [-0.509, -0.144] | 9 |
| grad_x_input | ERD_R_pearson | -0.363 | [-0.496, -0.214] | 9 |
| grad_x_input | ERD_R_spearman | -0.354 | [-0.488, -0.203] | 9 |
| grad_x_input | ERD_LR_pearson | -0.084 | [-0.311, 0.152] | 9 |
| grad_x_input | ERD_LR_spearman | -0.021 | [-0.215, 0.174] | 9 |
| grad_x_input | ERD_comb_pearson | -0.439 | [-0.568, -0.289] | 9 |
| grad_x_input | ERD_comb_spearman | -0.447 | [-0.591, -0.277] | 9 |
| grad_x_input | vs_grad_pearson | 0.999 | [0.997, 1.000] | 9 |
| grad_x_input | vs_grad_spearman | 0.999 | [0.981, 1.000] | 9 |
| integrated_gradients | ERD_L_pearson | -0.389 | [-0.537, -0.217] | 9 |
| integrated_gradients | ERD_L_spearman | -0.326 | [-0.495, -0.134] | 9 |
| integrated_gradients | ERD_R_pearson | -0.354 | [-0.487, -0.204] | 9 |
| integrated_gradients | ERD_R_spearman | -0.353 | [-0.488, -0.203] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.072 | [-0.303, 0.166] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.007 | [-0.208, 0.194] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.424 | [-0.553, -0.274] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.438 | [-0.577, -0.275] | 9 |
| integrated_gradients | vs_grad_pearson | 0.996 | [0.991, 0.998] | 9 |
| integrated_gradients | vs_grad_spearman | 0.987 | [0.982, 0.991] | 9 |
| smoothgrad | ERD_L_pearson | -0.382 | [-0.533, -0.207] | 9 |
| smoothgrad | ERD_L_spearman | -0.315 | [-0.491, -0.114] | 9 |
| smoothgrad | ERD_R_pearson | -0.360 | [-0.489, -0.215] | 9 |
| smoothgrad | ERD_R_spearman | -0.353 | [-0.479, -0.212] | 9 |
| smoothgrad | ERD_LR_pearson | -0.066 | [-0.293, 0.167] | 9 |
| smoothgrad | ERD_LR_spearman | -0.001 | [-0.192, 0.192] | 9 |
| smoothgrad | ERD_comb_pearson | -0.424 | [-0.558, -0.269] | 9 |
| smoothgrad | ERD_comb_spearman | -0.427 | [-0.578, -0.248] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.207 | [-0.376, -0.025] | 0.069 | -0.239 | 0.113 | 9 |
| ERD_theta_R | -0.202 | [-0.416, 0.033] | 0.161 | -0.208 | 0.178 | 9 |
| ERD_theta_mag | 0.102 | [-0.161, 0.352] | 0.102 | 0.144 | 0.144 | 9 |
| ERD_mu_L | -0.307 | [-0.509, -0.073] | 0.019 | -0.234 | 0.008 | 9 |
| ERD_mu_R | -0.311 | [-0.475, -0.127] | 0.093 | -0.288 | 0.052 | 9 |
| ERD_mu_mag | 0.043 | [-0.278, 0.355] | 0.043 | 0.027 | 0.027 | 9 |
| ERD_beta_L | -0.320 | [-0.422, -0.211] | -0.021 | -0.277 | -0.082 | 9 |
| ERD_beta_R | -0.205 | [-0.423, 0.035] | 0.090 | -0.190 | 0.009 | 9 |
| ERD_beta_mag | -0.004 | [-0.273, 0.266] | -0.004 | -0.060 | -0.060 | 9 |
| ERS_beta_rebound_L | -0.159 | [-0.327, 0.018] | -0.188 | -0.108 | -0.145 | 9 |
| ERS_beta_rebound_R | -0.097 | [-0.319, 0.135] | -0.101 | -0.062 | -0.085 | 9 |
| ERS_beta_rebound_mag | -0.180 | [-0.357, 0.009] | -0.180 | -0.110 | -0.110 | 9 |
| ERD_lowgamma_L | -0.164 | [-0.395, 0.086] | -0.019 | -0.106 | 0.002 | 9 |
| ERD_lowgamma_R | -0.290 | [-0.447, -0.117] | -0.143 | -0.207 | -0.180 | 9 |
| ERD_lowgamma_mag | -0.095 | [-0.421, 0.252] | -0.095 | -0.112 | -0.112 | 9 |
| logpow_8_30 | 0.004 | [-0.196, 0.204] | 0.004 | 0.028 | 0.028 | 9 |
| r2_signed | 0.097 | [-0.224, 0.398] | 0.102 | 0.049 | 0.073 | 9 |
| r2_abs | 0.102 | [-0.210, 0.395] | 0.102 | 0.073 | 0.073 | 9 |
| TFR_ERD_dB_L | -0.392 | [-0.512, -0.257] | 0.076 | -0.326 | 0.056 | 9 |
| TFR_ERD_dB_R | -0.316 | [-0.523, -0.075] | 0.236 | -0.314 | 0.189 | 9 |
| TFR_ERD_dB_mag | 0.161 | [-0.144, 0.438] | 0.161 | 0.135 | 0.135 | 9 |
| ITC_4_30 | 0.019 | [-0.177, 0.213] | 0.019 | 0.014 | 0.014 | 9 |
| ERP_abs_0.2_0.8 | 0.047 | [-0.124, 0.216] | 0.047 | 0.089 | 0.089 | 9 |
| SlowPot_0.5_3 | 0.085 | [-0.117, 0.280] | -0.107 | 0.014 | -0.035 | 9 |
| ERD_beta_classspecific | — | — | 0.081 [-0.142, 0.297] | — | 0.007 | 9 |
| ERD_lowgamma_classspecific | — | — | -0.073 [-0.252, 0.111] | — | -0.100 | 9 |
| ERD_mu_classspecific | — | — | 0.088 [-0.117, 0.286] | — | 0.048 | 9 |
| ERD_theta_classspecific | — | — | 0.122 [-0.050, 0.287] | — | 0.136 | 9 |
| ERS_beta_rebound_classspecific | — | — | -0.118 [-0.263, 0.033] | — | -0.107 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.212 [-0.051, 0.447] | — | 0.158 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +23.92 ± 19.15 | +13.98 ± 11.56 | +5.25 ± 5.31 | 0.0064 | 0.0060 |
| 4 | +27.51 ± 21.04 | +19.33 ± 15.36 | +10.84 ± 9.87 | 0.0058 | 0.0038 |
| 6 | +27.53 ± 21.04 | +23.54 ± 19.09 | +12.98 ± 11.15 | 0.0108 | 0.0024 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.049 | [-0.150, 0.053] | 9 |
| grad_x_input | -0.045 | [-0.144, 0.054] | 9 |
| integrated_gradients | -0.055 | [-0.150, 0.041] | 9 |
| smoothgrad | -0.046 | [-0.147, 0.055] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: False): S01=original, S02=original, S03=original, S04=k64_long, S05=k64_long, S06=original, S07=original, S08=original, S09=original

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 63.39 |
| k64 | 62.91 |
| k64_long | 62.83 |
| k64_aug | 54.24 |
| k64_long_aug | 52.76 |

```
model: "EEGNetv4"
dataset: "BNCI2014_001"
sfreq_hz: 128
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 32, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64_long": {"band_hz": [4.0, 40.0], "window_s": [0.5, 4.0], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": false}, "k64_aug": {"band_hz": [4.0, 40.0], "window_s": [0.5, 2.5], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": true}, "k64_long_aug": {"band_hz": [4.0, 40.0], "window_s": [0.5, 4.0], "kernel_length": 64, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.0005, "label_smoothing": 0.05, "batch_size": 64, "max_epochs": 800, "patience": 120, "augment": true}}
checkpoint_rule: "lowest validation loss, early stopping with patience"
folds: "4 chronological blocks; test block i, validation block i-1, rest train"
seeds: [42]
```
