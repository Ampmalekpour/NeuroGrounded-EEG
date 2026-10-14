# Results report — EEGItNet/BNCI_001

Generated 2026-10-14 04:37 from the saved `results.json` files. TOST margin ±6 percentage points.

## 2class

### T1. Test accuracy (%) per subject — chance 50.0%

Subject accuracy = mean over the 4 folds, then over seeds (1 seed(s)). SD = across subjects. Random = mean of the random-montage draws.

| Method | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | Mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22ch | 70.83 | 59.38 | 72.57 | 70.83 | 60.42 | 61.11 | 67.01 | 86.11 | 67.01 | 68.36 | 8.25 |
| Ours | 64.58 | 58.33 | 70.14 | 70.83 | 63.19 | 61.11 | 69.79 | 82.29 | 67.36 | 67.52 | 7.02 |
| CSP | 69.10 | 53.82 | 69.10 | 68.06 | 61.46 | 55.90 | 70.49 | 86.11 | 69.44 | 67.05 | 9.49 |
| RLF | 68.75 | 54.86 | 70.49 | 75.00 | 59.38 | 57.29 | 68.06 | 85.07 | 64.58 | 67.05 | 9.44 |
| MI | 55.90 | 52.08 | 70.49 | 64.58 | 62.85 | 58.68 | 66.67 | 82.29 | 65.62 | 64.35 | 8.82 |
| Fixed sensorimotor | 67.36 | 55.21 | 78.12 | 66.67 | 62.15 | 63.54 | 66.67 | 83.68 | 63.89 | 67.48 | 8.55 |
| ERD-only | 69.10 | 59.72 | 72.57 | 71.88 | 63.54 | 60.42 | 69.79 | 81.25 | 67.01 | 68.36 | 6.72 |
| Random (mean of draws) | 63.33 | 55.00 | 70.21 | 66.74 | 60.42 | 58.75 | 65.14 | 81.46 | 64.17 | 65.02 | 7.62 |

### T2. Montages and profiles

Full list per subject in `T2_montages.csv` (all 12-channel montages are leave-one-subject-out).

| Method | Distinct montages across subjects | Most frequent montage |
|---|---|---|
| Ours | 9 | CP4 P2 FC3 P1 CP3 C5 Cz C1 CP1 C3 C4 CP2 |
| CSP | 9 | CP4 C4 CP3 P2 FC3 FC2 FCz Fz CP2 FC4 FC1 C2 |
| RLF | 9 | CP4 C3 Fz CP3 FCz CP1 C4 C1 P1 P2 FC2 CP2 |
| MI | 9 | P2 C2 C3 Cz Pz FCz FC4 C4 CP2 CP1 CP3 C6 |
| Fixed sensorimotor | 1 | FC3 FC1 FC2 FC4 C3 C1 Cz C2 C4 CP3 CPz CP4 |
| ERD-only | 9 | P2 CP4 C3 CPz POz Pz CP2 P1 CP3 C4 CP1 Cz |

### T3. Equivalence of each 12-channel montage with 22 channels (paired TOST)

Difference = montage − 22ch, in percentage points. Equivalent means the 90% CI lies inside the margin; it does not mean identical accuracy.

| Method | n | Mean diff (pp) | SD | 90% CI | TOST p | Equivalent (±6 pp) | Worst subject (pp) | Subjects losing >6 pp | Equivalent at margins (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 9 | -0.85 | 2.95 | [-2.68, +0.98] | 0.0004 | yes | -6.25 | 1 | 3,4,5,6,8,10 |
| CSP | 9 | -1.31 | 3.25 | [-3.32, +0.70] | 0.0013 | yes | -5.56 | 0 | 4,5,6,8,10 |
| RLF | 9 | -1.31 | 2.61 | [-2.93, +0.31] | 0.0003 | yes | -4.51 | 0 | 3,4,5,6,8,10 |
| MI | 9 | -4.01 | 5.04 | [-7.14, -0.89] | 0.1353 | no | -14.93 | 3 | 8,10 |
| Fixed sensorimotor | 9 | -0.89 | 3.45 | [-3.03, +1.25] | 0.0011 | yes | -4.17 | 0 | 4,5,6,8,10 |
| ERD-only | 9 | +0.00 | 2.39 | [-1.48, +1.48] | <0.0001 | yes | -4.86 | 0 | 2,3,4,5,6,8,10 |
| Random (mean of draws) | 9 | -3.34 | 2.13 | [-4.66, -2.02] | 0.0028 | yes | -7.50 | 1 | 5,6,8,10 |

### T4. Saliency montage (Ours) vs every other 12-channel montage

Positive = Ours more accurate. Holm correction across the comparisons in this table.

| Ours vs | n | Mean diff (pp) | 95% CI | paired t p | Holm | Wilcoxon p | Holm | Ours better / tie / worse |
|---|---|---|---|---|---|---|---|---|
| CSP | 9 | +0.46 | [-2.22, +3.14] | 0.7007 | 1.0000 | 0.7344 | 1.0000 | 5 / 0 / 4 |
| RLF | 9 | +0.46 | [-2.15, +3.08] | 0.6938 | 1.0000 | 0.9414 | 1.0000 | 5 / 0 / 4 |
| MI | 9 | +3.16 | [+0.70, +5.63] | 0.0182 | 0.0909 | 0.0234 | 0.1172 | 7 / 1 / 1 |
| Fixed sensorimotor | 9 | +0.04 | [-3.05, +3.13] | 0.9777 | 1.0000 | 0.6523 | 1.0000 | 5 / 0 / 4 |
| ERD-only | 9 | -0.85 | [-2.20, +0.50] | 0.1847 | 0.7389 | 0.2500 | 1.0000 | 3 / 1 / 5 |
| Random (mean of draws) | 9 | +2.49 | [+1.30, +3.69] | 0.0014 | 0.0081 | 0.0078 | 0.0469 | 8 / 0 / 1 |

### T5. Confusion, class balance and majority baseline (chance 50.0%)

Pooled over all subjects, folds and seeds; full matrices in `T5_confusion_matrices.csv`. The last column flags models that collapsed to one class.

| Method | Pooled acc | Per-class recall | Predicted share per class | Mean majority-class acc | Folds predicting one class ≥90% |
|---|---|---|---|---|---|
| 22ch | 68.36 | left_hand 69.5, right_hand 67.2 | left_hand 51.2, right_hand 48.8 | 50.00 | 0 / 36 |
| Ours | 67.52 | left_hand 67.4, right_hand 67.7 | left_hand 49.8, right_hand 50.2 | 50.00 | 0 / 36 |
| CSP | 67.05 | left_hand 68.8, right_hand 65.4 | left_hand 51.7, right_hand 48.3 | 50.00 | 1 / 36 |
| RLF | 67.05 | left_hand 69.3, right_hand 64.8 | left_hand 52.2, right_hand 47.8 | 50.00 | 1 / 36 |
| MI | 64.35 | left_hand 66.4, right_hand 62.3 | left_hand 52.1, right_hand 47.9 | 50.00 | 2 / 36 |
| Fixed sensorimotor | 67.48 | left_hand 69.0, right_hand 66.0 | left_hand 51.5, right_hand 48.5 | 50.00 | 2 / 36 |
| ERD-only | 68.36 | left_hand 68.9, right_hand 67.8 | left_hand 50.5, right_hand 49.5 | 50.00 | 0 / 36 |
| Random (mean of draws) | 65.02 | left_hand 66.6, right_hand 63.5 | left_hand 51.6, right_hand 48.4 | 50.00 | 7 / 180 |

### T6. Seed stability

| Method | Seeds | Grand mean per seed (%) | Mean across seeds | SD across seeds |
|---|---|---|---|---|
| 22ch | 1 | 42: 68.36 | 68.36 | — |
| Ours | 1 | 42: 67.52 | 67.52 | — |
| CSP | 1 | 42: 67.05 | 67.05 | — |
| RLF | 1 | 42: 67.05 | 67.05 | — |
| MI | 1 | 42: 64.35 | 64.35 | — |
| Fixed sensorimotor | 1 | 42: 67.48 | 67.48 | — |
| ERD-only | 1 | 42: 68.36 | 68.36 | — |
| Random (mean of draws) | 1 | 42: 65.02 | 65.02 | — |

### T7. Saliency vs ERD/ERS alignment (participant level)

magnitude / class_specific / lateralisation compare unsigned saliency with desynchronisation strength (−ERD) and are sign-invariant; `legacy_signed_LR` is the old statistic, shown only for comparison (its sign is not interpretable). Mean and CI via Fisher z. Per-subject values: `results/full22/alignment_metrics.json`.

| Attribution | Metric | Mean r (Fisher) | 95% CI | SD of r | n subjects |
|---|---|---|---|---|---|
| grad | magnitude_pearson | 0.004 | [-0.115, 0.121] | 0.152 | 9 |
| grad | class_specific_pearson | 0.016 | [-0.085, 0.116] | 0.129 | 9 |
| grad | lateralisation_pearson | -0.199 | [-0.465, 0.101] | 0.358 | 9 |
| grad | magnitude_spearman | 0.029 | [-0.113, 0.170] | 0.181 | 9 |
| grad | class_specific_spearman | 0.004 | [-0.103, 0.112] | 0.138 | 9 |
| grad | lateralisation_spearman | -0.139 | [-0.387, 0.127] | 0.324 | 9 |
| grad | legacy_signed_LR_pearson | -0.086 | [-0.288, 0.124] | 0.263 | 9 |
| grad_x_input | magnitude_pearson | -0.027 | [-0.175, 0.122] | 0.190 | 9 |
| grad_x_input | class_specific_pearson | 0.007 | [-0.115, 0.130] | 0.158 | 9 |
| grad_x_input | lateralisation_pearson | -0.527 | [-0.751, -0.194] | 0.358 | 9 |
| grad_x_input | magnitude_spearman | 0.018 | [-0.143, 0.178] | 0.204 | 9 |
| grad_x_input | class_specific_spearman | -0.003 | [-0.119, 0.113] | 0.149 | 9 |
| grad_x_input | lateralisation_spearman | -0.488 | [-0.710, -0.179] | 0.330 | 9 |
| grad_x_input | legacy_signed_LR_pearson | -0.099 | [-0.304, 0.115] | 0.268 | 9 |
| integrated_gradients | magnitude_pearson | -0.049 | [-0.188, 0.092] | 0.180 | 9 |
| integrated_gradients | class_specific_pearson | -0.011 | [-0.138, 0.117] | 0.165 | 9 |
| integrated_gradients | lateralisation_pearson | -0.640 | [-0.798, -0.400] | 0.234 | 9 |
| integrated_gradients | magnitude_spearman | -0.020 | [-0.168, 0.128] | 0.188 | 9 |
| integrated_gradients | class_specific_spearman | -0.015 | [-0.138, 0.108] | 0.159 | 9 |
| integrated_gradients | lateralisation_spearman | -0.620 | [-0.797, -0.345] | 0.226 | 9 |
| integrated_gradients | legacy_signed_LR_pearson | -0.125 | [-0.302, 0.060] | 0.235 | 9 |
| smoothgrad | magnitude_pearson | 0.003 | [-0.115, 0.121] | 0.152 | 9 |
| smoothgrad | class_specific_pearson | 0.017 | [-0.083, 0.117] | 0.129 | 9 |
| smoothgrad | lateralisation_pearson | -0.204 | [-0.464, 0.087] | 0.348 | 9 |
| smoothgrad | magnitude_spearman | 0.027 | [-0.111, 0.164] | 0.177 | 9 |
| smoothgrad | class_specific_spearman | 0.004 | [-0.106, 0.115] | 0.142 | 9 |
| smoothgrad | lateralisation_spearman | -0.149 | [-0.387, 0.107] | 0.312 | 9 |
| smoothgrad | legacy_signed_LR_pearson | -0.086 | [-0.289, 0.125] | 0.263 | 9 |

### T7b. Correlations as computed by the original scripts

corr(saliency, ERD_L / ERD_R / ERD_(L−R) / ERD_comb) and corr(saliency, plain-gradient saliency), participant level. Kept so the earlier numbers can be compared; an unsigned map is correlated with a signed one, so the sign is not 'agreement' (use T7).

| Attribution | Correlation | Mean r (Fisher) | 95% CI | n subjects |
|---|---|---|---|---|
| grad | ERD_L_pearson | -0.055 | [-0.247, 0.141] | 9 |
| grad | ERD_L_spearman | -0.028 | [-0.232, 0.178] | 9 |
| grad | ERD_R_pearson | 0.011 | [-0.081, 0.102] | 9 |
| grad | ERD_R_spearman | 0.008 | [-0.124, 0.140] | 9 |
| grad | ERD_LR_pearson | -0.086 | [-0.288, 0.124] | 9 |
| grad | ERD_LR_spearman | -0.061 | [-0.244, 0.127] | 9 |
| grad | ERD_comb_pearson | -0.033 | [-0.154, 0.089] | 9 |
| grad | ERD_comb_spearman | -0.029 | [-0.174, 0.116] | 9 |
| grad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| grad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |
| grad_x_input | ERD_L_pearson | -0.071 | [-0.275, 0.140] | 9 |
| grad_x_input | ERD_L_spearman | -0.037 | [-0.248, 0.178] | 9 |
| grad_x_input | ERD_R_pearson | 0.017 | [-0.093, 0.127] | 9 |
| grad_x_input | ERD_R_spearman | 0.014 | [-0.126, 0.153] | 9 |
| grad_x_input | ERD_LR_pearson | -0.099 | [-0.304, 0.115] | 9 |
| grad_x_input | ERD_LR_spearman | -0.062 | [-0.258, 0.139] | 9 |
| grad_x_input | ERD_comb_pearson | -0.035 | [-0.180, 0.112] | 9 |
| grad_x_input | ERD_comb_spearman | -0.031 | [-0.183, 0.123] | 9 |
| grad_x_input | vs_grad_pearson | 0.997 | [0.994, 0.998] | 9 |
| grad_x_input | vs_grad_spearman | 0.993 | [0.988, 0.996] | 9 |
| integrated_gradients | ERD_L_pearson | -0.067 | [-0.261, 0.131] | 9 |
| integrated_gradients | ERD_L_spearman | -0.033 | [-0.227, 0.164] | 9 |
| integrated_gradients | ERD_R_pearson | 0.047 | [-0.055, 0.148] | 9 |
| integrated_gradients | ERD_R_spearman | 0.046 | [-0.102, 0.191] | 9 |
| integrated_gradients | ERD_LR_pearson | -0.125 | [-0.302, 0.060] | 9 |
| integrated_gradients | ERD_LR_spearman | -0.100 | [-0.275, 0.082] | 9 |
| integrated_gradients | ERD_comb_pearson | -0.016 | [-0.166, 0.135] | 9 |
| integrated_gradients | ERD_comb_spearman | -0.020 | [-0.177, 0.138] | 9 |
| integrated_gradients | vs_grad_pearson | 0.985 | [0.968, 0.993] | 9 |
| integrated_gradients | vs_grad_spearman | 0.970 | [0.951, 0.981] | 9 |
| smoothgrad | ERD_L_pearson | -0.056 | [-0.249, 0.140] | 9 |
| smoothgrad | ERD_L_spearman | -0.027 | [-0.232, 0.180] | 9 |
| smoothgrad | ERD_R_pearson | 0.009 | [-0.082, 0.099] | 9 |
| smoothgrad | ERD_R_spearman | 0.007 | [-0.124, 0.138] | 9 |
| smoothgrad | ERD_LR_pearson | -0.086 | [-0.289, 0.125] | 9 |
| smoothgrad | ERD_LR_spearman | -0.058 | [-0.241, 0.129] | 9 |
| smoothgrad | ERD_comb_pearson | -0.035 | [-0.155, 0.087] | 9 |
| smoothgrad | ERD_comb_spearman | -0.029 | [-0.172, 0.116] | 9 |
| smoothgrad | vs_grad_pearson | 1.000 | [1.000, 1.000] | 9 |
| smoothgrad | vs_grad_spearman | 1.000 | [1.000, 1.000] | 9 |

### T10. EXPLORATORY: saliency vs other physiological maps

Curiosity only — not used for any choice; the paper's alignment is T7. Saliency = mean of the four attribution maps (`mean4`); per method in `T10_physiology_all_methods.csv`. r = corr(saliency, map), |map| = corr(saliency, |map|), class-specific = mean of corr(saliency_L, |map_L|) and corr(saliency_R, |map_R|). Fisher-z mean over subjects. Group maps: `figures/physiology_maps_group.png`.

| Map | Pearson r | 95% CI | Pearson r (|map|) | Spearman r | Spearman r (|map|) | n |
|---|---|---|---|---|---|---|
| ERD_theta_L | -0.062 | [-0.255, 0.136] | -0.094 | -0.062 | -0.084 | 9 |
| ERD_theta_R | -0.065 | [-0.254, 0.129] | 0.038 | -0.062 | 0.005 | 9 |
| ERD_theta_mag | 0.007 | [-0.205, 0.219] | 0.007 | -0.009 | -0.009 | 9 |
| ERD_mu_L | -0.077 | [-0.269, 0.121] | -0.064 | -0.059 | -0.067 | 9 |
| ERD_mu_R | -0.031 | [-0.253, 0.193] | 0.095 | -0.022 | 0.078 | 9 |
| ERD_mu_mag | 0.017 | [-0.082, 0.116] | 0.017 | -0.010 | -0.010 | 9 |
| ERD_beta_L | 0.030 | [-0.183, 0.240] | 0.101 | 0.018 | 0.095 | 9 |
| ERD_beta_R | 0.057 | [-0.104, 0.215] | 0.048 | 0.008 | 0.037 | 9 |
| ERD_beta_mag | 0.094 | [-0.114, 0.294] | 0.094 | 0.078 | 0.078 | 9 |
| ERS_beta_rebound_L | -0.005 | [-0.120, 0.109] | -0.023 | -0.017 | -0.015 | 9 |
| ERS_beta_rebound_R | -0.024 | [-0.219, 0.174] | 0.036 | -0.040 | -0.014 | 9 |
| ERS_beta_rebound_mag | -0.009 | [-0.149, 0.130] | -0.009 | -0.061 | -0.061 | 9 |
| ERD_lowgamma_L | 0.061 | [-0.154, 0.271] | 0.049 | 0.032 | 0.014 | 9 |
| ERD_lowgamma_R | -0.102 | [-0.327, 0.133] | -0.166 | -0.094 | -0.152 | 9 |
| ERD_lowgamma_mag | -0.061 | [-0.292, 0.178] | -0.061 | -0.086 | -0.086 | 9 |
| logpow_8_30 | -0.049 | [-0.204, 0.109] | -0.049 | -0.044 | -0.044 | 9 |
| r2_signed | 0.092 | [-0.142, 0.316] | 0.192 | 0.036 | 0.121 | 9 |
| r2_abs | 0.192 | [0.005, 0.365] | 0.192 | 0.121 | 0.121 | 9 |
| TFR_ERD_dB_L | -0.046 | [-0.248, 0.160] | 0.001 | -0.020 | 0.024 | 9 |
| TFR_ERD_dB_R | 0.049 | [-0.088, 0.184] | 0.101 | 0.010 | 0.140 | 9 |
| TFR_ERD_dB_mag | 0.070 | [-0.045, 0.183] | 0.070 | 0.120 | 0.120 | 9 |
| ITC_4_30 | 0.047 | [-0.115, 0.207] | 0.047 | 0.054 | 0.054 | 9 |
| ERP_abs_0.2_0.8 | 0.079 | [-0.088, 0.243] | 0.079 | 0.048 | 0.048 | 9 |
| SlowPot_0.5_3 | -0.050 | [-0.215, 0.118] | 0.037 | -0.043 | 0.008 | 9 |
| ERD_beta_classspecific | — | — | 0.069 [-0.094, 0.228] | — | 0.062 | 9 |
| ERD_lowgamma_classspecific | — | — | -0.057 [-0.209, 0.098] | — | -0.074 | 9 |
| ERD_mu_classspecific | — | — | 0.004 [-0.074, 0.081] | — | -0.002 | 9 |
| ERD_theta_classspecific | — | — | -0.035 [-0.209, 0.140] | — | -0.045 | 9 |
| ERS_beta_rebound_classspecific | — | — | 0.006 [-0.104, 0.116] | — | -0.023 | 9 |
| TFR_ERD_dB_classspecific | — | — | 0.041 [-0.049, 0.130] | — | 0.071 | 9 |

### T8a. Faithfulness: deletion test on the test block

Accuracy drop (intact − channels removed), mean ± SD across subjects. Ranking from validation saliency; removal = channel set to its standardised mean. A faithful ranking gives the largest drop for top-ranked channels.

| Removed channels k | Drop: top-ranked (pp) | Drop: random (pp) | Drop: bottom-ranked (pp) | top vs random p | top vs bottom p |
|---|---|---|---|---|---|
| 2 | +6.83 ± 3.57 | +3.26 ± 1.88 | +1.16 ± 1.25 | 0.0040 | 0.0021 |
| 4 | +9.76 ± 5.93 | +5.15 ± 3.51 | +1.93 ± 1.68 | 0.0027 | 0.0046 |
| 6 | +10.84 ± 5.89 | +7.59 ± 4.20 | +2.16 ± 4.53 | 0.0237 | 0.0029 |

### T8b. Model-randomisation sanity check

Similarity between saliency of the trained model and of an untrained model. High values mean the map reflects the input more than what the network learned.

| Attribution | Spearman(trained, untrained model) | 95% CI | n subjects |
|---|---|---|---|
| grad | -0.028 | [-0.087, 0.032] | 9 |
| grad_x_input | -0.040 | [-0.094, 0.015] | 9 |
| integrated_gradients | -0.029 | [-0.087, 0.029] | 9 |
| smoothgrad | -0.026 | [-0.083, 0.032] | 9 |

### T9. Configuration

Full snapshot: `T9_config.json`.

Training profile chosen on validation data only, leave-one-subject-out (unanimous: False): S01=cue_0_4, S02=cue_0_4, S03=cue_0_4, S04=cue_0_4, S05=cue_0_4, S06=cue_0_4, S07=cue_0_4, S08=cue_0_4, S09=no_aug

| Profile | Mean validation accuracy (%) |
|---|---|
| original | 67.98 |
| cue_0_4 | 68.94 |
| no_aug | 68.13 |
| drop_0.25 | 66.05 |

```
model: "EEGITNet"
dataset: "BNCI2014_001"
sfreq_hz: 250
ems: {"factor_new": 0.001, "init_block_size": 1000}
profiles: {"original": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "cue_0_4": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [0.0, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "no_aug": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.4, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": false, "mixup_p": 0.0, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}, "drop_0.25": {"band_hz": [4.0, 38.0], "iir": {"order": 6, "ftype": "cheby1", "rp": 0.5}, "window_s": [-0.5, 4.0], "dropout": 0.25, "lr": 0.001, "weight_decay": 0.001, "label_smoothing": 0.1, "batch_size": 64, "max_epochs": 350, "swa_start_frac": 0.7142857142857143, "augment": true, "mixup_p": 0.5, "mixup_alpha": 0.2, "tta": true, "grad_clip": 10.0, "patience": null}}
checkpoint_rule: "SWA average from swa_start_frac*max_epochs + BN update (no early stopping); validation block only scored, never used for checkpointing"
folds: "4 chronological blocks; test block i, validation block i-1, rest train (original: shuffled StratifiedKFold(4), no validation set)"
seeds: [42]
```

## 4class

_No results found in `/home/user/NeuroGrounded-EEG/EEGItNet/BNCI_001/4class/results`._
