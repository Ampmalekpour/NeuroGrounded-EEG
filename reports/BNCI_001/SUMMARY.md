# Cross-model summary — EEGNetV4/BNCI_001, EEGItNet/BNCI_001

Generated 2026-10-14 04:37. Per-model details: `<model folder>/report/REPORT.md`.

## C1. Test accuracy (%), mean ± SD across subjects

| Model | Class | 22ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | 72.88 ± 17.55 | 69.44 ± 18.15 | 70.95 ± 17.40 | 71.18 ± 17.19 | 70.02 ± 17.07 | 69.06 ± 18.42 | 69.60 ± 17.28 | 68.44 ± 17.48 |
| EEGNetV4 | 4class | 58.97 ± 24.61 | 59.65 ± 22.06 | 58.66 ± 21.04 | 55.23 ± 21.63 | 57.66 ± 22.13 | 60.40 ± 21.74 | 60.51 ± 20.79 | 57.97 ± 20.54 |
| EEGItNet | 2class | 68.36 ± 8.25 | 67.52 ± 7.02 | 67.05 ± 9.49 | 67.05 ± 9.44 | 64.35 ± 8.82 | 67.48 ± 8.55 | 68.36 ± 6.72 | 65.02 ± 7.62 |

## C2. Saliency–ERD correlations (Pearson, Fisher mean over subjects)

First three: sign-invariant metrics used in the paper. Last two: as in the original scripts.

| Model | Class | Attribution | magnitude r | class-specific r | lateralisation r | orig ERD_(L−R) r | orig ERD_comb r |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | grad | 0.100 | 0.170 | -0.184 | -0.079 | -0.194 |
| EEGNetV4 | 2class | grad_x_input | 0.075 | 0.156 | -0.382 | -0.090 | -0.194 |
| EEGNetV4 | 2class | integrated_gradients | 0.056 | 0.136 | -0.353 | -0.083 | -0.169 |
| EEGNetV4 | 2class | smoothgrad | 0.100 | 0.172 | -0.181 | -0.078 | -0.196 |
| EEGNetV4 | 4class | grad | 0.109 | 0.354 | 0.168 | -0.068 | -0.422 |
| EEGNetV4 | 4class | grad_x_input | 0.086 | 0.350 | 0.112 | -0.084 | -0.439 |
| EEGNetV4 | 4class | integrated_gradients | 0.095 | 0.328 | 0.047 | -0.072 | -0.424 |
| EEGNetV4 | 4class | smoothgrad | 0.108 | 0.355 | 0.169 | -0.066 | -0.424 |
| EEGItNet | 2class | grad | 0.004 | 0.016 | -0.199 | -0.086 | -0.033 |
| EEGItNet | 2class | grad_x_input | -0.027 | 0.007 | -0.527 | -0.099 | -0.035 |
| EEGItNet | 2class | integrated_gradients | -0.049 | -0.011 | -0.640 | -0.125 | -0.016 |
| EEGItNet | 2class | smoothgrad | 0.003 | 0.017 | -0.204 | -0.086 | -0.035 |

## C3. EXPLORATORY: corr(mean saliency, physiological map), Pearson, Fisher mean

Curiosity only; not used anywhere in the pipeline.

| Map | EEGItNet 2class | EEGNetV4 2class | EEGNetV4 4class |
|---|---|---|---|
| ERD_theta_L | -0.062 | -0.059 | -0.207 |
| ERD_theta_R | -0.065 | -0.005 | -0.202 |
| ERD_theta_mag | 0.007 | -0.011 | 0.102 |
| ERD_mu_L | -0.077 | -0.171 | -0.307 |
| ERD_mu_R | -0.031 | -0.079 | -0.311 |
| ERD_mu_mag | 0.017 | 0.117 | 0.043 |
| ERD_beta_L | 0.030 | -0.127 | -0.320 |
| ERD_beta_R | 0.057 | -0.058 | -0.205 |
| ERD_beta_mag | 0.094 | 0.034 | -0.004 |
| ERS_beta_rebound_L | -0.005 | -0.010 | -0.159 |
| ERS_beta_rebound_R | -0.024 | -0.013 | -0.097 |
| ERS_beta_rebound_mag | -0.009 | 0.012 | -0.180 |
| ERD_lowgamma_L | 0.061 | -0.075 | -0.164 |
| ERD_lowgamma_R | -0.102 | -0.200 | -0.290 |
| ERD_lowgamma_mag | -0.061 | -0.093 | -0.095 |
| logpow_8_30 | -0.049 | -0.127 | 0.004 |
| r2_signed | 0.092 | 0.138 | 0.097 |
| r2_abs | 0.192 | 0.348 | 0.102 |
| TFR_ERD_dB_L | -0.046 | -0.210 | -0.392 |
| TFR_ERD_dB_R | 0.049 | -0.138 | -0.316 |
| TFR_ERD_dB_mag | 0.070 | 0.106 | 0.161 |
| ITC_4_30 | 0.047 | -0.017 | 0.019 |
| ERP_abs_0.2_0.8 | 0.079 | -0.049 | 0.047 |
| SlowPot_0.5_3 | -0.050 | 0.055 | 0.085 |
