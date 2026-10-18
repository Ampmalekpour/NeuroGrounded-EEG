# Cross-model summary — EEGNetV4\HGD, EEGItNet\HGD, Deep4net\HGD, EEGConformer\HGD

Generated 2026-10-18 01:23. Per-model details: `<model folder>/report*/REPORT.md` (results folder `<class>/results_session/`).

## C1. Test accuracy (%), mean ± SD across subjects

| Model | Class | 22ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | 84.89 ± 17.69 | 81.47 ± 17.13 | 80.43 ± 16.95 | 85.29 ± 16.57 | 83.31 ± 14.74 | 80.13 ± 19.91 | 84.56 ± 14.63 | 84.45 ± 14.37 |
| EEGNetV4 | 4class | 91.39 ± 8.62 | 92.62 ± 8.68 | 92.82 ± 7.64 | 90.82 ± 9.59 | 91.83 ± 6.50 | 90.39 ± 9.32 | 91.67 ± 8.93 | 90.32 ± 8.14 |

## C2. Saliency–ERD correlations (Pearson, Fisher mean over subjects)

First three: sign-invariant metrics used in the paper. Last two: as in the original scripts.

| Model | Class | Attribution | magnitude r | class-specific r | lateralisation r | orig ERD_(L−R) r | orig ERD_comb r |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | grad | 0.049 | 0.051 | -0.073 | 0.040 | -0.049 |
| EEGNetV4 | 2class | grad_x_input | 0.034 | 0.038 | -0.249 | 0.136 | -0.052 |
| EEGNetV4 | 2class | integrated_gradients | 0.025 | 0.035 | -0.208 | 0.140 | -0.051 |
| EEGNetV4 | 2class | smoothgrad | 0.049 | 0.051 | -0.072 | 0.040 | -0.049 |
| EEGNetV4 | 4class | grad | 0.390 | 0.401 | -0.118 | 0.191 | -0.445 |
| EEGNetV4 | 4class | grad_x_input | 0.354 | 0.287 | -0.203 | 0.180 | -0.403 |
| EEGNetV4 | 4class | integrated_gradients | 0.320 | 0.252 | -0.199 | 0.179 | -0.382 |
| EEGNetV4 | 4class | smoothgrad | 0.391 | 0.403 | -0.111 | 0.190 | -0.445 |

## C3. EXPLORATORY: corr(mean saliency, physiological map), Pearson, Fisher mean

Curiosity only; not used anywhere in the pipeline.

| Map | EEGNetV4 2class | EEGNetV4 4class |
|---|---|---|
| ERD_theta_L | -0.026 | -0.058 |
| ERD_theta_R | 0.093 | -0.095 |
| ERD_theta_mag | 0.070 | 0.152 |
| ERD_mu_L | -0.025 | -0.320 |
| ERD_mu_R | -0.061 | -0.372 |
| ERD_mu_mag | 0.039 | 0.237 |
| ERD_beta_L | -0.194 | -0.469 |
| ERD_beta_R | -0.181 | -0.487 |
| ERD_beta_mag | 0.159 | 0.463 |
| ERS_beta_rebound_L | 0.030 | -0.251 |
| ERS_beta_rebound_R | -0.022 | -0.273 |
| ERS_beta_rebound_mag | 0.091 | 0.317 |
| ERD_lowgamma_L | -0.159 | -0.416 |
| ERD_lowgamma_R | -0.174 | -0.451 |
| ERD_lowgamma_mag | 0.117 | 0.307 |
| ERD_highgamma_L | 0.069 | 0.030 |
| ERD_highgamma_R | 0.107 | 0.164 |
| ERD_highgamma_mag | 0.007 | 0.137 |
| logpow_8_30 | 0.343 | 0.171 |
| r2_signed | -0.088 | -0.025 |
| r2_abs | 0.079 | -0.043 |
| TFR_ERD_dB_L | -0.136 | -0.486 |
| TFR_ERD_dB_R | -0.161 | -0.518 |
| TFR_ERD_dB_mag | 0.069 | 0.459 |
| ITC_4_30 | 0.229 | 0.229 |
| ERP_abs_0.2_0.8 | 0.083 | -0.026 |
| SlowPot_0.5_3 | -0.220 | -0.005 |
