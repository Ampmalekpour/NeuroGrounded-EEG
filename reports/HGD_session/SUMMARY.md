# Cross-model summary — EEGNetV4\HGD, EEGItNet\HGD, Deep4net\HGD, EEGConformer\HGD

Generated 2026-10-19 19:33. Per-model details: `<model folder>/report*/REPORT.md` (results folder `<class>/results_session/`).

## C1. Test accuracy (%), mean ± SD across subjects

| Model | Class | 22ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | 81.07 ± 16.34 | 81.30 ± 15.57 | 80.42 ± 18.83 | 76.66 ± 17.74 | 76.08 ± 19.93 | 77.50 ± 18.98 | 79.60 ± 19.28 | 77.12 ± 18.82 |
| EEGNetV4 | 4class | 83.94 ± 12.26 | 83.84 ± 11.31 | 85.57 ± 10.21 | 83.94 ± 11.24 | 83.40 ± 9.46 | 81.03 ± 15.43 | 84.46 ± 10.40 | 82.47 ± 10.86 |

## C2. Saliency–ERD correlations (Pearson, Fisher mean over subjects)

First three: sign-invariant metrics used in the paper. Last two: as in the original scripts.

| Model | Class | Attribution | magnitude r | class-specific r | lateralisation r | orig ERD_(L−R) r | orig ERD_comb r |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | grad | 0.282 | 0.235 | -0.009 | 0.097 | -0.247 |
| EEGNetV4 | 2class | grad_x_input | 0.095 | -0.000 | -0.235 | 0.047 | -0.005 |
| EEGNetV4 | 2class | integrated_gradients | 0.087 | -0.023 | -0.210 | 0.025 | 0.016 |
| EEGNetV4 | 2class | smoothgrad | 0.281 | 0.235 | -0.008 | 0.097 | -0.247 |
| EEGNetV4 | 4class | grad | 0.401 | 0.380 | -0.083 | 0.150 | -0.417 |
| EEGNetV4 | 4class | grad_x_input | 0.365 | 0.241 | -0.189 | 0.138 | -0.361 |
| EEGNetV4 | 4class | integrated_gradients | 0.344 | 0.220 | -0.174 | 0.135 | -0.346 |
| EEGNetV4 | 4class | smoothgrad | 0.402 | 0.381 | -0.083 | 0.150 | -0.418 |

## C3. EXPLORATORY: corr(mean saliency, physiological map), Pearson, Fisher mean

Curiosity only; not used anywhere in the pipeline.

| Map | EEGNetV4 2class | EEGNetV4 4class |
|---|---|---|
| ERD_theta_L | 0.003 | -0.100 |
| ERD_theta_R | 0.028 | -0.095 |
| ERD_theta_mag | 0.084 | 0.145 |
| ERD_mu_L | -0.064 | -0.300 |
| ERD_mu_R | -0.092 | -0.333 |
| ERD_mu_mag | 0.105 | 0.279 |
| ERD_beta_L | -0.235 | -0.453 |
| ERD_beta_R | -0.251 | -0.469 |
| ERD_beta_mag | 0.240 | 0.460 |
| ERS_beta_rebound_L | 0.013 | -0.274 |
| ERS_beta_rebound_R | -0.073 | -0.293 |
| ERS_beta_rebound_mag | 0.193 | 0.339 |
| ERD_lowgamma_L | -0.250 | -0.408 |
| ERD_lowgamma_R | -0.298 | -0.429 |
| ERD_lowgamma_mag | 0.159 | 0.340 |
| ERD_highgamma_L | 0.003 | 0.018 |
| ERD_highgamma_R | 0.075 | 0.153 |
| ERD_highgamma_mag | -0.073 | 0.128 |
| logpow_8_30 | 0.130 | 0.181 |
| r2_signed | 0.019 | -0.028 |
| r2_abs | 0.234 | 0.042 |
| TFR_ERD_dB_L | -0.201 | -0.452 |
| TFR_ERD_dB_R | -0.219 | -0.477 |
| TFR_ERD_dB_mag | 0.210 | 0.449 |
| ITC_4_30 | 0.081 | 0.216 |
| ERP_abs_0.2_0.8 | -0.024 | -0.045 |
| SlowPot_0.5_3 | -0.041 | 0.031 |
