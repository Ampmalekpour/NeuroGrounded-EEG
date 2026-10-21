# Cross-model summary — EEGNetV4\HGD, EEGItNet\HGD, Deep4net\HGD, EEGConformer\HGD

Generated 2026-10-21 20:28. Per-model details: `<model folder>/report*/REPORT.md` (results folder `<class>/results_session/`).

## C1. Test accuracy (%), mean ± SD across subjects

| Model | Class | Full montage (22ch BNCI/HGD, 64ch PhysioNet) | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | 83.04 ± 15.46 | 81.07 ± 14.43 | 83.93 ± 14.12 | 84.02 ± 12.89 | 80.36 ± 17.06 | 80.71 ± 15.95 | 83.84 ± 15.38 | 79.59 ± 16.43 |
| EEGNetV4 | 4class | 87.05 ± 6.79 | 85.36 ± 7.00 | 86.38 ± 7.89 | 85.62 ± 6.86 | 86.29 ± 7.22 | 84.51 ± 7.99 | 86.03 ± 5.94 | 84.13 ± 7.43 |
| Deep4net | 2class | 75.63 ± 20.45 | 76.52 ± 17.79 | 75.62 ± 20.06 | 75.80 ± 20.14 | 75.71 ± 18.53 | 78.39 ± 18.39 | 76.52 ± 19.52 | 74.07 ± 19.45 |
| Deep4net | 4class | 80.67 ± 11.52 | 80.36 ± 11.12 | 79.24 ± 11.12 | 80.18 ± 10.88 | 80.04 ± 11.49 | 78.39 ± 11.88 | 78.30 ± 10.78 | 79.24 ± 11.12 |

## C2. Saliency–ERD correlations (Pearson, Fisher mean over subjects)

First three: sign-invariant metrics used in the paper. Last two: as in the original scripts.

| Model | Class | Attribution | magnitude r | class-specific r | lateralisation r | orig ERD_(L−R) r | orig ERD_comb r |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | grad | 0.293 | 0.277 | -0.192 | 0.183 | -0.296 |
| EEGNetV4 | 2class | grad_x_input | 0.136 | 0.076 | -0.368 | 0.129 | -0.092 |
| EEGNetV4 | 2class | integrated_gradients | 0.129 | 0.050 | -0.335 | 0.116 | -0.070 |
| EEGNetV4 | 2class | smoothgrad | 0.293 | 0.277 | -0.192 | 0.183 | -0.296 |
| EEGNetV4 | 4class | grad | 0.451 | 0.426 | -0.030 | 0.196 | -0.471 |
| EEGNetV4 | 4class | grad_x_input | 0.419 | 0.297 | -0.122 | 0.182 | -0.423 |
| EEGNetV4 | 4class | integrated_gradients | 0.389 | 0.263 | -0.109 | 0.182 | -0.402 |
| EEGNetV4 | 4class | smoothgrad | 0.452 | 0.428 | -0.025 | 0.196 | -0.472 |
| Deep4net | 2class | grad | 0.307 | 0.351 | -0.081 | 0.100 | -0.360 |
| Deep4net | 2class | grad_x_input | 0.222 | 0.243 | -0.277 | 0.055 | -0.252 |
| Deep4net | 2class | integrated_gradients | 0.224 | 0.242 | -0.375 | 0.034 | -0.254 |
| Deep4net | 2class | smoothgrad | 0.307 | 0.351 | -0.083 | 0.100 | -0.360 |
| Deep4net | 4class | grad | 0.412 | 0.416 | 0.002 | 0.111 | -0.422 |
| Deep4net | 4class | grad_x_input | 0.390 | 0.350 | -0.203 | 0.096 | -0.393 |
| Deep4net | 4class | integrated_gradients | 0.411 | 0.345 | -0.316 | 0.083 | -0.400 |
| Deep4net | 4class | smoothgrad | 0.412 | 0.416 | 0.005 | 0.110 | -0.422 |

## C3. EXPLORATORY: corr(mean saliency, physiological map), Pearson, Fisher mean

Curiosity only; not used anywhere in the pipeline.

| Map | Deep4net 2class | Deep4net 4class | EEGNetV4 2class | EEGNetV4 4class |
|---|---|---|---|---|
| ERD_theta_L | -0.081 | -0.092 | 0.008 | -0.095 |
| ERD_theta_R | -0.168 | -0.099 | -0.011 | -0.112 |
| ERD_theta_mag | 0.178 | 0.038 | 0.109 | 0.165 |
| ERD_mu_L | -0.257 | -0.368 | -0.096 | -0.334 |
| ERD_mu_R | -0.250 | -0.372 | -0.169 | -0.369 |
| ERD_mu_mag | 0.155 | 0.276 | 0.122 | 0.292 |
| ERD_beta_L | -0.313 | -0.445 | -0.282 | -0.515 |
| ERD_beta_R | -0.310 | -0.436 | -0.318 | -0.524 |
| ERD_beta_mag | 0.302 | 0.445 | 0.276 | 0.516 |
| ERS_beta_rebound_L | 0.000 | -0.158 | -0.036 | -0.290 |
| ERS_beta_rebound_R | -0.329 | -0.194 | -0.165 | -0.291 |
| ERS_beta_rebound_mag | 0.272 | 0.298 | 0.228 | 0.352 |
| ERD_lowgamma_L | -0.210 | -0.320 | -0.284 | -0.495 |
| ERD_lowgamma_R | -0.283 | -0.372 | -0.341 | -0.498 |
| ERD_lowgamma_mag | 0.230 | 0.271 | 0.208 | 0.406 |
| ERD_highgamma_L | 0.087 | 0.058 | -0.044 | 0.001 |
| ERD_highgamma_R | 0.133 | 0.191 | 0.076 | 0.131 |
| ERD_highgamma_mag | 0.139 | 0.108 | -0.045 | 0.124 |
| logpow_8_30 | 0.111 | 0.056 | 0.157 | 0.140 |
| r2_signed | 0.218 | -0.057 | 0.020 | -0.071 |
| r2_abs | 0.336 | 0.138 | 0.220 | 0.048 |
| TFR_ERD_dB_L | -0.336 | -0.494 | -0.238 | -0.517 |
| TFR_ERD_dB_R | -0.322 | -0.493 | -0.277 | -0.533 |
| TFR_ERD_dB_mag | 0.306 | 0.495 | 0.239 | 0.506 |
| ITC_4_30 | 0.098 | 0.064 | 0.129 | 0.219 |
| ERP_abs_0.2_0.8 | 0.020 | -0.121 | -0.008 | -0.027 |
| SlowPot_0.5_3 | -0.051 | -0.081 | -0.084 | -0.002 |
