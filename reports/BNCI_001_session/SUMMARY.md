# Cross-model summary — EEGNetV4\BNCI_001, EEGItNet\BNCI_001, Deep4net\BNCI_001, EEGConformer\BNCI_001

Generated 2026-10-16 09:59. Per-model details: `<model folder>/report*/REPORT.md` (results folder `<class>/results_session/`).

## C1. Test accuracy (%), mean ± SD across subjects

| Model | Class | 22ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|---|
| Deep4net | 4class | 45.14 ± 19.57 | 45.33 ± 18.51 | 44.37 ± 17.97 | 44.98 ± 18.32 | 44.75 ± 18.43 | 44.25 ± 20.00 | 44.91 ± 19.09 | 43.78 ± 17.68 |
| EEGConformer | 4class | 64.85 ± 15.81 | 60.46 ± 15.80 | 58.83 ± 16.08 | 58.41 ± 17.18 | 60.57 ± 16.77 | 58.95 ± 17.39 | 59.30 ± 15.37 | 58.06 ± 16.47 |

## C2. Saliency–ERD correlations (Pearson, Fisher mean over subjects)

First three: sign-invariant metrics used in the paper. Last two: as in the original scripts.

| Model | Class | Attribution | magnitude r | class-specific r | lateralisation r | orig ERD_(L−R) r | orig ERD_comb r |
|---|---|---|---|---|---|---|---|
| Deep4net | 4class | grad | 0.308 | 0.111 | 0.447 | 0.053 | -0.112 |
| Deep4net | 4class | grad_x_input | 0.293 | 0.108 | 0.262 | 0.041 | -0.124 |
| Deep4net | 4class | integrated_gradients | 0.292 | 0.090 | 0.018 | 0.052 | -0.113 |
| Deep4net | 4class | smoothgrad | 0.308 | 0.110 | 0.446 | 0.052 | -0.112 |
| EEGConformer | 4class | grad | 0.001 | 0.153 | 0.061 | -0.144 | -0.131 |
| EEGConformer | 4class | grad_x_input | -0.031 | 0.154 | 0.056 | -0.140 | -0.142 |
| EEGConformer | 4class | integrated_gradients | -0.089 | 0.127 | 0.061 | -0.017 | -0.133 |
| EEGConformer | 4class | smoothgrad | -0.000 | 0.149 | 0.069 | -0.144 | -0.127 |

## C3. EXPLORATORY: corr(mean saliency, physiological map), Pearson, Fisher mean

Curiosity only; not used anywhere in the pipeline.

| Map | Deep4net 4class | EEGConformer 4class |
|---|---|---|
| ERD_theta_L | -0.074 | -0.013 |
| ERD_theta_R | -0.333 | -0.015 |
| ERD_theta_mag | 0.105 | -0.075 |
| ERD_mu_L | -0.137 | -0.196 |
| ERD_mu_R | -0.247 | -0.106 |
| ERD_mu_mag | 0.427 | -0.078 |
| ERD_beta_L | 0.027 | -0.106 |
| ERD_beta_R | 0.057 | 0.012 |
| ERD_beta_mag | 0.109 | 0.015 |
| ERS_beta_rebound_L | -0.171 | -0.003 |
| ERS_beta_rebound_R | 0.006 | -0.095 |
| ERS_beta_rebound_mag | -0.046 | -0.005 |
| ERD_lowgamma_L | -0.089 | -0.042 |
| ERD_lowgamma_R | -0.003 | -0.070 |
| ERD_lowgamma_mag | -0.065 | 0.075 |
| logpow_8_30 | 0.128 | -0.152 |
| r2_signed | 0.002 | 0.145 |
| r2_abs | 0.408 | 0.146 |
| TFR_ERD_dB_L | -0.063 | -0.207 |
| TFR_ERD_dB_R | -0.149 | -0.042 |
| TFR_ERD_dB_mag | 0.233 | 0.119 |
| ITC_4_30 | 0.112 | -0.025 |
| ERP_abs_0.2_0.8 | 0.256 | -0.070 |
| SlowPot_0.5_3 | -0.218 | 0.021 |
