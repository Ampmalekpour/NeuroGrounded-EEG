# Cross-model summary — EEGNetV4/BNCI_001, EEGItNet/BNCI_001, Deep4net/BNCI_001, EEGConformer/BNCI_001

Generated 2026-10-17 10:50. Per-model details: `<model folder>/report*/REPORT.md` (results folder `<class>/results_session/`).

## C1. Test accuracy (%), mean ± SD across subjects

| Model | Class | 22ch | Ours | CSP | RLF | MI | Fixed sensorimotor | ERD-only | Random (mean of draws) |
|---|---|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | 67.13 ± 19.34 | 66.28 ± 18.29 | 69.98 ± 19.41 | 66.82 ± 18.59 | 66.20 ± 18.65 | 71.45 ± 18.43 | 64.58 ± 19.36 | 67.16 ± 17.40 |
| EEGNetV4 | 4class | 62.73 ± 20.64 | 60.88 ± 19.28 | 57.68 ± 22.23 | 57.64 ± 20.88 | 59.84 ± 19.10 | 61.88 ± 19.74 | 60.26 ± 20.24 | 58.90 ± 21.00 |
| EEGItNet | 2class | 71.06 ± 9.92 | 71.60 ± 10.48 | 69.06 ± 11.21 | 68.06 ± 12.30 | 66.67 ± 8.20 | 67.82 ± 11.06 | 70.91 ± 10.08 | 65.93 ± 9.21 |
| EEGItNet | 4class | 50.96 ± 11.79 | 50.58 ± 10.73 | 50.04 ± 10.44 | 49.31 ± 11.98 | 50.08 ± 10.47 | 48.19 ± 12.01 | 49.81 ± 10.91 | 48.06 ± 10.21 |
| Deep4net | 2class | 65.59 ± 16.67 | 65.28 ± 15.60 | 66.44 ± 16.13 | 63.66 ± 15.78 | 63.27 ± 14.32 | 65.35 ± 16.69 | 65.20 ± 15.18 | 64.43 ± 14.40 |
| Deep4net | 4class | 45.14 ± 19.57 | 45.33 ± 18.51 | 44.37 ± 17.97 | 44.98 ± 18.32 | 44.75 ± 18.43 | 44.25 ± 20.00 | 44.91 ± 19.09 | 43.78 ± 17.68 |
| EEGConformer | 2class | 78.70 ± 13.70 | 76.39 ± 14.82 | 76.93 ± 15.53 | 75.85 ± 13.99 | 74.92 ± 14.85 | 76.93 ± 16.17 | 76.93 ± 14.39 | 76.05 ± 14.77 |
| EEGConformer | 4class | 64.85 ± 15.81 | 60.46 ± 15.80 | 58.83 ± 16.08 | 58.41 ± 17.18 | 60.57 ± 16.77 | 58.95 ± 17.39 | 59.30 ± 15.37 | 58.06 ± 16.47 |

## C2. Saliency–ERD correlations (Pearson, Fisher mean over subjects)

First three: sign-invariant metrics used in the paper. Last two: as in the original scripts.

| Model | Class | Attribution | magnitude r | class-specific r | lateralisation r | orig ERD_(L−R) r | orig ERD_comb r |
|---|---|---|---|---|---|---|---|
| EEGNetV4 | 2class | grad | 0.118 | 0.224 | 0.011 | -0.187 | -0.218 |
| EEGNetV4 | 2class | grad_x_input | 0.062 | 0.216 | -0.070 | -0.221 | -0.220 |
| EEGNetV4 | 2class | integrated_gradients | 0.072 | 0.219 | -0.078 | -0.225 | -0.224 |
| EEGNetV4 | 2class | smoothgrad | 0.118 | 0.224 | 0.012 | -0.187 | -0.218 |
| EEGNetV4 | 4class | grad | 0.215 | 0.313 | 0.292 | -0.164 | -0.314 |
| EEGNetV4 | 4class | grad_x_input | 0.192 | 0.307 | 0.251 | -0.173 | -0.323 |
| EEGNetV4 | 4class | integrated_gradients | 0.201 | 0.297 | 0.238 | -0.172 | -0.308 |
| EEGNetV4 | 4class | smoothgrad | 0.216 | 0.314 | 0.296 | -0.164 | -0.315 |
| EEGItNet | 2class | grad | 0.066 | 0.143 | 0.020 | -0.130 | -0.155 |
| EEGItNet | 2class | grad_x_input | 0.015 | 0.136 | -0.066 | -0.136 | -0.151 |
| EEGItNet | 2class | integrated_gradients | 0.031 | 0.142 | -0.122 | -0.124 | -0.163 |
| EEGItNet | 2class | smoothgrad | 0.065 | 0.141 | 0.016 | -0.131 | -0.154 |
| EEGItNet | 4class | grad | 0.184 | 0.178 | 0.093 | -0.055 | -0.167 |
| EEGItNet | 4class | grad_x_input | 0.172 | 0.164 | 0.022 | -0.068 | -0.176 |
| EEGItNet | 4class | integrated_gradients | 0.153 | 0.140 | -0.077 | -0.075 | -0.155 |
| EEGItNet | 4class | smoothgrad | 0.183 | 0.178 | 0.093 | -0.055 | -0.167 |
| Deep4net | 2class | grad | 0.115 | 0.136 | -0.156 | -0.005 | -0.164 |
| Deep4net | 2class | grad_x_input | 0.082 | 0.114 | -0.349 | -0.014 | -0.151 |
| Deep4net | 2class | integrated_gradients | 0.083 | 0.116 | -0.420 | -0.016 | -0.153 |
| Deep4net | 2class | smoothgrad | 0.114 | 0.136 | -0.199 | -0.005 | -0.166 |
| Deep4net | 4class | grad | 0.308 | 0.111 | 0.447 | 0.053 | -0.112 |
| Deep4net | 4class | grad_x_input | 0.293 | 0.108 | 0.262 | 0.041 | -0.124 |
| Deep4net | 4class | integrated_gradients | 0.292 | 0.090 | 0.018 | 0.052 | -0.113 |
| Deep4net | 4class | smoothgrad | 0.308 | 0.110 | 0.446 | 0.052 | -0.112 |
| EEGConformer | 2class | grad | 0.053 | 0.198 | 0.015 | -0.180 | -0.223 |
| EEGConformer | 2class | grad_x_input | 0.033 | 0.197 | 0.060 | -0.185 | -0.219 |
| EEGConformer | 2class | integrated_gradients | 0.024 | 0.166 | 0.127 | -0.137 | -0.180 |
| EEGConformer | 2class | smoothgrad | 0.053 | 0.198 | 0.003 | -0.180 | -0.223 |
| EEGConformer | 4class | grad | 0.001 | 0.153 | 0.061 | -0.144 | -0.131 |
| EEGConformer | 4class | grad_x_input | -0.031 | 0.154 | 0.056 | -0.140 | -0.142 |
| EEGConformer | 4class | integrated_gradients | -0.089 | 0.127 | 0.061 | -0.017 | -0.133 |
| EEGConformer | 4class | smoothgrad | -0.000 | 0.149 | 0.069 | -0.144 | -0.127 |

## C3. EXPLORATORY: corr(mean saliency, physiological map), Pearson, Fisher mean

Curiosity only; not used anywhere in the pipeline.

| Map | Deep4net 2class | Deep4net 4class | EEGConformer 2class | EEGConformer 4class | EEGItNet 2class | EEGItNet 4class | EEGNetV4 2class | EEGNetV4 4class |
|---|---|---|---|---|---|---|---|---|
| ERD_theta_L | -0.126 | -0.074 | 0.040 | -0.013 | -0.174 | -0.036 | -0.139 | -0.203 |
| ERD_theta_R | -0.170 | -0.333 | 0.012 | -0.015 | -0.129 | -0.102 | -0.041 | -0.093 |
| ERD_theta_mag | -0.088 | 0.105 | -0.016 | -0.075 | 0.125 | 0.105 | 0.028 | 0.087 |
| ERD_mu_L | -0.129 | -0.137 | -0.287 | -0.196 | -0.168 | -0.175 | -0.296 | -0.285 |
| ERD_mu_R | -0.087 | -0.247 | -0.166 | -0.106 | -0.086 | -0.145 | -0.143 | -0.157 |
| ERD_mu_mag | 0.187 | 0.427 | 0.032 | -0.078 | 0.116 | 0.117 | 0.188 | 0.125 |
| ERD_beta_L | -0.038 | 0.027 | -0.096 | -0.106 | -0.102 | -0.028 | -0.110 | -0.273 |
| ERD_beta_R | -0.022 | 0.057 | 0.089 | 0.012 | -0.028 | 0.093 | 0.010 | -0.108 |
| ERD_beta_mag | -0.027 | 0.109 | 0.048 | 0.015 | 0.055 | 0.030 | 0.110 | -0.032 |
| ERS_beta_rebound_L | -0.139 | -0.171 | 0.021 | -0.003 | -0.136 | -0.005 | -0.092 | -0.153 |
| ERS_beta_rebound_R | -0.107 | 0.006 | -0.142 | -0.095 | -0.030 | 0.108 | -0.068 | -0.064 |
| ERS_beta_rebound_mag | -0.150 | -0.046 | -0.038 | -0.005 | -0.108 | 0.012 | -0.065 | -0.151 |
| ERD_lowgamma_L | -0.202 | -0.089 | -0.149 | -0.042 | -0.033 | -0.089 | -0.081 | -0.085 |
| ERD_lowgamma_R | -0.079 | -0.003 | -0.088 | -0.070 | -0.053 | 0.090 | -0.063 | -0.193 |
| ERD_lowgamma_mag | -0.068 | -0.065 | 0.033 | 0.075 | 0.034 | 0.056 | -0.040 | -0.096 |
| logpow_8_30 | -0.163 | 0.128 | -0.107 | -0.152 | -0.101 | 0.092 | -0.009 | 0.051 |
| r2_signed | 0.040 | 0.002 | 0.237 | 0.145 | 0.118 | 0.069 | 0.284 | 0.124 |
| r2_abs | 0.139 | 0.408 | 0.282 | 0.146 | 0.187 | 0.101 | 0.363 | 0.113 |
| TFR_ERD_dB_L | -0.159 | -0.063 | -0.237 | -0.207 | -0.165 | -0.103 | -0.241 | -0.332 |
| TFR_ERD_dB_R | -0.147 | -0.149 | -0.034 | -0.042 | -0.070 | -0.043 | -0.110 | -0.211 |
| TFR_ERD_dB_mag | 0.058 | 0.233 | 0.135 | 0.119 | 0.066 | 0.131 | 0.055 | 0.172 |
| ITC_4_30 | -0.185 | 0.112 | -0.141 | -0.025 | -0.054 | 0.040 | -0.130 | -0.056 |
| ERP_abs_0.2_0.8 | -0.047 | 0.256 | -0.121 | -0.070 | -0.068 | 0.034 | -0.093 | 0.044 |
| SlowPot_0.5_3 | 0.071 | -0.218 | 0.002 | 0.021 | -0.166 | -0.207 | -0.056 | 0.040 |
