# Invalid: first EEGNetV4 HGD session-protocol run (kept for the record only)

Do not use these numbers. Two problems, both fixed in the code since:

1. **Only subjects 1–9 were run** (the HGD code had inherited the BNCI default of 9 subjects).
2. **For subjects 6–14 the MOABB "1test" recording is an exact copy of "0train"**
   (`analysis/check_hgd_data.py`, `hgd_data_check.csv`: identical signal fingerprints, trial counts and
   durations). Subjects 6–9 were therefore tested on their own training data. The code now drops duplicated
   recordings and holds out the chronologically last 160 trials (at most 20 %) for those subjects.

Also fixed: about 15 % of the single-seed 2-class models collapsed to one class, because the
lowest-validation-loss checkpoint was the untrained epoch-1 model. Checkpoints are now chosen on validation
accuracy and never before epoch 20.

Folders: `EEGNetV4/HGD/*/results_session_invalid_v1/`, `*/logs_session_invalid_v1/`,
`EEGNetV4/HGD/report_session_invalid_v1/`.
