# EEGNetv4 on HGD (Schirrmeister2017)

Same structure and scripts as `Deep4net/HGD`: `eegnetv4_hgd_common.py` plus `2class/` and `4class/`
folders with `_saliency` (main: 22ch + Ours), `_csp`, `_relieff`, `_mi` and `_controls`.

`eegnetv4_hgd_common.py` differs from the Deep4Net common module only in the model builder and in:
* no exponential moving standardisation (the original EEGNetv4 scripts did not use it);
* **350 maximum epochs** (the original EEGNetv4 scripts used 350; the other models use 600).

Model: `EEGNetv4(F1=8, D=2, F2=16, kernel_length=64, third_kernel_size=(8, 4), drop_prob=0.30)`.
The builder tries the current argument names first and falls back to the older ones
(`in_chans`, `n_classes`, `input_window_samples`). If your braindecode no longer ships `EEGNetv4`
it uses the renamed `EEGNet` with the same hyper-parameters; the model name stored in every
`results.json` says which class was actually used.

`legacy_unrevised/` holds the earlier CSP / MI / ReliefF scripts, unchanged.
Run instructions: see the repository `README.md`.
