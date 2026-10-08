# EEG-ITNet on HGD (Schirrmeister2017)

Same structure and scripts as `Deep4net/HGD`. `eegitnet_hgd_common.py` differs from the Deep4Net
common module only in the model line and in not using exponential moving standardisation (the
original EEG-ITNet scripts did not use it). Run the scripts inside `2class/` or `4class/`;
see the repository `README.md`.
