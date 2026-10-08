# Deep4Net on HGD (Schirrmeister2017)

* `deep4net_hgd_common.py` - shared code (labels, preprocessing, folds, training, LOSO montages,
  ERD/ERS references, alignment metrics, result files). Not run directly.
* `2class/` and `4class/` - five scripts each: `_saliency` (main: 22ch + Ours), `_csp`, `_relieff`,
  `_mi`, `_controls`.
* `legacy_unrevised/` - the earlier 4-class CSP/MI/ReliefF scripts, unchanged.

Deep4Net keeps the exponential moving standardisation used in the original scripts.
Run instructions: see the repository `README.md`.
