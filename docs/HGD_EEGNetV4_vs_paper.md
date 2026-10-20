# HGD, EEGNetV4: official split (14 subjects) vs the paper

Results: `EEGNetV4/HGD/report_session/` (3-seed ensemble, 22 channels, official 880 / 160 train / test runs,
4–38 Hz profiles). Paper: Tables 1 and 4. The earlier hold-out run is kept in `*_holdout`.

## Mean accuracy (%), 14 subjects

| 4-class | Paper | Hold-out run | **Official split** |
|---|---|---|---|
| 22ch | 95.85 | 83.94 | **87.05** |
| Ours (saliency, 12 ch) | 96.52 | 83.84 | **85.36** |
| CSP | 94.89 | 85.57 | 86.38 |
| ReliefF | 97.22 | 83.94 | 85.62 |
| MI | 96.98 | 83.40 | 86.29 |

| 2-class | Paper (feet vs left hand) | **Official split (left vs right)** |
|---|---|---|
| 22ch | 99.69 | 83.04 |
| Ours | 99.56 | 81.07 |
| CSP / RLF / MI | 99.41 / 99.56 / 99.61 | 83.93 / 84.02 / 80.36 |

## Per subject, 4-class, 22ch

| | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper | 82.5 | 83.8 | 80.7 | 96.1 | 99.0 | 100 | 100 | 99.9 | 100 | 99.9 | 100 | 100 | 99.9 | 100 |
| Ours | 75.0 | 84.4 | 97.5 | 98.1 | 85.0 | 87.5 | 79.4 | 88.8 | 85.6 | 88.1 | 78.1 | 88.8 | 87.5 | 95.0 |

- S01–S05: paper 88.4, ours 88.0 (same recordings, different split).
- S06–S14: paper about 100, ours 86.5. The paper's value comes from duplicated test recordings.

## Why it differs from the paper

- The paper's S06–S14 were scored on copies of the training data (see `docs/HGD_RESULTS_ANALYSIS.md`).
- The paper's "2-class" is feet vs left hand. Our 4-class model gets 98.8 % on that pair
  (paper 99.7 %) and 85.1 % on left vs right.
- Validation accuracy (86.1 %) matches test accuracy (87.1 %): no over-fitting to the validation set.
- Published reference: about 91–93 % with 44 channels and no low-pass (Schirrmeister et al. 2017).

## Reviewer-relevant results

- **Equivalence (TOST, ±6 pp, vs 22ch):**
  - 4-class: all 7 montages equivalent (Ours −1.70 pp, p = 0.0003).
  - 2-class: 5 of 7 equivalent. MI (p = 0.098) and random (p = 0.135) are not. Ours: −1.96 pp, p = 0.017.
- **Ours vs other 12-channel montages:** no significant difference after Holm correction (all p > 0.17).
  Ours is not better than CSP, ReliefF, MI or ERD-only (4-class −1.0, −0.3, −0.9, −0.7 pp).
  It is above fixed sensorimotor and random montages, but not significantly.
- **Faithfulness (4-class, k = 2):** removing the top-ranked channels costs 33.1 pp, random 15.3 pp and
  bottom-ranked 2.8 pp (p < 0.0001). The ranking is meaningful.
- **Randomisation check:** correlation with an untrained network about 0.
- **Saliency–ERD alignment (4-class):** magnitude r = 0.45, class-specific r = 0.43 (grad, SmoothGrad).
  Lateralisation about 0 to −0.12; in 2-class it is negative (−0.19 to −0.37).
  Saliency finds the active channels but not the side.
- **Seeds:** single-seed means 84.4–85.5 % for 22ch, ensemble 87.05 %.
