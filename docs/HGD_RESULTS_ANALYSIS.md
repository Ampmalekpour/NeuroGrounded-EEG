# HGD, EEGNetV4: why the accuracy is lower than in the paper, and what can be improved

Data: `EEGNetV4/HGD/report_session/` (14 subjects, 3-seed ensemble, 22 channels).
Paper: Tables 1 and 4.

## 1. Our numbers match the paper wherever the paper's test is valid

| 4-class, 22ch | Paper | Ours |
|---|---|---|
| S01–S05 (genuine test recordings) | 88.4 | 88.0 |
| S06–S14 | ≈ 100 for every subject | 81.7 |
| Mean of 14 | 95.8 | 83.9 |

- For S01–S05 the two pipelines agree. Our code is not the problem.
- For S06–S14 the paper's ≈ 100 % comes from the duplicated test files, and from the original code pooling
  train and test before a block split.
  - For these subjects MOABB's "1test" recording is an exact copy of "0train" (`analysis/hgd_data_check.csv`).
  - After pooling, the last two blocks are an exact copy of the first two, so each test trial has an identical
    copy in the training data.

## 2. The paper's "2-class" HGD task is feet vs left hand

braindecode numbers the classes alphabetically (feet=0, left_hand=1, rest=2, right_hand=3), so the original
labels `[0, 1]` are feet vs left hand. The 4-class model scores 98.5 % on that pair, against 99.7 % in the paper.

| Pair, from our 4-class model | Accuracy |
|---|---|
| feet vs left hand (the paper's "2-class") | 98.5 % |
| left vs right hand (our 2-class) | 80.0 % |

Left vs right is the harder, genuine task, and the paper's 2-class numbers are not comparable with it.

## 3. What is left that is real

- **Hold-out size.** For S06–S14 the test set is the last 160 trials of one recording (S11: 32 trials), so
  the numbers are noisy. S11 has only 160 trials in total.
- **Signal-limited subjects.** S01, S07, S09 and S11 are near chance for left vs right, in every montage and
  every seed. This is not a training failure; the 4-class model reaches only 56–65 % on their left/right pair.
- **Band.**
  - Every profile except the Conformer's keeps only 4–38 Hz (the manuscript's setting).
  - Schirrmeister et al. (2017) used a 4 Hz high-pass only on this dataset and report about 91–93 % (4-class,
    official test set, 44 channels).
  - The gap to our 84 % comes partly from the missing high gamma and partly from 22 instead of 44 channels.
- **Validation is honest.** The mean validation accuracy (83.8 %) matches the test accuracy (83.9 %), so there
  is no sign of over-fitting to the validation set.

## 4. What you can do

1. **Repair the local data cache (found: the server files are fine).**
   - Checked on the real machine: the cached `train\6.edf` and `test\6.edf` were byte-identical, but the
     server holds distinct files (train about 0.9 GB, test about 0.16 GB). The cached S11 "train" file was
     the small test file.
   - So the HGD results in `results_session/` used a 160-trial hold-out for S06–S14 only because of the
     broken cache. With the right files, all 14 subjects use the official train and test runs.
   - Repair (re-downloads only the wrong files):

     ```powershell
     python analysis\fix_hgd_cache.py --check     # report only
     python analysis\fix_hgd_cache.py             # fix
     python analysis\check_hgd_data.py            # must now show distinct recordings for all 14 subjects
     ```

   - Keep the earlier results. Rename them before re-running (see `docs/HGD_RERUN.md`).
2. **Full-bandwidth sensitivity run (new, opt-in).**

   ```powershell
   python run_all.py --dataset hgd --protocol wide --models eegitnet
   ```

   - It uses 250 Hz and 4–100 Hz, which is Schirrmeister's HGD setting, and writes to `results_session_wide/`.
   - It never touches `results_session/`.
   - Report it next to the 4–38 Hz result, not instead of it. Executed movements produce EMG, so high-gamma
     decoding is not purely cortical.
3. **Do not expect 95–100 %.** That level was only reached through the duplicated data.

## 5. Not recommended

- Dropping the hold-out and using the duplicated test file.
- Choosing profiles on the test set.
- Tuning until S06–S14 look like the paper.
