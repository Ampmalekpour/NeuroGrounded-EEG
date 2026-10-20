# HGD: re-running after the cache repair

The earlier EEGNetV4 results used a 160-trial hold-out for S06–S14, because the cached test recordings
were copies of the training recordings. They stay in the repository as `*_holdout` folders. The
official-split results replace them in the paper.

## 1. Repair and check the data (once)

```powershell
cd D:\EEG\NeuroGrounded-EEG
git pull origin claude/practical-goldberg-gh2t5q
python analysis\fix_hgd_cache.py --check    # every line must end with "ok" (or run without --check to fix)
python analysis\check_hgd_data.py           # train and test fingerprints must differ for all 14 subjects
```

`run_all.py` repeats a cheap version of this check before an HGD run and stops if the cached train and test
files have the same size, or if the results folder still holds hold-out results.

## 2. Keep the earlier EEGNetV4 results

```powershell
foreach ($c in "2class","4class") {
  Rename-Item "EEGNetV4\HGD\$c\results_session" "results_session_holdout"
  Rename-Item "EEGNetV4\HGD\$c\logs_session"    "logs_session_holdout"
}
Rename-Item "EEGNetV4\HGD\report_session" "report_session_holdout"
Rename-Item "reports\HGD_session" "HGD_session_holdout"
Rename-Item "run_all_status_hgd_session.json" "run_all_status_hgd_session_holdout.json"
```

(`Rename-Item` takes only the new NAME as its second argument, not a path.)

## 3. Run the whole EEGNetV4 suite (one command)

```powershell
python EEGNetV4\HGD\run_suite.py
```

- This is the same as `python run_all.py --dataset hgd --models eegnetv4`.
- It runs tune → saliency → csp → relieff → mi → controls → tables, for 2class and 4class.
- If it stops, run the same command again. Finished steps are skipped.
- Keep the computer awake: `powercfg /change standby-timeout-ac 0` (the earlier 4class controls step took 16 h,
  which suggests sleep or interruptions).
- Results: `EEGNetV4\HGD\report_session\REPORT.md` (T0 lists trials per subject, T5 per-subject confusions)
  and `reports\HGD_session\SUMMARY.md`.

## 4. Other models, and the full-bandwidth sensitivity run

```powershell
python EEGItNet\HGD\run_suite.py
python Deep4net\HGD\run_suite.py
python EEGConformer\HGD\run_suite.py
python EEGNetV4\HGD\run_suite.py --protocol wide     # optional: 250 Hz, 4-100 Hz, results_session_wide/
```
