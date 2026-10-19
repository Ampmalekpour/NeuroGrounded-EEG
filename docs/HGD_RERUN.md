# HGD: re-running after the cache repair

The earlier HGD results (EEGNetV4, hold-out for S06–S14) stay in the repository as `*_holdout` folders.
They are valid for what they are, but the official-split results replace them in the paper.

```powershell
cd D:\EEG\NeuroGrounded-EEG
git pull origin claude/practical-goldberg-gh2t5q
python analysis\fix_hgd_cache.py
python analysis\check_hgd_data.py          # the table must show different fingerprints for train and test

# keep the earlier EEGNetV4 results
foreach ($c in "2class","4class") {
  Rename-Item EEGNetV4\HGD\$c\results_session EEGNetV4\HGD\$c\results_session_holdout
  Rename-Item EEGNetV4\HGD\$c\logs_session    EEGNetV4\HGD\$c\logs_session_holdout
}
Rename-Item EEGNetV4\HGD\report_session EEGNetV4\HGD\report_session_holdout
Rename-Item reports\HGD_session reports\HGD_session_holdout
Rename-Item run_all_status_hgd_session.json run_all_status_hgd_session_holdout.json

python run_all.py --dataset hgd --models eegnetv4        # official split, 880 train / 160 test trials
```

Then run the other models (`eegitnet`, `deep4net`, `eegconformer`) the same way. Optional full-bandwidth
sensitivity run: add `--protocol wide`.
