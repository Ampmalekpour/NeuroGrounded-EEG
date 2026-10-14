# Deep4Net on BNCI2014-001 (BCI Competition IV-2a)

Same structure, steps, tables and physiology block as `EEGNetV4/BNCI_001` and `EEGItNet/BNCI_001`.
`2class` = left vs right hand, `4class` = feet / left hand / right hand / tongue.

## Run

```powershell
cd D:\EEG\NeuroGrounded-EEG
python run_all.py --models deep4net      # only Deep4Net (session protocol)
python run_all.py                    # all four models: EEGNetV4, EEG-ITNet, Deep4Net, EEGConformer
```

Results: `Deep4net/BNCI_001/report_session/REPORT.md` and `reports/BNCI_001_session/SUMMARY.md`.

## Training recipe (profile `original`)

The paper's BNCI recipe (Sec. 3.4), shared with EEG-ITNet:

| Setting | Value |
|---|---|
| Preprocessing | 250 Hz, Chebyshev-I (order 6, 0.5 dB) 4–38 Hz IIR, exponential moving standardisation |
| Input window | −0.5 … 4.0 s around the cue (1125 samples) |
| Training | 350 epochs, batch 64, AdamW lr 1e-3 / wd 1e-3, cosine annealing, grad clip 10 |
| Regularisation | noise / scaling / channel-dropout augmentation, mixup (p 0.5, α 0.2), label smoothing 0.1 |
| Final model | SWA average from epoch 250 + BatchNorm update |
| Prediction | test-time augmentation (3 views), then the seed ensemble |
| Model | braindecode `Deep4Net`, `final_conv_length="auto"`, dropout 0.5 |

The legacy Deep4Net BNCI script (`legacy_unrevised/Deep4Net_BNCI_SWT_Pipeline.py`) used exactly this
recipe. Its problems were the same as EEG-ITNet's: saliency on the test fold, global montages,
shuffled folds without validation, and attribution methods chosen by ERD correlation. All are fixed here
(docs/REVIEW_CROSSWALK.md, sections G and H).

## Fair tuning

`TUNE_CANDIDATES` = `original`, `cue_0_4` (window 0–4 s). `no_aug` is also defined; add it with
`$env:EEG_TUNE_CANDIDATES="original,cue_0_4,no_aug"`.
