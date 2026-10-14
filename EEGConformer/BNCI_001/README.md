# EEGConformer on BNCI2014-001 (BCI Competition IV-2a)

Same structure, steps, tables and physiology block as `EEGNetV4/BNCI_001` and `EEGItNet/BNCI_001`.
`2class` = left vs right hand, `4class` = feet / left hand / right hand / tongue.

## Run

```powershell
cd D:\EEG\NeuroGrounded-EEG
python run_all.py --models eegconformer      # only EEGConformer (session protocol)
python run_all.py                    # all four models: EEGNetV4, EEG-ITNet, Deep4Net, EEGConformer
```

Results: `EEGConformer/BNCI_001/report_session/REPORT.md` and `reports/BNCI_001_session/SUMMARY.md`.

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
| Model | braindecode `EEGConformer`: 40 temporal filters of 25 samples, pooling 75 / stride 15; 2class depth 2, 4 heads, dropout 0.6; 4class depth 6, 10 heads, dropout 0.5 (sizes from EEGConformer/HGD) |

No BNCI EEGConformer script was provided. The paper states one BNCI recipe for all models, so that recipe
is used here.

## Fair tuning

`TUNE_CANDIDATES` = `original`, `lr_2e-4` (the original Conformer learning rate; transformers often
prefer it). `cue_0_4` is also defined; add it with `$env:EEG_TUNE_CANDIDATES="original,lr_2e-4,cue_0_4"`.
