# EEGConformer on HGD (Schirrmeister2017)

Same layout as the other models: `eegconformer_hgd_common.py` plus `2class/` and `4class/` folders with
`_saliency` (main: 22ch + Ours), `_csp`, `_relieff`, `_mi` and `_controls`.

Unlike the other models, EEGConformer keeps the pipeline of its original scripts:

| | EEGConformer |
|---|---|
| sampling / band | 500 Hz (native), 4-125 Hz IIR band-pass |
| reference | average reference over the 22 channels, computed **before** any channel subsetting |
| epoch | 0-4 s after the cue = 2000 samples; z-scored with training-fold mean/std |
| training | AdamW (lr 1e-3), batch 32, **fixed 100 epochs**, no early stopping |
| augmentation | segmentation-and-reconstruction (8 segments), every batch doubled |
| scheduler | ReduceLROnPlateau on accuracy, factor 0.5 |
| 2-class model | dropout 0.6, attention depth 2, heads 4, patience 15 |
| 4-class model | dropout 0.5, attention depth 6, heads 10, patience 10 |
| CSP regulariser | `reg = 1e-4` (average-referenced data are rank-deficient) |

What changed relative to the original scripts: the learning-rate schedule and the checkpoint now use the
**validation** block (the originals used the test fold), saliency is computed on validation trials, montages
are leave-one-subject-out, and the 22-channel data are referenced once so every montage shares the same
reference. See `docs/REVIEW_CROSSWALK.md` (section D) for the full list.

**Check before running the 4-class folder.** The only 4-class Conformer script provided was a fixed-montage
run (12 channels, subjects 10-14), so the 4-class settings above come from it. Edit `CONFORMER_CFG` in
`eegconformer_hgd_common.py` if the 4-class experiments should use other settings.

Compute: 22-channel, 2000-sample inputs with a doubled batch for 100 epochs per fold - use a GPU.
The model builder accepts both current (`num_layers`, `num_heads`) and older (`att_depth`, `att_heads`)
braindecode argument names. Run instructions: see the repository `README.md`.
