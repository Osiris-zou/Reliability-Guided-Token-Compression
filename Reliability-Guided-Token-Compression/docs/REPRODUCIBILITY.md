# Reproducibility protocol

This document summarizes how the repository maps to the manuscript experiments. Output files should be saved under `outputs/`; that directory is ignored by Git.

## ImageNet-1K classification tables

- ViT fixed-beta evaluation: `experiments/imagenet/eval_vit_methods_stable.py`
- DeiT fixed-beta and beta-sweep evaluation: `experiments/imagenet/eval_deit_generality.py`
- Paired bootstrap analysis: `experiments/imagenet/bootstrap_vit_paired.py` or `experiments/imagenet/paired_bootstrap_from_npy.py`
- Throughput diagnostics: `experiments/imagenet/benchmark_deit_throughput_stable.py` and diagnostic scripts

Keep the same preprocessing, batch size, merging rate, beta value, and full/ToMe/Ours pairing when reproducing table rows.

## Stable Diffusion transfer analysis

Use scripts in `experiments/stable_diffusion/` to generate paired samples or compute LPIPS/MS-SSIM/FID-related summaries. The reported comparison should be interpreted as transferability and non-degradation rather than a statistically decisive quality improvement.

## ADE20K Segmenter-B/16 dense prediction transfer

Use `experiments/dense_prediction/eval_segmenter_tome_ours_ade20k.py`. This experiment tests whether the reliability-guided ranking can be integrated into a ViT-based dense prediction pipeline without retraining. It is not an optimized segmentation acceleration benchmark because the decoder requires restoration of the full patch grid.

## Figure generation

Use `experiments/figures/` with locally generated CSV/NPY/JSON files. Generated figures should be saved under `outputs/figures/` and should not be committed unless intentionally needed for documentation.
