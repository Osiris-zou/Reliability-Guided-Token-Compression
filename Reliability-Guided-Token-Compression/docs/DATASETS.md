# Dataset and checkpoint layout

This repository does not redistribute datasets, checkpoints, generated images, or result logs.

## ImageNet-1K validation

Expected structure:

```text
/path/to/imagenet/val/
  n01440764/*.JPEG
  n01443537/*.JPEG
  ...
```

The scripts use `torchvision.datasets.ImageFolder` or equivalent sorted folder indexing.

## ADE20K

Expected structure for the Segmenter appendix experiment:

```text
/path/to/ADEChallengeData2016/
  images/validation/*.jpg
  annotations/validation/*.png
```

ADE20K label value 0 is treated as ignore/background; labels 1..150 are converted to class indices 0..149.

## Segmenter checkpoint

Expected structure:

```text
/path/to/segmenter/checkpoints/seg_base_mask/
  checkpoint.pth
  variant.yml
```

The `variant.yml` file must be in the same directory as `checkpoint.pth`.

## Stable Diffusion

Stable Diffusion weights are not included. Configure the local path to the model checkpoint or model directory according to the script arguments and your local environment.
