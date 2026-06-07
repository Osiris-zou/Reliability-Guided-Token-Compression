# Third-party notices and model acknowledgements

This repository contains research code for reliability-guided token compression. It does not redistribute ImageNet-1K, ADE20K, Stable Diffusion weights, Segmenter checkpoints, generated images, or experiment result logs.

The project builds on and acknowledges the following resources:

- Token Merging (ToMe): the repository follows a ToMe-style bipartite soft matching and ViT patching design. Some files in `tome/` are adapted from or structured after upstream ToMe-style code and retain original copyright headers where present.
- ToMe for Stable Diffusion / ToMeSD-style merge-unmerge: the diffusion extension follows the merge-unmerge idea for applying token merging inside selected Stable Diffusion self-attention blocks.
- Vision Transformer (ViT) and DeiT: classification backbones used for ImageNet-1K experiments.
- timm: model definitions, pretrained checkpoint loading, and ViT/DeiT utilities.
- Stable Diffusion / Latent Diffusion Models: generative backbone used for the transferability experiment. No model weights are redistributed.
- Segmenter: Seg-B-Mask/16 ADE20K checkpoint and `variant.yml` are used for the dense prediction appendix. No checkpoint or configuration file is redistributed here.
- PyTorch, torchvision, NumPy, SciPy, Pillow, tqdm, Matplotlib, LPIPS, MS-SSIM, and related libraries for inference, metrics, and analysis.
- ImageNet-1K and ADE20K datasets. Users must obtain them from official sources and comply with their usage terms.

Users are responsible for checking and following all third-party licenses, dataset terms, and pretrained-model usage conditions.
