import argparse
from pathlib import Path

import torch

import segm.utils.torch as ptu
from segm.model.factory import load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the official Segmenter checkpoint can be loaded and run."
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    variant = checkpoint.parent / "variant.yml"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not variant.is_file():
        raise FileNotFoundError(
            f"variant.yml must be in the same directory as the checkpoint: {variant}"
        )

    use_cuda = args.device.startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    ptu.set_gpu_mode(use_cuda)
    device = torch.device(args.device if use_cuda else "cpu")
    ptu.device = device

    print("========== Segmenter baseline verification ==========")
    print(f"Checkpoint : {checkpoint}")
    print(f"Variant    : {variant}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA build : {torch.version.cuda}")
    print(f"Device     : {device}")
    if use_cuda:
        print(f"GPU        : {torch.cuda.get_device_name(device)}")

    model, model_variant = load_model(str(checkpoint))
    model = model.to(device).eval()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters : {parameters / 1e6:.2f} M")
    print(f"Patch size : {model.patch_size}")
    print(f"Classes    : {model.n_cls}")
    print(f"Backbone   : {model_variant['net_kwargs'].get('backbone')}")
    print(f"Decoder    : {model_variant['net_kwargs']['decoder'].get('name')}")

    x = torch.randn(1, 3, args.height, args.width, device=device)

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        y = model(x)

    if use_cuda:
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"Peak memory: {peak_memory:.3f} GB")

    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Finite output: {bool(torch.isfinite(y).all())}")

    expected_shape = (1, 150, args.height, args.width)
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape {tuple(y.shape)}; expected {expected_shape}."
        )

    print("Segmenter checkpoint load and forward pass succeeded.")


if __name__ == "__main__":
    main()
