import argparse

SCRIPT_VERSION = "segmenter-ade20k-v2-vitnorm-20260607"
import csv
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import segm.utils.torch as ptu
from segm.model.factory import load_model


NORMALIZATION_STATS = {
    # Segmenter's improved ViT checkpoints use mean/std = 0.5/0.5.
    # Values are represented on the 0..255 image scale used by this script.
    "vit": {
        "mean": torch.tensor([127.5, 127.5, 127.5]).view(3, 1, 1),
        "std": torch.tensor([127.5, 127.5, 127.5]).view(3, 1, 1),
    },
    # DeiT checkpoints use the standard ImageNet normalization.
    "deit": {
        "mean": torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1),
        "std": torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the official Segmenter checkpoint on ADE20K without "
            "depending on the legacy MMSegmentation evaluation stack."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--ade20k-root", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--short-side", default=512, type=int)
    parser.add_argument("--max-long-side", default=2048, type=int)
    parser.add_argument("--window-size", default=512, type=int)
    parser.add_argument("--window-stride", default=480, type=int)
    parser.add_argument("--window-batch-size", default=4, type=int)
    parser.add_argument(
        "--max-images",
        default=0,
        type=int,
        help="0 evaluates all validation images; use a small value for a smoke test.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--normalization",
        choices=["auto", "vit", "deit"],
        default="auto",
        help="Use 'vit' for Seg-B-Mask/16. 'auto' reads variant.yml.",
    )
    parser.add_argument(
        "--output-json",
        default="segmenter_baseline_ade20k.json",
        type=str,
    )
    parser.add_argument(
        "--output-csv",
        default="segmenter_baseline_ade20k.csv",
        type=str,
    )
    return parser.parse_args()


def resolve_validation_pairs(root: Path) -> List[Tuple[Path, Path]]:
    image_dir = root / "images" / "validation"
    mask_dir = root / "annotations" / "validation"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Validation mask directory not found: {mask_dir}")

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No JPG files found in {image_dir}")

    pairs = []
    missing = []
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
        else:
            missing.append(mask_path)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} masks are missing. First missing mask: {missing[0]}"
        )

    return pairs


def resize_shape(
    height: int,
    width: int,
    short_side: int,
    max_long_side: int,
) -> Tuple[int, int]:
    short = min(height, width)
    long = max(height, width)

    scale = min(short_side / short, max_long_side / long)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return new_height, new_width


def load_and_preprocess(
    image_path: Path,
    short_side: int,
    max_long_side: int,
    normalization: str,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size

        new_height, new_width = resize_shape(
            original_height,
            original_width,
            short_side,
            max_long_side,
        )
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy()

    tensor = torch.from_numpy(array).permute(2, 0, 1)

    if normalization not in NORMALIZATION_STATS:
        raise ValueError(
            f"Unsupported normalization '{normalization}'. "
            f"Available choices: {sorted(NORMALIZATION_STATS)}"
        )

    stats = NORMALIZATION_STATS[normalization]
    tensor = (tensor - stats["mean"]) / stats["std"]
    return tensor.unsqueeze(0), (original_height, original_width)


def sliding_positions(length: int, window_size: int, stride: int) -> List[int]:
    if length <= window_size:
        return [0]

    positions = list(range(0, length - window_size + 1, stride))
    final_position = length - window_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def pad_to_window(
    image: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, height, width = image.shape
    pad_height = max(0, window_size - height)
    pad_width = max(0, window_size - width)

    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), value=0.0)

    return image, (height, width)


@torch.inference_mode()
def sliding_window_predict(
    model,
    image: torch.Tensor,
    original_shape: Tuple[int, int],
    device: torch.device,
    n_classes: int,
    window_size: int,
    window_stride: int,
    window_batch_size: int,
    amp: bool,
) -> torch.Tensor:
    image = image.to(device, non_blocking=True)
    image, valid_shape = pad_to_window(image, window_size)

    _, _, height, width = image.shape
    h_positions = sliding_positions(height, window_size, window_stride)
    w_positions = sliding_positions(width, window_size, window_stride)
    anchors = [(h, w) for h in h_positions for w in w_positions]

    logit_sum = torch.zeros(
        (1, n_classes, height, width),
        dtype=torch.float32,
        device=device,
    )
    count = torch.zeros(
        (1, 1, height, width),
        dtype=torch.float32,
        device=device,
    )

    for start in range(0, len(anchors), window_batch_size):
        batch_anchors = anchors[start : start + window_batch_size]
        crops = torch.cat(
            [
                image[
                    :,
                    :,
                    h : h + window_size,
                    w : w + window_size,
                ]
                for h, w in batch_anchors
            ],
            dim=0,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(crops)

        logits = logits.float()

        for index, (h, w) in enumerate(batch_anchors):
            logit_sum[
                :,
                :,
                h : h + window_size,
                w : w + window_size,
            ] += logits[index : index + 1]
            count[
                :,
                :,
                h : h + window_size,
                w : w + window_size,
            ] += 1.0

    logits = logit_sum / count.clamp_min(1.0)

    valid_height, valid_width = valid_shape
    logits = logits[:, :, :valid_height, :valid_width]

    logits = F.interpolate(
        logits,
        size=original_shape,
        mode="bilinear",
        align_corners=False,
    )

    return logits.argmax(dim=1)[0].cpu()


def load_ground_truth(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as mask:
        gt = np.asarray(mask, dtype=np.int64).copy()
    return gt


def update_confusion_matrix(
    confusion: np.ndarray,
    prediction: np.ndarray,
    ground_truth_raw: np.ndarray,
    n_classes: int,
) -> None:
    # ADE20K benchmark masks use 0 as ignore/background and 1..150 as classes.
    valid = (
        (ground_truth_raw > 0)
        & (ground_truth_raw <= n_classes)
        & (prediction >= 0)
        & (prediction < n_classes)
    )

    ground_truth = ground_truth_raw[valid] - 1
    prediction = prediction[valid]

    encoded = n_classes * ground_truth + prediction
    confusion += np.bincount(
        encoded,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)


def calculate_metrics(confusion: np.ndarray) -> dict:
    diagonal = np.diag(confusion).astype(np.float64)
    gt_area = confusion.sum(axis=1).astype(np.float64)
    pred_area = confusion.sum(axis=0).astype(np.float64)
    union = gt_area + pred_area - diagonal

    class_iou = np.divide(
        diagonal,
        union,
        out=np.full_like(diagonal, np.nan),
        where=union > 0,
    )
    class_accuracy = np.divide(
        diagonal,
        gt_area,
        out=np.full_like(diagonal, np.nan),
        where=gt_area > 0,
    )

    total = confusion.sum()
    pixel_accuracy = float(diagonal.sum() / total) if total > 0 else float("nan")

    return {
        "aAcc": 100.0 * pixel_accuracy,
        "mAcc": 100.0 * float(np.nanmean(class_accuracy)),
        "mIoU": 100.0 * float(np.nanmean(class_iou)),
        "evaluated_classes": int(np.isfinite(class_iou).sum()),
        "valid_pixels": int(total),
        "class_iou_percent": [
            None if not np.isfinite(value) else 100.0 * float(value)
            for value in class_iou
        ],
    }


def save_results(result: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    csv_fields = [
        "checkpoint",
        "images",
        "aAcc",
        "mAcc",
        "mIoU",
        "evaluated_classes",
        "valid_pixels",
        "elapsed_seconds",
        "images_per_second",
        "short_side",
        "max_long_side",
        "window_size",
        "window_stride",
        "window_batch_size",
        "amp",
        "normalization",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({key: result[key] for key in csv_fields})


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    variant = checkpoint.parent / "variant.yml"
    ade20k_root = Path(args.ade20k_root).resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not variant.is_file():
        raise FileNotFoundError(
            f"variant.yml must be beside checkpoint.pth: {variant}"
        )

    use_cuda = args.device.startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    ptu.set_gpu_mode(use_cuda)
    device = torch.device(args.device if use_cuda else "cpu")
    ptu.device = device

    pairs = resolve_validation_pairs(ade20k_root)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    model, variant_data = load_model(str(checkpoint))
    model = model.to(device).eval()
    n_classes = int(model.n_cls)

    normalization = (
        str(variant_data.get("dataset_kwargs", {}).get("normalization", "vit"))
        if args.normalization == "auto"
        else args.normalization
    )
    if normalization not in NORMALIZATION_STATS:
        raise ValueError(
            f"Checkpoint requests unsupported normalization '{normalization}'. "
            f"Available choices: {sorted(NORMALIZATION_STATS)}"
        )

    print("========== Segmenter ADE20K baseline evaluation ==========")
    print(f"Script version       : {SCRIPT_VERSION}")
    print(f"Checkpoint          : {checkpoint}")
    print(f"ADE20K root         : {ade20k_root}")
    print(f"Images              : {len(pairs)}")
    print(f"Classes             : {n_classes}")
    print(f"Device              : {device}")
    print(f"GPU                 : {torch.cuda.get_device_name(device) if use_cuda else 'CPU'}")
    print(f"Short side          : {args.short_side}")
    print(f"Maximum long side   : {args.max_long_side}")
    print(f"Window size/stride  : {args.window_size}/{args.window_stride}")
    print(f"Window batch size   : {args.window_batch_size}")
    print(f"Normalization       : {normalization}")
    print(f"AMP                 : {args.amp}")
    print("==========================================================")

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    start_time = time.perf_counter()

    progress = tqdm(pairs, desc="ADE20K validation", ncols=110)
    for index, (image_path, mask_path) in enumerate(progress, start=1):
        image, original_shape = load_and_preprocess(
            image_path,
            args.short_side,
            args.max_long_side,
            normalization,
        )

        prediction = sliding_window_predict(
            model=model,
            image=image,
            original_shape=original_shape,
            device=device,
            n_classes=n_classes,
            window_size=args.window_size,
            window_stride=args.window_stride,
            window_batch_size=args.window_batch_size,
            amp=args.amp,
        ).numpy()

        ground_truth = load_ground_truth(mask_path)

        if prediction.shape != ground_truth.shape:
            raise RuntimeError(
                f"Shape mismatch for {image_path.name}: "
                f"prediction={prediction.shape}, gt={ground_truth.shape}"
            )

        update_confusion_matrix(
            confusion,
            prediction,
            ground_truth,
            n_classes,
        )

        if index % 20 == 0 or index == len(pairs):
            partial = calculate_metrics(confusion)
            progress.set_postfix(
                mIoU=f"{partial['mIoU']:.2f}",
                aAcc=f"{partial['aAcc']:.2f}",
            )

    elapsed = time.perf_counter() - start_time
    metrics = calculate_metrics(confusion)

    result = {
        "checkpoint": str(checkpoint),
        "images": len(pairs),
        "aAcc": metrics["aAcc"],
        "mAcc": metrics["mAcc"],
        "mIoU": metrics["mIoU"],
        "evaluated_classes": metrics["evaluated_classes"],
        "valid_pixels": metrics["valid_pixels"],
        "elapsed_seconds": elapsed,
        "images_per_second": len(pairs) / elapsed if elapsed > 0 else float("nan"),
        "short_side": args.short_side,
        "max_long_side": args.max_long_side,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "window_batch_size": args.window_batch_size,
        "amp": bool(args.amp),
        "normalization": normalization,
        "class_iou_percent": metrics["class_iou_percent"],
        "variant_backbone": variant_data["net_kwargs"].get("backbone"),
        "variant_decoder": variant_data["net_kwargs"]["decoder"].get("name"),
    }

    print("\n========== Final ADE20K result ==========")
    print(f"Images : {result['images']}")
    print(f"aAcc   : {result['aAcc']:.3f}%")
    print(f"mAcc   : {result['mAcc']:.3f}%")
    print(f"mIoU   : {result['mIoU']:.3f}%")
    print(f"Time   : {result['elapsed_seconds']:.2f} s")
    print(f"E2E FPS: {result['images_per_second']:.3f}")
    print("=========================================")

    save_results(
        result,
        Path(args.output_json),
        Path(args.output_csv),
    )
    print(f"Saved JSON: {Path(args.output_json).resolve()}")
    print(f"Saved CSV : {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
