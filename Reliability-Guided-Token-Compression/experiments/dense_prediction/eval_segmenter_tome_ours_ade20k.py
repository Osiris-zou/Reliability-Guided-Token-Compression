import argparse
import csv
import json
import math
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import segm.utils.torch as ptu
from segm.model.factory import load_model


SCRIPT_VERSION = "segmenter-tome-ours-ade20k-v1-20260607"

NORMALIZATION_STATS = {
    "vit": {
        "mean": torch.tensor([127.5, 127.5, 127.5]).view(3, 1, 1),
        "std": torch.tensor([127.5, 127.5, 127.5]).view(3, 1, 1),
    },
    "deit": {
        "mean": torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1),
        "std": torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1),
    },
}


# ============================================================
# Dataset and metric utilities
# ============================================================
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
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        pairs.append((image_path, mask_path))

    return pairs


def resize_shape(height: int, width: int, short_side: int, max_long_side: int):
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
):
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

    stats = NORMALIZATION_STATS[normalization]
    tensor = (tensor - stats["mean"]) / stats["std"]

    return tensor.unsqueeze(0), (original_height, original_width)


def load_ground_truth(mask_path: Path):
    with Image.open(mask_path) as mask:
        return np.asarray(mask, dtype=np.int64).copy()


def sliding_positions(length: int, window_size: int, stride: int):
    if length <= window_size:
        return [0]

    positions = list(range(0, length - window_size + 1, stride))
    final_position = length - window_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def pad_to_window(image: torch.Tensor, window_size: int):
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
):
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
                image[:, :, h : h + window_size, w : w + window_size]
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
            logit_sum[:, :, h : h + window_size, w : w + window_size] += logits[
                index : index + 1
            ]
            count[:, :, h : h + window_size, w : w + window_size] += 1.0

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


def update_confusion_matrix(
    confusion: np.ndarray,
    prediction: np.ndarray,
    ground_truth_raw: np.ndarray,
    n_classes: int,
):
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


def calculate_metrics(confusion: np.ndarray):
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


# ============================================================
# ToMe / Ours merge plan
# ============================================================
@dataclass
class MergePlan:
    original_tokens: int
    removed_tokens: int
    unm_idx: torch.Tensor
    src_idx: torch.Tensor
    dst_idx: torch.Tensor

    def _expand(self, index: torch.Tensor, reference: torch.Tensor):
        shape = [index.shape[0], index.shape[1]] + list(reference.shape[2:])
        view_shape = [index.shape[0], index.shape[1]] + [1] * (reference.ndim - 2)
        return index.view(*view_shape).expand(*shape)

    def merge_sum(self, x: torch.Tensor):
        if self.removed_tokens <= 0:
            return x

        src = x[:, ::2]
        dst = x[:, 1::2]

        unm = torch.gather(src, dim=1, index=self._expand(self.unm_idx, src))
        selected = torch.gather(src, dim=1, index=self._expand(self.src_idx, src))

        dst = dst.scatter_reduce(
            dim=1,
            index=self._expand(self.dst_idx, selected),
            src=selected,
            reduce="sum",
            include_self=True,
        )

        return torch.cat([unm, dst], dim=1)

    def merge_weighted(self, x: torch.Tensor, size: Optional[torch.Tensor]):
        if size is None:
            size = torch.ones_like(x[..., :1])

        if self.removed_tokens <= 0:
            return x, size

        merged_sum = self.merge_sum(x * size)
        merged_size = self.merge_sum(size)
        merged = merged_sum / merged_size.clamp_min(1e-6)
        return merged, merged_size

    def unmerge(self, x: torch.Tensor):
        if self.removed_tokens <= 0:
            return x

        unm_count = self.unm_idx.shape[1]

        unm = x[:, :unm_count]
        dst = x[:, unm_count:]

        reconstructed_src = torch.gather(
            dst,
            dim=1,
            index=self._expand(self.dst_idx, dst),
        )

        output = torch.zeros(
            x.shape[0],
            self.original_tokens,
            *x.shape[2:],
            dtype=x.dtype,
            device=x.device,
        )

        output[:, 1::2] = dst

        even_unm = 2 * self.unm_idx
        even_src = 2 * self.src_idx

        output.scatter_(dim=1, index=self._expand(even_unm, unm), src=unm)
        output.scatter_(
            dim=1,
            index=self._expand(even_src, reconstructed_src),
            src=reconstructed_src,
        )

        return output


def build_merge_plan(
    metric: torch.Tensor,
    r: int,
    method: str,
    beta: float,
    protect_cls: bool = True,
):
    batch, token_count, _ = metric.shape

    source_count = (token_count + 1) // 2
    target_count = token_count // 2
    protected_source = 1 if protect_cls else 0

    actual_r = min(max(int(r), 0), max(0, source_count - protected_source), target_count)

    if actual_r <= 0:
        empty = torch.empty(batch, 0, 1, dtype=torch.long, device=metric.device)
        all_source = torch.arange(
            source_count, dtype=torch.long, device=metric.device
        ).view(1, source_count, 1).expand(batch, -1, -1)
        return MergePlan(
            original_tokens=token_count,
            removed_tokens=0,
            unm_idx=all_source,
            src_idx=empty,
            dst_idx=empty,
        )

    with torch.no_grad():
        metric = F.normalize(metric, dim=-1, eps=1e-6)
        source = metric[:, ::2]
        target = metric[:, 1::2]

        scores = source @ target.transpose(-1, -2)
        scores = torch.nan_to_num(
            scores,
            nan=-math.inf,
            posinf=math.inf,
            neginf=-math.inf,
        )

        top1_score, target_assignment = scores.max(dim=-1)

        if method == "tome":
            ranking_score = top1_score
        elif method == "ours":
            top_k = min(2, scores.shape[-1])
            values = scores.topk(top_k, dim=-1).values
            best = values[..., 0]
            second = values[..., 1] if top_k == 2 else torch.zeros_like(best)
            margin = best - second
            margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
            ranking_score = top1_score + float(beta) * margin
        else:
            raise ValueError(f"Unknown method: {method}")

        if protect_cls:
            ranking_score[:, 0] = -math.inf

        edge_order = ranking_score.argsort(dim=-1, descending=True).unsqueeze(-1)

        src_idx = edge_order[:, :actual_r]
        unm_idx = edge_order[:, actual_r:].sort(dim=1).values
        dst_idx = target_assignment.unsqueeze(-1).gather(dim=1, index=src_idx)

    return MergePlan(
        original_tokens=token_count,
        removed_tokens=actual_r,
        unm_idx=unm_idx,
        src_idx=src_idx,
        dst_idx=dst_idx,
    )


def restore_tokens(x: torch.Tensor, history: List[MergePlan]):
    restored = x
    for plan in reversed(history):
        restored = plan.unmerge(restored)
    return restored


# ============================================================
# Segmenter patching
# ============================================================
def call_attention(attn, x):
    try:
        return attn(x)
    except TypeError:
        return attn(x, None)


def compute_key_metric(attn, normalized_tokens: torch.Tensor):
    if hasattr(attn, "qkv"):
        qkv = attn.qkv(normalized_tokens)
        batch, tokens, three_c = qkv.shape
        channels = three_c // 3

        num_heads = int(getattr(attn, "num_heads", getattr(attn, "heads", 1)))
        head_dim = channels // num_heads

        key = qkv[:, :, channels : 2 * channels]
        if key.shape[-1] % num_heads == 0:
            key = key.reshape(batch, tokens, num_heads, head_dim)
            return key.mean(dim=2)
        return key

    return normalized_tokens


def block_forward_with_merge(self, x: torch.Tensor):
    info = self._rg_info
    method = info["method"]
    r = int(info["r_schedule"].pop(0)) if info["r_schedule"] else 0
    beta = float(info["beta"])

    norm1 = self.norm1(x)
    metric = compute_key_metric(self.attn, norm1)

    attn_out = call_attention(self.attn, norm1)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]

    drop_path1 = getattr(self, "drop_path1", getattr(self, "drop_path", torch.nn.Identity()))
    drop_path2 = getattr(self, "drop_path2", getattr(self, "drop_path", torch.nn.Identity()))

    x = x + drop_path1(attn_out)

    if r > 0:
        plan = build_merge_plan(
            metric=metric,
            r=r,
            method=method,
            beta=beta,
            protect_cls=bool(info["protect_cls"]),
        )
        x, info["size"] = plan.merge_weighted(x, info["size"])
        info["history"].append(plan)
        info["final_tokens"] = int(x.shape[1])

    x = x + drop_path2(self.mlp(self.norm2(x)))
    return x


def patch_encoder_forward(encoder):
    if hasattr(encoder, "_rg_original_forward"):
        return

    encoder._rg_original_forward = encoder.forward

    def patched_forward(self, *args, **kwargs):
        info = self._rg_info
        info["r_schedule"] = list(info["r_base_schedule"])
        info["size"] = None
        info["history"] = []
        info["final_tokens"] = None

        out = self._rg_original_forward(*args, **kwargs)

        def restore_output(obj: Any):
            if isinstance(obj, torch.Tensor) and obj.ndim == 3 and info["history"]:
                return restore_tokens(obj, info["history"])
            if isinstance(obj, tuple):
                return tuple(restore_output(item) for item in obj)
            if isinstance(obj, list):
                return [restore_output(item) for item in obj]
            return obj

        return restore_output(out)

    encoder.forward = types.MethodType(patched_forward, encoder)


def apply_segmenter_token_merging(model, method: str, r: int, beta: float):
    if not hasattr(model, "encoder"):
        raise RuntimeError("The loaded Segmenter model does not expose model.encoder.")

    encoder = model.encoder
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise RuntimeError("The Segmenter encoder does not expose encoder.blocks.")

    protect_cls = hasattr(encoder, "cls_token")
    depth = len(blocks)

    encoder._rg_info = {
        "method": method,
        "r": int(r),
        "beta": float(beta),
        "r_base_schedule": [int(r)] * depth,
        "r_schedule": [int(r)] * depth,
        "size": None,
        "history": [],
        "final_tokens": None,
        "protect_cls": protect_cls,
    }

    patch_encoder_forward(encoder)

    patched_blocks = 0
    for block in blocks:
        required = ["norm1", "attn", "norm2", "mlp"]
        if not all(hasattr(block, name) for name in required):
            raise RuntimeError(
                f"Cannot patch block {block.__class__.__name__}; missing one of {required}."
            )

        if not hasattr(block, "_rg_original_forward"):
            block._rg_original_forward = block.forward
        block._rg_info = encoder._rg_info
        block.forward = types.MethodType(block_forward_with_merge, block)
        patched_blocks += 1

    print(
        f"[Patch] method={method}, r={r}, beta={beta}, "
        f"blocks={patched_blocks}, protect_cls={protect_cls}"
    )


# ============================================================
# Evaluation and benchmarking
# ============================================================
@dataclass
class EvalResult:
    method: str
    r: int
    beta: float
    images: int
    aAcc: float
    mAcc: float
    mIoU: float
    evaluated_classes: int
    valid_pixels: int
    elapsed_seconds: float
    e2e_fps: float
    throughput_mean: float
    throughput_std: float
    throughput_median: float
    throughput_min: float
    throughput_max: float
    speedup: float
    peak_memory_gb: float
    memory_reduction: float
    final_tokens: int
    normalization: str
    script_version: str


def evaluate_ade20k(
    model,
    pairs,
    device,
    n_classes,
    normalization,
    args,
):
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

        update_confusion_matrix(confusion, prediction, ground_truth, n_classes)

        if index % 20 == 0 or index == len(pairs):
            partial = calculate_metrics(confusion)
            progress.set_postfix(
                mIoU=f"{partial['mIoU']:.2f}",
                aAcc=f"{partial['aAcc']:.2f}",
            )

    elapsed = time.perf_counter() - start_time
    metrics = calculate_metrics(confusion)

    return metrics, elapsed


@torch.inference_mode()
def benchmark_once(model, device, batch_size, warmup, runs, amp):
    model.eval()
    x = torch.randn(batch_size, 3, 512, 512, device=device)

    for _ in range(warmup):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        for _ in range(runs):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                _ = model(x)

        end.record()
        torch.cuda.synchronize(device)
        elapsed = start.elapsed_time(end) / 1000.0
    else:
        start_time = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        elapsed = time.perf_counter() - start_time

    return batch_size * runs / elapsed


def benchmark_model(model, device, batch_size, warmup, runs, repeats, amp):
    values = []
    for i in range(repeats):
        value = benchmark_once(model, device, batch_size, warmup, runs, amp)
        values.append(value)
        print(f"Throughput repeat {i + 1}/{repeats}: {value:.3f} img/s")

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


@torch.inference_mode()
def measure_peak_memory(model, device, batch_size, amp):
    if device.type != "cuda":
        return -1.0

    x = torch.randn(batch_size, 3, 512, 512, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    ):
        _ = model(x)

    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024**3))


def save_rows(rows, csv_path: Path, json_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    row_dicts = [asdict(row) for row in rows]

    if row_dicts:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(row_dicts[0].keys()))
            writer.writeheader()
            writer.writerows(row_dicts)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(row_dicts, file, indent=2, ensure_ascii=False)

    print(f"Saved CSV : {csv_path.resolve()}")
    print(f"Saved JSON: {json_path.resolve()}")


def load_segmenter_model(checkpoint: Path, device: torch.device):
    model, variant = load_model(str(checkpoint))
    model = model.to(device).eval()
    normalization = str(variant.get("dataset_kwargs", {}).get("normalization", "vit"))
    if normalization not in NORMALIZATION_STATS:
        raise ValueError(
            f"Unsupported normalization '{normalization}'. "
            f"Available: {sorted(NORMALIZATION_STATS)}"
        )
    return model, variant, normalization


def run_setting(
    args,
    pairs,
    device,
    method: str,
    r: int,
    beta: float,
    full_throughput: Optional[float],
    full_memory: Optional[float],
):
    checkpoint = Path(args.checkpoint).resolve()

    print("\n" + "=" * 100)
    print(f"Method={method} | r={r} | beta={beta}")
    print("=" * 100)

    model, variant, normalization = load_segmenter_model(checkpoint, device)

    if method in {"tome", "ours"}:
        apply_segmenter_token_merging(model, method=method, r=r, beta=beta)

    n_classes = int(model.n_cls)

    metrics, elapsed = evaluate_ade20k(
        model=model,
        pairs=pairs,
        device=device,
        n_classes=n_classes,
        normalization=normalization,
        args=args,
    )

    if args.skip_throughput:
        throughput = {
            "mean": -1.0,
            "std": 0.0,
            "median": -1.0,
            "min": -1.0,
            "max": -1.0,
        }
    else:
        throughput = benchmark_model(
            model=model,
            device=device,
            batch_size=args.benchmark_batch_size,
            warmup=args.throughput_warmup,
            runs=args.throughput_runs,
            repeats=args.throughput_repeats,
            amp=args.amp,
        )

    peak_memory = -1.0
    if not args.skip_memory:
        peak_memory = measure_peak_memory(
            model=model,
            device=device,
            batch_size=args.benchmark_batch_size,
            amp=args.amp,
        )

    if full_throughput and full_throughput > 0 and throughput["mean"] > 0:
        speedup = throughput["mean"] / full_throughput
    else:
        speedup = 1.0

    if full_memory and full_memory > 0 and peak_memory > 0:
        memory_reduction = (full_memory - peak_memory) / full_memory * 100.0
    else:
        memory_reduction = 0.0

    final_tokens = 1025
    if method in {"tome", "ours"}:
        info = getattr(model.encoder, "_rg_info", {})
        if info.get("final_tokens") is not None:
            final_tokens = int(info["final_tokens"])

    result = EvalResult(
        method={"full": "Full", "tome": "ToMe", "ours": "Ours"}[method],
        r=int(r),
        beta=float(beta),
        images=len(pairs),
        aAcc=float(metrics["aAcc"]),
        mAcc=float(metrics["mAcc"]),
        mIoU=float(metrics["mIoU"]),
        evaluated_classes=int(metrics["evaluated_classes"]),
        valid_pixels=int(metrics["valid_pixels"]),
        elapsed_seconds=float(elapsed),
        e2e_fps=float(len(pairs) / elapsed if elapsed > 0 else float("nan")),
        throughput_mean=float(throughput["mean"]),
        throughput_std=float(throughput["std"]),
        throughput_median=float(throughput["median"]),
        throughput_min=float(throughput["min"]),
        throughput_max=float(throughput["max"]),
        speedup=float(speedup),
        peak_memory_gb=float(peak_memory),
        memory_reduction=float(memory_reduction),
        final_tokens=int(final_tokens),
        normalization=normalization,
        script_version=SCRIPT_VERSION,
    )

    print("\nResult:")
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def parse_beta_map(items):
    mapping = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid beta-map item: {item}. Expected r:beta.")
        r_text, beta_text = item.split(":", 1)
        mapping[int(r_text)] = float(beta_text)
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Segmenter ADE20K Full / ToMe / Ours dense prediction evaluation."
    )

    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--ade20k-root", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)

    parser.add_argument("--r-list", nargs="+", type=int, default=[16, 32, 48])
    parser.add_argument(
        "--beta-map",
        nargs="+",
        default=["16:0.015", "32:0.035", "48:0.050"],
    )

    parser.add_argument("--short-side", default=512, type=int)
    parser.add_argument("--max-long-side", default=2048, type=int)
    parser.add_argument("--window-size", default=512, type=int)
    parser.add_argument("--window-stride", default=480, type=int)
    parser.add_argument("--window-batch-size", default=4, type=int)
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--benchmark-batch-size", default=1, type=int)
    parser.add_argument("--throughput-warmup", default=20, type=int)
    parser.add_argument("--throughput-runs", default=50, type=int)
    parser.add_argument("--throughput-repeats", default=5, type=int)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")

    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--full-throughput", default=-1.0, type=float)
    parser.add_argument("--full-memory", default=-1.0, type=float)

    parser.add_argument(
        "--out-csv",
        default="segmenter_tome_ours_ade20k_results.csv",
        type=str,
    )
    parser.add_argument(
        "--out-json",
        default="segmenter_tome_ours_ade20k_results.json",
        type=str,
    )

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    variant = checkpoint.parent / "variant.yml"
    ade20k_root = Path(args.ade20k_root).resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not variant.is_file():
        raise FileNotFoundError(f"variant.yml not found beside checkpoint: {variant}")

    use_cuda = args.device.startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    ptu.set_gpu_mode(use_cuda)
    device = torch.device(args.device if use_cuda else "cpu")
    ptu.device = device

    pairs = resolve_validation_pairs(ade20k_root)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    beta_map = parse_beta_map(args.beta_map)

    print("========== Segmenter ADE20K ToMe/Ours Experiment ==========")
    print(f"Script version       : {SCRIPT_VERSION}")
    print(f"Checkpoint           : {checkpoint}")
    print(f"ADE20K root          : {ade20k_root}")
    print(f"Images               : {len(pairs)}")
    print(f"Device               : {device}")
    print(f"GPU                  : {torch.cuda.get_device_name(device) if use_cuda else 'CPU'}")
    print(f"r list               : {args.r_list}")
    print(f"beta map             : {beta_map}")
    print(f"Window size/stride   : {args.window_size}/{args.window_stride}")
    print(f"Window batch size    : {args.window_batch_size}")
    print(f"Benchmark batch size : {args.benchmark_batch_size}")
    print(f"AMP                  : {args.amp}")
    print("===========================================================")

    rows: List[EvalResult] = []

    full_throughput = args.full_throughput if args.full_throughput > 0 else None
    full_memory = args.full_memory if args.full_memory > 0 else None

    if not args.skip_full:
        full = run_setting(
            args=args,
            pairs=pairs,
            device=device,
            method="full",
            r=0,
            beta=0.0,
            full_throughput=None,
            full_memory=None,
        )
        rows.append(full)
        save_rows(rows, Path(args.out_csv), Path(args.out_json))

        full_throughput = full.throughput_mean
        full_memory = full.peak_memory_gb

    for r in args.r_list:
        tome = run_setting(
            args=args,
            pairs=pairs,
            device=device,
            method="tome",
            r=r,
            beta=0.0,
            full_throughput=full_throughput,
            full_memory=full_memory,
        )
        rows.append(tome)
        save_rows(rows, Path(args.out_csv), Path(args.out_json))

        if r not in beta_map:
            raise ValueError(f"No beta supplied for r={r}. beta_map={beta_map}")

        ours = run_setting(
            args=args,
            pairs=pairs,
            device=device,
            method="ours",
            r=r,
            beta=beta_map[r],
            full_throughput=full_throughput,
            full_memory=full_memory,
        )
        rows.append(ours)
        save_rows(rows, Path(args.out_csv), Path(args.out_json))

    print("All requested Segmenter ADE20K experiments completed.")


if __name__ == "__main__":
    main()
