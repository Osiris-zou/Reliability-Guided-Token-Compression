import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import timm

# Put this script in the same directory as eval_deit_generality.py.
# It reuses the same patch implementation used in your DeiT experiments.
from eval_deit_generality import apply_patch, beta_for_r, compute_final_tokens


def set_stable_flags():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device(device_arg: str):
    if "cuda" in device_arg and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable. Use CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model(model_name: str, method: str, r: int, beta: float, pretrained: bool, device):
    model = timm.create_model(model_name, pretrained=pretrained)

    if method == "full_patched":
        model = apply_patch(model=model, method="tome", beta=0.0, prop_attn=True)
        model.r = 0
    elif method == "tome":
        model = apply_patch(model=model, method="tome", beta=0.0, prop_attn=True)
        model.r = int(r)
    elif method == "ours":
        model = apply_patch(model=model, method="ours", beta=float(beta), prop_attn=True)
        model.r = int(r)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.to(device).eval()
    return model


def get_final_tokens(model_name: str, method: str, r: int):
    rr = 0 if method == "full_patched" else int(r)
    m = timm.create_model(model_name, pretrained=False)
    final_tokens = compute_final_tokens(m, rr)
    del m
    return int(final_tokens)


@torch.no_grad()
def robust_chunk_benchmark(
    model,
    x,
    warmup: int,
    chunks: int,
    chunk_size: int,
    trim_ratio: float,
    use_amp: bool,
):
    """
    Robust model-only latency benchmark.

    Instead of measuring one long block per repeat, this measures many short chunks.
    Each chunk contains chunk_size forwards. This exposes instability and allows
    median / trimmed-mean statistics.
    """
    device = x.device
    batch_size = x.shape[0]

    # Long warmup for small models.
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    ms_per_forward = []

    for i in range(chunks):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(chunk_size):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _ = model(x)
            end.record()

            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            for _ in range(chunk_size):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _ = model(x)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        one_ms = elapsed_ms / chunk_size
        ms_per_forward.append(float(one_ms))

        if (i + 1) % max(1, chunks // 10) == 0:
            print(f"chunk {i + 1:03d}/{chunks}: {one_ms:.4f} ms/forward")

    arr = np.asarray(ms_per_forward, dtype=np.float64)

    # Robust statistics.
    p10 = float(np.percentile(arr, 10))
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p90 = float(np.percentile(arr, 90))

    lo = np.percentile(arr, trim_ratio * 100)
    hi = np.percentile(arr, (1.0 - trim_ratio) * 100)
    trimmed = arr[(arr >= lo) & (arr <= hi)]

    mean_ms = float(arr.mean())
    std_ms = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    median_ms = p50
    trimmed_mean_ms = float(trimmed.mean()) if len(trimmed) > 0 else mean_ms
    trimmed_std_ms = float(trimmed.std(ddof=1)) if len(trimmed) > 1 else 0.0

    # IQR outlier count.
    iqr = p75 - p25
    lower_fence = p25 - 1.5 * iqr
    upper_fence = p75 + 1.5 * iqr
    outlier_mask = (arr < lower_fence) | (arr > upper_fence)
    outlier_count = int(outlier_mask.sum())

    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_std": std_ms,
        "latency_ms_median": median_ms,
        "latency_ms_trimmed_mean": trimmed_mean_ms,
        "latency_ms_trimmed_std": trimmed_std_ms,
        "latency_ms_p10": p10,
        "latency_ms_p25": p25,
        "latency_ms_p50": p50,
        "latency_ms_p75": p75,
        "latency_ms_p90": p90,
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
        "outlier_count": outlier_count,
        "throughput_mean": float(batch_size * 1000.0 / mean_ms),
        "throughput_median": float(batch_size * 1000.0 / median_ms),
        "throughput_trimmed_mean": float(batch_size * 1000.0 / trimmed_mean_ms),
        "latency_ms_all": [float(v) for v in ms_per_forward],
    }


def run_single(args):
    set_stable_flags()
    device = get_device(args.device)

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    beta = beta_for_r(args.r) if args.method == "ours" else 0.0
    final_tokens = get_final_tokens(args.model_name, args.method, args.r)

    print("\n========== Single Robust Benchmark ==========")
    print(f"model       : {args.model_name}")
    print(f"method      : {args.method}")
    print(f"r           : {args.r}")
    print(f"beta        : {beta}")
    print(f"batch size  : {args.batch_size}")
    print(f"final tokens: {final_tokens}")
    print(f"warmup      : {args.warmup}")
    print(f"chunks      : {args.chunks}")
    print(f"chunk size  : {args.chunk_size}")
    print(f"trim ratio  : {args.trim_ratio}")
    print("=============================================\n")

    model = build_model(
        model_name=args.model_name,
        method=args.method,
        r=args.r,
        beta=beta,
        pretrained=not args.no_pretrained,
        device=device,
    )
    x = torch.randn(args.batch_size, 3, 224, 224, device=device)

    result = robust_chunk_benchmark(
        model=model,
        x=x,
        warmup=args.warmup,
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        trim_ratio=args.trim_ratio,
        use_amp=args.amp,
    )

    row = {
        "model": args.model_name,
        "method": args.method,
        "r": int(args.r),
        "beta": float(beta),
        "batch_size": int(args.batch_size),
        "final_tokens": int(final_tokens),
        **result,
    }

    print("\n========== Robust Result ==========")
    for k, v in row.items():
        if k != "latency_ms_all":
            print(f"{k}: {v}")
    print("===================================")
    print("RESULT_JSON:" + json.dumps(row, ensure_ascii=False))


def build_cmd(args, method: str, r: int, batch_size: int):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--model-name", args.model_name,
        "--method", method,
        "--r", str(r),
        "--batch-size", str(batch_size),
        "--warmup", str(args.warmup),
        "--chunks", str(args.chunks),
        "--chunk-size", str(args.chunk_size),
        "--trim-ratio", str(args.trim_ratio),
        "--device", args.device,
    ]
    if args.amp:
        cmd.append("--amp")
    if args.no_pretrained:
        cmd.append("--no-pretrained")
    if args.hf_offline:
        cmd.append("--hf-offline")
    return cmd


def write_csv(path: str, rows: List[Dict]):
    fieldnames = [
        "model", "method", "r", "beta", "batch_size", "final_tokens",
        "latency_ms_mean", "latency_ms_std",
        "latency_ms_median", "latency_ms_trimmed_mean", "latency_ms_trimmed_std",
        "latency_ms_p10", "latency_ms_p25", "latency_ms_p50",
        "latency_ms_p75", "latency_ms_p90", "latency_ms_min", "latency_ms_max",
        "outlier_count",
        "throughput_mean", "throughput_median", "throughput_trimmed_mean",
        "speedup_trimmed_vs_full",
        "latency_ms_all",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_sweep(args):
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    rows = []

    # Keep order deterministic. Run full baseline first for each batch size.
    for batch_size in args.batch_sizes:
        print(f"\n\n================ Batch size {batch_size} ================")

        settings = [("full_patched", 0)]
        for r in args.r_list:
            settings.append(("tome", r))
            settings.append(("ours", r))

        full_trimmed_tps = None

        for method, r in settings:
            cmd = build_cmd(args, method, r, batch_size)
            print("\n[CMD]", " ".join(cmd))

            proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            print(proc.stdout)

            result = None
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT_JSON:"):
                    result = json.loads(line[len("RESULT_JSON:"):])
                    break

            if result is None:
                raise RuntimeError(f"Cannot parse result for method={method}, r={r}, batch={batch_size}")

            if method == "full_patched":
                full_trimmed_tps = float(result["throughput_trimmed_mean"])
                result["speedup_trimmed_vs_full"] = 1.0
            else:
                result["speedup_trimmed_vs_full"] = float(result["throughput_trimmed_mean"]) / full_trimmed_tps

            rows.append(result)
            write_csv(args.out_csv, rows)

    print(f"\nSaved robust diagnostic CSV to: {args.out_csv}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--single", action="store_true")
    parser.add_argument("--model-name", type=str, default="deit_small_patch16_224")
    parser.add_argument("--method", type=str, default="full_patched",
                        choices=["full_patched", "tome", "ours"])
    parser.add_argument("--r", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--r-list", type=int, nargs="+", default=[4, 8, 12, 16, 20, 25])

    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--chunks", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--trim-ratio", type=float, default=0.20)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--hf-offline", action="store_true")

    parser.add_argument("--out-csv", type=str, default="deit_small_robust_latency_diagnostic.csv")

    args = parser.parse_args()

    if args.single:
        run_single(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
