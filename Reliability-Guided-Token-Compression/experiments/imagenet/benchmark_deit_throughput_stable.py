import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import timm

# Reuse the exact patching implementation from eval_deit_generality.py.
# Place this script in the same directory as eval_deit_generality.py.
from eval_deit_generality import apply_patch, beta_for_r, compute_final_tokens


def get_device(device_arg: str) -> torch.device:
    """
    Select CUDA device if available; otherwise fall back to CPU.
    """
    if "cuda" in device_arg and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_patched_model(
    model_name: str,
    method: str,
    r: int,
    beta: float,
    pretrained: bool,
    device: torch.device,
):
    """
    Build one patched timm model for throughput measurement.

    method:
        full_patched: ToMe-patched model with r=0
        tome        : original ToMe ranking
        ours        : confidence-aware calibrated ranking
    """
    if method == "full_patched":
        patch_method = "tome"
        model_r = 0
        model_beta = 0.0
    elif method == "tome":
        patch_method = "tome"
        model_r = int(r)
        model_beta = 0.0
    elif method == "ours":
        patch_method = "ours"
        model_r = int(r)
        model_beta = float(beta)
    else:
        raise ValueError(f"Unsupported method: {method}")

    model = timm.create_model(model_name, pretrained=pretrained)
    model = apply_patch(
        model=model,
        method=patch_method,
        beta=model_beta,
        prop_attn=True,
    )
    model.r = model_r
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def benchmark_cuda_event(
    model,
    device: torch.device,
    batch_size: int,
    input_size=(3, 224, 224),
    warmup: int = 50,
    runs: int = 200,
    repeats: int = 5,
    use_amp: bool = False,
):
    """
    Model-only throughput benchmark.

    It uses random input tensors, excludes dataloader and preprocessing overhead,
    and uses CUDA Event timing on GPU.
    """
    model.eval()
    x = torch.randn(batch_size, *input_size, device=device)

    throughputs = []

    # Let cuDNN select stable kernels.
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    for rep in range(repeats):
        # Warmup.
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            starter.record()
            for _ in range(runs):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _ = model(x)
            ender.record()

            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)
            elapsed_sec = elapsed_ms / 1000.0
        else:
            start = time.perf_counter()
            for _ in range(runs):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _ = model(x)
            elapsed_sec = time.perf_counter() - start

        tps = batch_size * runs / elapsed_sec
        throughputs.append(float(tps))
        print(f"[BENCH] repeat {rep + 1}/{repeats}: {tps:.2f} images/sec")

    arr = np.asarray(throughputs, dtype=np.float64)

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "all": [float(v) for v in throughputs],
    }


def run_one(args):
    """
    Run one model/method/r setting.
    """
    device = get_device(args.device)

    if args.method == "ours":
        beta = args.beta if args.beta is not None else beta_for_r(args.r)
    else:
        beta = 0.0

    # Compute final tokens with an unpatched lightweight model.
    schedule_model = timm.create_model(args.model_name, pretrained=False)
    final_tokens = compute_final_tokens(schedule_model, 0 if args.method == "full_patched" else args.r)
    del schedule_model

    print("\n========== Throughput Benchmark ==========")
    print(f"model       : {args.model_name}")
    print(f"method      : {args.method}")
    print(f"r           : {0 if args.method == 'full_patched' else args.r}")
    print(f"beta        : {beta}")
    print(f"final tokens: {final_tokens}")
    print(f"batch size  : {args.batch_size}")
    print(f"warmup/runs : {args.warmup}/{args.runs}")
    print(f"repeats     : {args.repeats}")
    print(f"device      : {device}")
    print(f"pretrained  : {not args.no_pretrained}")
    print("==========================================\n")

    model = build_patched_model(
        model_name=args.model_name,
        method=args.method,
        r=args.r,
        beta=beta,
        pretrained=not args.no_pretrained,
        device=device,
    )

    bench = benchmark_cuda_event(
        model=model,
        device=device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        repeats=args.repeats,
        use_amp=args.amp,
    )

    result = {
        "model": args.model_name,
        "method": args.method,
        "r": 0 if args.method == "full_patched" else int(args.r),
        "beta": beta,
        "final_tokens": int(final_tokens),
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "runs": int(args.runs),
        "repeats": int(args.repeats),
        "throughput_mean": bench["mean"],
        "throughput_std": bench["std"],
        "throughput_median": bench["median"],
        "throughput_min": bench["min"],
        "throughput_max": bench["max"],
        "throughput_all": bench["all"],
    }

    print("\n========== Result ==========")
    for k, v in result.items():
        print(f"{k}: {v}")
    print("============================\n")
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))

    return result


def build_child_command(args, model_name: str, method: str, r: int):
    """
    Build a subprocess command for one setting.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--model-name", model_name,
        "--method", method,
        "--r", str(r),
        "--batch-size", str(args.batch_size),
        "--warmup", str(args.warmup),
        "--runs", str(args.runs),
        "--repeats", str(args.repeats),
        "--device", args.device,
    ]

    if args.amp:
        cmd.append("--amp")
    if args.no_pretrained:
        cmd.append("--no-pretrained")
    if args.hf_offline:
        cmd.append("--hf-offline")

    # For Ours, use fixed beta map unless user forces a beta.
    if method == "ours" and args.beta is not None:
        cmd.extend(["--beta", str(args.beta)])

    return cmd


def run_all_by_subprocess(args):
    """
    Run multiple settings in separate subprocesses.

    This avoids contamination from long-running processes, CUDA cache state,
    and model replacement side effects.
    """
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    methods = args.methods
    models = args.models
    r_list = args.r_list

    rows = []

    for model_name in models:
        print(f"\n\n========== Model group: {model_name} ==========")

        # Always run full baseline first for speedup calculation.
        settings = []
        if "full_patched" in methods:
            settings.append(("full_patched", 0))

        for r in r_list:
            if "tome" in methods:
                settings.append(("tome", r))
            if "ours" in methods:
                settings.append(("ours", r))

        baseline_mean = None
        baseline_median = None

        for method, r in settings:
            print(f"\n========== Launch subprocess: {model_name}, {method}, r={r} ==========")
            cmd = build_child_command(args, model_name, method, r)
            print("[CMD]", " ".join(cmd))

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
                raise RuntimeError(f"Failed to parse RESULT_JSON for {model_name}, {method}, r={r}")

            if method == "full_patched":
                baseline_mean = result["throughput_mean"]
                baseline_median = result["throughput_median"]

            if baseline_mean is not None and baseline_mean > 0:
                result["speedup_mean"] = result["throughput_mean"] / baseline_mean
            else:
                result["speedup_mean"] = 1.0

            if baseline_median is not None and baseline_median > 0:
                result["speedup_median"] = result["throughput_median"] / baseline_median
            else:
                result["speedup_median"] = 1.0

            rows.append(result)

            # Save incrementally after every setting.
            write_csv(args.out_csv, rows)

    print(f"\nSaved all throughput results to: {args.out_csv}")


def write_csv(path: str, rows: List[Dict]):
    """
    Incrementally save CSV.
    """
    if len(rows) == 0:
        return

    fieldnames = [
        "model",
        "method",
        "r",
        "beta",
        "final_tokens",
        "batch_size",
        "warmup",
        "runs",
        "repeats",
        "throughput_mean",
        "throughput_std",
        "throughput_median",
        "throughput_min",
        "throughput_max",
        "speedup_mean",
        "speedup_median",
        "throughput_all",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()

    # Mode.
    parser.add_argument("--single", action="store_true",
                        help="Run a single setting. Internal use for subprocess mode.")

    # Single-setting arguments.
    parser.add_argument("--model-name", type=str, default="deit_small_patch16_224")
    parser.add_argument("--method", type=str, default="full_patched",
                        choices=["full_patched", "tome", "ours"])
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--beta", type=float, default=None)

    # Multi-setting arguments.
    parser.add_argument("--models", type=str, nargs="+",
                        default=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["full_patched", "tome", "ours"],
                        choices=["full_patched", "tome", "ours"])
    parser.add_argument("--r-list", type=int, nargs="+",
                        default=[4, 8, 12, 16, 20, 25])

    # Benchmark arguments.
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")

    # HuggingFace cache.
    parser.add_argument("--hf-offline", action="store_true",
                        help="Set HF_HUB_OFFLINE=1 before subprocesses.")

    # Output.
    parser.add_argument("--out-csv", type=str, default="deit_throughput_stable.csv")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.single:
        run_one(args)
    else:
        run_all_by_subprocess(args)


if __name__ == "__main__":
    main()
