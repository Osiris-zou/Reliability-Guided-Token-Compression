import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import timm

# Put this script in the same directory as eval_deit_generality.py.
# It reuses the exact patch implementation used in your DeiT experiments.
from eval_deit_generality import apply_patch, beta_for_r, compute_final_tokens


# -----------------------------
# Utility
# -----------------------------
def set_stable_cuda_flags():
    """
    Make CUDA benchmark behavior more stable.
    """
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device(device_arg: str) -> torch.device:
    """
    Select CUDA device if available.
    """
    if "cuda" in device_arg and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model(model_name: str, method: str, r: int, beta: float, pretrained: bool, device: torch.device):
    """
    Build one model for diagnostic benchmark.

    method:
        full_native  : original timm model without ToMe patch
        full_patched : ToMe-patched model with r = 0
        tome         : ToMe with r > 0
        ours         : proposed confidence-aware method with r > 0
    """
    model = timm.create_model(model_name, pretrained=pretrained)

    if method == "full_native":
        model.to(device).eval()
        return model

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
        raise ValueError(f"Unsupported method: {method}")

    model.to(device).eval()
    return model


def get_final_tokens(model_name: str, method: str, r: int) -> int:
    """
    Compute final token count.
    """
    if method in ["full_native", "full_patched"]:
        rr = 0
    else:
        rr = r

    m = timm.create_model(model_name, pretrained=False)
    final_tokens = compute_final_tokens(m, rr)
    del m
    return int(final_tokens)


@torch.no_grad()
def benchmark_forward_ms(model, x, warmup: int, runs: int, repeats: int, use_amp: bool = False):
    """
    Measure model-only forward latency and throughput using CUDA events.

    Returns:
        mean_ms, std_ms, median_ms, throughput_mean, all_ms
    """
    device = x.device
    batch_size = x.shape[0]

    all_ms = []

    for rep in range(repeats):
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(runs):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _ = model(x)
            end.record()

            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            for _ in range(runs):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _ = model(x)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        per_forward_ms = elapsed_ms / runs
        all_ms.append(float(per_forward_ms))
        print(f"repeat {rep + 1}/{repeats}: {per_forward_ms:.4f} ms/forward, "
              f"{batch_size * 1000.0 / per_forward_ms:.2f} img/s")

    arr = np.asarray(all_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    std_ms = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    median_ms = float(np.median(arr))
    tps_mean = float(batch_size * 1000.0 / mean_ms)

    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_std": std_ms,
        "latency_ms_median": median_ms,
        "throughput_mean": tps_mean,
        "latency_ms_all": [float(v) for v in all_ms],
    }


def categorize_op(op_name: str) -> str:
    """
    Categorize profiler operators into rough runtime groups.

    The merge-related group is the key diagnostic target.
    """
    name = op_name.lower()

    if any(k in name for k in ["topk", "sort", "argsort"]):
        return "merge_sort_topk"
    if any(k in name for k in ["scatter", "gather", "index_select", "index", "where", "cat"]):
        return "merge_gather_scatter"
    if any(k in name for k in ["norm", "linalg_vector_norm", "div", "clamp", "isnan", "isinf"]):
        return "merge_norm_stabilize"
    if any(k in name for k in ["bmm", "matmul", "mm", "addmm", "linear"]):
        return "matmul_linear"
    if "softmax" in name:
        return "attention_softmax"
    if any(k in name for k in ["layer_norm", "native_layer_norm"]):
        return "layer_norm"
    if any(k in name for k in ["gelu", "dropout"]):
        return "mlp_activation_dropout"
    if any(k in name for k in ["copy", "contiguous", "transpose", "permute", "reshape", "view", "slice"]):
        return "tensor_reshape_copy"
    return "other"


@torch.no_grad()
def profile_ops(model, x, warmup: int, steps: int, use_amp: bool = False):
    """
    Run PyTorch profiler and aggregate CUDA time by operation group.

    This is slower than normal benchmarking, so use it only for a few settings.
    """
    device = x.device

    for _ in range(warmup):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    op_rows = []
    group_time = {}

    for evt in prof.key_averages():
        # CUDA time is in microseconds. CPU time is also in microseconds.
        cuda_ms = float(getattr(evt, "cuda_time_total", 0.0)) / 1000.0
        cpu_ms = float(getattr(evt, "cpu_time_total", 0.0)) / 1000.0
        count = int(evt.count)

        group = categorize_op(evt.key)
        group_time[group] = group_time.get(group, 0.0) + cuda_ms

        op_rows.append({
            "op": evt.key,
            "group": group,
            "cuda_time_total_ms": cuda_ms,
            "cpu_time_total_ms": cpu_ms,
            "count": count,
        })

    total_cuda_ms = sum(group_time.values())
    group_rows = []
    for group, ms in sorted(group_time.items(), key=lambda kv: kv[1], reverse=True):
        group_rows.append({
            "group": group,
            "cuda_time_total_ms": ms,
            "cuda_time_pct": (ms / total_cuda_ms * 100.0) if total_cuda_ms > 0 else 0.0,
        })

    op_rows = sorted(op_rows, key=lambda r: r["cuda_time_total_ms"], reverse=True)

    return group_rows, op_rows


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    """
    Save rows to CSV.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_latency_sweep(args):
    """
    Run throughput/latency sweep over r and methods.
    """
    set_stable_cuda_flags()
    device = get_device(args.device)

    rows = []

    for batch_size in args.batch_sizes:
        print(f"\n\n================ Batch size = {batch_size} ================")
        x = torch.randn(batch_size, 3, 224, 224, device=device)

        # Full baselines.
        baseline_ms = None
        baseline_tps = None

        settings = [("full_patched", 0)]
        for r in args.r_list:
            settings.append(("tome", r))
            settings.append(("ours", r))

        for method, r in settings:
            beta = beta_for_r(r) if method == "ours" else 0.0
            final_tokens = get_final_tokens(args.model_name, method, r)

            print("\n----------------------------------------")
            print(f"model={args.model_name}, method={method}, r={r}, beta={beta}, "
                  f"batch={batch_size}, final_tokens={final_tokens}")
            print("----------------------------------------")

            model = build_model(
                model_name=args.model_name,
                method=method,
                r=r,
                beta=beta,
                pretrained=not args.no_pretrained,
                device=device,
            )

            result = benchmark_forward_ms(
                model=model,
                x=x,
                warmup=args.warmup,
                runs=args.runs,
                repeats=args.repeats,
                use_amp=args.amp,
            )

            if method == "full_patched":
                baseline_ms = result["latency_ms_mean"]
                baseline_tps = result["throughput_mean"]

            latency_speedup = baseline_ms / result["latency_ms_mean"] if baseline_ms else 1.0
            throughput_speedup = result["throughput_mean"] / baseline_tps if baseline_tps else 1.0

            row = {
                "model": args.model_name,
                "batch_size": batch_size,
                "method": method,
                "r": r,
                "beta": beta,
                "final_tokens": final_tokens,
                "latency_ms_mean": result["latency_ms_mean"],
                "latency_ms_std": result["latency_ms_std"],
                "latency_ms_median": result["latency_ms_median"],
                "throughput_mean": result["throughput_mean"],
                "latency_speedup": latency_speedup,
                "throughput_speedup": throughput_speedup,
                "latency_ms_all": json.dumps(result["latency_ms_all"]),
            }
            rows.append(row)

            # Save incrementally.
            write_csv(
                args.out_latency_csv,
                rows,
                [
                    "model", "batch_size", "method", "r", "beta", "final_tokens",
                    "latency_ms_mean", "latency_ms_std", "latency_ms_median",
                    "throughput_mean", "latency_speedup", "throughput_speedup",
                    "latency_ms_all",
                ],
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nSaved latency sweep to: {args.out_latency_csv}")


def run_profile_selected(args):
    """
    Profile selected r values and methods to estimate whether merge overhead dominates.
    """
    set_stable_cuda_flags()
    device = get_device(args.device)
    x = torch.randn(args.profile_batch_size, 3, 224, 224, device=device)

    group_all = []
    op_all = []

    settings = [("full_patched", 0)]
    for r in args.profile_r_list:
        for method in args.profile_methods:
            settings.append((method, r))

    for method, r in settings:
        beta = beta_for_r(r) if method == "ours" else 0.0
        final_tokens = get_final_tokens(args.model_name, method, r)

        print("\n========================================")
        print(f"[PROFILE] model={args.model_name}, method={method}, r={r}, "
              f"beta={beta}, batch={args.profile_batch_size}, final_tokens={final_tokens}")
        print("========================================")

        model = build_model(
            model_name=args.model_name,
            method=method,
            r=r,
            beta=beta,
            pretrained=not args.no_pretrained,
            device=device,
        )

        group_rows, op_rows = profile_ops(
            model=model,
            x=x,
            warmup=args.profile_warmup,
            steps=args.profile_steps,
            use_amp=args.amp,
        )

        for row in group_rows:
            row.update({
                "model": args.model_name,
                "batch_size": args.profile_batch_size,
                "method": method,
                "r": r,
                "beta": beta,
                "final_tokens": final_tokens,
            })
            group_all.append(row)

        for row in op_rows[:args.top_ops]:
            row.update({
                "model": args.model_name,
                "batch_size": args.profile_batch_size,
                "method": method,
                "r": r,
                "beta": beta,
                "final_tokens": final_tokens,
            })
            op_all.append(row)

        print("Top CUDA-time groups:")
        for row in group_rows[:10]:
            print(f"  {row['group']:<24s} {row['cuda_time_total_ms']:.3f} ms "
                  f"({row['cuda_time_pct']:.2f}%)")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(
        args.out_profile_group_csv,
        group_all,
        [
            "model", "batch_size", "method", "r", "beta", "final_tokens",
            "group", "cuda_time_total_ms", "cuda_time_pct",
        ],
    )

    write_csv(
        args.out_profile_ops_csv,
        op_all,
        [
            "model", "batch_size", "method", "r", "beta", "final_tokens",
            "op", "group", "cuda_time_total_ms", "cpu_time_total_ms", "count",
        ],
    )

    print(f"\nSaved profile group summary to: {args.out_profile_group_csv}")
    print(f"Saved top op summary to: {args.out_profile_ops_csv}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="deit_small_patch16_224")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # Latency sweep.
    parser.add_argument("--run-latency-sweep", action="store_true")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--r-list", type=int, nargs="+", default=[4, 8, 12, 16, 20, 25])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out-latency-csv", type=str, default="deit_small_latency_diagnostic.csv")

    # Profiler.
    parser.add_argument("--run-profile", action="store_true")
    parser.add_argument("--profile-batch-size", type=int, default=32)
    parser.add_argument("--profile-r-list", type=int, nargs="+", default=[8, 12, 16, 25])
    parser.add_argument("--profile-methods", type=str, nargs="+", default=["tome", "ours"], choices=["tome", "ours"])
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--top-ops", type=int, default=80)
    parser.add_argument("--out-profile-group-csv", type=str, default="deit_small_profile_groups.csv")
    parser.add_argument("--out-profile-ops-csv", type=str, default="deit_small_profile_ops.csv")

    args = parser.parse_args()

    if not args.run_latency_sweep and not args.run_profile:
        args.run_latency_sweep = True

    if args.run_latency_sweep:
        run_latency_sweep(args)

    if args.run_profile:
        run_profile_selected(args)


if __name__ == "__main__":
    main()
