import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import timm
from timm.models.vision_transformer import Attention, Block, VisionTransformer

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except Exception:
    HAS_FVCORE = False


FIXED_BETA_MAP = {
    4: 0.000,
    8: 0.035,
    12: 0.050,
    16: 0.015,
    20: 0.040,
    25: 0.020,
}

DEFAULT_BETA_SWEEP = [0.000, 0.005, 0.010, 0.015, 0.020, 0.035, 0.050, 0.075]


def parse_r(num_layers: int, r: int) -> List[int]:
    return [int(r)] * num_layers


def beta_for_r(r: int) -> float:
    """Reuse ViT-B/16 beta settings for the fixed-beta DeiT transfer experiment."""
    if r not in FIXED_BETA_MAP:
        raise ValueError(f"No fixed-beta value specified for r={r}.")
    return FIXED_BETA_MAP[r]


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def build_val_loader(data_path: str, batch_size: int, num_workers: int):
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_path, transform=val_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate_accuracy(model, data_loader, device, save_correct: bool = False):
    model.eval()
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    correct_top1_all = []
    for images, target in tqdm(data_loader, desc="Evaluating", ncols=100):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.size(0)
        total += batch_size
        top1_sum += acc1.item() * batch_size
        top5_sum += acc5.item() * batch_size
        if save_correct:
            pred = output.argmax(dim=1)
            correct_top1_all.append(pred.eq(target).detach().cpu().numpy().astype(np.int8))
    top1 = top1_sum / total
    top5 = top5_sum / total
    if save_correct:
        return top1, top5, np.concatenate(correct_top1_all, axis=0)
    return top1, top5, None


@torch.no_grad()
def benchmark_throughput_once(model, device, batch_size=64, input_size=(3, 224, 224), runs=40, throw_out=0.25):
    model.eval().to(device)
    x = torch.randn(batch_size, *input_size, device=device)
    warmup = int(runs * throw_out)
    total_images = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for i in range(runs):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            total_images = 0
        _ = model(x)
        total_images += batch_size
    if device.type == "cuda":
        torch.cuda.synchronize()
    return total_images / (time.time() - start)


def benchmark_throughput(model, device, batch_size=64, input_size=(3, 224, 224), runs=40, throw_out=0.25, repeats=1):
    values = []
    for rep in range(repeats):
        v = benchmark_throughput_once(model, device, batch_size, input_size, runs, throw_out)
        values.append(v)
        print(f"Throughput repeat {rep + 1}/{repeats}: {v:.2f} img/s")
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) if repeats > 1 else 0.0)


def compute_gflops(model, device):
    if not HAS_FVCORE:
        print("[Warning] fvcore is not installed. GFLOPs will be -1.")
        return -1.0
    model.eval().to(device)
    dummy = torch.randn(1, 3, 224, 224, device=device)
    try:
        flops = FlopCountAnalysis(model, dummy)
        return float(flops.total() / 1e9)
    except Exception as e:
        print(f"[Warning] FLOPs computation failed: {e}")
        return -1.0


def infer_initial_tokens_and_protected(model) -> Tuple[int, int]:
    num_patches = getattr(model.patch_embed, "num_patches", None)
    if num_patches is None:
        grid_size = getattr(model.patch_embed, "grid_size", None)
        if grid_size is None:
            raise RuntimeError("Cannot infer number of patch tokens.")
        num_patches = int(grid_size[0] * grid_size[1])
    class_token = getattr(model, "cls_token", None) is not None
    distill_token = hasattr(model, "dist_token") and getattr(model, "dist_token") is not None
    protected = int(class_token) + int(distill_token)
    return int(num_patches) + protected, protected


def compute_final_tokens(model, r: int) -> int:
    tokens, protected = infer_initial_tokens_and_protected(model)
    for _ in range(len(model.blocks)):
        r_actual = min(int(r), max(0, (tokens - protected) // 2))
        tokens -= r_actual
    return int(tokens)


def do_nothing(x, mode="mean"):
    return x


def bipartite_soft_matching_tome(metric: torch.Tensor, r: int, class_token: bool = False, distill_token: bool = False) -> Tuple[Callable, Callable]:
    protected = int(class_token) + int(distill_token)
    t = metric.shape[1]
    r = min(int(r), max(0, (t - protected) // 2))
    if r <= 0:
        return do_nothing, do_nothing
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a = metric[..., ::2, :]
        b = metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        node_max = torch.nan_to_num(node_max, nan=-math.inf, posinf=math.inf, neginf=-math.inf)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src = x[..., ::2, :]
        dst = x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src_selected = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(dim=-2, index=dst_idx.expand(n, r, c), src=src_selected, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    return merge, do_nothing


def bipartite_soft_matching_ours(metric: torch.Tensor, r: int, beta: float, class_token: bool = False, distill_token: bool = False) -> Tuple[Callable, Callable]:
    protected = int(class_token) + int(distill_token)
    t = metric.shape[1]
    r = min(int(r), max(0, (t - protected) // 2))
    if r <= 0:
        return do_nothing, do_nothing
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a = metric[..., ::2, :]
        b = metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        topk = min(2, scores.shape[-1])
        top_values = scores.topk(k=topk, dim=-1).values
        top1_score = top_values[..., 0]
        top2_score = top_values[..., 1] if topk == 2 else torch.zeros_like(top1_score)
        margin = top1_score - top2_score
        node_max = torch.nan_to_num(node_max, nan=-math.inf, posinf=math.inf, neginf=-math.inf)
        margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
        calibrated_score = torch.nan_to_num(node_max + beta * margin, nan=-math.inf, posinf=math.inf, neginf=-math.inf)
        if class_token:
            calibrated_score[..., 0] = -math.inf
        edge_idx = calibrated_score.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src = x[..., ::2, :]
        dst = x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src_selected = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(dim=-2, index=dst_idx.expand(n, r, c), src=src_selected, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    return merge, do_nothing


def merge_wavg(merge_fn: Callable, x: torch.Tensor, size: Optional[torch.Tensor] = None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge_fn(x * size, mode="sum")
    size = merge_fn(size, mode="sum")
    x = x / size.clamp_min(1e-6)
    return x, size


class PatchedToMeBlock(Block):
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)
        r = self._tome_info["r"].pop(0)
        if r > 0:
            method = self._tome_info["method"]
            beta = self._tome_info["beta"]
            if method == "tome":
                merge_fn, _ = bipartite_soft_matching_tome(metric, r, self._tome_info["class_token"], self._tome_info["distill_token"])
            elif method == "ours":
                merge_fn, _ = bipartite_soft_matching_ours(metric, r, beta, self._tome_info["class_token"], self._tome_info["distill_token"])
            else:
                raise ValueError(f"Unknown method: {method}")
            x, self._tome_info["size"] = merge_wavg(merge_fn, x, self._tome_info["size"])
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchedToMeAttention(Attention):
    def forward(self, x: torch.Tensor, size: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        metric = k.mean(dim=1)
        return x, metric


def make_tome_class(transformer_class):
    class PatchedVisionTransformer(transformer_class):
        def forward(self, *args, **kwargs):
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            return super().forward(*args, **kwargs)
    return PatchedVisionTransformer


def apply_patch(model: VisionTransformer, method: str, beta: float, prop_attn: bool = True):
    model.__class__ = make_tome_class(model.__class__)
    model.r = 0
    class_token = getattr(model, "cls_token", None) is not None
    distill_token = hasattr(model, "dist_token") and getattr(model, "dist_token") is not None
    model._tome_info = {
        "r": model.r,
        "size": None,
        "prop_attn": prop_attn,
        "class_token": class_token,
        "distill_token": distill_token,
        "method": method,
        "beta": beta,
    }
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = PatchedToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = PatchedToMeAttention
    return model


@dataclass
class EvalResult:
    model: str
    method: str
    r: int
    beta: float
    top1: float
    top5: float
    gflops: float
    flops_red: float
    throughput_mean: float
    throughput_std: float
    speedup: float
    final_tokens: int


def create_model(model_name: str, pretrained: bool):
    return timm.create_model(model_name, pretrained=pretrained)


def run_one_setting(model_name: str, method: str, r: int, beta: float, data_loader, device, batch_size: int, throughput_runs: int, throughput_repeats: int, full_gflops: Optional[float], full_throughput: Optional[float], pretrained: bool = True, eval_gflops: bool = True, eval_throughput: bool = True, save_correct: bool = False):
    print("\n" + "=" * 88)
    print(f"Model: {model_name} | Method: {method} | r={r} | beta={beta}")
    print("=" * 88)
    base_model_for_schedule = create_model(model_name, pretrained=False)
    final_tokens = compute_final_tokens(base_model_for_schedule, r)
    del base_model_for_schedule
    model = create_model(model_name, pretrained=pretrained)
    model = apply_patch(model=model, method=method, beta=beta, prop_attn=True)
    model.r = int(r)
    model.to(device)
    model.eval()
    top1, top5, correct_top1 = evaluate_accuracy(model, data_loader, device, save_correct=save_correct)
    gflops = compute_gflops(model, device) if eval_gflops else -1.0
    if eval_throughput:
        throughput_mean, throughput_std = benchmark_throughput(model, device, batch_size=batch_size, runs=throughput_runs, repeats=throughput_repeats)
    else:
        throughput_mean, throughput_std = -1.0, 0.0
    flops_red = (full_gflops - gflops) / full_gflops * 100.0 if full_gflops is not None and full_gflops > 0 and gflops > 0 else 0.0
    speedup = throughput_mean / full_throughput if full_throughput is not None and full_throughput > 0 and throughput_mean > 0 else 1.0
    result = EvalResult(model_name, method, int(r), float(beta), float(top1), float(top5), float(gflops), float(flops_red), float(throughput_mean), float(throughput_std), float(speedup), int(final_tokens))
    print(result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, correct_top1


def result_to_dict(result: EvalResult) -> Dict:
    return {
        "model": result.model,
        "method": result.method,
        "r": result.r,
        "beta": result.beta,
        "top1": result.top1,
        "top5": result.top5,
        "gflops": result.gflops,
        "flops_red": result.flops_red,
        "throughput_mean": result.throughput_mean,
        "throughput_std": result.throughput_std,
        "speedup": result.speedup,
        "final_tokens": result.final_tokens,
    }


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV to: {path}")


def paired_bootstrap_ci(tome_correct: np.ndarray, ours_correct: np.ndarray, num_bootstrap: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    assert tome_correct.shape == ours_correct.shape
    rng = np.random.default_rng(seed)
    n = tome_correct.shape[0]
    diff = ours_correct.astype(np.float32) - tome_correct.astype(np.float32)
    mean_delta = float(diff.mean() * 100.0)
    boot_values = np.empty(num_bootstrap, dtype=np.float32)
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = diff[idx].mean() * 100.0
    ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
    return mean_delta, float(ci_low), float(ci_high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-val", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", default=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--r-list", type=int, nargs="+", default=[4, 8, 12, 16, 20, 25])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--throughput-runs", type=int, default=40)
    parser.add_argument("--throughput-repeats", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out-csv", type=str, default="deit_generality_results_full_r.csv")
    parser.add_argument("--run-fixed", action="store_true", help="Run fixed-beta transfer experiment.")
    parser.add_argument("--run-beta-sweep", action="store_true", help="Run DeiT beta sensitivity sweep.")
    parser.add_argument("--sweep-r-list", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--beta-list", type=float, nargs="+", default=DEFAULT_BETA_SWEEP)
    parser.add_argument("--sweep-out-csv", type=str, default="deit_beta_sweep_results.csv")
    parser.add_argument("--bootstrap-r-list", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--run-bootstrap", action="store_true", help="Save per-image correctness and compute paired bootstrap CI for fixed-beta settings.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-out-csv", type=str, default="deit_bootstrap_results.csv")
    parser.add_argument("--skip-gflops", action="store_true", help="Skip GFLOPs profiling.")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip throughput benchmark.")
    args = parser.parse_args()
    if not args.run_fixed and not args.run_beta_sweep:
        args.run_fixed = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pretrained = not args.no_pretrained
    print("========== DeiT Generality Experiment ==========")
    print(f"ImageNet val       : {args.imagenet_val}")
    print(f"Models             : {args.models}")
    print(f"r list             : {args.r_list}")
    print(f"Fixed beta map     : {FIXED_BETA_MAP}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Device             : {device}")
    print(f"Pretrained         : {pretrained}")
    print(f"Run fixed          : {args.run_fixed}")
    print(f"Run beta sweep     : {args.run_beta_sweep}")
    print(f"Run bootstrap      : {args.run_bootstrap}")
    print("================================================")
    data_loader = build_val_loader(args.imagenet_val, args.batch_size, args.num_workers)
    fixed_rows: List[Dict] = []
    correctness_cache: Dict[Tuple[str, int, str], np.ndarray] = {}
    tome_top1_cache: Dict[Tuple[str, int], float] = {}
    if args.run_fixed:
        for model_name in args.models:
            full_result, _ = run_one_setting(model_name, "tome", 0, 0.0, data_loader, device, args.batch_size, args.throughput_runs, args.throughput_repeats, None, None, pretrained, not args.skip_gflops, not args.skip_throughput, False)
            full_gflops = full_result.gflops
            full_throughput = full_result.throughput_mean
            full_top1 = full_result.top1
            full_result.method = "Full patched"
            row = result_to_dict(full_result)
            row["acc_drop"] = 0.0
            fixed_rows.append(row)
            for r in args.r_list:
                save_correct = args.run_bootstrap and (r in args.bootstrap_r_list)
                tome_result, tome_correct = run_one_setting(model_name, "tome", r, 0.0, data_loader, device, args.batch_size, args.throughput_runs, args.throughput_repeats, full_gflops, full_throughput, pretrained, not args.skip_gflops, not args.skip_throughput, save_correct)
                tome_result.method = "ToMe"
                tome_top1_cache[(model_name, r)] = tome_result.top1
                row = result_to_dict(tome_result)
                row["acc_drop"] = full_top1 - tome_result.top1
                fixed_rows.append(row)
                if save_correct:
                    correctness_cache[(model_name, r, "ToMe")] = tome_correct
                beta = beta_for_r(r)
                ours_result, ours_correct = run_one_setting(model_name, "ours", r, beta, data_loader, device, args.batch_size, args.throughput_runs, args.throughput_repeats, full_gflops, full_throughput, pretrained, not args.skip_gflops, not args.skip_throughput, save_correct)
                ours_result.method = "Ours"
                row = result_to_dict(ours_result)
                row["acc_drop"] = full_top1 - ours_result.top1
                fixed_rows.append(row)
                if save_correct:
                    correctness_cache[(model_name, r, "Ours")] = ours_correct
        fixed_fieldnames = ["model", "method", "r", "beta", "top1", "top5", "acc_drop", "gflops", "flops_red", "throughput_mean", "throughput_std", "speedup", "final_tokens"]
        write_csv(args.out_csv, fixed_rows, fixed_fieldnames)
    if args.run_bootstrap:
        bootstrap_rows = []
        for model_name in args.models:
            for r in args.bootstrap_r_list:
                kt = (model_name, r, "ToMe")
                ko = (model_name, r, "Ours")
                if kt not in correctness_cache or ko not in correctness_cache:
                    print(f"[Bootstrap] Missing correctness for {model_name}, r={r}. Skipped.")
                    continue
                mean_delta, ci_low, ci_high = paired_bootstrap_ci(correctness_cache[kt], correctness_cache[ko], args.bootstrap_samples, args.bootstrap_seed)
                bootstrap_rows.append({
                    "model": model_name,
                    "r": r,
                    "tome_top1": float(correctness_cache[kt].mean() * 100.0),
                    "ours_top1": float(correctness_cache[ko].mean() * 100.0),
                    "delta_top1_pp": mean_delta,
                    "ci95_low_pp": ci_low,
                    "ci95_high_pp": ci_high,
                    "bootstrap_samples": args.bootstrap_samples,
                })
        bootstrap_fieldnames = ["model", "r", "tome_top1", "ours_top1", "delta_top1_pp", "ci95_low_pp", "ci95_high_pp", "bootstrap_samples"]
        write_csv(args.bootstrap_out_csv, bootstrap_rows, bootstrap_fieldnames)
    if args.run_beta_sweep:
        sweep_rows = []
        for model_name in args.models:
            for r in args.sweep_r_list:
                if (model_name, r) not in tome_top1_cache:
                    tome_result, _ = run_one_setting(model_name, "tome", r, 0.0, data_loader, device, args.batch_size, args.throughput_runs, 1, None, None, pretrained, False, False, False)
                    tome_top1_cache[(model_name, r)] = tome_result.top1
                tome_top1 = tome_top1_cache[(model_name, r)]
                for beta in args.beta_list:
                    ours_result, _ = run_one_setting(model_name, "ours", r, beta, data_loader, device, args.batch_size, args.throughput_runs, 1, None, None, pretrained, False, False, False)
                    sweep_rows.append({
                        "model": model_name,
                        "r": r,
                        "beta": beta,
                        "tome_top1": tome_top1,
                        "ours_top1": ours_result.top1,
                        "ours_top5": ours_result.top5,
                        "delta_top1_pp": ours_result.top1 - tome_top1,
                        "final_tokens": ours_result.final_tokens,
                    })
        sweep_fieldnames = ["model", "r", "beta", "tome_top1", "ours_top1", "ours_top5", "delta_top1_pp", "final_tokens"]
        write_csv(args.sweep_out_csv, sweep_rows, sweep_fieldnames)
    print("\nAll requested experiments completed.")


if __name__ == "__main__":
    main()
