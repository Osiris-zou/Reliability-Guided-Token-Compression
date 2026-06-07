import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import timm
import tome

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# Dataset and transform
# =========================================================
class ImageNetFolderDataset(torch.utils.data.Dataset):
    """
    ImageNet validation folder dataset.

    Expected structure:
        val/
          n01440764/
          n01443537/
          ...
    Class folders are sorted alphabetically, matching standard ImageFolder behavior.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG", ".BMP")

        assert os.path.isdir(root_dir), f"Validation path does not exist: {root_dir}"

        self.classes = self._find_classes()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _find_classes(self) -> List[str]:
        classes = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                classes.append(item)
        classes.sort()
        return classes

    def _make_dataset(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            label = self.class_to_idx[class_name]
            for filename in os.listdir(class_dir):
                if filename.endswith(self.extensions):
                    samples.append((os.path.join(class_dir, filename), label))
        return samples

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load image: {path}, error: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)


def build_transform(preprocess: str):
    """
    Use the same preprocessing protocol as the main ViT evaluation.

    inception:
        mean/std = 0.5/0.5, used by the previous ViT table in this project.
    imagenet:
        standard ImageNet mean/std.
    """
    if preprocess == "inception":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif preprocess == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unsupported preprocess: {preprocess}")

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# =========================================================
# Weight loading
# =========================================================
def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print("[WARN] torch.load(weights_only=True) failed.")
        print(f"[WARN] Error: {e}")
        print("[WARN] Try weights_only=False. Only use trusted checkpoints.")
        return torch.load(path, map_location="cpu", weights_only=False)


def unwrap_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[CKPT] Found wrapper key: {key}")
                return ckpt[key]
    return ckpt


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str):
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith(prefix) for k in keys[: min(10, len(keys))]):
        print(f"[CKPT] Strip prefix: {prefix}")
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_weights_smart(model: torch.nn.Module, weights_path: str, device: torch.device):
    assert os.path.exists(weights_path), f"Weights file does not exist: {weights_path}"

    ckpt = safe_torch_load(weights_path)
    state_dict = unwrap_checkpoint(ckpt)
    state_dict = strip_prefix_if_present(state_dict, "module.")

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for k, v in state_dict.items():
        if k not in model_state:
            skipped.append((k, "unexpected_key"))
            continue
        if hasattr(v, "shape") and tuple(v.shape) != tuple(model_state[k].shape):
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered_state[k] = v

    load_info = model.load_state_dict(filtered_state, strict=False)

    print("\n========== Load Result ==========")
    print(f"[LOAD] Loaded tensors: {len(filtered_state)}")
    print(f"[LOAD] Skipped tensors: {len(skipped)}")
    print(f"[LOAD] Missing keys count: {len(load_info.missing_keys)}")
    print(f"[LOAD] Unexpected keys count: {len(load_info.unexpected_keys)}")
    if "head.weight" in load_info.missing_keys or "head.bias" in load_info.missing_keys:
        print("[WARNING] head.weight/head.bias were NOT loaded. The classifier head may be random.")
    else:
        print("[LOAD] head.weight/head.bias loaded successfully.")
    print("=================================\n")

    model.to(device)
    return model


# =========================================================
# Model construction
# =========================================================
def build_vit_model(
    model_name: str,
    num_classes: int,
    weights: str,
    method: str,
    r: int,
    beta: float,
    prop_attn: bool,
    device: torch.device,
):
    """
    Build a ViT model with the same ToMe/Ours patch interface used in eval_vit_methods_stable.py.

    method:
        tome : original similarity-only ToMe
        ours : proposed confidence-aware method, named 'etmrl' in the existing patch code
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    if method == "tome":
        merge_method = "tome"
        model_r = int(r)
    elif method == "ours":
        merge_method = "etmrl"
        model_r = int(r)
    else:
        raise ValueError(f"Unsupported method: {method}")

    tome.patch.timm(
        model,
        trace_source=False,
        prop_attn=prop_attn,
        merge_method=merge_method,
    )
    model.r = model_r

    if hasattr(model, "_tome_info"):
        # Some patch versions read beta from _tome_info; keep it explicit.
        model._tome_info["beta"] = float(beta)
        model._tome_info["merge_method"] = merge_method

    print("\n========== Build Model ==========")
    print(f"model_name   : {model_name}")
    print(f"method       : {method}")
    print(f"merge_method : {merge_method}")
    print(f"r            : {r}")
    print(f"beta         : {beta}")
    print(f"prop_attn    : {prop_attn}")
    print("=================================\n")

    model = load_weights_smart(model, weights, device)
    model.eval()
    return model


# =========================================================
# Evaluation and bootstrap
# =========================================================
@torch.no_grad()
def evaluate_correctness(model, data_loader, device, use_amp: bool = False):
    """
    Return Top-1, Top-5, and per-image Top-1 correctness array.
    """
    model.eval()

    total = 0
    correct1 = 0
    correct5 = 0
    correct_top1_all = []

    for images, labels in tqdm(data_loader, desc="Evaluating", ncols=100):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)

        _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))

        correct1 += correct[:, :1].sum().item()
        correct5 += correct[:, :5].sum().item()
        total += labels.size(0)

        pred1 = pred[:, 0]
        correct_top1_all.append(pred1.eq(labels).detach().cpu().numpy().astype(np.int8))

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    correct_top1 = np.concatenate(correct_top1_all, axis=0)

    return float(top1), float(top5), correct_top1


def paired_bootstrap_ci(
    tome_correct: np.ndarray,
    ours_correct: np.ndarray,
    num_bootstrap: int = 10000,
    seed: int = 0,
):
    """
    Paired bootstrap on image-level correctness difference:
        diff_i = ours_correct_i - tome_correct_i

    Returned values are percentage points.
    """
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


def parse_beta_map(items: List[str]) -> Dict[int, float]:
    """
    Parse beta map from command line, e.g.
        --beta-map 8:0.035 12:0.050 16:0.015
    """
    mapping = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid beta-map item: {item}. Expected r:beta.")
        r_str, b_str = item.split(":", 1)
        mapping[int(r_str)] = float(b_str)
    return mapping


def write_csv(path: str, rows: List[Dict]):
    fieldnames = [
        "model",
        "r",
        "beta",
        "tome_top1",
        "tome_top5",
        "ours_top1",
        "ours_top5",
        "delta_top1_pp",
        "ci95_low_pp",
        "ci95_high_pp",
        "bootstrap_samples",
        "num_images",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV to: {path}")


def cache_path(cache_dir: str, model_name: str, method: str, r: int, beta: float):
    safe_beta = str(beta).replace(".", "p")
    filename = f"{model_name}_{method}_r{r}_b{safe_beta}_correct.npz"
    return Path(cache_dir) / filename


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--r-list", type=int, nargs="+", required=True)
    parser.add_argument("--beta-map", type=str, nargs="+", required=True,
                        help="Format: r:beta, e.g. 8:0.035 12:0.050 16:0.015")

    parser.add_argument("--preprocess", type=str, default="inception", choices=["inception", "imagenet"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--prop-attn", action="store_true")

    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)

    parser.add_argument("--cache-dir", type=str, default="vit_bootstrap_correctness_cache")
    parser.add_argument("--out-csv", type=str, default="vit_bootstrap_results.csv")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    beta_map = parse_beta_map(args.beta_map)

    os.makedirs(args.cache_dir, exist_ok=True)

    print("========== ViT Paired Bootstrap ==========")
    print(f"data path       : {args.data_path}")
    print(f"model           : {args.model_name}")
    print(f"weights         : {args.weights}")
    print(f"r list          : {args.r_list}")
    print(f"beta map        : {beta_map}")
    print(f"preprocess      : {args.preprocess}")
    print(f"batch size      : {args.batch_size}")
    print(f"num workers     : {args.num_workers}")
    print(f"device          : {device}")
    print(f"prop_attn       : {args.prop_attn}")
    print(f"bootstrap       : {args.bootstrap_samples}")
    print("==========================================\n")

    transform = build_transform(args.preprocess)
    dataset = ImageNetFolderDataset(args.data_path, transform=transform)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    data_loader = DataLoader(dataset, **loader_kwargs)

    print("\n========== Dataset ==========")
    print(f"samples: {len(dataset)}")
    print(f"classes: {len(dataset.classes)}")
    print("=============================\n")

    rows = []

    for r in args.r_list:
        if r not in beta_map:
            raise ValueError(f"Missing beta value for r={r}. Add it to --beta-map.")

        beta = beta_map[r]

        correctness = {}
        metrics = {}

        for method in ["tome", "ours"]:
            method_beta = 0.0 if method == "tome" else beta
            cpath = cache_path(args.cache_dir, args.model_name, method, r, method_beta)

            if cpath.exists():
                print(f"[CACHE] Loading {cpath}")
                data = np.load(cpath)
                top1 = float(data["top1"])
                top5 = float(data["top5"])
                correct = data["correct"].astype(np.int8)
            else:
                model = build_vit_model(
                    model_name=args.model_name,
                    num_classes=args.num_classes,
                    weights=args.weights,
                    method=method,
                    r=r,
                    beta=method_beta,
                    prop_attn=args.prop_attn,
                    device=device,
                )

                top1, top5, correct = evaluate_correctness(
                    model=model,
                    data_loader=data_loader,
                    device=device,
                    use_amp=args.amp,
                )

                np.savez_compressed(cpath, top1=np.array(top1), top5=np.array(top5), correct=correct)
                print(f"[CACHE] Saved {cpath}")

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            metrics[method] = {"top1": top1, "top5": top5}
            correctness[method] = correct

            print(f"[RESULT] r={r}, method={method}, top1={top1:.4f}, top5={top5:.4f}")

        mean_delta, ci_low, ci_high = paired_bootstrap_ci(
            tome_correct=correctness["tome"],
            ours_correct=correctness["ours"],
            num_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )

        row = {
            "model": args.model_name,
            "r": r,
            "beta": beta,
            "tome_top1": metrics["tome"]["top1"],
            "tome_top5": metrics["tome"]["top5"],
            "ours_top1": metrics["ours"]["top1"],
            "ours_top5": metrics["ours"]["top5"],
            "delta_top1_pp": mean_delta,
            "ci95_low_pp": ci_low,
            "ci95_high_pp": ci_high,
            "bootstrap_samples": args.bootstrap_samples,
            "num_images": int(correctness["tome"].shape[0]),
        }
        rows.append(row)

        print("\n========== Bootstrap Result ==========")
        print(json.dumps(row, indent=2, ensure_ascii=False))
        print("======================================\n")

        write_csv(args.out_csv, rows)

    print("All requested ViT bootstrap experiments completed.")


if __name__ == "__main__":
    main()
