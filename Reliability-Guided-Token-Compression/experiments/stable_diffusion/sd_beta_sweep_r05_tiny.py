import os
import sys
import csv
import time
import importlib.util
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer


# ============================================================
# 1. 需要你直接修改的参数
# ============================================================

CKPT_PATH = r"E:\zp\pytorch-stable-diffusion-main\pytorch-stable-diffusion-main\data\v1-5-pruned-emaonly.ckpt"

TOKENIZER_DIR = r"E:\zp\pytorch-stable-diffusion-main\pytorch-stable-diffusion-main\data"

PROMPT_FILE = r"E:\zp\pytorch-stable-diffusion-main\pytorch-stable-diffusion-main\imagenet_prompts.txt"

OUT_DIR = r"E:\zp\pytorch-stable-diffusion-main\pytorch-stable-diffusion-main\results\beta_sweep_r05_tiny"

DEVICE = "cuda"

N_INFERENCE_STEPS = 50
CFG_SCALE = 7.5
SAMPLER_NAME = "ddpm"

MERGE_RATIO = 0.5
MERGE_MAX_DOWNSAMPLE = 1
MERGE_SX = 2
MERGE_SY = 2
MERGE_USE_RAND = True

# 只生成 2 张图：默认取前 2 个 prompt，每个 prompt 用 seed=0
MAX_PROMPTS = 20
SEEDS = [0]

# beta 扫描范围
BETA_LIST = [
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
    0.050,
    0.060,
    0.070,
]

SKIP_EXISTING = True


# ============================================================
# 2. 本地模块导入
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SD_DIR = PROJECT_ROOT / "sd"
TOMESD_DIR = PROJECT_ROOT / "tomesd"

for p in [PROJECT_ROOT, SD_DIR, TOMESD_DIR]:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_py_module(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find python file: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load_py_module(
    module_name="sd_pipeline",
    file_path=SD_DIR / "pipeline.py",
)

model_loader = load_py_module(
    module_name="sd_model_loader",
    file_path=SD_DIR / "model_loader.py",
)

preload_models_from_standard_weights = model_loader.preload_models_from_standard_weights


# ============================================================
# 3. 指标模块
# ============================================================

try:
    import lpips
except ImportError:
    raise ImportError(
        "缺少 lpips，请先安装：pip install lpips"
    )

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    raise ImportError(
        "缺少 pytorch-msssim，请先安装：pip install pytorch-msssim"
    )


def np_image_to_tensor_01(img_np: np.ndarray, device: torch.device):
    """
    numpy uint8 [H, W, 3] -> torch float [1, 3, H, W], range [0, 1]
    """
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    x = torch.from_numpy(img_np).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    return x


def compute_pair_metrics(img_ref_np, img_test_np, lpips_model, device):
    """
    Full 作为 reference，计算 test 相对 Full 的 LPIPS 和 MS-SSIM。
    """
    ref_01 = np_image_to_tensor_01(img_ref_np, device)
    test_01 = np_image_to_tensor_01(img_test_np, device)

    # LPIPS 需要 [-1, 1]
    ref_lpips = ref_01 * 2.0 - 1.0
    test_lpips = test_01 * 2.0 - 1.0

    with torch.no_grad():
        lpips_val = lpips_model(ref_lpips, test_lpips).mean().item()
        msssim_val = ms_ssim(ref_01, test_01, data_range=1.0, size_average=True).item()

    return lpips_val, msssim_val


# ============================================================
# 4. 工具函数
# ============================================================

def read_prompts(prompt_file: str, max_prompts: int):
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompts = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                prompts.append(text)

    prompts = prompts[:max_prompts]

    if len(prompts) == 0:
        raise ValueError("No prompts found. Please check imagenet_prompts.txt.")

    return prompts


def safe_name(text: str, max_len: int = 80):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            keep.append(ch)
        elif ch == " ":
            keep.append("_")
    return "".join(keep)[:max_len]


def save_np_image(img_np, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_np).save(save_path)


def generate_one_image(
    models,
    tokenizer,
    device,
    prompt: str,
    seed: int,
    merge_method: str,
    ratio: float,
    beta: float,
):
    """
    生成单张图片，并统计时间和显存。
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()

    img_np = pipeline.generate(
        prompt=prompt,
        uncond_prompt="",
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=CFG_SCALE,
        sampler_name=SAMPLER_NAME,
        n_inference_steps=N_INFERENCE_STEPS,
        models=models,
        seed=seed,
        device=device,
        idle_device=None,
        tokenizer=tokenizer,

        merge_method=merge_method,
        merge_ratio=ratio,
        merge_beta=beta,
        merge_max_downsample=MERGE_MAX_DOWNSAMPLE,
        merge_sx=MERGE_SX,
        merge_sy=MERGE_SY,
        merge_use_rand=MERGE_USE_RAND,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()

    time_sec = end - start

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    else:
        peak_mem_gb = 0.0

    return img_np, time_sec, peak_mem_gb


def write_rows(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "beta",
                "class_id",
                "prompt",
                "seed",
                "image_path",
                "time_sec",
                "peak_mem_gb",
                "lpips_vs_full",
                "ms_ssim_vs_full",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(csv_path: Path, summary_rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "beta",
                "num_images",
                "avg_time_sec",
                "avg_peak_mem_gb",
                "avg_lpips_vs_full",
                "avg_ms_ssim_vs_full",
                "score_for_selection",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


# ============================================================
# 5. 主流程
# ============================================================

def main():
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    print("[INFO] Loading prompts...")
    prompts = read_prompts(PROMPT_FILE, MAX_PROMPTS)
    print(f"[INFO] prompts = {len(prompts)}, seeds = {SEEDS}")

    print("[INFO] Loading tokenizer...")
    tokenizer = CLIPTokenizer(
        os.path.join(TOKENIZER_DIR, "vocab.json"),
        merges_file=os.path.join(TOKENIZER_DIR, "merges.txt"),
    )

    print("[INFO] Loading Stable Diffusion models...")
    models = preload_models_from_standard_weights(CKPT_PATH, device)

    print("[INFO] Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    # ------------------------------------------------------------
    # Step 1: 先生成 Full 参考图
    # ------------------------------------------------------------
    full_cache = {}
    all_rows = []

    print("\n" + "=" * 80)
    print("[STEP 1] Generate Full reference images")
    print("=" * 80)

    for class_id, prompt in tqdm(list(enumerate(prompts)), desc="Full"):
        prompt_tag = safe_name(prompt)

        for seed in SEEDS:
            save_path = out_root / "full" / f"class_{class_id:04d}_seed{seed}_{prompt_tag}.png"

            if SKIP_EXISTING and save_path.exists():
                img_np = np.array(Image.open(save_path).convert("RGB"))
                time_sec = 0.0
                peak_mem_gb = 0.0
            else:
                img_np, time_sec, peak_mem_gb = generate_one_image(
                    models=models,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    seed=seed,
                    merge_method="full",
                    ratio=0.0,
                    beta=0.0,
                )
                save_np_image(img_np, save_path)

            full_cache[(class_id, seed)] = img_np

            all_rows.append({
                "method": "full",
                "beta": "",
                "class_id": class_id,
                "prompt": prompt,
                "seed": seed,
                "image_path": str(save_path),
                "time_sec": f"{time_sec:.4f}",
                "peak_mem_gb": f"{peak_mem_gb:.4f}",
                "lpips_vs_full": "0.000000",
                "ms_ssim_vs_full": "1.000000",
            })

    # ------------------------------------------------------------
    # Step 2: 生成 ToMe baseline
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 2] Generate ToMe baseline images")
    print("=" * 80)

    tome_metrics = []

    for class_id, prompt in tqdm(list(enumerate(prompts)), desc="ToMe"):
        prompt_tag = safe_name(prompt)

        for seed in SEEDS:
            save_path = out_root / "tome" / f"class_{class_id:04d}_seed{seed}_{prompt_tag}.png"

            if SKIP_EXISTING and save_path.exists():
                img_np = np.array(Image.open(save_path).convert("RGB"))
                time_sec = 0.0
                peak_mem_gb = 0.0
            else:
                img_np, time_sec, peak_mem_gb = generate_one_image(
                    models=models,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    seed=seed,
                    merge_method="tome",
                    ratio=MERGE_RATIO,
                    beta=0.0,
                )
                save_np_image(img_np, save_path)

            ref_np = full_cache[(class_id, seed)]
            lpips_val, msssim_val = compute_pair_metrics(
                ref_np,
                img_np,
                lpips_model,
                device,
            )

            tome_metrics.append((time_sec, peak_mem_gb, lpips_val, msssim_val))

            all_rows.append({
                "method": "tome",
                "beta": "",
                "class_id": class_id,
                "prompt": prompt,
                "seed": seed,
                "image_path": str(save_path),
                "time_sec": f"{time_sec:.4f}",
                "peak_mem_gb": f"{peak_mem_gb:.4f}",
                "lpips_vs_full": f"{lpips_val:.6f}",
                "ms_ssim_vs_full": f"{msssim_val:.6f}",
            })

    # ------------------------------------------------------------
    # Step 3: 遍历 beta，生成 Ours
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 3] Sweep beta for Ours")
    print("=" * 80)

    beta_to_metrics = {}

    for beta in BETA_LIST:
        beta_name = f"beta_{beta:.3f}".replace(".", "p")
        print(f"\n[RUN] Ours beta = {beta:.3f}")

        cur_metrics = []

        for class_id, prompt in tqdm(list(enumerate(prompts)), desc=f"Ours beta={beta:.3f}"):
            prompt_tag = safe_name(prompt)

            for seed in SEEDS:
                save_path = out_root / "ours" / beta_name / f"class_{class_id:04d}_seed{seed}_{prompt_tag}.png"

                if SKIP_EXISTING and save_path.exists():
                    img_np = np.array(Image.open(save_path).convert("RGB"))
                    time_sec = 0.0
                    peak_mem_gb = 0.0
                else:
                    img_np, time_sec, peak_mem_gb = generate_one_image(
                        models=models,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        seed=seed,
                        merge_method="ours",
                        ratio=MERGE_RATIO,
                        beta=beta,
                    )
                    save_np_image(img_np, save_path)

                ref_np = full_cache[(class_id, seed)]
                lpips_val, msssim_val = compute_pair_metrics(
                    ref_np,
                    img_np,
                    lpips_model,
                    device,
                )

                cur_metrics.append((time_sec, peak_mem_gb, lpips_val, msssim_val))

                all_rows.append({
                    "method": "ours",
                    "beta": f"{beta:.3f}",
                    "class_id": class_id,
                    "prompt": prompt,
                    "seed": seed,
                    "image_path": str(save_path),
                    "time_sec": f"{time_sec:.4f}",
                    "peak_mem_gb": f"{peak_mem_gb:.4f}",
                    "lpips_vs_full": f"{lpips_val:.6f}",
                    "ms_ssim_vs_full": f"{msssim_val:.6f}",
                })

        beta_to_metrics[beta] = cur_metrics

    # ------------------------------------------------------------
    # Step 4: 汇总结果
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 4] Write summary")
    print("=" * 80)

    detail_csv = out_root / "beta_sweep_detail.csv"
    summary_csv = out_root / "beta_sweep_summary.csv"

    write_rows(detail_csv, all_rows)

    summary_rows = []

    def summarize(method, beta, metrics):
        arr = np.array(metrics, dtype=np.float64)
        avg_time = float(arr[:, 0].mean())
        avg_mem = float(arr[:, 1].mean())
        avg_lpips = float(arr[:, 2].mean())
        avg_msssim = float(arr[:, 3].mean())

        # 简单选择分数：LPIPS 越低越好，MS-SSIM 越高越好
        # 这里用 lpips + (1 - ms_ssim) 作为综合偏移，越低越好。
        score = avg_lpips + (1.0 - avg_msssim)

        return {
            "method": method,
            "beta": "" if beta is None else f"{beta:.3f}",
            "num_images": len(metrics),
            "avg_time_sec": f"{avg_time:.4f}",
            "avg_peak_mem_gb": f"{avg_mem:.4f}",
            "avg_lpips_vs_full": f"{avg_lpips:.6f}",
            "avg_ms_ssim_vs_full": f"{avg_msssim:.6f}",
            "score_for_selection": f"{score:.6f}",
        }

    summary_rows.append(
        summarize("tome", None, tome_metrics)
    )

    for beta in BETA_LIST:
        summary_rows.append(
            summarize("ours", beta, beta_to_metrics[beta])
        )

    write_summary(summary_csv, summary_rows)

    print(f"[DONE] Detail CSV saved to: {detail_csv}")
    print(f"[DONE] Summary CSV saved to: {summary_csv}")

    print("\n========== Beta Sweep Summary ==========")
    for row in summary_rows:
        print(
            f"{row['method']:>5} | "
            f"beta={row['beta'] if row['beta'] else '-':>6} | "
            f"LPIPS={row['avg_lpips_vs_full']} | "
            f"MS-SSIM={row['avg_ms_ssim_vs_full']} | "
            f"Time={row['avg_time_sec']}s | "
            f"Mem={row['avg_peak_mem_gb']}GB | "
            f"Score={row['score_for_selection']}"
        )

    ours_rows = [r for r in summary_rows if r["method"] == "ours"]
    best_row = min(ours_rows, key=lambda x: float(x["score_for_selection"]))

    print("\n========== Recommended Beta ==========")
    print(
        f"Best beta = {best_row['beta']} | "
        f"LPIPS={best_row['avg_lpips_vs_full']} | "
        f"MS-SSIM={best_row['avg_ms_ssim_vs_full']} | "
        f"Score={best_row['score_for_selection']}"
    )


if __name__ == "__main__":
    main()