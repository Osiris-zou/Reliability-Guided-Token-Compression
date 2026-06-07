import os
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# 1. 需要你直接修改的参数
# ============================================================

# 6000 张生成图所在目录
RESULT_ROOT = r"E:\zp\pytorch-stable-diffusion-main\pytorch-stable-diffusion-main\results\sd_r05"

# generation_log.csv 路径
GEN_LOG_CSV = r"E:\zp\pytorch-stable-diffusion-main\pytorch-stable-diffusion-main\results\sd_r05\generation_log.csv"

# ImageNet val 真实图片目录，用于计算 FID
# 这里可以是 ImageNet val 的完整目录，也可以是你抽取的 5000 张参考图目录
# 例如：
# IMAGENET_REF_DIR = r"D:\imagenet-1k\val"
IMAGENET_REF_DIR = r"D:\imagenet-1k\val_fid_5000_flat_512"

# 是否计算 FID
# 第一次测试脚本时可以先设为 False，确认 LPIPS/MS-SSIM 正常后再设为 True
COMPUTE_FID = True

# 设备
DEVICE = "cuda"

# LPIPS / MS-SSIM 批大小
PAIR_BATCH_SIZE = 16

# 调试用：只计算前 N 对图片；正式计算设为 None
MAX_PAIRS_FOR_DEBUG = None
# MAX_PAIRS_FOR_DEBUG = 20


# ============================================================
# 2. 输出路径
# ============================================================

OUT_SUMMARY_CSV = os.path.join(RESULT_ROOT, "metrics_summary.csv")
OUT_PAIR_DETAIL_CSV = os.path.join(RESULT_ROOT, "pair_metrics_detail.csv")


# ============================================================
# 3. 依赖导入
# ============================================================

try:
    import lpips
except ImportError:
    raise ImportError("缺少 lpips，请先运行：pip install lpips")

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    raise ImportError("缺少 pytorch-msssim，请先运行：pip install pytorch-msssim")

try:
    from pytorch_fid import fid_score
except ImportError:
    fid_score = None
    print("[WARN] 未安装 pytorch-fid，如果需要 FID，请运行：pip install pytorch-fid")


# ============================================================
# 4. 工具函数
# ============================================================

def list_pngs(folder):
    """
    读取指定文件夹下所有 png 图片，并按文件名排序。
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(folder.glob("*.png"))
    return files


def load_image_tensor_01(path):
    """
    读取图片，转换为 torch Tensor。
    输出范围：[0, 1]
    形状：[3, H, W]
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def load_batch(paths, device):
    """
    批量读取图片。
    输出：[B, 3, H, W]，范围 [0, 1]
    """
    tensors = [load_image_tensor_01(p) for p in paths]
    batch = torch.stack(tensors, dim=0).to(device)
    return batch


def read_generation_log(csv_path):
    """
    读取 generation_log.csv。
    返回每种方法的 time 和 mem 统计。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"generation_log.csv not found: {csv_path}")

    records = {
        "full": [],
        "tome": [],
        "ours": [],
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            method = row.get("method", "").strip()

            if method not in records:
                continue

            if row.get("time_sec") == "ERROR":
                continue

            try:
                time_sec = float(row["time_sec"])
                peak_mem_gb = float(row["peak_mem_gb"])
            except Exception:
                continue

            records[method].append({
                "time_sec": time_sec,
                "peak_mem_gb": peak_mem_gb,
            })

    return records


def summarize_time_mem(records):
    """
    汇总平均时间、平均显存、最大显存。
    """
    summary = {}

    for method, rows in records.items():
        if len(rows) == 0:
            summary[method] = {
                "num_images": 0,
                "avg_time_sec": math.nan,
                "avg_peak_mem_gb": math.nan,
                "max_peak_mem_gb": math.nan,
            }
            continue

        times = np.array([r["time_sec"] for r in rows], dtype=np.float64)
        mems = np.array([r["peak_mem_gb"] for r in rows], dtype=np.float64)

        summary[method] = {
            "num_images": len(rows),
            "avg_time_sec": float(times.mean()),
            "avg_peak_mem_gb": float(mems.mean()),
            "max_peak_mem_gb": float(mems.max()),
        }

    return summary


def compute_lpips_msssim_vs_full(method_name, full_dir, method_dir, device, lpips_model):
    """
    计算某个方法相对于 Full 的 LPIPS 和 MS-SSIM。

    配对方式：
    full/class_0001_seed0_xxx.png
    method/class_0001_seed0_xxx.png
    """
    full_files = list_pngs(full_dir)

    if MAX_PAIRS_FOR_DEBUG is not None:
        full_files = full_files[:MAX_PAIRS_FOR_DEBUG]

    lpips_values = []
    msssim_values = []
    detail_rows = []

    valid_full_paths = []
    valid_method_paths = []

    for full_path in full_files:
        method_path = Path(method_dir) / full_path.name

        if not method_path.exists():
            print(f"[WARN] Missing matched image: {method_path}")
            continue

        valid_full_paths.append(full_path)
        valid_method_paths.append(method_path)

    print(f"[INFO] {method_name}: matched pairs = {len(valid_full_paths)}")

    for start in tqdm(range(0, len(valid_full_paths), PAIR_BATCH_SIZE), desc=f"LPIPS/MS-SSIM {method_name}"):
        end = min(start + PAIR_BATCH_SIZE, len(valid_full_paths))

        full_batch_paths = valid_full_paths[start:end]
        method_batch_paths = valid_method_paths[start:end]

        full_01 = load_batch(full_batch_paths, device)
        method_01 = load_batch(method_batch_paths, device)

        full_lpips = full_01 * 2.0 - 1.0
        method_lpips = method_01 * 2.0 - 1.0

        with torch.no_grad():
            lpips_batch = lpips_model(full_lpips, method_lpips)
            lpips_batch = lpips_batch.view(-1).detach().cpu().numpy()

            for i in range(full_01.shape[0]):
                msssim_val = ms_ssim(
                    full_01[i:i + 1],
                    method_01[i:i + 1],
                    data_range=1.0,
                    size_average=True,
                ).item()

                lpips_val = float(lpips_batch[i])

                lpips_values.append(lpips_val)
                msssim_values.append(float(msssim_val))

                detail_rows.append({
                    "method": method_name,
                    "image_name": full_batch_paths[i].name,
                    "full_path": str(full_batch_paths[i]),
                    "method_path": str(method_batch_paths[i]),
                    "lpips_vs_full": lpips_val,
                    "ms_ssim_vs_full": float(msssim_val),
                })

    avg_lpips = float(np.mean(lpips_values)) if len(lpips_values) > 0 else math.nan
    avg_msssim = float(np.mean(msssim_values)) if len(msssim_values) > 0 else math.nan

    return avg_lpips, avg_msssim, detail_rows


def compute_fid_for_folder(gen_dir, ref_dir, device):
    """
    计算生成图目录与参考图目录之间的 FID。
    """
    if not COMPUTE_FID:
        return math.nan

    if fid_score is None:
        print("[WARN] pytorch-fid 未安装，跳过 FID。")
        return math.nan

    gen_dir = str(gen_dir)
    ref_dir = str(ref_dir)

    if not os.path.exists(gen_dir):
        raise FileNotFoundError(f"Generated image folder not found: {gen_dir}")

    if not os.path.exists(ref_dir):
        raise FileNotFoundError(f"ImageNet reference folder not found: {ref_dir}")

    print(f"[INFO] Computing FID:")
    print(f"       generated = {gen_dir}")
    print(f"       reference = {ref_dir}")

    fid = fid_score.calculate_fid_given_paths(
        paths=[gen_dir, ref_dir],
        batch_size=32,
        device=device,
        dims=2048,
        num_workers=0,
    )

    return float(fid)


def write_pair_detail_csv(rows, save_path):
    """
    保存每一对图像的 LPIPS / MS-SSIM 明细。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "image_name",
                "full_path",
                "method_path",
                "lpips_vs_full",
                "ms_ssim_vs_full",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows, save_path):
    """
    保存最终总表。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "ratio",
                "beta",
                "num_images",
                "fid",
                "lpips_vs_full",
                "ms_ssim_vs_full",
                "avg_time_sec",
                "speedup_vs_full",
                "avg_peak_mem_gb",
                "max_peak_mem_gb",
                "mem_reduction_vs_full_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 5. 主函数
# ============================================================

def main():
    result_root = Path(RESULT_ROOT)

    full_dir = result_root / "full"
    tome_dir = result_root / "tome"
    ours_dir = result_root / "ours"

    for folder in [full_dir, tome_dir, ours_dir]:
        if not folder.exists():
            raise FileNotFoundError(f"Generated image folder not found: {folder}")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------
    # Step 1: 统计 Time / Mem
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 1] Summarize time and memory")
    print("=" * 80)

    gen_records = read_generation_log(GEN_LOG_CSV)
    tm_summary = summarize_time_mem(gen_records)

    for method, s in tm_summary.items():
        print(
            f"{method:>5} | "
            f"num={s['num_images']} | "
            f"avg_time={s['avg_time_sec']:.4f}s | "
            f"avg_mem={s['avg_peak_mem_gb']:.4f}GB | "
            f"max_mem={s['max_peak_mem_gb']:.4f}GB"
        )

    # ------------------------------------------------------------
    # Step 2: LPIPS / MS-SSIM
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 2] Compute LPIPS and MS-SSIM vs Full")
    print("=" * 80)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    tome_lpips, tome_msssim, tome_detail = compute_lpips_msssim_vs_full(
        method_name="tome",
        full_dir=full_dir,
        method_dir=tome_dir,
        device=device,
        lpips_model=lpips_model,
    )

    ours_lpips, ours_msssim, ours_detail = compute_lpips_msssim_vs_full(
        method_name="ours",
        full_dir=full_dir,
        method_dir=ours_dir,
        device=device,
        lpips_model=lpips_model,
    )

    pair_detail_rows = tome_detail + ours_detail
    write_pair_detail_csv(pair_detail_rows, OUT_PAIR_DETAIL_CSV)

    print(f"[DONE] Pair detail saved to: {OUT_PAIR_DETAIL_CSV}")

    # ------------------------------------------------------------
    # Step 3: FID
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 3] Compute FID")
    print("=" * 80)

    fid_full = compute_fid_for_folder(full_dir, IMAGENET_REF_DIR, device)
    fid_tome = compute_fid_for_folder(tome_dir, IMAGENET_REF_DIR, device)
    fid_ours = compute_fid_for_folder(ours_dir, IMAGENET_REF_DIR, device)

    # ------------------------------------------------------------
    # Step 4: 汇总总表
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 4] Write final metrics summary")
    print("=" * 80)

    full_time = tm_summary["full"]["avg_time_sec"]
    full_mem = tm_summary["full"]["avg_peak_mem_gb"]

    def make_row(method, ratio, beta, fid, lpips_val, msssim_val):
        avg_time = tm_summary[method]["avg_time_sec"]
        avg_mem = tm_summary[method]["avg_peak_mem_gb"]
        max_mem = tm_summary[method]["max_peak_mem_gb"]

        if method == "full":
            speedup = 1.0
            mem_red = 0.0
        else:
            speedup = full_time / avg_time
            mem_red = (full_mem - avg_mem) / full_mem * 100.0

        return {
            "method": method,
            "ratio": ratio,
            "beta": beta,
            "num_images": tm_summary[method]["num_images"],
            "fid": f"{fid:.6f}" if not math.isnan(fid) else "",
            "lpips_vs_full": f"{lpips_val:.6f}" if not math.isnan(lpips_val) else "",
            "ms_ssim_vs_full": f"{msssim_val:.6f}" if not math.isnan(msssim_val) else "",
            "avg_time_sec": f"{avg_time:.6f}",
            "speedup_vs_full": f"{speedup:.6f}",
            "avg_peak_mem_gb": f"{avg_mem:.6f}",
            "max_peak_mem_gb": f"{max_mem:.6f}",
            "mem_reduction_vs_full_percent": f"{mem_red:.6f}",
        }

    summary_rows = [
        make_row(
            method="full",
            ratio="0.0",
            beta="-",
            fid=fid_full,
            lpips_val=0.0,
            msssim_val=1.0,
        ),
        make_row(
            method="tome",
            ratio="0.5",
            beta="-",
            fid=fid_tome,
            lpips_val=tome_lpips,
            msssim_val=tome_msssim,
        ),
        make_row(
            method="ours",
            ratio="0.5",
            beta="0.005",
            fid=fid_ours,
            lpips_val=ours_lpips,
            msssim_val=ours_msssim,
        ),
    ]

    write_summary_csv(summary_rows, OUT_SUMMARY_CSV)

    print(f"[DONE] Metrics summary saved to: {OUT_SUMMARY_CSV}")

    print("\n========== Final SD Metrics ==========")
    for row in summary_rows:
        print(
            f"{row['method']:>5} | "
            f"ratio={row['ratio']:>4} | "
            f"beta={row['beta']:>5} | "
            f"FID={row['fid']} | "
            f"LPIPS={row['lpips_vs_full']} | "
            f"MS-SSIM={row['ms_ssim_vs_full']} | "
            f"Time={row['avg_time_sec']}s | "
            f"Speedup={row['speedup_vs_full']}x | "
            f"Mem={row['avg_peak_mem_gb']}GB | "
            f"MemRed={row['mem_reduction_vs_full_percent']}%"
        )


if __name__ == "__main__":
    main()