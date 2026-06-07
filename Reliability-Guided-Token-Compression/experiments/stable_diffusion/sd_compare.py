import os
import sys
import time
import importlib.util
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPTokenizer


# ============================================================
# 1. 项目路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SD_DIR = PROJECT_ROOT / "sd"
TOMESD_DIR = PROJECT_ROOT / "tomesd"

if not SD_DIR.exists():
    raise FileNotFoundError(f"Cannot find sd folder: {SD_DIR}")

if not TOMESD_DIR.exists():
    raise FileNotFoundError(f"Cannot find tomesd folder: {TOMESD_DIR}")

# 把 sd、tomesd、项目根目录加入 Python 搜索路径
for p in [PROJECT_ROOT, SD_DIR, TOMESD_DIR]:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================
# 2. 按文件路径导入本地模块，避免 PyCharm 标红和相对导入问题
# ============================================================

def load_py_module(module_name: str, file_path: Path):
    """
    按指定文件路径加载 Python 模块。
    这样不依赖 from xxx import xxx 的包路径解析。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find python file: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# 加载 sd/pipeline.py
pipeline = load_py_module(
    module_name="sd_pipeline",
    file_path=SD_DIR / "pipeline.py",
)

# 加载 sd/model_loader.py
model_loader = load_py_module(
    module_name="sd_model_loader",
    file_path=SD_DIR / "model_loader.py",
)

# 从 model_loader.py 中取出模型加载函数
preload_models_from_standard_weights = model_loader.preload_models_from_standard_weights

# ============================================================
# 用户配置
# ============================================================

CKPT_PATH = r"data\v1-5-pruned-emaonly.ckpt"
TOKENIZER_DIR = r"data"

OUT_DIR = r"results"

DEVICE = "cuda"

PROMPT = "a photo of a dog"
UNCOND_PROMPT = ""

SEED = 42
N_STEPS = 50
CFG_SCALE = 7.5

RATIO = 0.6
BETA = 0.015


def save_image(np_img, save_path):
    img = Image.fromarray(np_img)
    img.save(save_path)
    print(f"[SAVE] {save_path}")


def run_one(method, ratio, beta, models, tokenizer, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()

    image = pipeline.generate(
        prompt=PROMPT,
        uncond_prompt=UNCOND_PROMPT,
        models=models,
        tokenizer=tokenizer,
        device=device,
        seed=SEED,
        n_inference_steps=N_STEPS,
        cfg_scale=CFG_SCALE,

        merge_method=method,
        merge_ratio=ratio,
        merge_beta=beta,
        merge_max_downsample=1,
        merge_sx=2,
        merge_sy=2,
        merge_use_rand=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()

    elapsed = end - start

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    else:
        peak_mem = 0.0

    return image, elapsed, peak_mem


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading tokenizer...")
    tokenizer = CLIPTokenizer(
        os.path.join(TOKENIZER_DIR, "vocab.json"),
        merges_file=os.path.join(TOKENIZER_DIR, "merges.txt"),
    )

    print("[INFO] Loading SD models...")
    models = preload_models_from_standard_weights(CKPT_PATH, device)

    experiments = [
        {
            "name": "full",
            "method": "full",
            "ratio": 0.0,
            "beta": 0.0,
        },
        {
            "name": f"tome_r{RATIO}",
            "method": "tome",
            "ratio": RATIO,
            "beta": 0.0,
        },
        {
            "name": f"ours_r{RATIO}_b{BETA}",
            "method": "ours",
            "ratio": RATIO,
            "beta": BETA,
        },
    ]

    csv_path = os.path.join(OUT_DIR, "sd_compare_results.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,ratio,beta,time_sec,peak_mem_gb\n")

        for exp in experiments:
            print("\n" + "=" * 60)
            print(
                f"[RUN] method={exp['method']}, "
                f"ratio={exp['ratio']}, beta={exp['beta']}"
            )
            print("=" * 60)

            img, elapsed, peak_mem = run_one(
                method=exp["method"],
                ratio=exp["ratio"],
                beta=exp["beta"],
                models=models,
                tokenizer=tokenizer,
                device=device,
            )

            save_path = os.path.join(OUT_DIR, f"{exp['name']}.png")
            save_image(img, save_path)

            f.write(
                f"{exp['method']},{exp['ratio']},{exp['beta']},"
                f"{elapsed:.4f},{peak_mem:.4f}\n"
            )

            print(
                f"[RESULT] method={exp['method']} | "
                f"time={elapsed:.4f}s | peak_mem={peak_mem:.4f}GB"
            )

    print(f"\n[DONE] CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()