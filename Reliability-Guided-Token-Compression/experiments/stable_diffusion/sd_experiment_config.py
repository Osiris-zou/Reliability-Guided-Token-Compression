# sd_experiment_config.py

OUTPUT_ROOT = "results/sd_r05"

METHODS = [
    {"name": "full", "merge_method": "full", "ratio": 0.0, "beta": 0.0},
    {"name": "tome", "merge_method": "tome", "ratio": 0.5, "beta": 0.0},
    {"name": "ours", "merge_method": "ours", "ratio": 0.5, "beta": 0.005},  # 或你最终选定的beta
]

N_IMAGES_PER_CLASS = 2
N_INFERENCE_STEPS = 50
CFG_SCALE = 7.5
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

MERGE_MAX_DOWNSAMPLE = 1
MERGE_SX = 2
MERGE_SY = 2
MERGE_USE_RAND = True

SEEDS = [0, 1]   # 每个类别生成2张
DEVICE = "cuda"