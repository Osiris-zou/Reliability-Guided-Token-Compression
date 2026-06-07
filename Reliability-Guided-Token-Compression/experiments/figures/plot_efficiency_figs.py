import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) 全局风格设置
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 颜色可以自己微调
COLOR_FULL = "#1f77b4"   # Full 五角星
COLOR_BAR  = "#ff8c1a"   # GFLOPs 柱子（ToMe / Ours 共用）
COLOR_TOME = "#f39c12"   # ToMe 折线
COLOR_OURS = "#2ecc71"   # Ours 折线

BAR_WIDTH = 0.42         # 柱子更细一点
STAR_SIZE = 220          # 五角星大小
LINE_WIDTH = 2.2
MARKER_SIZE = 7

# =========================
# 2) 数据区（请替换成你的最终结果）
# =========================
# 说明：
# - full_gflops / full_tps：原始模型
# - rs：所有要展示的 r
# - merge_gflops：ToMe/Ours 的 GFLOPs（两者相同，所以只保留一组）
# - tome_tps / ours_tps：ToMe 和 Ours 的 model-only throughput
#
# 如果某个模型没有某些 r，就只填你真实跑过的 r

DATA = {
    "ViT-B/16": {
        "full_gflops": 17.58,
        "full_tps": 1153.37,
        "rs": [4, 8, 12, 16, 20, 25],
        "merge_gflops": [15.34, 13.12, 10.92, 8.78, 7.14, 5.80],
        # 下面吞吐量请替换成你的最终值
        "tome_tps": [1192.38, 1430.93, 1713.21, 2077.79, 2487.78, 2932.20],
        "ours_tps": [1178.25, 1406.70, 1681.17, 2040.54, 2424.03, 2848.76],
    },

    "ViT-L/16": {
        "full_gflops": 61.60,
        "full_tps": 247.74,
        "rs": [4, 8],
        "merge_gflops": [46.10, 31.00],
        # 下面吞吐量请替换成你的最终值
        "tome_tps": [318.48, 474.62],
        "ours_tps": [315.20, 467.96],
    },

    "DeiT-S/16": {
        "full_gflops": 4.61,
        "full_tps": 3146.45,
        "rs": [4, 8, 12, 16, 20, 25],
        "merge_gflops": [4.02, 3.43, 2.85, 2.30, 1.88, 1.53],
        # 下面吞吐量请替换成你稳定测试后的最终值
        "tome_tps": [3145.09, 3495.93, 3386.76, 3326.15, 3276.43, 3227.47],
        "ours_tps": [3028.47, 3097.76, 2983.53, 2939.86, 2894.08, 2887.90],
    },

    "DeiT-B/16": {
        "full_gflops": 17.58,
        "full_tps": 1167.02,
        "rs": [4, 8, 12, 16, 20, 25],
        "merge_gflops": [15.34, 13.12, 10.92, 8.78, 7.14, 5.80],
        # 下面吞吐量请替换成你的最终值
        "tome_tps": [1208.06, 1442.48, 1732.12, 2098.87, 2505.07, 2946.83],
        "ours_tps": [1188.17, 1413.55, 1692.60, 2046.72, 2432.34, 2844.45],
    },
}

PANEL_TITLES = {
    "ViT-B/16": "(a) ViT-B/16",
    "ViT-L/16": "(b) ViT-L/16",
    "DeiT-S/16": "(c) DeiT-S/16",
    "DeiT-B/16": "(d) DeiT-B/16",
}

# =========================
# 3) 单个 GFLOPs 子图
# =========================
def plot_gflops_panel(ax, model_name, cfg):
    rs = cfg["rs"]
    full_gflops = cfg["full_gflops"]
    merge_gflops = cfg["merge_gflops"]

    x = np.arange(len(rs))

    # 只画 ToMe/Ours 共享的 GFLOPs 柱子
    bars = ax.bar(
        x, merge_gflops,
        width=BAR_WIDTH,
        color=COLOR_BAR,
        edgecolor="black",
        linewidth=0.8,
        label="ToMe / Ours"
    )

    # 在左边放一个 full 的五角星
    star_x = -0.75
    ax.scatter(
        star_x, full_gflops,
        marker="*",
        s=STAR_SIZE,
        color=COLOR_FULL,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="Full"
    )

    # Full 标注
    ax.text(
        star_x, full_gflops * 1.02,
        "Full",
        ha="center", va="bottom", fontsize=10
    )

    # 每个柱子上标 GFLOPs reduction
    for xi, yi in zip(x, merge_gflops):
        reduction = (1.0 - yi / full_gflops) * 100.0
        ax.text(
            xi, yi + 0.02 * full_gflops,
            f"-{reduction:.1f}%",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_title(PANEL_TITLES[model_name], fontsize=15, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$r={r}$" for r in rs], fontsize=12)
    ax.set_xlabel(r"Merging rate $r$", fontsize=13)
    ax.set_ylabel("GFLOPs", fontsize=13)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 给左边五角星留空间
    ax.set_xlim(star_x - 0.35, len(rs) - 0.4)

# =========================
# 4) 单个 Throughput 子图
# =========================
def plot_throughput_panel(ax, model_name, cfg):
    rs = cfg["rs"]
    full_tps = cfg["full_tps"]
    tome_tps = cfg["tome_tps"]
    ours_tps = cfg["ours_tps"]

    x = np.arange(len(rs))
    star_x = -0.75

    # Full 五角星
    ax.scatter(
        star_x, full_tps,
        marker="*",
        s=STAR_SIZE,
        color=COLOR_FULL,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="Full"
    )
    ax.text(
        star_x, full_tps * 1.02,
        "Full",
        ha="center", va="bottom", fontsize=10
    )

    # ToMe 曲线
    ax.plot(
        x, tome_tps,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        color=COLOR_TOME,
        label="ToMe"
    )

    # Ours 曲线
    ax.plot(
        x, ours_tps,
        marker="D",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        color=COLOR_OURS,
        label="Ours"
    )

    # 可选：在 Ours 上标 speedup（相对 Full）
    for xi, yi in zip(x, ours_tps):
        speedup = yi / full_tps
        ax.text(
            xi, yi * 1.01,
            f"{speedup:.2f}×",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_title(PANEL_TITLES[model_name], fontsize=15, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$r={r}$" for r in rs], fontsize=12)
    ax.set_xlabel(r"Merging rate $r$", fontsize=13)
    ax.set_ylabel("Images / sec", fontsize=13)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(star_x - 0.35, len(rs) - 0.4)

# =========================
# 5) 绘制 GFLOPs 图
# =========================
def draw_gflops_figure(save_path="figure7_gflops_revised.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    model_order = ["ViT-B/16", "ViT-L/16", "DeiT-S/16", "DeiT-B/16"]

    for ax, model_name in zip(axes, model_order):
        plot_gflops_panel(ax, model_name, DATA[model_name])

    # 只在第一个子图取 legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02)
    )

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# 6) 绘制 Throughput 图
# =========================
def draw_throughput_figure(save_path="figure7_throughput_revised.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    model_order = ["ViT-B/16", "ViT-L/16", "DeiT-S/16", "DeiT-B/16"]

    for ax, model_name in zip(axes, model_order):
        plot_throughput_panel(ax, model_name, DATA[model_name])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02)
    )

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# 7) 主函数
# =========================
if __name__ == "__main__":
    draw_gflops_figure("figure7_gflops_revised.png")
    draw_throughput_figure("figure7_throughput_revised.png")