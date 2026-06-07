import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import timm
import tome
from PIL import Image
from torchvision import transforms

from fig6_debug_utils import (
    compute_edge_debug_with_scores,
    imagenet_denormalize,
    source_idx_to_grid_xy,
)


# ============================================================
# Global figure style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 13,
})


# ============================================================
# Basic utilities
# ============================================================

def load_image(path: str, device: torch.device):
    """
    作用：
    读取一张 ImageNet 验证图片，并使用 timm ViT/DeiT 的常规 ImageNet 预处理。
    """
    pil_img = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    x = transform(pil_img).unsqueeze(0).to(device)
    return x


def run_model_and_get_debug(model, x, r, beta):
    """
    作用：
    前向运行一次模型，读取指定层保存的 metric，
    然后离线计算 ToMe 与 Ours 的 edge-ranking 差异。
    """
    model._tome_info["fig6_debug_metric"] = None

    with torch.no_grad():
        _ = model(x)

    metric = model._tome_info.get("fig6_debug_metric", None)

    if metric is None:
        raise RuntimeError(
            "No fig6_debug_metric was saved. "
            "请检查 tome/patch/timm.py 中是否已经加入 fig6_debug_metric 保存逻辑。"
        )

    debug = compute_edge_debug_with_scores(
        metric=metric,
        r=r,
        beta=beta,
        class_token=True,
        distill_token=False,
    )

    if debug is None:
        raise RuntimeError("debug is None. 请检查 r 是否过大。")

    return debug


def to_numpy_1d(x):
    """
    作用：
    把 tensor 转成 numpy 一维数组。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mean_value(values, indices):
    """
    作用：
    计算一组 token 的平均值。
    """
    if len(indices) == 0:
        return 0.0
    if isinstance(values, torch.Tensor):
        return float(values[indices].mean().item())
    return float(np.asarray(values)[indices].mean())


# ============================================================
# Spatial token plotting
# ============================================================

def scatter_source_tokens(
    ax,
    image_np,
    source_indices,
    color,
    label=None,
    marker="o",
    size=105,
    grid_size=14,
    alpha=0.95,
    linewidth=0.7,
):
    """
    作用：
    把 source token 在原图上的位置画出来。
    """
    h, w = image_np.shape[:2]
    cell_w = w / grid_size
    cell_h = h / grid_size

    xs, ys = [], []

    for src_idx in source_indices:
        pos = source_idx_to_grid_xy(
            src_idx,
            grid_size=grid_size,
            class_token=True,
        )

        if pos is None:
            continue

        gx, gy = pos
        xs.append((gx + 0.5) * cell_w)
        ys.append((gy + 0.5) * cell_h)

    if len(xs) > 0:
        ax.scatter(
            xs,
            ys,
            s=size,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=linewidth,
            alpha=alpha,
            label=label,
            zorder=5,
        )


def get_low_margin_indices(debug, percentile=25, max_tokens=12):
    """
    作用：
    选出低 margin source tokens。
    这些 token 表示 top-1 和 top-2 target similarity 接近，
    即 merge decision 更不确定。
    """
    margin = to_numpy_1d(debug["margin"])

    valid = []

    for src_idx, m in enumerate(margin):
        pos = source_idx_to_grid_xy(
            src_idx,
            grid_size=14,
            class_token=True,
        )

        if pos is None:
            continue

        if not np.isfinite(m):
            continue

        valid.append((src_idx, float(m)))

    if len(valid) == 0:
        return []

    margins = np.array([item[1] for item in valid])
    threshold = np.percentile(margins, percentile)

    low = [(idx, m) for idx, m in valid if m <= threshold]
    low = sorted(low, key=lambda x: x[1])[:max_tokens]

    return [idx for idx, _ in low]


def plot_image_diagnosis(ax, image_np, debug):
    """
    作用：
    左图：在原图上叠加 low-margin tokens 和 changed selected tokens。
    """
    ax.imshow(image_np)
    ax.set_title("Spatial diagnosis", fontsize=16, pad=6)
    ax.axis("off")
    ax.set_anchor("C")

    low_margin_indices = get_low_margin_indices(
        debug,
        percentile=25,
        max_tokens=12,
    )

    scatter_source_tokens(
        ax,
        image_np=image_np,
        source_indices=low_margin_indices,
        color="red",
        label="Low margin",
        marker="s",
        size=90,
        alpha=0.75,
        linewidth=0.55,
    )

    scatter_source_tokens(
        ax,
        image_np=image_np,
        source_indices=debug["common_src"],
        color="lightgray",
        label="Common",
        marker="o",
        size=34,
        alpha=0.78,
        linewidth=0.45,
    )

    scatter_source_tokens(
        ax,
        image_np=image_np,
        source_indices=debug["tome_only_src"],
        color="orange",
        label="ToMe only",
        marker="o",
        size=135,
        alpha=0.98,
        linewidth=0.75,
    )

    scatter_source_tokens(
        ax,
        image_np=image_np,
        source_indices=debug["ours_only_src"],
        color="limegreen",
        label="Ours only",
        marker="o",
        size=135,
        alpha=0.98,
        linewidth=0.75,
    )


# ============================================================
# Edge-ranking diagnosis
# ============================================================

def set_compact_xlim(ax, sim_all, sim_focus=None, left_pad=0.008, right_pad=0.012):
    """
    作用：
    压缩 Edge-ranking diagnosis 的横轴范围，避免左侧空白过大。

    逻辑：
    - 用所有候选点的分位数确定主体范围；
    - 强制保留 ToMe-only 和 Ours-only 等关键点；
    - 如果范围过窄，则保留最小可读宽度。
    """
    sim_all = np.asarray(sim_all, dtype=np.float64)
    sim_all = sim_all[np.isfinite(sim_all)]

    if len(sim_all) == 0:
        ax.set_xlim(0.0, 1.0)
        return 0.0, 1.0

    x_low = np.percentile(sim_all, 8)
    x_high = np.percentile(sim_all, 99.5)

    if sim_focus is not None and len(sim_focus) > 0:
        sim_focus = np.asarray(sim_focus, dtype=np.float64)
        sim_focus = sim_focus[np.isfinite(sim_focus)]
        if len(sim_focus) > 0:
            x_low = min(x_low, float(sim_focus.min()) - left_pad)
            x_high = max(x_high, float(sim_focus.max()) + right_pad)

    x_low = max(0.0, x_low - left_pad)
    x_high = min(1.0, x_high + right_pad)

    # 防止横轴范围过窄导致点挤成一团。
    min_width = 0.10
    if x_high - x_low < min_width:
        mid = 0.5 * (x_low + x_high)
        x_low = max(0.0, mid - min_width / 2)
        x_high = min(1.0, mid + min_width / 2)

    ax.set_xlim(x_low, x_high)
    return x_low, x_high


def pick_arrow_pair(sim, margin, tome_only_idx, ours_only_idx):
    """
    作用：
    选择一对真实 token 点来画箭头，保证箭头从橙色 ToMe-only 点
    指向绿色 Ours-only 点，而不是指向均值位置。

    选择规则：
    - 起点：ToMe-only 中 margin 最低的点；
    - 终点：Ours-only 中 margin 更高且 similarity 最接近起点的点；
    - 如果没有 margin 更高的 Ours-only，则选 Ours-only 中 margin 最大的点。
    """
    tome_only_idx = np.asarray(tome_only_idx, dtype=int)
    ours_only_idx = np.asarray(ours_only_idx, dtype=int)

    if len(tome_only_idx) == 0 or len(ours_only_idx) == 0:
        return None

    t_idx = tome_only_idx[np.argmin(margin[tome_only_idx])]

    better = [i for i in ours_only_idx if margin[i] > margin[t_idx]]

    if len(better) > 0:
        o_idx = min(better, key=lambda i: abs(sim[i] - sim[t_idx]))
    else:
        o_idx = ours_only_idx[np.argmax(margin[ours_only_idx])]

    return int(t_idx), int(o_idx)


def add_replacement_arrow(ax, sim, margin, tome_only_idx, ours_only_idx):
    """
    作用：
    在 Edge-ranking diagnosis 中画一个代表性替换箭头。
    箭头表示：从 ToMe-only 低 margin 候选，替换到 Ours-only 高 margin 候选。
    """
    pair = pick_arrow_pair(sim, margin, tome_only_idx, ours_only_idx)
    if pair is None:
        return

    t_idx, o_idx = pair

    x0, y0 = float(sim[t_idx]), float(margin[t_idx])
    x1, y1 = float(sim[o_idx]), float(margin[o_idx])

    # 箭头严格连接橙点与绿点。
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            lw=2.0,
            color="#333333",
            alpha=0.82,
            shrinkA=6,
            shrinkB=6,
        ),
        zorder=7,
    )

    # 文字放到箭头左侧空白区域，避免压在线上。
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_span = xmax - xmin
    y_span = ymax - ymin

    # 优先放在线段左侧；如果太靠左，则贴近坐标轴左内侧。
    text_x = min(x0, x1) - 0.14 * x_span
    text_x = max(xmin + 0.035 * x_span, text_x)

    text_y = y0 + 0.58 * (y1 - y0)
    text_y = min(max(text_y, ymin + 0.18 * y_span), ymax - 0.12 * y_span)

    ax.text(
        text_x,
        text_y,
        "margin\nincrease",
        fontsize=11,
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.18",
            fc="white",
            ec="none",
            alpha=0.78,
        ),
        zorder=8,
    )


def plot_ranking_scatter(ax, debug):
    """
    作用：
    中图：在 similarity-margin 空间中画所有 candidate source tokens。

    横轴：original similarity score
    纵轴：top-1 / top-2 confidence margin
    """
    similarity = to_numpy_1d(debug["similarity"])
    margin = to_numpy_1d(debug["margin"])

    valid = np.isfinite(similarity) & np.isfinite(margin)
    all_idx = np.arange(len(similarity))
    valid_idx = all_idx[valid]

    tome_only = np.array(debug["tome_only_src"], dtype=int)
    ours_only = np.array(debug["ours_only_src"], dtype=int)
    common = np.array(debug["common_src"], dtype=int)

    selected = set(debug["tome_src"]) | set(debug["ours_src"])
    unselected = np.array([i for i in valid_idx if i not in selected], dtype=int)

    ax.set_title("Edge-ranking diagnosis", fontsize=16, pad=6)
    ax.set_anchor("C")

    if len(unselected) > 0:
        ax.scatter(
            similarity[unselected],
            margin[unselected],
            s=22,
            c="lightgray",
            alpha=0.42,
            label="Unselected",
            edgecolors="none",
            zorder=1,
        )

    if len(common) > 0:
        ax.scatter(
            similarity[common],
            margin[common],
            s=44,
            c="gray",
            alpha=0.76,
            label="Common",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

    if len(tome_only) > 0:
        ax.scatter(
            similarity[tome_only],
            margin[tome_only],
            s=118,
            c="orange",
            alpha=0.98,
            label="ToMe only",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )

    if len(ours_only) > 0:
        ax.scatter(
            similarity[ours_only],
            margin[ours_only],
            s=118,
            c="limegreen",
            alpha=0.98,
            label="Ours only",
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
        )

    # 先设置 y 轴，再压缩 x 轴。
    valid_margin = margin[valid]
    if len(valid_margin) > 0:
        y_high = max(
            float(np.percentile(valid_margin, 99.5)),
            float(margin[tome_only].max()) if len(tome_only) > 0 else 0.0,
            float(margin[ours_only].max()) if len(ours_only) > 0 else 0.0,
        )
        y_high = max(y_high * 1.12, 0.08)
        ax.set_ylim(-0.03 * y_high, y_high)

    focus_list = []
    if len(tome_only) > 0:
        focus_list.append(similarity[tome_only])
    if len(ours_only) > 0:
        focus_list.append(similarity[ours_only])
    if len(common) > 0:
        focus_list.append(similarity[common])

    focus_sim = np.concatenate(focus_list) if len(focus_list) > 0 else similarity[valid]
    set_compact_xlim(ax, sim_all=similarity[valid], sim_focus=focus_sim)

    add_replacement_arrow(
        ax=ax,
        sim=similarity,
        margin=margin,
        tome_only_idx=tome_only,
        ours_only_idx=ours_only,
    )

    ax.set_xlabel("Similarity score", fontsize=13, labelpad=2)
    ax.set_ylabel("Top-1 / top-2 margin", fontsize=13, labelpad=0)
    ax.yaxis.set_label_coords(-0.075, 0.5)

    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.32)


# ============================================================
# Margin summary and row summary table
# ============================================================

def plot_margin_summary(ax, debug):
    """
    作用：
    右图：柱状图展示 ToMe-only 与 Ours-only 的平均 margin。
    关键改动：
    - y 轴放到右侧，避免和中间 Edge-ranking diagnosis 重叠。
    """
    ax.set_title("Margin summary", fontsize=16, pad=6)

    tome_only = debug["tome_only_src"]
    ours_only = debug["ours_only_src"]

    tome_margin = mean_value(debug["margin"], tome_only)
    ours_margin = mean_value(debug["margin"], ours_only)

    labels = ["ToMe-only", "Ours-only"]
    values = [tome_margin, ours_margin]
    x = np.arange(2)

    ax.bar(
        x,
        values,
        width=0.52,
        color=["orange", "limegreen"],
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    # 把 y 轴放到右侧，解决和中间图重叠的问题。
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("Mean margin", fontsize=13, labelpad=8)
    ax.tick_params(axis="y", labelsize=11, pad=2)
    ax.tick_params(axis="x", labelsize=11, pad=2)

    ymax = max(max(values) * 1.35, 0.05)
    ax.set_ylim(0, ymax)

    for i, value in enumerate(values):
        ax.text(
            i,
            value + ymax * 0.04,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.grid(axis="y", linestyle="--", alpha=0.32)


def add_group_summary_text(ax_text, debug, r, beta, debug_layer):
    """
    作用：
    在每组图的正下方添加紧凑 summary 表格。
    表格居中缩短，避免横跨整张图。
    """
    ax_text.axis("off")

    tome_only = debug["tome_only_src"]
    ours_only = debug["ours_only_src"]

    tome_cal = mean_value(debug["ours_score"], tome_only)
    ours_cal = mean_value(debug["ours_score"], ours_only)

    headers = [
        "Setting",
        "Changed tokens",
        "Margin gap",
        "Mean calibrated score",
    ]

    values = [
        f"r={r}, β={beta}, layer={debug_layer}",
        f"{len(tome_only)}/{debug['r_actual']} ({debug['changed_ratio']:.2f})",
        f"{debug['margin_gap']:.3f}",
        f"ToMe-only={tome_cal:.3f}   |   Ours-only={ours_cal:.3f}",
    ]

    table = ax_text.table(
        cellText=[headers, values],
        cellLoc="center",
        loc="center",
        colWidths=[0.24, 0.20, 0.16, 0.40],

        # x, y, width, height
        # 表格宽度 0.76，比之前更短，和文字长度更匹配。
        bbox=[0.12, 0.07, 0.76, 0.82],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 1.20)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.6)
        cell.get_text().set_fontname("Times New Roman")

        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-json", type=str, required=True)
    parser.add_argument("--image-index", type=int, nargs="+", default=[0, 3, 7, 4])
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--r", type=int, default=12)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--debug-layer", type=int, default=7)
    parser.add_argument("--out", type=str, default="fig6_ranking_4groups_final_v6.png")
    parser.add_argument("--summary-json", type=str, default="fig6_ranking_4groups_final_v6.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.candidates_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    selected_candidates = [candidates[i] for i in args.image_index]

    print("========== Plot Fig. 6 Ranking Diagnostic ==========")
    print(f"model       : {args.model_name}")
    print(f"r           : {args.r}")
    print(f"beta        : {args.beta}")
    print(f"debug layer : {args.debug_layer}")
    print(f"output      : {args.out}")
    print("selected images:")
    for item in selected_candidates:
        print(item["path"])
    print("====================================================")

    model = timm.create_model(args.model_name, pretrained=args.pretrained).to(device)
    model.eval()

    # 应用 ToMe patch。这里 forward 时不实际 merge，只保存同一层 metric。
    tome.patch.timm(model, trace_source=False)
    model.r = 0

    model._tome_info["fig6_debug"] = True
    model._tome_info["fig6_debug_layer"] = args.debug_layer
    model._tome_info["fig6_debug_metric"] = None

    num_groups = len(selected_candidates)

    # =========================
    # Final compact layout
    # =========================
    fig = plt.figure(figsize=(15.2, 3.62 * num_groups))

    outer_gs = GridSpec(
        nrows=num_groups,
        ncols=1,
        figure=fig,
        hspace=0.14,
    )

    summary = []

    all_handles = []
    all_labels = []

    for g, item in enumerate(selected_candidates):
        path = item["path"]

        x = load_image(path, device)
        image_np = imagenet_denormalize(x)

        debug = run_model_and_get_debug(
            model=model,
            x=x,
            r=args.r,
            beta=args.beta,
        )

        # 每个 group 内部：
        # 第一行：三张图；
        # 第二行：summary table。
        inner_gs = outer_gs[g].subgridspec(
            nrows=2,
            ncols=3,
            height_ratios=[1.0, 0.22],

            # 关键：
            # 中间 Edge-ranking diagnosis 不再过宽，右图 y 轴也移动到右侧。
            width_ratios=[0.66, 1.18, 0.82],

            # wspace 比之前略大，避免左图、中图、右图互相压线。
            hspace=0.34,
            wspace=0.16,
        )

        ax_img = fig.add_subplot(inner_gs[0, 0])
        ax_scatter = fig.add_subplot(inner_gs[0, 1])
        ax_bar = fig.add_subplot(inner_gs[0, 2])
        ax_text = fig.add_subplot(inner_gs[1, :])

        plot_image_diagnosis(
            ax_img,
            image_np=image_np,
            debug=debug,
        )

        plot_ranking_scatter(
            ax_scatter,
            debug=debug,
        )

        plot_margin_summary(
            ax_bar,
            debug=debug,
        )

        add_group_summary_text(
            ax_text,
            debug=debug,
            r=args.r,
            beta=args.beta,
            debug_layer=args.debug_layer,
        )

        handles1, labels1 = ax_img.get_legend_handles_labels()
        handles2, labels2 = ax_scatter.get_legend_handles_labels()

        for h, l in zip(handles1 + handles2, labels1 + labels2):
            all_handles.append(h)
            all_labels.append(l)

        summary_item = {
            "path": path,
            "r": args.r,
            "beta": args.beta,
            "debug_layer": args.debug_layer,
            "r_actual": debug["r_actual"],
            "changed_ratio": debug["changed_ratio"],
            "num_tome_only": len(debug["tome_only_src"]),
            "num_ours_only": len(debug["ours_only_src"]),
            "tome_only_margin": debug["tome_only_margin"],
            "ours_only_margin": debug["ours_only_margin"],
            "margin_gap": debug["margin_gap"],
            "tome_only_src": debug["tome_only_src"],
            "ours_only_src": debug["ours_only_src"],
        }
        summary.append(summary_item)

        print(f"\nImage: {path}")
        print(f"changed_ratio    : {debug['changed_ratio']:.4f}")
        print(f"ToMe-only margin : {debug['tome_only_margin']:.6f}")
        print(f"Ours-only margin : {debug['ours_only_margin']:.6f}")
        print(f"margin_gap       : {debug['margin_gap']:.6f}")
        print(f"ToMe-only src    : {debug['tome_only_src']}")
        print(f"Ours-only src    : {debug['ours_only_src']}")

    # =========================
    # Global legend
    # =========================
    handle_dict = {}
    for h, l in zip(all_handles, all_labels):
        handle_dict[l] = h

    desired_order = ["Low margin", "Common", "ToMe only", "Ours only", "Unselected"]
    handles_final = []
    labels_final = []

    for name in desired_order:
        if name in handle_dict:
            labels_final.append(name)
            handles_final.append(handle_dict[name])

    if len(handles_final) > 0:
        fig.legend(
            handles_final,
            labels_final,
            loc="lower center",
            ncol=5,
            fontsize=13,
            frameon=True,
            fancybox=True,
            markerscale=1.55,
            handletextpad=0.65,
            columnspacing=1.45,
            borderpad=0.55,

            # 图例上移，且字号更大。
            bbox_to_anchor=(0.5, 0.064),
        )

    # 手动控制整体边距，不使用 tight_layout，避免表格被压缩。
    fig.subplots_adjust(
        left=0.042,
        right=0.970,
        top=0.982,
        bottom=0.083,
    )

    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved figure to: {args.out}")
    print(f"Saved summary to: {args.summary_json}")


if __name__ == "__main__":
    main()
