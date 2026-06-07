# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
import math
from typing import Callable, Tuple

def do_nothing(x, mode="mean"):
    return x


def knn_augment(
        tokens: torch.Tensor,
        k: int = 1,
        distance_metric: str = "euclidean"
) -> torch.Tensor:
    """
    向量化版本的 KNN 最近邻融合
    """
    batch_size, num_tokens, token_dim = tokens.shape

    # 计算距离矩阵
    if distance_metric == "euclidean":
        dists = torch.cdist(tokens, tokens, p=2)
    elif distance_metric == "manhattan":
        dists = torch.cdist(tokens, tokens, p=1)
    elif distance_metric == "chebyshev":
        dists = torch.cdist(tokens, tokens, p=float('inf'))
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    # 创建掩码排除自身
    mask = torch.eye(num_tokens, device=tokens.device).bool().unsqueeze(0)
    dists.masked_fill_(mask, float('inf'))

    # 获取最近的k个邻居索引
    _, knn_indices = torch.topk(dists, k=k, dim=-1, largest=False)

    # 修复索引操作 - 正确收集邻居tokens
    batch_idx = torch.arange(batch_size, device=tokens.device).view(batch_size, 1, 1)
    token_idx = knn_indices.unsqueeze(-1).expand(-1, -1, -1, token_dim)

    # 使用 gather 正确收集邻居
    neighbors = tokens.unsqueeze(1).expand(-1, num_tokens, -1, -1)
    neighbors = neighbors.gather(dim=2, index=token_idx)

    # 计算邻居均值 [batch_size, num_tokens, token_dim]
    neighbors_mean = neighbors.mean(dim=2)

    # 融合tokens
    augmented_tokens = 0.9 * tokens + 0.1 * neighbors_mean

    return augmented_tokens


def contextual_window_augment(
        tokens: torch.Tensor,
        window_size: int = 4
) -> torch.Tensor:
    """
    动态上下文窗口融合：
    - 首尾单 token 直接保留。
    - 中间部分按 window_size 分组，计算窗口内均值后加权融合。
    """
    n, t, c = tokens.shape
    augmented_tokens = tokens.clone()

    if t <= 1:
        return tokens

    # 第一个 token 单独处理
    if t > 1:
        # 第二个 token 的窗口为 [0, 1, 2]（如果存在）
        if t >= 3:
            window = tokens[:, 0:3, :]
            augmented_tokens[:, 1, :] = 0.8 * tokens[:, 1, :] + 0.2 * window.mean(dim=1)

    # 中间部分按窗口分组
    for i in range(2, t - 2, window_size):
        if i + window_size <= t:
            window = tokens[:, i:i + window_size, :]
            mean = window.mean(dim=1, keepdim=True)
            augmented_tokens[:, i:i + window_size, :] = 0.8 * window + 0.2 * mean.expand(-1, window_size, -1)

    # 最后一个 token 单独处理
    if t >= 2:
        # 倒数第二个 token 的窗口为 [t-3, t-2, t-1]（如果存在）
        if t >= 3:
            window = tokens[:, -3:, :]
            augmented_tokens[:, -2, :] = 0.8 * tokens[:, -2, :] + 0.2 * window.mean(dim=1)

    return augmented_tokens


def bipartite_soft_matching_xincheng(
        metric: torch.Tensor,
        r: int,
        class_token: bool = False,
        distill_token: bool = False,
        window_size: int = 4,
        knn_k: int = 1
) -> Tuple[Callable, Callable]:
    """
    改进的双边软匹配：
    1. 对 A/B 组 token 分别进行上下文窗口增强。
    2. 对每个 token 进行 KNN 最近邻融合。
    3. 计算相似度时加入 L2 距离惩罚项。
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return lambda x: x, lambda x: x

    with torch.no_grad():
        # Step 1: 上下文窗口增强
        metric = contextual_window_augment(metric, window_size=window_size)

        # Step 2: KNN 最近邻融合
        metric = knn_augment(metric, k=knn_k, distance_metric="euclidean")

        # Step 3: 分组并计算相似度（加入 L2 距离惩罚）
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]

        # 相似度得分（余弦相似度）
        sim_scores = a @ b.transpose(-1, -2)

        # L2 距离惩罚项（归一化到 [0, 1]）
        dist_penalty = torch.cdist(a, b, p=2)
        dist_penalty = dist_penalty / dist_penalty.max()

        # 综合得分 = 相似度 - λ * 距离（λ 为平衡系数）
        combined_scores = sim_scores - 0.3 * dist_penalty

        if class_token:
            combined_scores[..., 0, :] = -math.inf
        if distill_token:
            combined_scores[..., :, 0] = -math.inf

        # 选择 top-r 最相似的配对
        node_max, node_idx = combined_scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # 未合并的 token
        src_idx = edge_idx[..., :r, :]  # 待合并的 token（A 组）
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # 目标 token（B 组）

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size

def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
