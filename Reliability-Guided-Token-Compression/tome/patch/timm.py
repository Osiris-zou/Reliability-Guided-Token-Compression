# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from ..merge import merge_source, merge_wavg, bipartite_soft_matching_xincheng
from ..utils import parse_r

class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
# 在标准的attention后，通过调用bipartite_soft_matching函数，根据提供的metric（度量）来调整token。这个过程可能还会涉及到token的大小和来源信息的追踪。
#drop_path操作通常用于模型正则化，减少过拟合。
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)
    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # attn_size从类的_tome_info字典中获取，这个字典在模型初始化或某些配置时填充。prop_attn标志决定是否考虑attention的比例。
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        # self.attn(self.norm1(x), attn_size)调用attention层，并可能使用normalized的输入x和可能的attention尺寸attn_size。
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        # 将attention的输出与输入x相加，通过_drop_path1进行可能的dropout处理。
        x = x + self._drop_path1(x_attn)
        # 从_tome_info中获取并更新r，这代表此次应用ToMe操作期间要合并的token数量。
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            # bipartite_soft_matching是ToMe的核心操作，它基于metric（由attention层输出）和其他参数进行token合并。
            #merge, _ = bipartite_soft_matching(
            merge, _=bipartite_soft_matching_xincheng(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            # 如果启用了trace_source，将更新token的来源信息。
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            # 使用merge_wavg进行加权平均合并，更新x和token的尺寸信息。
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x

class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """
# 这是对Attention模块的一个修改版本，增加了对比例化attention的支持。在这个版本中，attention的计算会根据token的尺寸进行调整（如果提供了尺寸信息的话）。
# 它还修改了返回值，除了返回处理后的token之外，还返回了key向量的均值。
# 父类 Attention 的 forward 方法。
    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
            #
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        """
        修改：
        - 初始化 r、token 尺寸和 token 来源。
        """
# 类的文档注释说明了它的主要修改：初始化用于 ToMe 功能的相关变量，如 r（需要合并的token数量），size（token的尺寸），和 source（token的来源）。
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            # 在执行原有的 forward 方法之前，此方法首先设置 _tome_info 字典中的 r，这通过调用 parse_r 函数完成，该函数根据模型中的 blocks 数量和当前的 r 值来计算新的 r。
            self._tome_info["size"] = None
            # 此外，它将 _tome_info 字典中的 size 和 source 设置为 None，准备在模型的其他部分中根据需要进行更新。
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    """
        应用 ToMe 变换到这个 transformer 模型。之后，使用 model.r 设置 r 的值。

        如果你想了解每个 token 的来源（例如，为了可视化），设置 trace_source 为 true。
        来源信息之后将会在 model._tome_info["source"] 中可用。

        对于比例化注意力，设置 prop_attn 为 True。这仅在从现成模型评估时必要。
        对于训练和从自训练的 MAE 模型评估时，设置这个为 False。
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)
# trace_stource 用于决定是否追踪每个 token 的来源信息，prop_attn 用于决定是否应用比例化注意力。
    model.__class__ = ToMeVisionTransformer
    model.r = 0
# 初始化 model.r 为 0 和 _tome_info 字典，包括 r, size, source, trace_source, prop_attn 等关键信息。
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }
# 设置 class_token 和 distill_token 标志，以指示模型是否具有类 token 或蒸馏 token。
    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True
# 如果模型具有蒸馏 token，更新 _tome_info 中的 distill_token 标志。
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
