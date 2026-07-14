#这个是convnext和segformer一起的
# Copyright (c) OpenMMLab. All rights reserved.
"""Backbone definitions for TAFE-Net (AAAI 2026), serving the MFFE frequency mining module.

This file gathers both ConvNeXt and SegFormer backbones; the two core classes actually used by MFFE are:

- ``HubVisionTransformer0521``: the low-frequency Transformer branch of MFFE =
  paper eq(7) {F_l1, F_l2} = TransformerBlocks(Il_hat) (input Il_hat is the
  low-freq VFIM output; here fed as the low-freq view / RGB modality stream).
  SegFormer style, num_layers=[3,4,6,3]. It is a multi-modal hub that takes
  multiple modalities (e.g. modals=['img','img1']); in the per-stage forward it
  first does patch embedding, then uses token-select to fuse the multi-modal
  features into a single stream by weighting, and finally feeds them into the
  Transformer encoder layers, outputting per-stage low-freq features F_l for 4
  stages (two of which are the {F_l1, F_l2} consumed by scSE fusion eq(8)).
- ``ConvNeXt_0521``: the high-frequency CNN branch of MFFE = paper eq(6)
  {F_h1, F_h2} = CNNBlocks(Ih_hat) (input Ih_hat is the high-freq VFIM output;
  ConvNeXt-V2 tiny, use_grn=True), adapted to 16-channel JPEG-DCT input; it is
  also the CNN baseline of TAFE-Net* in the paper, outputting per-stage high-freq
  features F_h for 4 stages (two of which are the {F_h1, F_h2} used by eq(8)).

The remaining classes in this file (e.g. the various ModalSelector /
PatchEmbedParallel / AsymCMNeXt_0521, etc.) are historical experiments or
spare components and are not annotated individually here.
"""
import math
import warnings
from typing import Sequence
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from ..builder import build_loss
from typing import List, Optional, Tuple, Dict
from ..utils import resize
from mmcv.cnn.bricks import DropPath
import functools
from functools import partial
import timm
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmengine.utils import to_2tuple

from mmseg.models.toys.ffm import FeatureFusionModule as FFM
from mmseg.models.toys.ffm import FeatureRectifyModule as FRM
from mmseg.models.toys.ffm import ChannelEmbed
from mmseg.models.toys.mspa import MSPABlock

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from ..toys.consistency import ToSimiVolume,CrossAttnBlock,ToSimiVolumeEx
from ..toys.fusers import NATFuserBlock
from ..backbones.mit import TransformerEncoderLayer
from mmpretrain.models.backbones.convnext import ConvNeXtBlock
from mmpretrain.models.backbones.convnext import build_norm_layer as build_norm_layer1
from ..utils import InvertedResidualV3 as InvertedResidual
from ..toys.pim_v1 import PRIM1


# =============================================================================
# TRAINING-FLOW / MFFE encoder branches  (file: aysm_cmnext_0521.py)
# -----------------------------------------------------------------------------
# This file collects the custom backbone building blocks used by TAFE-Net's
# MFFE (Multi-Frequency Feature Extractor), the paper "Frequency Mining
# Empowered by Text Aggregation" (AAAI 2026). MFFE encodes the two frequency
# views produced by the VFIM module; the paper states "we separately employ a
# CNN backbone and a Transformer backbone to encode I_hat_h and I_hat_l".
#
# Only TWO classes in this file are annotated for the paper<->code mapping:
#
#   * HubVisionTransformer0521 = the MFFE *Transformer branch*. A MiT/SegFormer
#     style pyramid encoder (OverlapPatchEmbed + TransformerEncoderLayer stages)
#     that encodes the VFIM low-frequency output I_hat_l into the low-freq
#     features {F_l1, F_l2}. It is modality-aware (optional per-token modal
#     re-weighting via `tokenselect`). ACTIVE: registered with
#     @MODELS.register_module() and instantiated by the DocTamper backbone
#     (AsymCMNeXt_0521, below) as the low-frequency Transformer branch.
#
#   * ConvNeXt_0521 = the MFFE *CNN branch* definition for the high-frequency
#     view I_hat_h -> {F_h1, F_h2}. A standard 4-stage ConvNeXt v1/v2.
#     PAPER DIFF: in the DocTamper backbone (AsymCMNeXt_0521 / AsymCMNeXt_0524) this
#     class is NOT instantiated -- the CNN branch is built from a timm
#     'convnextv2_tiny' instead (self.extra1_branch = timm.create_model(...),
#     see AsymCMNeXt_0521 below). ConvNeXt_0521 is therefore the config-declared
#     but INACTIVE CNN-branch build.
#
# All other classes here are alternative / dead helper variants that are not on
# the DocTamper training path, and are left un-annotated.
# =============================================================================


# INACTIVE on the DocTamper path (not reached from HubVisionTransformer0521): SegFormer/MiT
# depth-wise 3x3 conv token mixer for the custom Mix-FFN below; only reached from MLP ->
# the never-instantiated ChannelProcessing.
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

# INACTIVE on the DocTamper path: mixed 3x3 / 5x5 depth-wise conv (split-channel); used only by
# MixCFN, which is never instantiated.
class MixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups // 2,
            bias=bias,
        )
        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=groups - groups // 2,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x = torch.cat([x1, x2], dim=1)
        return x


# INACTIVE on the DocTamper path: SegFormer-style Mix-CFN feed-forward (fc -> MixConv2d -> fc);
# never instantiated (the active transformer branch uses mit.TransformerEncoderLayer's FFN).
class MixCFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        feedforward_channels: Optional[int] = None,
        out_features: Optional[int] = None,
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = feedforward_channels or in_features
        self.with_cp = with_cp
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = MixConv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            dilation=1,
            bias=True,
        )
        self.act = act_func()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        def _inner_forward(x: Tensor) -> Tensor:
            x = self.fc1(x)
            B, N, C = x.shape
            x = self.conv(x.transpose(1, 2).view(B, C, H, W))
            x = self.act(x)
            x = self.fc2(x.flatten(2).transpose(-1, -2))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


# INACTIVE on the DocTamper path: SegFormer Mix-FFN (fc -> DWConv -> fc); used only by the
# never-instantiated ChannelProcessing.
class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, H, W):
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


# INACTIVE on the DocTamper path: a local copy of SegFormer's Mix-FFN; never instantiated here
# because HubVisionTransformer0521 delegates its FFN to mit.TransformerEncoderLayer.
class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


# INACTIVE on the DocTamper path: SegFormer/MiT-style efficient (linear) self-attention block
# (softmax-normalised q,k with channel pooling); never instantiated anywhere in this file.
class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1, -2)
        _, _, Nk, Ck = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))

        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(
            1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x


# INACTIVE (token-select disabled): per-modal score predictor for tokenselect; in the DocTamper
# config extra_score_predictor is left empty and tokenselect is bypassed, so it is never built.
class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        ) for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_


# TODO: Make it theoretically correct
# INACTIVE (token-select disabled): alternative modal-selection head; never instantiated
# (all extra_score_predictor appends are commented out).
class ModalSelector(nn.Module):
    def __init__(self, embed_dims=128, num_modals=4, sr_ratio=1):
        super().__init__()
        self.num_modals = num_modals
        self.simi_projs = nn.ModuleList([
            nn.ModuleList([
                ToSimiVolume(embed_dims, mode='dot', norm=True),
                nn.Sequential(
                    nn.Conv2d(4, 4, 1),
                    nn.ReLU(),
                )
            ]) for _ in range(num_modals)])

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_block = nn.ModuleList([
                nn.ModuleList([
                    Conv2d(
                        in_channels=embed_dims,
                        out_channels=embed_dims,
                        kernel_size=sr_ratio,
                        stride=sr_ratio),
                    nn.LayerNorm(embed_dims),
                ]) for _ in range(num_modals)])

        self.channel_mixer = nn.Sequential(
            nn.Conv2d(4 * num_modals, 4 * num_modals * 4, 1),
            nn.ReLU(),
            nn.Conv2d(4 * num_modals * 4, num_modals, 1),
        )


    def forward(self, x):
        B, C, H, W = x[0].shape
        rs_shape = (H // self.sr_ratio, W // self.sr_ratio)

        simi_vols = []
        for i in range(self.num_modals):
            temp = x[i]
            if self.sr_ratio > 1:
                temp = self.sr_block[i][0](temp)
                temp = nchw_to_nlc(temp)
                temp = self.sr_block[i][1](temp)
                temp = nlc_to_nchw(temp, rs_shape)
            temp = self.simi_projs[i][0](temp)
            simi_vols.append([])
            simi_vols[i].append(torch.mean(temp, dim=1, keepdim=True))
            simi_vols[i].append(torch.var(temp, dim=1, keepdim=True))
            simi_vols[i].append(torch.max(temp, dim=1, keepdim=True)[0])
            simi_vols[i].append(torch.min(temp, dim=1, keepdim=True)[0])

            simi_vols[i] = self.simi_projs[i][1](torch.cat(simi_vols[i], dim=1))

        out = self.channel_mixer(torch.cat(simi_vols, dim=1))
        out = torch.softmax(out, dim=1)

        if self.sr_ratio > 1:
            out = F.interpolate(out, size=(H, W), mode='nearest')

        return out


# INACTIVE (token-select disabled): cross-attention modal-selection variant; never instantiated.
class CrossAttentionModalSelector(nn.Module):
    """
    不同方式的多模态选择/融合思路示例：
    1) 可学习门控 (gating)
    2) 多头跨模态注意力 (cross-attn)
    3) 输出每个模态的注意力分布
    """
    def __init__(self, embed_dims=128, num_modals=4, use_sr=True, sr_ratio=2, num_heads=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_modals = num_modals
        self.use_sr = use_sr
        self.sr_ratio = sr_ratio

        # (可选) SR（spatial reduction），用简单的Conv + BN 来做
        if self.use_sr and self.sr_ratio > 1:
            self.downsamples = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio),
                    nn.BatchNorm2d(embed_dims),
                    nn.ReLU(inplace=True)
                ) for _ in range(num_modals)
            ])
        else:
            self.downsamples = None

        # 跨模态注意力模块
        # 为简化，这里我们只写一个CrossAttnBlock，也可以堆叠多个或更复杂网络
        self.cross_attn_block = CrossAttnBlock(embed_dim=embed_dims, num_heads=num_heads)

        # 可学习门控，每个模态一个参数 (也可改成向量门控)
        self.gates = nn.Parameter(torch.ones(num_modals))

        # 输出投影，将融合结果映射到 1 个通道，用于计算 softmax 权重
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1),
        )

    def forward(self, x_list):
        """
        x_list: 包含 num_modals 个张量，每个形状 [B, C, H, W]
        返回: [B, num_modals, H, W]  (表示各模态在每个像素位置上的权重分布)
        """
        B = x_list[0].shape[0]
        H, W = x_list[0].shape[2], x_list[0].shape[3]

        # 1) (可选) spatial reduction
        feats = []
        for i in range(self.num_modals):
            feat = x_list[i]
            if self.downsamples is not None:
                feat = self.downsamples[i](feat)  # [B, C, H//sr, W//sr]
            feats.append(feat)

        # 新的高宽 (可能跟原图相同，也可能缩小了)
        new_H, new_W = feats[0].shape[2], feats[0].shape[3]

        # 2) flatten + gating
        #    把 [B, C, new_H, new_W] -> [B, new_H*new_W, C]
        #    并乘上可学习门控 gate
        feat_list = []
        for i, feat in enumerate(feats):
            gate_i = F.relu(self.gates[i])  # [scalar], forced to be positive
            feat_i = feat.permute(0, 2, 3, 1).contiguous()           # [B, H', W', C]
            feat_i = feat_i.view(B, new_H * new_W, self.embed_dims)  # [B, N, C]
            feat_i = feat_i * gate_i
            feat_list.append(feat_i)

        # 3) 拼接所有模态的特征在一起
        #    key_value 形状 [B, N * num_modals, C]
        key_value = torch.cat(feat_list, dim=1)

        # 4) 对每个模态做 cross-attn：让第 i 个模态去 attend 其它模态(以及自身)
        #    这里简化处理：直接 all-to-all 交叉注意力
        out_maps = []
        for i in range(self.num_modals):
            query = feat_list[i]  # [B, N, C]
            # query 与 key_value 做多头交叉注意力
            fused = self.cross_attn_block(query, key_value)  # [B, N, C]
            # reshape 回 [B, C, H', W']
            fused = fused.view(B, new_H, new_W, self.embed_dims).permute(0, 3, 1, 2)
            # 再用 1x1 conv 映射到单通道
            score_map = self.output_conv(fused)  # [B, 1, H', W']
            out_maps.append(score_map)

        # 5) 将 num_modals 个输出在通道维度拼接并 softmax
        #    out => [B, num_modals, H', W']
        out = torch.cat(out_maps, dim=1)
        out = F.softmax(out, dim=1)

        # 6) 如果做过 downsample，需要上采样到 (H, W)
        if self.downsamples is not None:
            out = F.interpolate(out, size=(H, W), mode='nearest')

        return out


# INACTIVE (token-select disabled): modal-selection variant; never instantiated.
class ModalSelectorV2(nn.Module):
    def __init__(self, embed_dims=128, num_modals=4, sr_ratio=1):
        super().__init__()
        self.num_modals = num_modals
        self.simi_projs = nn.ModuleList([
            nn.ModuleList([
                ToSimiVolume(embed_dims, mode='cosine', norm=True),
                # nn.Sequential(
                #     nn.Conv2d(4, 4, 1),
                #     nn.ReLU(),
                # )
            ]) for _ in range(num_modals)])

        self.nonlinear = nn.Sequential(
            nn.Conv2d(4, 4 * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(4 * 4, 1, 1),
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_block = nn.ModuleList([
                nn.ModuleList([
                    Conv2d(
                        in_channels=embed_dims,
                        out_channels=embed_dims,
                        kernel_size=sr_ratio,
                        stride=sr_ratio),
                    nn.LayerNorm(embed_dims),
                ]) for _ in range(num_modals)])


    def forward(self, x):
        B, C, H, W = x[0].shape
        rs_shape = (H // self.sr_ratio, W // self.sr_ratio)

        simi_vols = []
        for i in range(self.num_modals):
            temp = x[i]
            if self.sr_ratio > 1:
                temp = self.sr_block[i][0](temp)
                temp = nchw_to_nlc(temp)
                temp = self.sr_block[i][1](temp)
                temp = nlc_to_nchw(temp, rs_shape)
            temp = self.simi_projs[i][0](temp)
            simi_vols.append([])
            simi_vols[i].append(torch.mean(temp, dim=1, keepdim=True))
            simi_vols[i].append(torch.var(temp, dim=1, keepdim=True))
            simi_vols[i].append(torch.max(temp, dim=1, keepdim=True)[0])
            simi_vols[i].append(torch.min(temp, dim=1, keepdim=True)[0])

            simi_vols[i] = self.nonlinear(torch.cat(simi_vols[i], dim=1))

        out = torch.cat(simi_vols, dim=1)
        out = torch.softmax(out, dim=1)

        if self.sr_ratio > 1:
            out = F.interpolate(out, size=(H, W), mode='nearest')

        return out


# INACTIVE (token-select disabled): modal-selection variant; never instantiated.
class ModalSelectorV3(nn.Module):
    """
    改进示例:
    1) 可选下采样
    2) 相似度计算 (ToSimiVolumeEx)
    3) 统计: mean, var, max, min, median (5通道)
    4) 可学习门控: 每个模态一个标量 gate
    5) 更深的融合网络 (带残差)
    6) 在通道维度softmax
    """
    def __init__(self, 
                 embed_dims=128, 
                 num_modals=4, 
                 sr_ratio=1,
                 mode='cosine',
                 use_norm=True,
                 flatten=True):
        super().__init__()
        self.num_modals = num_modals
        self.embed_dims = embed_dims
        self.sr_ratio = sr_ratio

        # (可选) 下采样
        if sr_ratio > 1:
            self.sr_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio),
                    nn.BatchNorm2d(embed_dims),
                    nn.ReLU(inplace=True)
                ) for _ in range(num_modals)
            ])
        else:
            self.sr_blocks = None

        # 相似度投影模块: 每个模态一个
        self.simi_projs = nn.ModuleList([
            ToSimiVolumeEx(embed_dims, 
                           mode=mode, 
                           norm=use_norm, 
                           flatten=flatten) 
            for _ in range(num_modals)
        ])

        # 可学习门控参数, 每个模态一个
        # 初始化为1, 并使用 ReLU 保证非负
        self.modal_gates = nn.Parameter(torch.ones(num_modals))

        # 融合网络: 将统计后的5个通道( mean,var,max,min,median ) -> 1通道
        # 这里用更深一点的结构带 residual
        in_ch = 5
        hidden_ch = 16
        # self.fuse_block = nn.Sequential(
        #     nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_ch, 1, kernel_size=1),
        # )

        # # 简单残差 (可选)
        self.res_conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)

        self.nonlinear = nn.Sequential(
            nn.Conv2d(5, 5 * 5, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(5 * 5, 1, 1),
        )

    def forward(self, x_list):
        """
        x_list: list[torch.Tensor], len = num_modals
                每个张量 [B, C=embed_dims, H, W]
        return: [B, num_modals, H, W]  (softmax后的权重)
        """
        B, C, H, W = x_list[0].shape
        if self.sr_blocks is not None:
            rs_shape = (H // self.sr_ratio, W // self.sr_ratio)
        else:
            rs_shape = (H, W)

        out_list = []
        for i in range(self.num_modals):
            feat = x_list[i]
            # 1) 下采样
            if self.sr_blocks is not None:
                feat = self.sr_blocks[i](feat)  # => [B, C, H//sr, W//sr]

            # 2) 相似度
            simi = self.simi_projs[i](feat)  # => [B, N, H//sr, W//sr] or [B, C', H', W']

            # 3) 可学习门控: 直接在 simi 上乘一个标量
            gate_val = F.relu(self.modal_gates[i])  # keep non-negative
            simi = simi * gate_val

            # 4) 提取统计信息 (通道维度=1?)
            #    需根据上一步的输出形状来选择 dim
            #    这里假设 "flatten=True" => simi.shape: [B, N, H', W']
            #    统计沿 channel dim=1
            mean_map = torch.mean(simi, dim=1, keepdim=True)
            var_map = torch.var(simi, dim=1, keepdim=True)
            max_map, _ = torch.max(simi, dim=1, keepdim=True)
            min_map, _ = torch.min(simi, dim=1, keepdim=True)
            # median: PyTorch>=1.7.0  支持 torch.median(dim=...)
            median_map = torch.median(simi, dim=1, keepdim=True)[0]

            # cat => [B, 5, H', W']
            stats = torch.cat([mean_map, var_map, max_map, min_map, median_map], dim=1)

            # 5) 使用更深的融合网络 (带残差)
            fused = self.nonlinear(stats)      # [B, 1, H', W']
            res = self.res_conv(stats)          # [B, 1, H', W']
            fused = fused + res                 # simple residual addition

            out_list.append(fused)  # each modality => [B, 1, H', W']

        # 6) 拼接 => [B, num_modals, H', W']
        out = torch.cat(out_list, dim=1)

        # 7) softmax
        out = F.softmax(out, dim=1)

        # 8) 如果下采样过, 做上采样回原分辨率
        if self.sr_blocks is not None:
            out = F.interpolate(out, size=(H, W), mode='nearest')

        return out


# INACTIVE on the DocTamper path: applies one shared module to a list of per-modal tensors; used
# only by the dead PatchEmbedParallel and InvertedResidualSiamese.
class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


# INACTIVE on the DocTamper path: channel-first LayerNorm; used only by LayerNormParallel and the
# unregistered PPXVisionTransformer3.
class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# INACTIVE on the DocTamper path: per-modal channel-first LayerNorm; used only by the dead
# PatchEmbedParallel.
class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


# INACTIVE on the DocTamper path: per-modal overlap patch-embed; used only by the unregistered
# PPXVisionTransformer3. HubVisionTransformer0521 was refactored (see its __init__, the
# 2025-06-20 note) to use the external mmseg PatchEmbed so pretrained weights load.
class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, patch_size, stride, padding))  # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W




# INACTIVE on the DocTamper path: per-modal wrapper around mmseg PatchEmbed; referenced only inside a
# triple-quoted / commented block in HubVisionTransformer0521.__init__, which builds the plain
# external PatchEmbed instead.
class DetailedPatchEmbedParallel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 kernel_size=7,
                 stride=None,
                 dilation=1,
                 num_modals=4,
                 to_hw=True,
                 norm_cfg=dict(type='LN')):
        super(DetailedPatchEmbedParallel, self).__init__()

        assert num_modals > 0
        self.to_hw = to_hw
        self.projs = []
        for i in range(num_modals):
            self.projs.append(
                PatchEmbed(
                    in_channels=in_channels,
                    embed_dims=embed_dims,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    dilation=dilation,
                    norm_cfg=norm_cfg)
            )
        self.projs = nn.ModuleList(self.projs)


    def forward(self, x: list):
        outs = []
        for i in range(len(x)):
            out, hw_shape = self.projs[i](x[i])
            if self.to_hw:
                outs.append(nlc_to_nchw(out, hw_shape))
            else:
                outs.append(out)

        return outs, hw_shape






# TODO: Neighborhood Attention Based Rectifier
class NABR(nn.Module):
    def __init__(self, embed_dim=128, num_modals=4):
        super().__init__()
        pass
    def forward(self, x):
        pass


# INACTIVE on the DocTamper path: per-modal MobileNetV3 InvertedResidual stack. Built in
# HubVisionTransformer0521.__init__ only when modals_proj=True, but the DocTamper config leaves
# modals_proj=False (-> AnyIdentity) and modal_convs is never used in forward.
class InvertedResidualParallel(nn.Module):
    def __init__(self, num_modals=4, **kwargs):
        super().__init__()
        self.num_modals = num_modals
        self.blocks = nn.ModuleList([InvertedResidual(**kwargs) for _ in range(num_modals)])

    def forward(self, x_parallel):
        out = []
        for i in range(len(x_parallel)):
            out.append(self.blocks[i](x_parallel[i]))

        return out


class InvertedResidualSiamese(nn.Module):
    def __init__(self, num_modals=4, **kwargs):
        super().__init__()
        self.num_modals = num_modals
        self.blocks = ModuleParallel(InvertedResidual(**kwargs))

    def forward(self, x_parallel):
        return self.blocks(x_parallel)




# @MODELS.register_module()
# INACTIVE (dead): registration is commented out (see the line above), so this SegFormer/MSPA
# variant is never instantiated; it is the sole (dead) consumer of PatchEmbedParallel /
# ConvLayerNorm above.
class PPXVisionTransformer3(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 modals=['dct', 'srm'],
                 in_modals=None,
                 skip_patch_embed_stage=-1,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratios=(8,8,4,4),
                 # qkv_bias=True,
                 drop_rate=0.,
                 # attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 # act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
        self.modals = modals
        self.num_modals = len(modals)

        self.in_modals = in_modals
        self.skip_patch_embed_stage = skip_patch_embed_stage

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        if self.in_modals is not None:
            assert max(self.in_modals) <= self.num_modals
            assert len(self.in_modals) == self.num_stages
        else:
            self.in_modals = [self.num_modals] * self.num_stages

        cur = 0
        self.layers = ModuleList()
        self.extra_score_predictor = ModuleList([])
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbedParallel(
                c1=in_channels,
                c2=embed_dims_i,
                patch_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                num_modals=self.in_modals[i-1] if (i == self.skip_patch_embed_stage) else self.in_modals[i],
            )
            if self.in_modals[i] > 1:
                self.extra_score_predictor.append(PredictorConv(embed_dims_i, self.in_modals[i]))
            layer = ModuleList([
                MSPABlock(
                    dim=embed_dims_i,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + idx]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = ConvLayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        outs = []


        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            if self.in_modals[i] > 1:
                x = self.tokenselect(x, self.extra_score_predictor[i]) if self.in_modals[i] > 1 else x[0]
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs

    def tokenselect(self, x_ext, module):
        x_scores = module(x_ext)                            #
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]    # weighting
        x_f = functools.reduce(torch.max, x_ext)
        return x_f

@MODELS.register_module()
# --- MFFE CNN branch (paper role) -------------------------------------------
# ConvNeXt_0521 realizes the paper's "a CNN backbone (ConvNeXt) encodes the
# high-freq view I_hat_h -> {F_h1, F_h2}". It is a plain 4-stage ConvNeXt v1/v2
# pyramid: a stem (Conv, stride=stem_patch_size) plus 3 stride-2 downsample
# layers, each stage a stack of `depths[i]` ConvNeXtBlock at width channels[i]
# (giving H/4, H/8, H/16, H/32). `in_channels` = the I_hat_h channel count;
# `out_indices` select which stage feature maps are emitted as F_h*. The DocTamper
# 'tiny' arch is overridden to channels [64,128,320,512] (see arch_settings).
# PAPER DIFF: the DocTamper backbone AsymCMNeXt_0521 does NOT instantiate this class;
# it builds the CNN branch from a timm 'convnextv2_tiny' instead
# (self.extra1_branch = timm.create_model(...), see AsymCMNeXt_0521 below). So
# ConvNeXt_0521 is the config-declared but INACTIVE CNN-branch build.
class ConvNeXt_0521(BaseModule):
    """ConvNeXt v1&v2 backbone.

    A PyTorch implementation of `A ConvNet for the 2020s
    <https://arxiv.org/abs/2201.03545>`_ and
    `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    <http://arxiv.org/abs/2301.00808>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    To use ConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    # MFFE eq(6): {F_h1, F_h2} = CNNBlocks(Ih_hat) -- high-frequency CNN branch of MFFE / CNN baseline of
    # TAFE-Net*: ConvNeXt-V2 tiny (use_grn=True), adapted to 16-channel JPEG-DCT input, with 4 resolution
    # stages, outputting all 4 high-freq stage feature maps F_h by default (two of these are the {F_h1,F_h2}
    # fed into the scSE fusion eq(8)).
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 320, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        # 'tiny': {
        #     'depths': [3, 3, 9, 3],
        #     'channels': [96, 192, 384, 768]
        # },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [64, 128, 320, 512]
        },


        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [64, 128, 320, 512]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        """Build the ConvNeXt backbone = MFFE eq(6) CNNBlocks (high-freq branch).

        Obtains the per-stage depths/channels from ``arch`` (default 'tiny');
        ``in_channels`` interfaces with the 16-channel DCT input; ConvNeXt-V2
        requires ``use_grn=True`` and ``layer_scale_init_value=0.``. Builds the
        stem and 4 stages, each stage stacking several ConvNeXtBlocks and
        connected between stages by downsampling layers; only the stages
        specified by ``out_indices`` additionally get a normalization layer for
        output. The 4 stage outputs are the high-freq features F_h of eq(6).
        """
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer1(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer1(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer1(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)


    def forward(self, x):
        # x = I_hat_h (high-freq view). Each iter: downsample (stem = /4, then
        # /2 per stage) then run a ConvNeXt stage; the maps at out_indices form
        # the CNN-branch feature pyramid {F_h*}.
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            # eq(6): this stage's ConvNeXtBlocks encode the high-freq view into F_h
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))

        return tuple(outs)







@MODELS.register_module()
# --- MFFE Transformer branch (paper role) -----------------------------------
# HubVisionTransformer0521 realizes the paper's "a Transformer backbone
# (SegFormer/MiT) encodes the low-freq view I_hat_l -> {F_l1, F_l2}". It is a
# MiT/SegFormer-b* style pyramid: `num_stages` stages, each an OverlapPatchEmbed
# followed by `num_layers[i]` TransformerEncoderLayer blocks and a LayerNorm.
# With strides [4,2,2,2] the stages produce H/4, H/8, H/16, H/32 features;
# `out_indices` select which stages are returned as F_l*. It is MODALITY-AWARE:
# when a stage receives >1 modal input, per-token modal scores softly re-weight
# and fuse the modalities (see `tokenselect`) -- but on the DocTamper path this modal
# machinery is INACTIVE (in_stages=(0,0), tokenselect never called). ACTIVE role:
# instantiated as `backbone_extra` (the low-frequency Transformer branch) by the
# DocTamper backbone AsymCMNeXt_0524[_convback]; the legacy AsymCMNeXt_0521 below also
# builds it. NOTE: that DocTamper parent drives this module's per-stage `.layers[i]`
# directly, so HubVisionTransformer0521.forward() itself is bypassed (it would in
# fact error under in_modals=(2,2,2,2)). Hence the custom modal-selector / Mix-FFN
# blocks in this file are tagged INACTIVE below.
class HubVisionTransformer0521(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """
    # MFFE eq(7): {F_l1, F_l2} = TransformerBlocks(Il_hat) -- low-frequency Transformer branch of MFFE
    # (SegFormer style, num_layers=[3,4,6,3]). Il_hat is the low-freq VFIM output (paper role).
    # Acts as a multi-modal hub: takes multiple modalities (e.g. modals=['img','img1']);
    # each modality is patch-embedded separately, then token-select fuses them by
    # weighting into a single token stream, which passes through the Transformer
    # encoder layers, outputting 4 low-freq stage feature maps F_l (two of which are the
    # {F_l1, F_l2} fed into the scSE fusion eq(8)). in_modals specifies the
    # number of modalities actually involved at each stage.

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 modals=['dct', 'srm'],
                 in_modals=None,
                 skip_patch_embed_stage=-1,
                 modal_interact=False,
                 modals_proj = False,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        """Build the low-frequency multi-modal Transformer hub = MFFE eq(7) TransformerBlocks.

        Records the modality configuration (modals / num_modals / in_modals: the
        number of modalities involved per stage, all modalities by default);
        builds [PatchEmbed, stacked Transformer encoder layers, normalization
        layer] stage by stage according to num_layers; when a stage has
        in_modals>1, token-select together with a scorer fuses the multi-modal
        features by weighting, so a single-stream patch_embed per stage suffices.
        drop_path_rate is allocated to each layer with linear decay over the
        total number of layers.
        """
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
        self.modals = modals
        self.num_modals = len(modals)

        self.in_modals = in_modals
        self.skip_patch_embed_stage = skip_patch_embed_stage
        self.modal_interact = modal_interact
        self.modals_proj = modals_proj

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        # if self.num_modals > 1:
        #     self.extra_score_predictor = nn.ModuleList([PredictorConv(embed_dims * num_heads[i], self.num_modals) for i in range(len(num_layers))])
        if self.in_modals is not None:
            assert max(self.in_modals) <= self.num_modals
            assert len(self.in_modals) == self.num_stages
        else:
            self.in_modals = [self.num_modals] * self.num_stages

        cur = 0
        self.layers = ModuleList()
        #c创建另一个相同的分支
        self.layers_1=ModuleList()
        self.extra_score_predictor = ModuleList([])
        # self.modal_channel_mixers = ModuleList([])
        self.modal_convs = ModuleList([])
        # Build the pyramid stages that emit F_l*. Per-stage width
        # embed_dims_i = embed_dims * num_heads[i]; patch_sizes [7,3,3,3] /
        # strides [4,2,2,2] give resolutions H/4, H/8, H/16, H/32 (MiT layout).
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]

            '''
            patch_embed = DetailedPatchEmbedParallel(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                # padding=patch_sizes[i] // 2,
                # num_modals=self.in_modals[i-1] if (i == self.skip_patch_embed_stage) else self.in_modals[i],
                num_modals=1
            )
            '''

            #为了预训练权重能够加载上，使用最新的2025.0620进行更改
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)





            # patch_embed_1 = DetailedPatchEmbedParallel(
            #     in_channels=in_channels,
            #     embed_dims=embed_dims_i,
            #     kernel_size=patch_sizes[i],
            #     stride=strides[i],
            #     # padding=patch_sizes[i] // 2,
            #     num_modals=1,
            # )


            # if self.in_modals[i] > 1:
                # self.extra_score_predictor.append(ModalSelectorV2(embed_dims_i, self.in_modals[i], sr_ratios[i]))
                # self.extra_score_predictor.append(CrossAttentionModalSelector(embed_dims_i, self.in_modals[i], sr_ratios[i]))
                # self.extra_score_predictor.append(ModalSelectorV3(embed_dims_i, self.in_modals[i], sr_ratios[i]))
                # self.modal_channel_mixers.append(nn.ModuleList([nn.Conv2d(embed_dims_i, embed_dims_i, kernel_size=1, stride=1, padding=0) for _ in range(self.in_modals[i])]))
            # else:
                # self.extra_score_predictor.append(nn.Identity())

            # if self.modals_proj:
            #     self.modal_convs.append(
            #         InvertedResidualParallel(
            #             in_channels=embed_dims_i,
            #             out_channels=embed_dims_i,
            #             mid_channels= embed_dims_i * 2,
            #             se_cfg = dict(
            #                 channels=embed_dims_i * 2,
            #                 ratio=4,
            #                 act_cfg=(dict(type='ReLU'),
            #                          dict(type='HSigmoid', bias=3.0, divisor=6.0)))
            #         )
            #     )
            # else:
            #     self.modal_convs.append((AnyIdentity()))


            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])

            # layer_1=ModuleList([
            #     TransformerEncoderLayer(
            #         embed_dims=embed_dims_i,
            #         num_heads=num_heads[i],
            #         feedforward_channels=mlp_ratio * embed_dims_i,
            #         drop_rate=drop_rate,
            #         attn_drop_rate=attn_drop_rate,
            #         drop_path_rate=dpr[cur + idx],
            #         qkv_bias=qkv_bias,
            #         act_cfg=act_cfg,
            #         norm_cfg=norm_cfg,
            #         with_cp=with_cp,
            #         sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            # ])


            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            
            # norm_1=build_norm_layer(norm_cfg,embed_dims_i)[1]

            self.layers.append(ModuleList([patch_embed, layer, norm]))
            # self.layers_1.append(ModuleList([patch_embed_1,layer_1,norm_1]))

            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        """Encode the low-freq view I_hat_l into the MiT pyramid {F_l*}.

        x = I_hat_l (VFIM low-frequency output). For each stage: patch-embed
        (layer[0]) -> optional modality-aware token fusion (tokenselect) ->
        stacked TransformerEncoderLayers (layer[1]) -> LayerNorm (layer[2]) ->
        reshape NLC->NCHW. Feature maps at out_indices are returned as the
        low-freq features {F_l1, F_l2} consumed by the MFFE frequency fusion.
        """
        outs = []

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            if self.in_modals[i] > 1:
                # multi-modal token-select fuses the modality streams into one before encoding
                x = self.tokenselect(x, self.extra_score_predictor[i]) if self.in_modals[i] > 1 else x[0]
            for block in layer[1]:
                # eq(7): TransformerBlocks encode the low-freq tokens of this stage
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                # collect per-stage low-freq feature F_l (eq7); F_l1/F_l2 feed scSE eq(8)
                outs.append(x)

        return outs


    def tokenselect(self, x_ext, module):
        """Modality-aware token fusion for the low-freq Transformer branch.

        Given a list of per-modal feature maps, `module` predicts per-modal
        spatial scores [B, num_modals, H, W]; each modal feature is residually
        re-weighted (score * x + x) and the modalities are summed into one
        [B, C, H, W] map, then flattened back to token (NLC) form. With a single
        modal input it is a pass-through. This is what makes the branch
        "modality-aware".
        """
        if len(x_ext) == 1:
            x_f = x_ext[0]

        else:
            x_scores = module(x_ext)    # [B, num_modals, H, W]
            x_scores = x_scores.unsqueeze(2)    # [B, num_modals, 1, H, W]
            x_scores = x_scores.transpose(0, 1)    # [num_modals, B, 1, H, W]

            # print(x_scores.shape, x_ext[0].shape)

            for i in range(len(x_ext)):
                x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]    # weighting
            # x_f = functools.reduce(torch.max, x_ext)
            x_f = torch.sum(torch.stack(x_ext), dim=0)    # [B, C, H, W]

        x_f = x_f.flatten(2).transpose(1, 2)

        return x_f


# Trivial identity / pass-through placeholder. Built as HubVisionTransformer0521.modal_convs
# when modals_proj=False, but modal_convs is never referenced in forward, so it is effectively
# unused on the DocTamper path.
class AnyIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *kwargs):
        return kwargs

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


@MODELS.register_module()
class AsymCMNeXt_0521(BaseModule):
    """The backbone of CMNeXt but allow asymmetric input.

    This backbone is the Upgrade of `CMNeXt:'
    Delivering Arbitrary-Modal Semantic Segmentation
    modified from SegFormer

    """
    def __init__(self,
                 backbone_main: ConfigType,
                 backbone_extra: ConfigType,
                 backbone_extra1: ConfigType,
                 use_rectifier: bool,
                 rectifier: OptConfigType=None,
                 fuser: OptConfigType=None,
                 num_heads=[1,2,5,8],
                 in_stages=None,
                 extra_patch_embed: dict=None,
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None,
                 spatial_reshape=False,
                 no_select=False,
                 ):
        super().__init__(init_cfg=init_cfg)

        self.out_indices = out_indices
        self.spatial_reshape = spatial_reshape
        self.use_rectifier = use_rectifier
        self.fuser = fuser
        self.no_select = no_select

        self.main_branch = MODELS.build(backbone_main)
        self.extra_branch = MODELS.build(backbone_extra)



        # self.extra1_branch = MODELS.build(backbone_extra1)


        self.extra1_branch = timm.create_model(
            model_name='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
            features_only=False,
            pretrained=True,
            in_chans=16,
            pretrained_cfg=dict(file='/data3/yzq/code/RTM-shuang-2/convnextv2_tiny_22k_384_ema.pt'),
            pretrained_strict=False
        )
        self.extra1_branch.head = None






        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        self.loss_decode = build_loss(loss_decode)


        # aux_end=dict(
        # type='FCNHead',
        # in_channels=512,
        # # in_channels=320,
        # in_index=0,
        # channels=256,
        # num_convs=1,
        # concat_input=False,
        # dropout_ratio=0.1,
        # num_classes=2,
        # norm_cfg=dict(type='SyncBN', requires_grad=True),
        # align_corners=False,
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # ],)

        # self.aux_end=MODELS.build(aux_end)

        aux_end=dict(
                type='SegformerHead',
                # in_channels=[256, 256, 256, 256],
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                channels=256,
                dropout_ratio=0.1,
                num_classes=2,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                # sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=100000),
                loss_decode=[
                    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    dict(type='LovaszLoss', loss_weight=1.0, per_image=False, reduction='none'),
                ],)


        # self.aux_end=MODELS.build(aux_end)





        if in_stages is not None:
            assert len(in_stages) == self.extra_branch.num_modals
        self.in_stages = in_stages
        if extra_patch_embed is not None:
            self.extra_patch_embed = PatchEmbed(
                in_channels=extra_patch_embed['in_channels'],
                embed_dims=extra_patch_embed['embed_dims'],
                kernel_size=extra_patch_embed['kernel_size'],
                stride=extra_patch_embed['stride'],
                padding=extra_patch_embed['kernel_size'] // 2,
                norm_cfg=dict(type='LN', eps=1e-6),
            )
            self.use_extra_patch_embed = True
            self.reshape_extra_nchw = extra_patch_embed['reshape']
        else:
            self.use_extra_patch_embed = False


        self.num_stage_main = self.main_branch.num_layers.__len__()
        self.num_stage_extra = self.extra_branch.num_layers.__len__()

        assert self.num_stage_main >= self.num_stage_extra, 'main branch must have more stages than extra branch'
        self.shift_stage = self.num_stage_main - self.num_stage_extra


        num_heads = self.extra_branch.num_heads

        # fusion module
        self.FRMs = []

        self.conv11=[]
        self.conv11.append(nn.Conv2d(96, 64, 1, 1, 0))
        self.conv11.append(nn.Conv2d(192, 128, 1, 1, 0))
        self.conv11.append(nn.Conv2d(384, 320, 1, 1, 0))
        self.conv11.append(nn.Conv2d(768, 512, 1, 1, 0))


        self.FU = nn.ModuleList([nn.Sequential(SCSEModule(256), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))])
        self.FU.append(nn.Conv2d(128, 128, 1, 1, 0))
        # self.FU1 = nn.ModuleList([nn.Sequential(SCSEModule(256), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))])
        # self.FU1.append(nn.Conv2d(128, 128, 1, 1, 0))
        self.FU1 = nn.ModuleList([nn.Sequential(SCSEModule(128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))])
        self.FU1.append(nn.Conv2d(64, 64, 1, 1, 0))

        self.FU2 = nn.ModuleList([nn.Sequential(SCSEModule(256), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))])
        self.FU2.append(nn.Conv2d(128, 128, 1, 1, 0))
        # feature rectification module

        self.FU3 = nn.ModuleList([nn.Sequential(SCSEModule(640), nn.Conv2d(640, 320, 3, 1, 1), nn.BatchNorm2d(320), nn.ReLU(True))])
        self.FU3.append(nn.Conv2d(320, 320, 1, 1, 0))

        self.FU4 = nn.ModuleList([nn.Sequential(SCSEModule(1024), nn.Conv2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))])
        self.FU4.append(nn.Conv2d(512, 512, 1, 1, 0))

        self.FFMs = []

        
        self.prim1=PRIM1(15, 16, nn.BatchNorm2d)
        self.prim2=PRIM1(15, 16, nn.BatchNorm2d)

        # pre-fusion module
        embed_dims = [self.extra_branch.embed_dims * num_heads[i] for i in range(len(num_heads))]

        # for i in range(len(num_heads)):
        #     if self.use_rectifier:
        #         self.FRMs.append(FRM(dim=embed_dims[i], reduction=1))
        #     else:
        #         self.FRMs.append(AnyIdentity())
        #     if self.fuser is None:
        #         self.FFMs.append(
        #             NATFuserBlock(
        #                 a_channel=embed_dims[i],
        #                 b_channel=embed_dims[i],
        #                 num_head=self.extra_branch.num_heads[i],
        #                 kernel_size=5,
        #                 gated=True,
        #             )
        #         )
        #     elif isinstance(self.fuser, dict):
        #         self.fuser.update(dict(a_channel=embed_dims[i], b_channel=embed_dims[i]))
        #         if self.fuser['type'] == 'NATFuserBlock' or self.fuser['type'] == 'AdvancedNATFuserBlock':
        #             self.fuser.update(dict(num_head=self.extra_branch.num_heads[i]))
        #             fuser_block = MODELS.build(self.fuser)
        #         elif self.fuser['type'] == 'EfficientAttentionFuserBlock':
        #             self.fuser.update(dict(num_head=self.extra_branch.num_heads[i], sr_ratio=self.extra_branch.sr_ratios[i]))
        #             fuser_block = MODELS.build(self.fuser)
        #         elif self.fuser['type']=='FeatureFusionModule':
        #             fuser_block=FFM(dim=embed_dims[i],num_heads=self.extra_branch.num_heads[i])
        #         print(self.fuser)
        #         self.FFMs.append(
        #             fuser_block
        #         )


        self.FRMs = nn.ModuleList(self.FRMs)
        self.FFMs = nn.ModuleList(self.FFMs)
        self.conv11 = nn.ModuleList(self.conv11)


    def merge_inputs(self, x_a, x_b):
        if isinstance(x_a, list):
            merged = x_a
        else:
            merged = [x_a]

        if isinstance(x_b, list):
            merged = merged.extend(x_b)
        else:
            merged = merged.append(x_b)

        return merged
    
    # def compute_feature_distance(self, x_f, x_f_1):
    #     """
    #     Compute feature distance channel-wise between two inputs and generate an anomaly map.
    #     The anomaly map will have the same size as the input, including the channel count.

    #     Args:
    #         x_f (torch.Tensor): The first input tensor, shape (N, C, H, W).
    #         x_f_1 (torch.Tensor): The second input tensor, shape (N, C, H, W).

    #     Returns:
    #         torch.Tensor: Anomaly map with the same size as the input (N, C, H, W), 
    #                     where each channel corresponds to the difference for that channel.
    #     """
    #     # 获取通道数量
    #     num_channels = x_f.shape[1]  # C

    #     # 保存每个通道的差异结果
    #     channel_anomaly_maps = []

    #     # 遍历每个通道，逐通道计算余弦距离
    #     for c in range(num_channels):
    #         # 提取当前通道的特征图 (N, H, W)
    #         x_f_c = x_f[:, c, :, :].unsqueeze(1)  # 形状变为 (N, 1, H, W)
    #         x_f_1_c = x_f_1[:, c, :, :].unsqueeze(1)  # 形状变为 (N, 1, H, W)

    #         # 计算当前通道的余弦相似度
    #         cosine_similarity = F.cosine_similarity(x_f_c, x_f_1_c, dim=1, eps=1e-6)  # 输出形状为 (N, H, W)

    #         # 转换为余弦距离
    #         cosine_distance = 1 - cosine_similarity

    #         # 恢复成形状 (N, 1, H, W)，以便后续拼接
    #         channel_anomaly_map = cosine_distance.unsqueeze(1)  # 输出形状为 (N, 1, H, W)

    #         # 保存到结果列表
    #         channel_anomaly_maps.append(channel_anomaly_map)
    #         # import pdb;pdb.set_trace()
    #     # 将所有通道的差异结果拼接起来
    #     anomaly_map = torch.cat(channel_anomaly_maps, dim=1)  # 最终形状为 (N, C, H, W)

    #     return anomaly_map



    def compute_feature_distance(self, x_f, x_f_1):
        """
        Compute feature distance channel-wise between two inputs and generate an anomaly map.
        The anomaly map will have the same size as the input, including the channel count.

        Args:
            x_f (torch.Tensor): The first input tensor, shape (N, C, H, W).
            x_f_1 (torch.Tensor): The second input tensor, shape (N, C, H, W).

        Returns:
            torch.Tensor: Anomaly map with the same size as the input (N, C, H, W), 
                        where each channel corresponds to the difference for that channel.
        """
        # 获取通道数量
        num_channels = x_f.shape[1]  # C

        # 计算当前通道的余弦相似度
        cosine_similarity = F.cosine_similarity(x_f, x_f_1, dim=1, eps=1e-6)  # output shape (N, H, W)

            # 转换为余弦距离
        cosine_distance = 1 - cosine_similarity

            # 恢复成形状 (N, 1, H, W)，以便后续拼接
        channel_anomaly_map = cosine_distance.unsqueeze(1)  # output shape (N, 1, H, W)

        anomaly_map = channel_anomaly_map.repeat(1, num_channels, 1, 1)

        return anomaly_map



    # def compute_feature_distance(self, x_f, x_f_1):
    #     """
    #     Compute feature distance channel-wise between two inputs and generate an anomaly map.
    #     The anomaly map will have the same size as the input, including the channel count.

    #     Args:
    #         x_f (torch.Tensor): The first input tensor, shape (N, C, H, W).
    #         x_f_1 (torch.Tensor): The second input tensor, shape (N, C, H, W).

    #     Returns:
    #         torch.Tensor: Anomaly map with the same size as the input (N, C, H, W), 
    #                     where each channel corresponds to the L1 distance for that channel.
    #     """
    #     # 获取通道数量
    #     num_channels = x_f.shape[1]  # C

    #     # 保存每个通道的差异结果
    #     channel_anomaly_maps = []

    #     # 遍历每个通道，逐通道计算 L1 距离
    #     for c in range(num_channels):
    #         # 提取当前通道的特征图 (N, H, W)
    #         x_f_c = x_f[:, c, :, :].unsqueeze(1)  # 形状变为 (N, 1, H, W)
    #         x_f_1_c = x_f_1[:, c, :, :].unsqueeze(1)  # 形状变为 (N, 1, H, W)

    #         # 计算当前通道的 L1 距离
    #         l1_distance = torch.abs(x_f_c - x_f_1_c)  # 输出形状为 (N, 1, H, W)

    #         # 保存到结果列表
    #         channel_anomaly_maps.append(l1_distance)
    #     # 将所有通道的差异结果拼接起来
    #     anomaly_map = torch.cat(channel_anomaly_maps, dim=1)  # 最终形状为 (N, C, H, W)

    #     return anomaly_map



    def forward(self, x):
        
        outs = []

        x_cam = x[0]
        x_extra = x[1]

        # align channel
        for i in range(len(x_extra)):
            if x_extra[i].size()[1] != self.extra_branch.in_channels:
                if (x_extra[i].size()[1] == 1) & (x_cam.size()[2:] == x_extra[i].size()[2:]):
                    x_extra[i] = x_extra[i].repeat(1, self.extra_branch.in_channels, 1, 1)

        if self.in_stages is not None:
            if max(self.in_stages) > 0:
                in_stage = max(self.in_stages)

                x_extra_0 = []
                x_extra_m = []
                for i in range(len(self.in_stages)):
                    if self.in_stages[i] > 0:
                        x_extra_m.append(x_extra[i])
                    else:
                        x_extra_0.append(x_extra[i])

                x_extra = x_extra_0
            else:
                x_extra_1=x_extra[0]
                x_extra_1=torch.cat([x_extra_1,x[0]],dim=1)
                x_extra_1=self.prim1(x_extra_1)

                x_extra_2=x_extra[1]
                x_extra_2=torch.cat([x_extra_2,x[0]],dim=1)

                x_extra_2=self.prim2(x_extra_2)
                in_stage = -1
                x_extra_3=x_extra[2]
        else:
            in_stage = -1

        B = x_cam.shape[0]

        #没有执行
        for i in range(self.shift_stage):
            layer = self.main_branch.layers[i]
            x_cam, hw_shape = layer[0](x_cam)
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)
            if i in self.out_indices:
                outs.append(x_cam)

        #
        # cross recalibration
        for i in range(self.num_stage_extra):
            # import pdb;pdb.set_trace()
            layer = self.main_branch.layers[(i+self.shift_stage)]
            layer_extra = self.extra_branch.layers[i]
            
            # layer_extra_1=self.extra_branch.layers_1[i]



            x_cam, hw_shape = layer[0](x_cam)
            H, W = hw_shape
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)

            if self.in_stages is not None:
                if i == in_stage:
                    if self.use_extra_patch_embed:
                        if self.reshape_extra_nchw:
                            temp = []
                            for item in x_extra_m:
                                em, s = self.extra_patch_embed(item)
                                temp.append(nlc_to_nchw(em, s))
                            x_extra_m = temp
                        else:
                            x_extra_m = [self.extra_patch_embed(item)[0] for item in x_extra_m]

                        x_extra, _ = layer_extra[0](x_extra)
                        x_extra.extend(x_extra_m)
                    else:
                        x_extra.extend(x_extra_m)
                        x_extra, _ = layer_extra[0](x_extra)
                else:
                    # import pdb;pdb.set_trace()
                    
                    # x_extra_2=torch.cat([x_extra_2,x[0]])
                    x_extra_1, _ = layer_extra[0]([x_extra_1])
                    # x_extra_2,_ = layer_extra_1[0]([x_extra_2])

            else:
                x_extra, _ = layer_extra[0](x_extra)

            # if self.no_select:
            #     x_f = torch.stack(x_extra,dim=0).mean(dim=0)
            #     x_f = x_f.flatten(2).transpose(1, 2)
            # else:
            #     x_f = self.extra_branch.tokenselect(x_extra, self.extra_branch.extra_score_predictor[i])
            
            x_f=x_extra_1[0]
            x_f = x_f.flatten(2).transpose(1, 2)
            for block in layer_extra[1]:
                x_f = block(x_f, hw_shape)
            x_f = layer_extra[2](x_f)
            x_f = nlc_to_nchw(x_f, hw_shape)



            x_f_1=x_extra_2
            # stage=self.extra1_branch.stages[i]
            if i==0:
                x = self.extra1_branch.stem(x_f_1)
            x = self.extra1_branch.stages[i](x)
            # norm_layer=getattr(self.extra1_branch,f'norm{i}')
            # import pdb;pdb.set_trace()
            x_f_1=self.conv11[i](x)

            # x_f_end=self.compute_feature_distance(x_f,x_f_1)
            # import pdb;pdb.set_trace()
            # x_f_end=self.compute_feature_distance(x_f,x_f_1)

            #在ascformer_rtm_img_img1_true_cosine_frfm的20250319_145114中
            # x, x_f = self.FRMs[i](x_cam, x_f_end)          
            # x_fused = self.FFMs[i](x_cam, x_f_end)



            x_extra_all=[x_f,x_f_1]
            if i==0:
                
            #     x_f_end = self.extra_branch.tokenselect(x_extra_all, self.extra_branch.extra_score_predictor[i])
            #     x_f_end = nlc_to_nchw(x_f_end, hw_shape)

            # # x_f_end=x_f+x_f_1

            #     x, x_f = self.FRMs[i](x_cam, x_f_end)  

            #     x_fused = self.FFMs[i](x_cam, x_f_end)
                # x_f_end = self.extra_branch.tokenselect(x_extra_all, self.extra_branch.extra_score_predictor[i])
                # x_f_end = nlc_to_nchw(x_f_end, hw_shape)

            # x_f_end=x_f+x_f_1

                # x, x_f = self.FRMs[i](x_cam, x_f_end)  

                # x_fused = self.FFMs[i](x_cam, x_f_end)



                ext_1 = self.FU1[0](torch.cat((x_f, x_f_1), dim=1))
                x_fused = self.FU1[1](ext_1) + x_cam
                # x_cam=x_fused

            elif i==1:
                # import pdb;pdb.set_trace()
                # x_f_end = self.extra_branch.tokenselect(x_extra_all, self.extra_branch.extra_score_predictor[i])
                # x_f_end = nlc_to_nchw(x_f_end, hw_shape)
                # x, x_f = self.FRMs[i](x_cam, x_f_end)  

                # x_fused = self.FFMs[i](x_cam, x_f_end)
                # ext1 = self.FU1[0](torch.cat((x_f, x), dim=1))
                # ext1_1=self.FU1[1](ext1)
                ext1 = self.FU2[0](torch.cat((x_f, x_f_1), dim=1))
                ext1_1=self.FU2[1](ext1)


                ext = self.FU[0](torch.cat((x_cam, x_extra_3), dim=1))
                x_fused = self.FU[1](ext)+ext1_1
                # x_cam=x_fused


            # elif i==2:
            #     ext_1 = self.FU3[0](torch.cat((x_f, x_f_1), dim=1))
            #     x_fused = self.FU3[1](ext_1) + x_cam
            #     # x_cam=x_fused

            # elif i==3:
            #     ext_1 = self.FU4[0](torch.cat((x_f, x_f_1), dim=1))
            #     # x_fused = self.FU4[1](ext_1) + x_cam
            #     ext_1== self.FU4[1](ext_1)
            #     x_fused=ext_1+x_cam
                # x_cam=x_fused

            else:
                x_fused=x_cam


            # x_extra_all=[x_f,x_f_1]
            # if i==0:
                
            #     x_f_end = self.extra_branch.tokenselect(x_extra_all, self.extra_branch.extra_score_predictor[i])
            #     x_f_end = nlc_to_nchw(x_f_end, hw_shape)

            # # x_f_end=x_f+x_f_1

            #     x, x_f = self.FRMs[i](x_cam, x_f_end)  

            #     x_fused = self.FFMs[i](x_cam, x_f_end)

            # elif i==1:
            #     # import pdb;pdb.set_trace()
            #     ext = self.FU[0](torch.cat((x_cam, x_extra_3), dim=1))
            #     x_fused = self.FU[1](ext) + x
            # else:
            #     x_fused=x_cam





            # x_1,x_2=self.FRMs[i](x_cam, x_f_end)  
            # x_fused=self.FFMs[i](x_1,x_2)

            # x_fused=x_f_end

            if (i+self.shift_stage) in self.out_indices:
                outs.append(x_fused)

            # x_extra = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_extra_all] if self.extra_branch.num_modals > 1 else [x_f]
            
            #在4月4日改变
            x_extra=x_extra_all
            x_extra_1=x_extra[0]
            x_extra_2=x_extra[1]
            # x_extra_1=x_f
            # x_extra_2=x_f_1


        return outs



#-----------------------------------如果要在backbone当中加上loss的计算---------------------
'''
    def loss(self, x,data_samples):
        
        outs = []
        aux_input=[]



        x_cam = x[0]
        x_extra = x[1]

        # align channel
        for i in range(len(x_extra)):
            if x_extra[i].size()[1] != self.extra_branch.in_channels:
                if (x_extra[i].size()[1] == 1) & (x_cam.size()[2:] == x_extra[i].size()[2:]):
                    x_extra[i] = x_extra[i].repeat(1, self.extra_branch.in_channels, 1, 1)

        if self.in_stages is not None:
            if max(self.in_stages) > 0:
                in_stage = max(self.in_stages)

                x_extra_0 = []
                x_extra_m = []
                for i in range(len(self.in_stages)):
                    if self.in_stages[i] > 0:
                        x_extra_m.append(x_extra[i])
                    else:
                        x_extra_0.append(x_extra[i])

                x_extra = x_extra_0
            else:
                x_extra_1=x_extra[0]
                x_extra_1=torch.cat([x_extra_1,x[0]],dim=1)
                x_extra_1=self.prim1(x_extra_1)

                x_extra_2=x_extra[1]
                x_extra_2=torch.cat([x_extra_2,x[0]],dim=1)

                x_extra_2=self.prim2(x_extra_2)
                in_stage = -1
                x_extra_3=x_extra[2]
        else:
            in_stage = -1

        B = x_cam.shape[0]

        # Not executed
        for i in range(self.shift_stage):
            layer = self.main_branch.layers[i]
            x_cam, hw_shape = layer[0](x_cam)
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)
            if i in self.out_indices:
                outs.append(x_cam)

        #
        # cross recalibration
        for i in range(self.num_stage_extra):
            # import pdb;pdb.set_trace()
            layer = self.main_branch.layers[(i+self.shift_stage)]
            layer_extra = self.extra_branch.layers[i]
            
            # layer_extra_1=self.extra_branch.layers_1[i]



            x_cam, hw_shape = layer[0](x_cam)
            H, W = hw_shape
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)

            if self.in_stages is not None:
                if i == in_stage:
                    if self.use_extra_patch_embed:
                        if self.reshape_extra_nchw:
                            temp = []
                            for item in x_extra_m:
                                em, s = self.extra_patch_embed(item)
                                temp.append(nlc_to_nchw(em, s))
                            x_extra_m = temp
                        else:
                            x_extra_m = [self.extra_patch_embed(item)[0] for item in x_extra_m]

                        x_extra, _ = layer_extra[0](x_extra)
                        x_extra.extend(x_extra_m)
                    else:
                        x_extra.extend(x_extra_m)
                        x_extra, _ = layer_extra[0](x_extra)
                else:
                    # import pdb;pdb.set_trace()
                    
                    # x_extra_2=torch.cat([x_extra_2,x[0]])
                    x_extra_1, _ = layer_extra[0]([x_extra_1])
                    # x_extra_2,_ = layer_extra_1[0]([x_extra_2])

            else:
                x_extra, _ = layer_extra[0](x_extra)

            # if self.no_select:
            #     x_f = torch.stack(x_extra,dim=0).mean(dim=0)
            #     x_f = x_f.flatten(2).transpose(1, 2)
            # else:
            #     x_f = self.extra_branch.tokenselect(x_extra, self.extra_branch.extra_score_predictor[i])
            
            x_f=x_extra_1[0]
            x_f = x_f.flatten(2).transpose(1, 2)
            for block in layer_extra[1]:
                x_f = block(x_f, hw_shape)
            x_f = layer_extra[2](x_f)
            x_f = nlc_to_nchw(x_f, hw_shape)



            x_f_1=x_extra_2
            # stage=self.extra1_branch.stages[i]
            x = self.extra1_branch.downsample_layers[i](x_f_1)
            x = self.extra1_branch.stages[i](x)
            norm_layer=getattr(self.extra1_branch,f'norm{i}')
            # import pdb;pdb.set_trace()
            x_f_1=norm_layer(x)

            # x_f_end=self.compute_feature_distance(x_f,x_f_1)
            # import pdb;pdb.set_trace()
            # x_f_end=self.compute_feature_distance(x_f,x_f_1)

            # In run ascformer_rtm_img_img1_true_cosine_frfm/20250319_145114
            # x, x_f = self.FRMs[i](x_cam, x_f_end)          
            # x_fused = self.FFMs[i](x_cam, x_f_end)



            x_extra_all=[x_f,x_f_1]
            if i==0:
                
            #     x_f_end = self.extra_branch.tokenselect(x_extra_all, self.extra_branch.extra_score_predictor[i])
            #     x_f_end = nlc_to_nchw(x_f_end, hw_shape)

            # # x_f_end=x_f+x_f_1

            #     x, x_f = self.FRMs[i](x_cam, x_f_end)  

            #     x_fused = self.FFMs[i](x_cam, x_f_end)
                # x_f_end = self.extra_branch.tokenselect(x_extra_all, self.extra_branch.extra_score_predictor[i])
                # x_f_end = nlc_to_nchw(x_f_end, hw_shape)

            # x_f_end=x_f+x_f_1

                # x, x_f = self.FRMs[i](x_cam, x_f_end)  

                # x_fused = self.FFMs[i](x_cam, x_f_end)
                ext_1 = self.FU1[0](torch.cat((x_f, x_f_1), dim=1))
                ext_1 = self.FU1[1](ext_1) 
                x_fused= ext_1 + x_cam
                # x_cam=x_fused
                aux_input.append(ext_1)

            elif i==1:
                # import pdb;pdb.set_trace()
                # x_f_end = self.extra_branch.tokenselect(x_extra_all, self.extra_branch.extra_score_predictor[i])
                # x_f_end = nlc_to_nchw(x_f_end, hw_shape)
                # x, x_f = self.FRMs[i](x_cam, x_f_end)  

                # x_fused = self.FFMs[i](x_cam, x_f_end)
                # ext1 = self.FU1[0](torch.cat((x_f, x), dim=1))
                # ext1_1=self.FU1[1](ext1)
                ext1 = self.FU2[0](torch.cat((x_f, x_f_1), dim=1))
                ext1_1=self.FU2[1](ext1)
                
                aux_input.append(ext1_1)
                ext = self.FU[0](torch.cat((x_cam, x_extra_3), dim=1))
                x_fused = self.FU[1](ext)+ext1_1
                # x_cam=x_fused


            elif i==2:
                ext_1 = self.FU3[0](torch.cat((x_f, x_f_1), dim=1))
                ext_1 = self.FU3[1](ext_1) 
                x_fused=ext_1+ x_cam
                # x_cam=x_fused
                aux_input.append(ext_1)

            elif i==3:
                ext_1 = self.FU4[0](torch.cat((x_f, x_f_1), dim=1))
                ext_1== self.FU4[1](ext_1)
                x_fused=ext_1+x_cam
                # x_fused = self.FU4[1](ext_1) + x_cam
                # x_cam=x_fused

                aux_input.append(ext_1)

            else:
                x_fused=x_cam


            # x_extra_all=[x_f,x_f_1]
            # if i==0:
                
            #     x_f_end = self.extra_branch.tokenselect(x_extra_all, self.extra_branch.extra_score_predictor[i])
            #     x_f_end = nlc_to_nchw(x_f_end, hw_shape)

            # # x_f_end=x_f+x_f_1

            #     x, x_f = self.FRMs[i](x_cam, x_f_end)  

            #     x_fused = self.FFMs[i](x_cam, x_f_end)

            # elif i==1:
            #     # import pdb;pdb.set_trace()
            #     ext = self.FU[0](torch.cat((x_cam, x_extra_3), dim=1))
            #     x_fused = self.FU[1](ext) + x
            # else:
            #     x_fused=x_cam





            # x_1,x_2=self.FRMs[i](x_cam, x_f_end)  
            # x_fused=self.FFMs[i](x_1,x_2)

            # x_fused=x_f_end

            if (i+self.shift_stage) in self.out_indices:
                outs.append(x_fused)

            # x_extra = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_extra_all] if self.extra_branch.num_modals > 1 else [x_f]
            
            # Changed on April 4
            x_extra=x_extra_all
            x_extra_1=x_extra[0]
            x_extra_2=x_extra[1]
            # x_extra_1=x_f
            # x_extra_2=x_f_1
        add_out_end=self.aux_end(aux_input)
        loss1=self.loss_by_feat(add_out_end,data_samples)

        # loss1=F.mse_loss(ext_1, x_cam)
        # loss1={'loss_classfier': loss_backbone}
        return outs,{'loss_bacobone': loss1}


    def loss_by_feat(self, seg_logits: Tensor,
                        batch_data_samples: SampleList) -> dict:
            """Compute segmentation loss.

            Args:
                seg_logits (Tensor): The output from decode head forward function.
                batch_data_samples (List[:obj:`SegDataSample`]): The seg
                    data samples. It usually includes information such
                    as `metainfo` and `gt_sem_seg`.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """
            self.align_corners=False
            seg_label = self._stack_batch_gt(batch_data_samples)
            loss = dict()
            seg_logits = resize(
                input=seg_logits,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
  
            seg_weight = None
            seg_label = seg_label.squeeze(1)

            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
            for loss_decode in losses_decode:
                if loss_decode.loss_name not in loss:
                    # loss[loss_decode.loss_name] = loss_decode(
                    #     seg_logits,
                    #     seg_label,
                    #     weight=seg_weight,
                    #     ignore_index=255)
                    loss_value=loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=255)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=255)
            return loss_value

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
'''