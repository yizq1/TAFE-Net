"""Frequency-view extractors (TAFE-Net, AAAI 2026): produce the paper's I_hf / I_lf views.

This file implements the input-side frequency-view extractors of the paper (INPUT & FREQUENCY
VIEWS section). They generate the high/low-frequency views that VFIM concatenates with RGB Iv
(eq(1) I_h = Cat(Iv, I_hf); eq(2) I_l = Cat(Iv, I_lf)) and that MFFE routes to the CNN branch
(eq(6), high-freq) and the Transformer branch (eq(7), low-freq):
  - MultiScaleHighFrequencyExtractor  -> high-freq view I_hf (multi-scale, multi-band high-freq residual);
  - MultiScaleMultiLowFreqExtractor   -> low-freq view I_lf  (multi-scale, multi-band low-freq residual);
  - HighDctFrequencyExtractor         -> I_hf, single-scale single-band ("Dct") variant;
  - LowDctFrequencyExtractor          -> I_lf, single-scale single-band ("Dct") variant.
CONFIG MAP: the SegFormer config uses the MultiScale variants; the ConvNeXt config (TAFE-Net*)
uses the *Dct* variants. All four apply DCT frequency masking + IDCT to obtain a frequency residual
that exposes tampering/compression/splicing traces.
DEVIATION from paper: the paper derives I_hf/I_lf via a DCT tied to the Y channel; HERE the views are
computed by a FULL-IMAGE DCT on the (normalized) RGB image directly, with NO YCrCb / Y-channel
conversion. (The JPEG 8x8-block DCT coeff map D and quant table T of eq(5) FPH come from a separate
jpegio path, not from these classes.) So these outputs play the role of I_hf/I_lf but differ in exact
derivation from the paper text.
The remaining classes (SRMConv2d_simple / BayarConv2d / NoFilter / SimpleProjection / ResidualFilters)
form an optional spatial-/frequency-domain forensic filter toolbox and are not part of the paper pipeline.
"""
import warnings

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
# from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import ConvModule
import math
from mmseg.registry import MODELS


# =============================================================================
# TRAINING-FLOW / frequency-view extractors  (file: filters.py)
# -----------------------------------------------------------------------------
# This module is the grab-bag of pre-backbone "filter" modules that the DocTamper
# configs plug in ahead of the TAFE-Net backbone ("Frequency Mining Empowered by
# Text Aggregation", AAAI 2026). Only the TWO classes annotated below are on the
# active TAFE-Net training path; they turn the visual RGB image I_v into the
# paper's two DCT frequency "views":
#
#   * HighDctFrequencyExtractor -> paper HIGH-frequency view  I_hf  (H x W x 3)
#   * LowDctFrequencyExtractor  -> paper LOW-frequency  view  I_lf  (H x W x 3)
#
# Both do: whole-image 2D-DCT (orthonormal DCT-II via a cached cosine basis) ->
# zero a rectangular CORNER of the spectrum -> inverse DCT -> per-sample min-max
# norm. Downstream, these views are concatenated with I_v inside the backbone's
# VFIM (I_h = Cat(I_v, I_hf), I_l = Cat(I_v, I_lf), 6ch each) and feed the two
# MFFE frequency branches (ConvNeXt on the high view, SegFormer/MiT on the low).
#
# PAPER DIFF (applies to BOTH extractors below):
#   (1) No explicit YCrCb conversion: the DCT runs on the mean/std-normalized RGB
#       tensor, NOT on a [0,255] luminance (Y) image.
#   (2) The DCT is WHOLE-IMAGE, not the JPEG 8x8-block DCT the paper framing implies.
#   (3) The high/low band split is a rectangular CORNER mask (top-left vs
#       bottom-right), not an isotropic / radial frequency split.
#
# The remaining classes in this file (SRM / Bayar / Residual / MultiScale* /
# NoFilter / SimpleProjection, ...) are alternative or inactive filter variants;
# they are left untouched (not asserted to correspond to any paper module).
# =============================================================================




@MODELS.register_module()
class MultiScaleMultiLowFreqExtractor(nn.Module):
    """TAFE-Net low-freq view I_lf extractor (MultiScale variant; used by the SegFormer config).

    参数：
    alpha_list: list[float]，表示不同的alpha截断比例（越接近1越保留更少频率），
                可以用来获取不同的过滤带范围。
    scales: list[float]，表示要使用的多尺度倍数（相对于原图）
    """
    # Paper low-freq view I_lf extractor: zero out the bottom-right corner (high-freq region) after DCT, then IDCT, yielding multi-scale multi-band low-freq residual.
    # Output: per-scale/per-alpha low-freq residuals concatenated along the channel dim, with global min-max normalization, shape (B, C*len(alpha)*len(scales), H, W).
    # def __init__(self, alpha_list=[0.9, 0.95], scales=[1.0, 0.5]):
    # def __init__(self, alpha_list=[0.9, 0.95], scales=[1.0]):
    def __init__(self, alpha_list=[0.95], scales=[1.0]):
    # def __init__(self, alpha_list=[0.95], scales=[2.0,1.0, 0.5]):
        super(MultiScaleMultiLowFreqExtractor, self).__init__()
        
        # 检查alpha合法性
        for alpha in alpha_list:
            if alpha <= 0 or alpha >= 1:
                raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha_list = alpha_list
        
        # 检查scales合法性
        for s in scales:
            if s <= 0:
                raise ValueError("scale factor must be positive")
        self.scales = scales

        # 用于缓存不同大小H/W对应的DCT矩阵
        self.dct_matrix_cache = {}  # key: (H, W), value: (dct_matrix_h, dct_matrix_w)
        # self.pim=PRIM1_Text(64, 16, nn.BatchNorm2d, k_asym=7)

    def create_dct_matrix(self, N: int, device: torch.device):
        """
        生成长度为N的一维DCT基矩阵
        """
        n = torch.arange(N, dtype=torch.float32, device=device).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32, device=device).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N, device=device)) * \
                     torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        # 第0行特殊处理
        dct_matrix[0, :] = 1.0 / math.sqrt(N)
        return dct_matrix

    def get_dct_matrices(self, H: int, W: int, device: torch.device):
        """
        获取 (H, W) 对应的DCT矩阵 (dct_matrix_h, dct_matrix_w)，如果cache里有则取，否则创建
        """
        if (H, W) not in self.dct_matrix_cache:
            dct_h = self.create_dct_matrix(H, device)
            dct_w = self.create_dct_matrix(W, device)
            self.dct_matrix_cache[(H, W)] = (dct_h, dct_w)
        return self.dct_matrix_cache[(H, W)]

    def dct_2d(self, x: torch.Tensor):
        """
        对最后两个维度做2D DCT变换 (只支持 BCHW 或 CHW 格式)
        """
        B, C, H, W = x.shape
        dct_h, dct_w = self.get_dct_matrices(H, W, x.device)
        # (B, C, H, W) -> 逐通道应用 2D-DCT
        # 可以用矩阵相乘： X_dct = dct_h @ X @ dct_w^T
        # 这里需要先把通道视为 batch 维度展平处理，也可在for loop
        x = x.view(B*C, H, W)
        # dct over H dimension
        x = torch.matmul(dct_h, x)    # yields (H, W)
        # dct over W dimension
        x = torch.matmul(x, dct_w.t())
        return x.view(B, C, H, W)

    def idct_2d(self, x: torch.Tensor):
        """
        对最后两个维度做2D IDCT变换 (只支持 BCHW 或 CHW 格式)
        """
        B, C, H, W = x.shape
        dct_h, dct_w = self.get_dct_matrices(H, W, x.device)
        x = x.view(B*C, H, W)
        # X_idct = dct_h^T @ X @ dct_w
        x = torch.matmul(dct_h.t(), x)
        x = torch.matmul(x, dct_w)
        return x.view(B, C, H, W)

    def high_pass_filter(self, x: torch.Tensor, alpha: float):
        """
        简单示例：保留[x.shape - alpha*h : x.shape, alpha*w : x.shape]之外的频率
        即截除右下角 (alpha_h, alpha_w) 大小区域 (与原LowDctFrequencyExtractor中类似)
        """
        B, C, H, W = x.shape
        mask = torch.ones((H, W), device=x.device)
        alpha_h, alpha_w = int(alpha * H), int(alpha * W)
        # 这里相当于把右下角 alpha比例的频率置0
        mask[-alpha_h:, -alpha_w:] = 0
        # 扩展到与 x 维度对齐
        mask = mask.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)
        return x * mask

    def forward_single_scale(self, x: torch.Tensor):
        """
        针对某一个尺度上的图像，提取多频带特征并拼接
        """
        # x 形状: (B, C, H, W)
        # 先做 DCT
        x_dct = self.dct_2d(x)  # (B, C, H, W)

        # 对 alpha_list 中每个 alpha 做 high-pass
        # 然后 IDCT 复原，再拼接
        out_features = []
        for alpha in self.alpha_list:
            high_freq = self.high_pass_filter(x_dct, alpha)  # high-pass
            x_idct = self.idct_2d(high_freq)
            out_features.append(x_idct)

        # 将多频带结果在通道维度拼接
        # 例如：输入(C=3)，alpha_list长度=2 -> 输出 (C=3*2=6)
        x_cat = torch.cat(out_features, dim=1)
        return x_cat

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)

        # 存放多个尺度的结果
        multi_scale_features = []
        
        for scale in self.scales:
            if abs(scale - 1.0) < 1e-6:
                # 原始尺⼨
                scaled_x = x
            else:
                # 下采样
                scaled_x = F.interpolate(x,
                                         scale_factor=scale,
                                         mode='bilinear',
                                         align_corners=False)
            
            # 在该尺度下处理
            feat_scale = self.forward_single_scale(scaled_x)

            # 如果做了下采样，需要再插值回到原始尺寸，以便融合
            if abs(scale - 1.0) > 1e-6:
                feat_scale = F.interpolate(feat_scale, 
                                           size=(x.shape[-2], x.shape[-1]), 
                                           mode='bilinear', 
                                           align_corners=False)
            
            multi_scale_features.append(feat_scale)

        # 多个尺度提取结果在通道维度拼接
        xh = torch.cat(multi_scale_features, dim=1)
        
        # 做一次简单的global min-max归一化
        B, C, H, W = xh.shape
        min_vals = xh.view(B, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        max_vals = xh.view(B, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        xh = (xh - min_vals) / (max_vals - min_vals + 1e-8)

        return xh



@MODELS.register_module()
class MultiScaleHighFrequencyExtractor(nn.Module):
    """
    多尺度、多频带DCT高频特征提取。
    与原先HighDctFrequencyExtractor相比，主要做了以下改动：
      1. 允许传入多个 alpha（alpha_list）来分别提取不同的高频范围；
      2. 通过 scales 进行多尺度变换，在不同分辨率下捕捉高频信息；
      3. 不同尺度/频带的输出特征在通道维度拼接，然后做一次归一化。

    参数：
        alpha_list: List[float]，每个 alpha 表示高通截断比例。
                    例如 alpha=0.05 表示只保留右下角 (1-alpha)*H x (1-alpha)*W
                    的频率分量。多个alpha可以获取不同的高频带。
        scales: List[float]，多尺度列表，用于对输入图像缩放后再进行DCT高通处理。
    """
    # Paper high-freq view I_hf extractor: zero out the top-left corner (low-freq region) after DCT, then IDCT, yielding multi-scale multi-band high-freq residual.
    # Output: per-scale/per-alpha high-freq residuals concatenated along the channel dim, with global min-max normalization, shape (B, C*len(alpha)*len(scales), H, W).
    # def __init__(self, alpha_list=[0.05, 0.10], scales=[1.0, 0.5]):
    # def __init__(self, alpha_list=[0.05, 0.10], scales=[1.0]):
    def __init__(self, alpha_list=[0.05], scales=[1.0]):
    # def __init__(self, alpha_list=[0.05], scales=[2.0,1.0, 0.5]):
        super(MultiScaleHighFrequencyExtractor, self).__init__()

        # 检查 alpha_list 合法性
        if not isinstance(alpha_list, list):
            raise ValueError("alpha_list must be a list of float.")
        for alpha in alpha_list:
            if alpha <= 0 or alpha >= 1:
                raise ValueError("Each alpha must be between 0 and 1 (exclusive).")
        self.alpha_list = alpha_list

        # 检查 scales 合法性
        if not isinstance(scales, list):
            raise ValueError("scales must be a list of float.")
        for s in scales:
            if s <= 0:
                raise ValueError("Scale factor must be positive.")
        self.scales = scales

        # 缓存 (H, W) 对应的DCT矩阵
        self.dct_matrix_h = {}
        self.dct_matrix_w = {}

    def create_dct_matrix(self, N, device):
        """
        生成长度为 N 的一维 DCT 基矩阵
        """
        n = torch.arange(N, dtype=torch.float32, device=device).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32, device=device).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N, device=device)) * \
                     torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        # 第 0 行特殊处理
        dct_matrix[0, :] = 1.0 / math.sqrt(N)
        return dct_matrix

    def get_dct_matrices(self, H, W, device):
        """
        根据输入分辨率 H, W 获取 (dct_matrix_h, dct_matrix_w)，若缓存无则创建
        """
        if H not in self.dct_matrix_h:
            self.dct_matrix_h[H] = self.create_dct_matrix(H, device)
        if W not in self.dct_matrix_w:
            self.dct_matrix_w[W] = self.create_dct_matrix(W, device)
        return self.dct_matrix_h[H], self.dct_matrix_w[W]

    def dct_2d(self, x):
        """
        对最后两个维度进行 2D DCT 变换，输入 x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        dct_mat_h, dct_mat_w = self.get_dct_matrices(H, W, x.device)

        # 先将 (B, C) 展开，然后做矩阵乘法
        x = x.view(B*C, H, W)
        x = torch.matmul(dct_mat_h, x)         # [B*C, H, W]
        x = torch.matmul(x, dct_mat_w.t())     # [B*C, H, W]
        x = x.view(B, C, H, W)
        return x

    def idct_2d(self, x):
        """
        对最后两个维度进行 2D IDCT 变换，输入 x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        dct_mat_h, dct_mat_w = self.get_dct_matrices(H, W, x.device)

        x = x.view(B*C, H, W)
        # X_idct = dct_h^T @ X @ dct_w
        x = torch.matmul(dct_mat_h.t(), x)
        x = torch.matmul(x, dct_mat_w)
        x = x.view(B, C, H, W)
        return x

    def high_pass_filter(self, x, alpha):
        """
        与原先 HighDctFrequencyExtractor 类似，
        alpha 较小时表示只保留图像右下角那部分高频区域：
            mask[:alpha_h, :alpha_w] = 0
        """
        B, C, H, W = x.shape
        mask = torch.ones((H, W), device=x.device)
        alpha_h, alpha_w = int(alpha * H), int(alpha * W)
        # 将左上 (alpha_h, alpha_w) 置0，保留右下的高频分量
        mask[:alpha_h, :alpha_w] = 0

        # 扩展至 (B, C, H, W) 的形状，用于逐元素相乘
        mask = mask.unsqueeze(0).unsqueeze(0)
        return x * mask

    def forward_single_scale(self, x):
        """
        针对某一个尺度（已经下采样或原尺寸）进行多频带提取并将结果拼接。
        """
        # 先做 DCT
        x_dct = self.dct_2d(x)

        # 针对 alpha_list 进行多频带提取
        feat_list = []
        for alpha in self.alpha_list:
            # 高频滤波
            x_high = self.high_pass_filter(x_dct, alpha)
            # 逆DCT
            x_idct = self.idct_2d(x_high)
            feat_list.append(x_idct)

        # 多个频带结果在通道维度拼接
        # 如果原输入 C=3，alpha_list 长度=2，则输出 C=6
        out_feats = torch.cat(feat_list, dim=1)
        return out_feats

    def forward(self, x):
        """
        多尺度 & 多频带融合
        """
        B, C, H, W = x.shape

        multi_scale_feats = []
        for scale in self.scales:
            if abs(scale - 1.0) < 1e-6:
                # 原始尺寸
                scaled_x = x
            else:
                # 下采样
                new_h = int(H * scale)
                new_w = int(W * scale)
                scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # 在该尺度下做多频带提取
            feats_scale = self.forward_single_scale(scaled_x)

            # 如果做了下采样，需要再上采样回到原尺寸，以便在通道维度融合
            if abs(scale - 1.0) > 1e-6:
                feats_scale = F.interpolate(feats_scale, size=(H, W), mode='bilinear', align_corners=False)

            multi_scale_feats.append(feats_scale)

        # 将不同尺度的结果在通道维度拼接
        xh = torch.cat(multi_scale_feats, dim=1)  # [B, C*(num alpha_list)*(num scales), H, W]

        # 做一次简单的全局 min-max 归一化
        # 这样得到的高频输出类似 0~1 范围
        min_vals = xh.view(B, -1).min(dim=1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
        max_vals = xh.view(B, -1).max(dim=1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
        xh = (xh - min_vals) / (max_vals - min_vals + 1e-8)

        return xh


@MODELS.register_module()
class HighDctFrequencyExtractor(nn.Module):
    """TAFE-Net HIGH-frequency view extractor -> paper I_hf (H x W x 3).

    Given the visual RGB image I_v (x), runs a whole-image 2D-DCT, ZEROES the
    top-left DC/low-frequency corner of the spectrum (a high-pass band mask),
    inverts the DCT and per-sample min-max normalises. The 3-channel result I_hf
    is later Cat(I_v, I_hf) inside the backbone's VFIM and drives the MFFE
    high-frequency branch (the ConvNeXt encoder that produces {F_h1, F_h2}).

    PAPER DIFF: (1) no explicit YCrCb conversion (operates on the mean/std-
    normalized RGB tensor, not a [0,255] image); (2) whole-image DCT, not the
    JPEG 8x8-block DCT; (3) rectangular corner band mask, not an isotropic split.
    """
    def __init__(self, alpha=0.05):
        super(HighDctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None

    def create_dct_matrix(self, N):
        # Build the N x N orthonormal DCT-II basis matrix (cached per side length).
        # Entry (k, n) = sqrt(2/N) * cos(pi*k*(2n+1)/(2N)); this 1D transform is
        # applied separably along H and W below to realise the whole-image 2D-DCT.
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        # DC row (k=0) overwritten with 1/sqrt(N) so the basis stays orthonormal.
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        # Whole-image forward 2D-DCT: X_dct = A_h * x * A_w^T (separable over H, W).
        # PAPER DIFF: this is a WHOLE-IMAGE RGB DCT (not the JPEG 8x8-block DCT), and
        # x is the mean/std-normalized RGB tensor rather than a [0,255] Y channel; this
        # spectrum is NOT the paper's JPEG-luminance coefficient map D (which feeds FPH).
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    def idct_2d(self, x):
        # Inverse 2D-DCT: x = A_h^T * D * A_w. Rebuilds the spatial view after the
        # band mask has zeroed part of the DCT spectrum.
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    def high_pass_filter(self, x, alpha):
        # HIGH-pass band mask: keep every coefficient EXCEPT the top-left corner.
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        # Zero the [:alpha_h, :alpha_w] TOP-LEFT block = DC/low-freq corner -> drops
        # coarse structure, so the inverse DCT keeps only the fine (high-freq) detail.
        mask[:alpha_h, :alpha_w] = 0

        return x * mask

    def forward(self, x):
        # Build the paper HIGH-frequency view I_hf from I_v (x: B x 3 x H x W).
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        B = xh.shape[0]
        # Per-sample min-max normalisation into ~[0,1] over all C*H*W elements.
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)
        return xh




@MODELS.register_module()
class LowDctFrequencyExtractor(nn.Module):
    """TAFE-Net LOW-frequency view extractor -> paper I_lf (H x W x 3).

    Same pipeline as HighDctFrequencyExtractor, but the band mask ZEROES the
    BOTTOM-RIGHT high-frequency corner of the DCT spectrum (a low-pass mask), so
    the inverse DCT keeps the coarse structure. The 3-channel result I_lf is
    later Cat(I_v, I_lf) inside the backbone's VFIM and drives the MFFE
    low-frequency branch (the SegFormer/MiT encoder that produces {F_l1, F_l2}).

    PAPER DIFF: (1) no explicit YCrCb conversion (operates on the mean/std-
    normalized RGB tensor, not a [0,255] image); (2) whole-image DCT, not the
    JPEG 8x8-block DCT; (3) rectangular corner band mask, not an isotropic split.
    """
    def __init__(self, alpha=0.95):
        super(LowDctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None

    def create_dct_matrix(self, N):
        # Build the N x N orthonormal DCT-II basis matrix (cached per side length).
        # Entry (k, n) = sqrt(2/N) * cos(pi*k*(2n+1)/(2N)); this 1D transform is
        # applied separably along H and W below to realise the whole-image 2D-DCT.
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        # DC row (k=0) overwritten with 1/sqrt(N) so the basis stays orthonormal.
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        # Whole-image forward 2D-DCT: X_dct = A_h * x * A_w^T (separable over H, W).
        # PAPER DIFF: this is a WHOLE-IMAGE RGB DCT (not the JPEG 8x8-block DCT), and
        # x is the mean/std-normalized RGB tensor rather than a [0,255] Y channel; this
        # spectrum is NOT the paper's JPEG-luminance coefficient map D (which feeds FPH).
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    def idct_2d(self, x):
        # Inverse 2D-DCT: x = A_h^T * D * A_w. Rebuilds the spatial view after the
        # band mask has zeroed part of the DCT spectrum.
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    def high_pass_filter(self, x, alpha):
        # LOW-pass band mask (method name reused from the High class): keep every
        # coefficient EXCEPT the bottom-right corner, so only low freqs survive.
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        # Zero the [-alpha_h:, -alpha_w:] BOTTOM-RIGHT block = high-freq corner ->
        # removes fine detail, so the inverse DCT keeps the coarse (low-freq) view.
        mask[-alpha_h:, -alpha_w:] = 0

        return x * mask

    def forward(self, x):
        # Build the paper LOW-frequency view I_lf from I_v (x: B x 3 x H x W).
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        B = xh.shape[0]
        # Per-sample min-max normalisation into ~[0,1] over all C*H*W elements.
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)
        return xh




@MODELS.register_module()
class NoFilter(BaseModule):
    """Identity filter (returns the input as-is, used as a placeholder for no frequency processing)."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

@MODELS.register_module()
class SimpleProjection(BaseModule):
    """Lightweight projection module of two ConvModules (for channel transformation/feature refinement)."""
    def __init__(self, in_channels=3, hidden_channels=12, out_channels=3, kernel_size=3, norm_cfg=dict(type='BN')):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            norm_cfg=norm_cfg,
        )
        self.conv2 = ConvModule(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@MODELS.register_module()
class SRMConv2d_simple(BaseModule):
    """SRM noise-residual filter (3 fixed SRM high-pass kernel convolutions + Hardtanh clipping, spatial-domain forensics)."""

    def __init__(self, inc=3, learnable=False, extra_projection=False):
        super().__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

        self.extra_projection = extra_projection
        if self.extra_projection:
            self.proj = SimpleProjection(in_channels=inc, hidden_channels=16, out_channels=inc, kernel_size=5)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        if self.extra_projection:
            out = self.proj(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # stack the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters






@MODELS.register_module()
class BayarConv2d(BaseModule):
    """Bayar constrained convolution (learnable high-pass kernel with center fixed at -1 and remaining weights normalized to sum to 0, spatial-domain forensics)."""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super().__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.rgb2gray(x)
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)

        return x

    def rgb2gray(self, rgb):
        b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = torch.unsqueeze(gray, 1)
        return gray


@MODELS.register_module()
class ResidualFilters(BaseModule):
    """
    Residual Filters in preprocessing block
    """

    def __init__(self, inc=3, learnable=False, extra_projection=False):
        super().__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

        self.extra_projection = extra_projection
        if self.extra_projection:
            self.proj = SimpleProjection(in_channels=inc, hidden_channels=16, out_channels=inc, kernel_size=5)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        if self.extra_projection:
            out = self.proj(out)

        return out

    def _build_kernel(self, inc):

        filter1 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, -1, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter2 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter3 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]

        filter4 = [[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, -1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter5 = [[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, -2, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter6 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]


        filter1 = np.asarray(filter1, dtype=float)
        filter2 = np.asarray(filter2, dtype=float) /2.
        filter3 = np.asarray(filter3, dtype=float) / 4.
        filter4 = np.asarray(filter4, dtype=float)
        filter5 = np.asarray(filter5, dtype=float) / 2.
        filter6 = np.asarray(filter6, dtype=float) / 12.
        # stack the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3],
                   [filter4],
                   [filter5],
                   [filter6]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters



if __name__ ==  '__main__':
    srm = SRMConv2d_simple(inc=3, learnable=False)
