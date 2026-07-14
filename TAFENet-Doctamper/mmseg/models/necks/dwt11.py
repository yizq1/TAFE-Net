# =============================================================================
# TRAINING-FLOW / DFDE  (Direction-aware Frequency Decoupling Enhancement)
#   (file: mmseg/models/necks/dwt11.py)
# -----------------------------------------------------------------------------
# This file implements the neck that plays the role of the paper's DFDE stage
# in TAFE-Net ("Frequency Mining Empowered by Text Aggregation", AAAI 2026;
# SegFormer/MiT-b2 baseline). The DocTamper configs register and instantiate the
# `DWTFPN_dct_v6` class below; everything else in this file is either a helper
# it depends on, or a dead/unused variant kept from earlier experiments.
#
# WHERE IT SITS IN THE PIPELINE:
#   MFFE backbone -> (F_1, F_2, F_3, F_4)  -> [THIS NECK = DFDE] -> SegFormerHead
# The MFFE (Multi-Frequency Feature Extractor) emits the four multi-scale
# feature maps F_1..F_4 (F_1 highest resolution, F_4 lowest). DFDE decouples
# each F_i into directional frequency sub-bands, refines/aggregates them across
# scales, and returns a refined tuple (F'_1, F_2, F_3, F'_4) for the seg head.
#
# ##### CRITICAL LEGACY-NAMING NOTE (read before trusting any name here) #####
# The neck is called "DWTFPN_dct_v6" and almost every module is named DCT_*
# (DCT_RB / DCT_ARB / DCT_AGF / DCT_AGFG / DCT_Attention), yet the transform
# that is ACTUALLY used on the training path is a WAVELET, not a DCT:
#     DWTNeck.__init__ sets  self.DWT = DB4Wavelet()
# i.e. a db4 (Daubechies-4) discrete wavelet transform (4-tap h0/h1 filters,
# outer-producted into LL/LH/HL/HH 2D kernels, stride-2). The plain-DCT class
# `DCT`, the `OptimizedDCT_V2` class, the Haar-style `DWT` class, and the
# High/Low-DCT frequency extractors are ALL inactive variants -- they are not
# used by the DocTamper configs. So "DCT" in the class names is legacy naming; read
# it as "per-sub-band frequency module", and the sub-band decomposition is db4.
#
# PAPER <-> CODE CORRESPONDENCE (see per-class docstrings for detail):
#   DWTFPN_dct_v6 = DWTNeck (= paper DFDE) followed by a standard mmseg FPN.
#       PAPER DIFF: in the paper F'_1,F_2,F_3,F'_4 go straight to SegFormerHead;
#       here an extra FPN is inserted between DFDE and the head.
#   DWTNeck        = paper DFDE proper.
#   DB4Wavelet     = the db4 wavelet used to split F_i -> LL/LH/HL/HH.
#   ETM (T1..T4)   = per-scale input transform applied to F_1..F_4.
#   DCT_AGFG       = paper FFM (Frequency Fusion Module): 3 cascaded GFA that
#                    consolidate ONE sub-band type across the 4 scales.
#   DCT_AGF        = one GFA (Guidance-based Feature Aggregation) step: an ARB
#                    attention map + a guided filter (GF).
#   DCT_RB/DCT_ARB = direction-specific residual/attention block; LL & HH use a
#                    square 7x7 conv, LH uses Conv1x7 (vertical), HL uses Conv7x1
#                    (horizontal) -- matches the paper's direction-specific convs.
#   GF             = differentiable guided filter used inside each GFA.
#   AM/CA/SA       = channel + spatial attention used by DCT_ARB.
# See DWTNeck.forward for how F_H / F'_1 / F'_4 are assembled and the PAPER DIFF
# notes on the high-freq fusion (PixelShuffle) and the LL PONO branch.
# =============================================================================

from mmengine.model import BaseModule
from mmseg.registry import MODELS
import torch.nn as nn
import time
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import math


class HighDctFrequencyExtractor(nn.Module):
    # (dead/unused) inactive DCT high-pass variant; not on the DocTamper training path.
    """
    提取高频部分的DCT变换模块
    alpha 取值越小，保留的高频区域越大（因为会把低频mask掉）
    """
    def __init__(self, alpha=0.05):
        super(HighDctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None


    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        # x: (B, 1, H, W) 假设只处理单通道
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        # 先对 height 做DCT，再对 width 做DCT
        return torch.matmul(self.dct_matrix_h, torch.matmul(x.squeeze(1), self.dct_matrix_w.t())).unsqueeze(1)

    def idct_2d(self, x):
        # x: (B, 1, H, W)
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x.squeeze(1), self.dct_matrix_w)).unsqueeze(1)

    def high_pass_filter(self, x, alpha):
        # x: (B, 1, H, W)，H, W分别是DCT变换后的频域尺寸(与原图一致)
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        # 将左上角(低频区域)置0，保留高频
        mask[:alpha_h, :alpha_w] = 0
        return x * mask

    def forward(self, x):
        # x: (B, 1, H, W)
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        # 做简单归一化，防止值域太大或为负
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals + 1e-6)
        return xh


# (dead/unused) inactive DCT low-pass variant; not on the DocTamper training path.
class LowDctFrequencyExtractor(nn.Module):
    """
    提取低频部分的DCT变换模块
    alpha 取值越大，保留的低频区域就小(只留左上角更小区域)
    """
    def __init__(self, alpha=0.95):
        super(LowDctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None

    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h, torch.matmul(x.squeeze(1), self.dct_matrix_w.t())).unsqueeze(1)

    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x.squeeze(1), self.dct_matrix_w)).unsqueeze(1)

    def low_pass_filter(self, x, alpha):
        # 这里为了“保留低频”，实现思路可以把右下角(高频区域)置0
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        # 置0的是右下角的区域，保留左上角
        mask[-alpha_h:, -alpha_w:] = 0
        return x * mask

    def forward(self, x):
        xq = self.dct_2d(x)
        xq_low = self.low_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_low)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals + 1e-6)
        return xh


# (dead/unused) these *_static helpers are only called by the inactive `DCT` class
# below; the active neck uses DB4Wavelet, so none of them run on the DocTamper path.
# --- DCT 相关工具函数 (可以放在 DCT 类外部或作为静态方法) ---
def create_dct_matrix_static(N, device):
    n_range = torch.arange(N, dtype=torch.float32, device=device).reshape((1, N))
    k_range = torch.arange(N, dtype=torch.float32, device=device).reshape((N, 1))
    dct_matrix = torch.sqrt(torch.tensor(2.0 / N, device=device)) * torch.cos(math.pi * k_range * (2 * n_range + 1) / (2 * N))
    dct_matrix[0, :] = 1 / math.sqrt(N)
    return dct_matrix

def dct_2d_static(x_input, dct_matrix_h, dct_matrix_w):
    # x_input: (B, 1, H, W)
    x_squeezed = x_input.squeeze(1)
    dct_h = torch.matmul(dct_matrix_h, x_squeezed)
    dct_hw = torch.matmul(dct_h, dct_matrix_w.t())
    return dct_hw.unsqueeze(1)

def idct_2d_static(x_dct_coeffs, idct_matrix_h_t, idct_matrix_w): # note: args are the transposed H matrix and the original W matrix
    # x_dct_coeffs: (B, 1, H, W) DCT 系数
    x_squeezed = x_dct_coeffs.squeeze(1)
    idct_h = torch.matmul(idct_matrix_h_t, x_squeezed)
    idct_hw = torch.matmul(idct_h, idct_matrix_w)
    return idct_hw.unsqueeze(1)

def normalize_min_max(tensor):
    B = tensor.shape[0]
    min_vals = tensor.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
    max_vals = tensor.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
    return (tensor - min_vals) / (max_vals - min_vals + 1e-6)


# (dead/unused) inactive true-DCT sub-band split; the neck uses DB4Wavelet instead.
class DCT(nn.Module):
    """
    使用DCT将x分解为四个近似的频域子带：LL, LH, HL, HH。
    split_ratio: 用于划分四个频域象限的比例 (0到1之间)。
                 例如，0.5 表示在H/2, W/2处划分。
    normalize_outputs: 是否对每个子带进行min-max归一化。
    """
    def __init__(self, split_ratio=0.5, normalize_outputs=True):
        super(DCT, self).__init__()
        if not (0 < split_ratio < 1):
            raise ValueError("split_ratio 必须在 0 和 1 之间 (不包括边界)")
        self.split_ratio = split_ratio
        self.normalize_outputs = normalize_outputs
        self._dct_matrix_h_cache = None
        self._dct_matrix_w_cache = None
        self._idct_matrix_h_t_cache = None # cache of the transposed version
        self._idct_matrix_w_cache = None   # cache of the original version (the IDCT formula differs)


    def _get_dct_matrices(self, H, W, device):
        # 获取或创建并缓存DCT/IDCT矩阵
        # 为了IDCT，我们需要 D_h.T 和 D_w (不是 D_w.T)
        if self._dct_matrix_h_cache is None or self._dct_matrix_h_cache.size(0) != H or self._dct_matrix_h_cache.device != device:
            self._dct_matrix_h_cache = create_dct_matrix_static(H, device)
            self._idct_matrix_h_t_cache = self._dct_matrix_h_cache.t() # D_h.T for IDCT
        if self._dct_matrix_w_cache is None or self._dct_matrix_w_cache.size(0) != W or self._dct_matrix_w_cache.device != device:
            self._dct_matrix_w_cache = create_dct_matrix_static(W, device)
            self._idct_matrix_w_cache = self._dct_matrix_w_cache # D_w for IDCT (not D_w.T)

        return self._dct_matrix_h_cache, self._dct_matrix_w_cache, \
               self._idct_matrix_h_t_cache, self._idct_matrix_w_cache


    def forward(self, x):
        # x: (N, C, H, W)
        N_batch, C_channel, H_img, W_img = x.shape
        
        # 获取DCT/IDCT矩阵
        dct_mat_h, dct_mat_w, idct_mat_h_t, idct_mat_w = self._get_dct_matrices(H_img, W_img, x.device)

        # 重塑以独立处理每个通道
        x_reshaped = x.reshape(N_batch * C_channel, 1, H_img, W_img)
        
        # 1. 执行2D DCT
        dct_coeffs = dct_2d_static(x_reshaped, dct_mat_h, dct_mat_w) # (N*C, 1, H, W)

        # 2. 创建四个象限的掩码
        h_split = max(1, int(self.split_ratio * H_img)) # ensure at least 1
        w_split = max(1, int(self.split_ratio * W_img)) # ensure at least 1

        mask_ll = torch.zeros_like(dct_coeffs)
        mask_lh = torch.zeros_like(dct_coeffs)
        mask_hl = torch.zeros_like(dct_coeffs)
        mask_hh = torch.zeros_like(dct_coeffs)

        mask_ll[:, :, :h_split, :w_split] = 1
        mask_lh[:, :, :h_split, w_split:] = 1
        mask_hl[:, :, h_split:, :w_split] = 1
        mask_hh[:, :, h_split:, w_split:] = 1
        
        # 3. 提取各子带系数
        coeffs_ll = dct_coeffs * mask_ll
        coeffs_lh = dct_coeffs * mask_lh
        coeffs_hl = dct_coeffs * mask_hl
        coeffs_hh = dct_coeffs * mask_hh

        # 4. 对各子带系数执行2D IDCT
        out_ll = idct_2d_static(coeffs_ll, idct_mat_h_t, idct_mat_w)
        out_lh = idct_2d_static(coeffs_lh, idct_mat_h_t, idct_mat_w)
        out_hl = idct_2d_static(coeffs_hl, idct_mat_h_t, idct_mat_w)
        out_hh = idct_2d_static(coeffs_hh, idct_mat_h_t, idct_mat_w)

        if self.normalize_outputs:
            out_ll = normalize_min_max(out_ll)
            out_lh = normalize_min_max(out_lh)
            out_hl = normalize_min_max(out_hl)
            out_hh = normalize_min_max(out_hh)
            
        # 5. 恢复原始批次和通道维度
        out_ll = out_ll.view(N_batch, C_channel, H_img, W_img)
        out_lh = out_lh.view(N_batch, C_channel, H_img, W_img)
        out_hl = out_hl.view(N_batch, C_channel, H_img, W_img)
        out_hh = out_hh.view(N_batch, C_channel, H_img, W_img)
        
        return out_ll, out_lh, out_hl, out_hh




# (dead/unused) inactive optimized-DCT variant; not on the DocTamper training path.
# (dead/unused) FFT-based DCT-like split variant; never instantiated on the DocTamper path (neck uses DB4Wavelet).
class OptimizedDCT_V2(nn.Module):
    """
    优化的DCT实现V2版本。
    核心优化：
    1. 使用切片+填充代替掩码乘法，大幅降低峰值显存。
    2. 移除不再需要的掩码创建和缓存机制。
    3. 将子带处理逻辑封装，使代码更清晰，并按顺序执行，有利于内存回收。
    """
    def __init__(self, split_ratio=0.5, normalize_outputs=True):
        super().__init__()
        if not (0 < split_ratio < 1):
            raise ValueError("split_ratio must be between 0 and 1")
        self.split_ratio = split_ratio
        self.normalize_outputs = normalize_outputs

    def _normalize(self, tensor):
        """Min-max归一化，保持不变"""
        B = tensor.shape[0]
        # 使用视图操作以减少内存开销
        tensor_flat = tensor.view(B, -1)
        min_vals = torch.amin(tensor_flat, dim=1, keepdim=True).view(B, 1, 1, 1)
        max_vals = torch.amax(tensor_flat, dim=1, keepdim=True).view(B, 1, 1, 1)
        # 加上一个小的epsilon防止除以零
        return (tensor - min_vals) / (max_vals - min_vals + 1e-6)

    def _process_sub_band(self, x_freq, h_start, h_end, w_start, w_end):
        """
        处理单个频域子带。
        该方法通过切片、填充和逆变换来处理，避免创建完整的稀疏张量。
        """
        B, C, H, W = x_freq.shape
        
        # 1. 创建一个小的、仅包含目标频率系数的张量（切片）
        freq_sub = x_freq[:, :, h_start:h_end, w_start:w_end]
        
        # 2. 创建一个目标大小的全零张量
        padded_freq = torch.zeros_like(x_freq)
        
        # 3. 将切片出的频率系数填充到正确的位置
        padded_freq[:, :, h_start:h_end, w_start:w_end] = freq_sub
        
        # 4. 对填充后的张量进行逆傅里叶变换
        # .real会创建一个新的张量，这是必要的
        return torch.fft.ifft2(padded_freq, dim=(-2, -1)).real

    def forward(self, x):
        # 1. 执行FFT，这是不可避免的显存开销
        x_freq = torch.fft.fft2(x.float(), dim=(-2, -1))
        
        B, C, H, W = x.shape
        h_split = int(self.split_ratio * H)
        w_split = int(self.split_ratio * W)
        
        # 2. 顺序处理每个子带，这允许PyTorch的内存管理器回收上一个子带
        # 计算中产生的临时张量（如padded_freq）。
        ll = self._process_sub_band(x_freq, 0, h_split, 0, w_split)
        lh = self._process_sub_band(x_freq, 0, h_split, w_split, W)
        hl = self._process_sub_band(x_freq, h_split, H, 0, w_split)
        hh = self._process_sub_band(x_freq, h_split, H, w_split, W)
        
        if self.normalize_outputs:
            # 归一化也会产生临时张量，但影响远小于频域操作
            ll = self._normalize(ll)
            lh = self._normalize(lh)
            hl = self._normalize(hl)
            hh = self._normalize(hh)
            
        return ll, lh, hl, hh




class DB4Wavelet(nn.Module):
    """db4 (Daubechies-4) 2D discrete wavelet transform -- the ACTIVE sub-band
    split used by DWTNeck (= paper DFDE) to decompose each F_i into LL/LH/HL/HH.

    Despite the surrounding "DCT_*" naming in this file, THIS is the transform the
    paper refers to as the db4 wavelet. h0 (low-pass) and h1 (high-pass) are the
    4-tap db4 filters; their outer products give the four separable 2D kernels:
        LL = h0 x h0 (approx.),  LH = h0 x h1 (vertical detail),
        HL = h1 x h0 (horizontal detail),  HH = h1 x h1 (diagonal detail).
    Each kernel is applied with stride=2, so every sub-band is downsampled x2.
    """
    def __init__(self):
        super(DB4Wavelet, self).__init__()
        # Daubechies db4 小波滤波器系数
        # h0 = low-pass (approximation), h1 = high-pass (detail); 4 taps each.
        self.register_buffer('h0', torch.tensor([
            0.6830127, 1.1830127, 0.3169873, -0.1830127
        ], dtype=torch.float32))
        
        self.register_buffer('h1', torch.tensor([
            -0.1830127, -0.3169873, 1.1830127, -0.6830127
        ], dtype=torch.float32))
        
        # 构建2D卷积核
        self._build_wavelet_kernels()
    
    def _build_wavelet_kernels(self):
        # Build the four separable 2D kernels as outer products of the 1D filters.
        # 低通滤波器
        h0_2d = self.h0.unsqueeze(0) * self.h0.unsqueeze(1)  # LL
        h0h1_2d = self.h0.unsqueeze(0) * self.h1.unsqueeze(1)  # LH
        h1h0_2d = self.h1.unsqueeze(0) * self.h0.unsqueeze(1)  # HL
        h1_2d = self.h1.unsqueeze(0) * self.h1.unsqueeze(1)  # HH
        
        # 注册为buffer
        self.register_buffer('kernel_ll', h0_2d.unsqueeze(0).unsqueeze(0))
        self.register_buffer('kernel_lh', h0h1_2d.unsqueeze(0).unsqueeze(0))
        self.register_buffer('kernel_hl', h1h0_2d.unsqueeze(0).unsqueeze(0))
        self.register_buffer('kernel_hh', h1_2d.unsqueeze(0).unsqueeze(0))
    
    def forward(self, x):
        # x: [B, C, H, W] feature map F_i. Returns 4 sub-bands, each [B, C, ~H/2, ~W/2].
        B, C, H, W = x.shape

        # 确保尺寸是偶数  (pad to even H/W so the stride-2 conv tiles cleanly)
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            H = H + 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))
            W = W + 1

        # 使用reshape代替view
        # Fold channels into the batch dim so one single-channel kernel filters every channel.
        x_flat = x.reshape(-1, 1, H, W)  # fix: use reshape

        # 应用小波变换  (stride-2 => each sub-band is downsampled x2)
        ll = F.conv2d(x_flat, self.kernel_ll, stride=2, padding=1)  # LL: approximation (low freq)
        lh = F.conv2d(x_flat, self.kernel_lh, stride=2, padding=1)  # LH: vertical detail
        hl = F.conv2d(x_flat, self.kernel_hl, stride=2, padding=1)  # HL: horizontal detail
        hh = F.conv2d(x_flat, self.kernel_hh, stride=2, padding=1)  # HH: diagonal detail
        
        # 恢复batch和channel维度
        ll = ll.reshape(B, C, ll.shape[2], ll.shape[3])
        lh = lh.reshape(B, C, lh.shape[2], lh.shape[3])
        hl = hl.reshape(B, C, hl.shape[2], hl.shape[3])
        hh = hh.reshape(B, C, hh.shape[2], hh.shape[3])
        
        return ll, lh, hl, hh




# @MODELS.register_module()
class DWTNeck(BaseModule):
    """Paper's DFDE (Direction-aware Frequency Decoupling Enhancement).

    Consumes the MFFE outputs (F_1, F_2, F_3, F_4) and returns a refined tuple
    (F'_1, F_2, F_3, F'_4) for the seg head. Pipeline (see forward):
      1. T1..T4 (ETM) transform each F_i.
      2. DB4Wavelet splits every transformed F_i into LL/LH/HL/HH sub-bands.
      3. For EACH sub-band type, DCT_AGFG (= paper FFM) aggregates that sub-band
         across the 4 scales with THREE cascaded GFA steps.
      4. High-freq path: Cat(LH_agg, HL_agg, HH_agg) -> fusion -> upsample = paper
         F_H; then F'_1 = Conv1x1(Cat(F_H, F_1)).
         Low-freq path : LL_agg -> downsample -> F'_4 = Conv1x1(Cat(F_LL, F_4)).

    NAMING: `self.DWT = DB4Wavelet()` -- the transform is the db4 WAVELET even
    though the field/modules are named DWT/DCT_*. The commented-out `DWT()` and
    `DCT(...)` alternatives are inactive variants, not used by the DocTamper configs.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # T1..T4 = per-scale ETM input transforms applied to F_1..F_4 (paper: the
        # feature refinement before wavelet decomposition). All project to out_channels.
        self.T1 = ETM(in_channels[0], out_channels)
        self.T2 = ETM(in_channels[1], out_channels)
        self.T3 = ETM(in_channels[2], out_channels)
        self.T4 = ETM(in_channels[3], out_channels)

        # wavelet attention module
        # ACTIVE transform = db4 wavelet (F_i -> LL/LH/HL/HH). The two commented
        # lines below (Haar DWT / true DCT) are inactive variants, unused by DocTamper.
        # self.DWT = DWT()
        self.DWT=DB4Wavelet()
        # self.DWT=DCT()
        # self.DWT = DCT( # 命名仍为 DWT 以兼容，实际是新的 DCT
        #     split_ratio=0.5,
        #     normalize_outputs=True
        # )

        # self.ll_attention = DCT_Attention(out_channels, 'LL')
        # self.lh_attention = DCT_Attention(out_channels, 'LH')
        # self.hl_attention = DCT_Attention(out_channels, 'HL')
        # self.hh_attention = DCT_Attention(out_channels, 'HH')

        # One FFM (= DCT_AGFG) per sub-band type. Each consolidates its sub-band
        # across the 4 scales via 3 cascaded GFA. Direction is handled inside via
        # subband_type: LL/HH square conv, LH Conv1x7 (vertical), HL Conv7x1 (horizontal).
        self.AGFG_LL = DCT_AGFG(out_channels, 'LL')
        self.AGFG_LH = DCT_AGFG(out_channels, 'LH')
        self.AGFG_HL = DCT_AGFG(out_channels, 'HL')
        self.AGFG_HH = DCT_AGFG(out_channels, 'HH')

        # Downsampling for aggregated LL path
        # LL_agg comes out at F_1 resolution; three stride-2 convs bring it down x8 to F_4.
        self.LL_down = nn.Sequential(
            BasicConv2d(out_channels, out_channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(out_channels, out_channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(out_channels, out_channels, stride=2, kernel_size=3, padding=1)
        )
        # PixelShuffle(2) upsamples the 4C fused high-freq map by x2 (4C -> C).
        # PAPER DIFF: the paper fuses F_H with a plain Conv1x1; here the fuse is a
        # 3C->4C conv followed by PixelShuffle upsampling (see forward).
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        # Convolution layers for final fusion
        # For f4_LL path (low-resolution output) = paper F'_4 = Conv1x1(Cat(F_LL, F_4)).
        self.one_conv_f4_ll = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1)
        
        # For f1_HH path (high-resolution output)
        # We will concatenate aggregated LH, HL, HH and the original f1 feature
        # So, input channels = out_channels (for LH_agg) + out_channels (for HL_agg) + out_channels (for HH_agg) + out_channels (for f1)
        # This might be too many channels. Let's try fusing LH, HL, HH first.
        # Fuse the three high-freq sub-bands (LH,HL,HH) into the 4C map that
        # PixelShuffle later upsamples = paper F_H = Conv1x1(Cat(F_LH,F_HL,F_HH)).
        self.high_freq_fusion_conv = BasicConv2d(out_channels * 3, out_channels*4, kernel_size=1) # eq(13): Cat(F^LH,F^HL,F^HH) -> F^H
        # F'_1 = Conv1x1(Cat(F_H, F_1)): merge the fused high-freq map with F_1.
        self.one_conv_f1_fused_high = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1)

    def forward(self, inputs):
        """DFDE forward: (F_1,F_2,F_3,F_4) -> (F'_1, F_2, F_3, F'_4)."""
        assert len(inputs) == len(self.in_channels)
        # inputs = paper's MFFE outputs; f1 highest resolution, f4 lowest.
        f1_in, f2_in, f3_in, f4_in = inputs

        # 1. Initial Transformation  (ETM refine + project each F_i to out_channels)
        f1 = self.T1(f1_in) # Highest resolution
        f2 = self.T2(f2_in)
        f3 = self.T3(f3_in)
        f4 = self.T4(f4_in) # Lowest resolution

        # 2. DCT decomposition for each feature level
        # db4 wavelet split (naming says "DCT" but transform is the db4 wavelet):
        # each subbands_fX = (LL, LH, HL, HH), each ~half the spatial size of fX.
        subbands_f1 = self.DWT(f1)
        subbands_f2 = self.DWT(f2)
        subbands_f3 = self.DWT(f3)
        subbands_f4 = self.DWT(f4)

        # Helper to get specific sub-band from all levels
        # Order for AGFG: (guide_lowest_res_feature, ..., guide_highest_res_feature, data_to_refine)
        # No, AGFG(f_guide_res2, f_data_res1); AGFG(f_guide_res3, output_of_prev_AGF)
        # AGFG(f1,f2,f3,f4) where f1=data, f2=guide, f3=guide for next stage, f4=guide for final stage
        # So, AGFG_XX(wf4_xx, wf3_xx, wf2_xx, wf1_xx) means:
        # y1 = AGF1(wf3_xx, wf4_xx) -> wf4_xx is data, wf3_xx is guide
        # y2 = AGF2(wf2_xx, y1)    -> y1 is data, wf2_xx is guide
        # y3 = AGF3(wf1_xx, y2)    -> y2 is data, wf1_xx is guide
        # This refines the lowest resolution sub-band (wf4_xx) using higher-res sub-bands as guides.
        # f1_ll = self.ll_attention(subbands_f1[0])
        # f1_lh = self.lh_attention(subbands_f1[1])
        # f1_hl = self.hl_attention(subbands_f1[2])
        # f1_hh = self.hh_attention(subbands_f1[3])
        
        # 更新f1的子带
        enhanced_subbands_f1 = (subbands_f1[0], subbands_f1[1], subbands_f1[2], subbands_f1[3])



        # Per sub-band FFM: aggregate the SAME sub-band across the 4 scales.
        # Args are (f4_band, f3_band, f2_band, f1_band); DCT_AGFG refines from the
        # lowest-res (f4) sub-band up to f1's resolution via 3 cascaded GFA.
        LL_agg = self.AGFG_LL(subbands_f4[0], subbands_f3[0], subbands_f2[0], enhanced_subbands_f1[0])
        LH_agg = self.AGFG_LH(subbands_f4[1], subbands_f3[1], subbands_f2[1], enhanced_subbands_f1[1])
        HL_agg = self.AGFG_HL(subbands_f4[2], subbands_f3[2], subbands_f2[2], enhanced_subbands_f1[2])
        HH_agg = self.AGFG_HH(subbands_f4[3], subbands_f3[3], subbands_f2[3], enhanced_subbands_f1[3])
        
        # At this point, LL_agg, LH_agg, HL_agg, HH_agg should be at the resolution of f1 (highest res)
        # because AGFG refines towards the resolution of its last guide (wf1_xx).

        # 3. Low-resolution path -> paper F'_4 = Conv1x1(Cat(F_LL, F_4)).
        # LL_agg is at f1's resolution. We need to downsample it to f4's resolution.
        # The self.LL_down is designed for features at f1's resolution to be downsampled 3 times (8x).
        # If f1 is 64x64 and f4 is 8x8, this is correct.
        LL_agg_down = self.LL_down(LL_agg)

        # Ensure dimensions match f4 before concatenation
        if LL_agg_down.shape[-2:] != f4.shape[-2:]:
            LL_agg_down = F.interpolate(LL_agg_down, size=f4.shape[-2:], mode='bilinear', align_corners=True)
        
        f4_LL = torch.cat([LL_agg_down, f4], dim=1)
        f4_LL = self.one_conv_f4_ll(f4_LL)

        # 4. eq(13)+eq(14) high-freq path: fuse F^LH/F^HL/F^HH -> F^H, then Cat with F1 -> F1_prime
        # Fuse LH_agg, HL_agg, HH_agg. They are all at f1's resolution.
        fused_high_freq = torch.cat([LH_agg, HL_agg, HH_agg], dim=1)
        fused_high_freq_processed = self.high_freq_fusion_conv(fused_high_freq) # Now C=out_channels

        # Concatenate with original f1 feature
        # PAPER DIFF: the paper builds F_H with a plain Conv1x1; here F_H is upsampled
        # via PixelShuffle(2) (4C->C, x2) then bilinear-resized to F_1's resolution.
        # print(fused_high_freq_processed.size(),f1.size())
        fused_high_freq_processed = self.dePixelShuffle(fused_high_freq_processed)
        fused_high_freq_processed = F.interpolate(fused_high_freq_processed, (f1.size(2), f1.size(3)), mode='bilinear', align_corners=True)  
        f1_fused_high = torch.cat([fused_high_freq_processed, f1], dim=1)
        
        f1_fused_high = self.one_conv_f1_fused_high(f1_fused_high)

        # eq(16): output (F1_prime, F2, F3, F4_prime) -> FPN -> SegFormerHead
        return tuple([f1_fused_high, f2, f3, f4_LL])


from mmseg.models.necks.fpn import FPN

@MODELS.register_module()
class DWTFPN_dct_v6(BaseModule):
    """DocTamper-registered neck: DFDE (DWTNeck) followed by a standard mmseg FPN.

    This is the class the DocTamper configs instantiate. It equals the paper's DFDE
    stage plus an extra FPN.

    PAPER DIFF: in the paper the refined pyramid (F'_1, F_2, F_3, F'_4) is fed
    STRAIGHT to SegFormerHead. Here an extra standard FPN sits between DFDE and
    the head (self.fpn), all four inputs already unified to out_channels.
    """
    def __init__(self,
                 **kwargs):
        """Build DWTNeck (DFDE), then build the FPN after unifying the FPN input channels to out_channels."""
        super().__init__()
        # DFDE: takes the backbone's multi-scale in_channels, emits out_channels x4.
        self.neck = DWTNeck(in_channels=kwargs['in_channels'], out_channels=kwargs['out_channels'])
        # FPN then re-mixes those 4 equal-width maps (in_channels are now all out_channels).
        kwargs['in_channels'] = [kwargs['out_channels']] * 4
        self.fpn = FPN(**kwargs)

    def forward(self, x):
        # x = (F_1..F_4) from MFFE; neck -> (F'_1,F_2,F_3,F'_4); FPN -> head inputs.
        x = self.neck(x)
        return self.fpn(x)


class ConvLayer(nn.Module):
    """Reflection-padded plain conv (no norm, no activation). ACTIVE only inside
    DCT_ARB's LL branch: it builds the mean_conv1..3 / std_conv1..3 stacks that
    refine the per-pixel mean and std stripped out by PONO before they are
    re-injected into the LL sub-band. ReflectionPad2d(kernel_size//2) keeps the
    spatial size unchanged and avoids zero-padding artefacts on the low-freq stats."""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class BasicConv2d(nn.Module):
    """Conv->Norm->(optional ReLU) building block used throughout DWTNeck (= paper
    DFDE): every ETM branch, DCT_RB's direction convs, LL_down and
    high_freq_fusion_conv are built from it. A single bias-free Conv2d followed by a
    norm layer and (by default) a ReLU. `bn` selects the norm class (BatchNorm2d by
    default; DCT_RB passes InstanceNorm2d for the LL band); `need_relu=False` drops
    the activation, e.g. for the second conv of a residual block so the skip is added
    before non-linearity."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


# ETM: RFB-style input-transform used as DWTNeck.T1..T4 to project each backbone scale to out_channels before DB4 (paper DFDE input projection).
class ETM(nn.Module):
    """Per-scale input transform (T1..T4 in DWTNeck) applied to each F_i before
    the wavelet split. A multi-branch RFB/receptive-field block: a 1x1 branch
    plus three asymmetric (1xk then kx1) + dilated branches (k=3,5,7), concatenated
    and fused with a 3x3 conv, added to a 1x1 residual. Projects in_channels ->
    out_channels while enlarging the receptive field."""
    def __init__(self, in_channels, out_channels):
        super(ETM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# (dead/unused) inactive Haar-style DWT variant; DWTNeck uses DB4Wavelet instead.
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        _, _, h, w = x.size()

        # 如果高度或宽度是奇数，进行填充
        if h % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))  # pad one row along the height
        if w % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))  # pad one column along the width

        # DWT 分解
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # 计算 LL, LH, HL, HH
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4

        return ll, lh, hl, hh

class SA(nn.Module):
    """Spatial Attention: two 3x3 convs -> single-channel sigmoid map, multiplied
    back onto the input. Used by AM (inside DCT_ARB) to re-weight sub-band pixels."""
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y


class CA(nn.Module):
    """Channel Attention (ECA-style): global pool -> 1D conv over channels -> sigmoid
    gate. lf=True uses average pooling (low-freq), lf=False uses max pooling."""
    def __init__(self, lf=True):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class AM(nn.Module):
    """Attention Module = channel attention (CA) then spatial attention (SA),
    applied in sequence. DCT_ARB uses it to produce the ARB attention/guidance map
    that feeds the guided filter inside each GFA (DCT_AGF)."""
    def __init__(self, channels, lf):
        super(AM, self).__init__()
        self.CA = CA(lf=lf)
        self.SA = SA(channels)

    def forward(self, x):
        x = self.CA(x)
        x = self.SA(x)
        return x


# 新增：针对DCT子带特性的残差块
class DCT_RB(nn.Module):
    """Direction-specific residual block, one per sub-band type. This is where the
    paper's direction-specific convolutions live (naming says DCT, band is db4):
        LL & HH -> SQUARE 7x7 conv,
        LH (vertical detail)   -> Conv1x7,
        HL (horizontal detail) -> Conv7x1.
    LL uses InstanceNorm, the high-freq bands use BatchNorm. Two convs + identity
    skip. (Note: the module's own convention labels LH 'horizontal-high/vertical-low'
    in the Chinese comments, but the kernel shapes match the paper's LH=1x7 / HL=7x1.)
    """
    def __init__(self, channels, subband_type):
        super(DCT_RB, self).__init__()
        is_lowfreq = (subband_type == 'LL')
        norm_layer = nn.InstanceNorm2d if is_lowfreq else nn.BatchNorm2d
        
        # 不同子带使用不同的卷积策略
        # if subband_type == 'LL':
        #     # 低频使用较大感受野
        #     self.conv1 = BasicConv2d(channels, channels, 3, padding=1, bn=norm_layer)
        #     self.conv2 = BasicConv2d(channels, channels, 3, padding=1, bn=norm_layer, need_relu=False)
        # elif subband_type == 'LH':
        #     # 水平高频，垂直低频 - 使用水平方向的卷积
        #     self.conv1 = BasicConv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bn=norm_layer)
        #     self.conv2 = BasicConv2d(channels, channels, 3, padding=1, bn=norm_layer, need_relu=False)
        # elif subband_type == 'HL':
        #     # 水平低频，垂直高频 - 使用垂直方向的卷积
        #     self.conv1 = BasicConv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bn=norm_layer)
        #     self.conv2 = BasicConv2d(channels, channels, 3, padding=1, bn=norm_layer, need_relu=False)
        # else:  # 'HH'
        #     # 高频使用更小的卷积核
        #     self.conv1 = BasicConv2d(channels, channels, 3, padding=1, bn=norm_layer)
        #     self.conv2 = BasicConv2d(channels, channels, 3, padding=2, dilation=2, bn=norm_layer, need_relu=False)
        
        if subband_type == 'LH':
            # LH (vertical detail) -> Conv1x7 first conv = paper's vertical direction conv.
            # 水平高频，垂直低频 - 使用水平方向的卷积
            self.conv1 = BasicConv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), bn=norm_layer)
            self.conv2 = BasicConv2d(channels, channels, 7, padding=3, bn=norm_layer, need_relu=False)
        elif subband_type == 'HL':
            # HL (horizontal detail) -> Conv7x1 first conv = paper's horizontal direction conv.
            # 水平低频，垂直高频 - 使用垂直方向的卷积
            self.conv1 = BasicConv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), bn=norm_layer)
            self.conv2 = BasicConv2d(channels, channels, 7, padding=3, bn=norm_layer, need_relu=False)
        elif subband_type == 'HH':
            # HH (diagonal detail) -> SQUARE 7x7 conv (paper: square conv for HH).
            # 水平垂直都是高频 - 使用完整卷积
            self.conv1 = BasicConv2d(channels, channels, 7, padding=3, bn=norm_layer)
            self.conv2 = BasicConv2d(channels, channels, 7, padding=3, bn=norm_layer, need_relu=False)
        else:  # 'LL' or other cases
            # LL (approximation) -> SQUARE 7x7 conv (paper: square conv for LL).
            # 低频分量 - 使用标准卷积
            self.conv1 = BasicConv2d(channels, channels, 7, padding=3, bn=norm_layer)
            self.conv2 = BasicConv2d(channels, channels, 7, padding=3, bn=norm_layer, need_relu=False)



        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)


# DCT_ARB: DCT_RB + attention (AM); for LL it adds PONO normalize + mean/std refine (EXTRA ops, not in paper DFDE).
class DCT_ARB(nn.Module):
    """Attention Residual Block: a direction-specific residual block (DCT_RB)
    followed by the AM attention (CA+SA). Its output is the guidance/attention map
    that each GFA (DCT_AGF) feeds into the guided filter. For LL it additionally
    normalizes with PONO (see forward) and restores the learned mean/std.

    PAPER DIFF: the extra PONO normalize/denormalize on the LL band is an
    implementation detail not stated in the paper's FFDN-style GFA."""
    def __init__(self, channels, subband_type):
        super(DCT_ARB, self).__init__()
        self.subband_type = subband_type
        self.rb = DCT_RB(channels, subband_type)
        # self.attention = DCT_Attention(channels, subband_type)  # inactive variant
        self.attention=AM(channels, True)


        # 低频子带的特殊处理
        if subband_type == 'LL':
            self.mean_conv1 = ConvLayer(1, 16, 1, 1)
            self.mean_conv2 = ConvLayer(16, 16, 3, 1)
            self.mean_conv3 = ConvLayer(16, 1, 1, 1)
            
            self.std_conv1 = ConvLayer(1, 16, 1, 1)
            self.std_conv2 = ConvLayer(16, 16, 3, 1)
            self.std_conv3 = ConvLayer(16, 1, 1, 1)
    
    def PONO(self, x, epsilon=1e-5):
        # Positional Normalization: per-pixel normalize across channels, returning
        # the removed mean/std so they can be re-injected (LL branch only).
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
        output = (x - mean) / std
        return output, mean, std

    def forward(self, x):
        # LL: strip mean/std via PONO, refine them through small conv stacks.
        if self.subband_type == 'LL':
            # PONO (extra op, LL only, not in paper): normalize x; mean/std refined by convs and re-injected after RB+attention
            x, mean, std = self.PONO(x)
            mean = self.mean_conv3(self.mean_conv2(self.mean_conv1(mean)))
            std = self.std_conv3(self.std_conv2(self.std_conv1(std)))

        y = self.rb(x)
        y = self.attention(y)

        # LL: re-inject the refined statistics before returning.
        if self.subband_type == 'LL':
            return y * std + mean
        return y


class BoxFilter(nn.Module):
    """Fast box filter: the sliding-window sum over a (2r+1) square, computed in
    O(1) per pixel via an integral-image trick -- cumulative-sum then a fixed-offset
    difference along H (diff_x) and then W (diff_y). It is the workhorse of GF (the
    guided filter): every local mean the guided-filter linear model needs (mean_a,
    mean_ax, mean_ay, ... and the normalizer N) is one boxfilter call. (Standard
    guided-filter box filter; here it powers the paper's FFM/GFA guidance-based
    sub-band aggregation.) diff_x/diff_y assemble the window sum from the three
    boundary regions (left edge / interior difference / right edge)."""
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GF(nn.Module):
    """Differentiable (attention-weighted) Guided Filter -- the core of each GFA
    step (DCT_AGF). Given a low-res guide lr_x, a low-res target lr_y, a high-res
    guide hr_x and an attention map l_a (from DCT_ARB), it fits a local linear
    model (A, b) with box-filter statistics weighted by l_a, upsamples (A, b) to
    the high-res grid, and returns A*hr_x + b. This propagates a low-res refined
    sub-band up to the higher-res scale using the finer sub-band as guidance,
    which is the FFDN-style guidance-based aggregation the paper's FFM/GFA use.
    r is the box-filter radius; eps regularizes the linear-model denominator."""
    def __init__(self, r, eps=1e-8):
        super(GF, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        # lr_x: low-res guide, lr_y: low-res target, hr_x: high-res guide,
        # l_a: attention/guidance weights (from DCT_ARB). Returns hr-res refined map.
        # import pdb;pdb.set_trace()
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        # assert n_lrx == n_lry and n_lry == n_hrx
        # assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        # print(h_lrx,h_lry,w_lrx,w_lry)
        # assert h_lrx == h_lry and w_lrx == w_lry
        
        # assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ss=self.boxfilter(l_a)
        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A * hr_x + mean_b).float()


# (dead/unused) inactive attention variant; DCT_ARB uses AM instead of this.
# 新增：DCT特性感知的注意力机制
class DCT_Attention(nn.Module):
    """DCT频域特性感知的注意力机制"""
    def __init__(self, channels, subband_type='LL'):
        super(DCT_Attention, self).__init__()
        self.subband_type = subband_type
        
        # 创建可学习的频域权重 - 用于对不同频率位置加权
        self.freq_weights = nn.Parameter(torch.ones(2, 2))
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 针对不同子带定制
        if subband_type == 'LL':
            # 低频使用平均池化
            pool_type = nn.AdaptiveAvgPool2d
        else:
            # 高频使用最大池化
            pool_type = nn.AdaptiveMaxPool2d
            
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 应用频域感知的通道注意力
        B, C, H, W = x.shape
        
        # 构建频域权重掩码(根据子带特性调整权重)
        h_idx = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, 1, H, W).to(x.device)
        w_idx = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, 1, H, W).to(x.device)
        
        # 根据DCT特性和子带类型构建不同的频域权重掩码
        if self.subband_type == 'LL':
            # 低频子带 - 更重视左上角
            freq_mask = self.freq_weights[0, 0] * (1-h_idx) * (1-w_idx)
        elif self.subband_type == 'LH':
            # LH子带 - 更重视左下角
            freq_mask = self.freq_weights[0, 1] * (1-h_idx) * w_idx
        elif self.subband_type == 'HL':
            # HL子带 - 更重视右上角
            freq_mask = self.freq_weights[1, 0] * h_idx * (1-w_idx)
        else:  # 'HH'
            # HH子带 - 更重视右下角
            freq_mask = self.freq_weights[1, 1] * h_idx * w_idx
        
        # 应用通道注意力
        channel_weight = self.channel_attention(x)
        
        # 应用空间注意力 - 结合平均池化和最大池化特征
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weight = self.spatial_attention(spatial_feat)
        
        # 综合各种注意力权重
        return x * channel_weight * spatial_weight * freq_mask.expand_as(x)


class DCT_AGF(nn.Module):
    """One GFA (Guidance-based Feature Aggregation) step of the paper's FFM.
    It refines a low-res sub-band (low_level) using a higher-res sub-band of the
    same type (high_level) as guidance: DCT_ARB builds the attention map, the
    guided filter GF propagates the refinement to the higher-res grid, and a
    sub-band-specific post_process sharpens the result (identity for LL, edge
    enhance for LH/HL, dilated detail enhance for HH). Three of these are
    cascaded inside DCT_AGFG."""
    def __init__(self, channels, subband_type):
        super(DCT_AGF, self).__init__()
        self.ARB = DCT_ARB(channels, subband_type)
        self.GF = GF(r=2, eps=1e-2)
        self.subband_type = subband_type
        
        # 针对DCT的子带特性添加后处理
        if subband_type == 'LL':
            # 低频保持
            self.post_process = nn.Identity()
        elif subband_type in ['LH', 'HL']:
            # 中频边缘增强
            self.post_process = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(0.2)
            )
        else:  # 'HH'
            # 高频细节增强
            self.post_process = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
                nn.ReLU()
            )
    
    def forward(self, high_level, low_level):
        # high_level = higher-res guide sub-band; low_level = lower-res sub-band to refine.
        # 根据输入特征图尺寸决定下采样目标尺寸
        N, C, H, W = high_level.size()
        target_height = (H // 2 + 1) if H % 2 != 0 else (H // 2)
        target_width = (W // 2 + 1) if W % 2 != 0 else (W // 2)

        # 下采样高层特征  (low-res version of the guide, aligned to low_level's grid)
        high_level_small = F.interpolate(high_level, size=(target_height, target_width),
                                         mode='bilinear', align_corners=True)

        # 应用ARB进行调制，得到特征注意力  (ARB attention map = guided-filter weights)
        y = self.ARB(low_level)

        # 应用导向滤波进行区域特性保留  (GF lifts low_level to high_level's resolution)
        y = self.GF(high_level_small, low_level, high_level, y)

        # 应用子带特定的后处理  (sub-band-specific sharpening)
        return self.post_process(y)

# Paper's FFM (Frequency Fusion Module): consolidates ONE sub-band type across the
# 4 scales via THREE cascaded GFA (= DCT_AGF). Called once per sub-band (LL/LH/HL/HH)
# from DWTNeck. Correspondence to the FFDN FFM is in-spirit (guided-filter based),
# not a verbatim match.
class DCT_AGFG(nn.Module):
    """针对DCT子带特性的多层次特征融合"""
    def __init__(self, channels, subband_type):
        super(DCT_AGFG, self).__init__()
        self.subband_type = subband_type
        # GF1/GF2/GF3 = the three cascaded GFA steps (= paper's 3 GFA in the FFM).
        self.GF1 = DCT_AGF(channels, subband_type)
        self.GF2 = DCT_AGF(channels, subband_type)
        self.GF3 = DCT_AGF(channels, subband_type)
        
        # 特征融合模块
        if subband_type != 'LL':
            self.fusion = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        
        # 添加子带特有的增强处理
        if subband_type == 'LL':
            # 低频强化结构
            self.enhance = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif subband_type == 'LH':
            # 垂直高频增强
            self.enhance = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif subband_type == 'HL':
            # 水平高频增强
            self.enhance = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:  # 'HH'
            # 细节增强
            self.enhance = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, f1, f2, f3, f4):
        # Called from DWTNeck as (f4_band, f3_band, f2_band, f1_band); here the args
        # are named f1..f4 locally, with f1 = the lowest-res sub-band (from F_4) that
        # gets progressively guided up to the highest-res grid (from F_1).
        # 自下而上的特征融合过程  (cascade the 3 GFA: each guides the running result up one scale)
        y1 = self.GF1(f2, f1)
        y2 = self.GF2(f3, y1)
        y3 = self.GF3(f4, y2)

        # 根据子带类型确定是否进行跨级融合
        if self.subband_type == 'LL':
            # 低频子带保持结构信息，不需要跨级融合  (LL: just enhance, no skip fuse)
            y = self.enhance(y3)
        else:
            # 高频子带需要跨级融合以增强特定细节  (high-freq: fuse an early stage back in)
            y1_up = F.interpolate(y1, size=y3.shape[2:], mode='bilinear', align_corners=True)
            y = self.fusion(torch.cat([y3, y1_up], dim=1))
            y = self.enhance(y)

        return y
    

# (dead/unused) standalone smoke test run only under __main__; not part of training/inference.
def test_DWTFPN_dct_fast():
    print("==== Test DWTFPN_dct_fast ====")
    # 假设 in_channels=[32,64,128,256], out_channels=32, FPN 的参数可自定义
    model = DWTFPN_dct_v6(
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        add_extra_convs=False
    )
    model.eval()
    # 构造4个不同通道输入
    x = [
        torch.rand(2, 64, 65, 65),
        torch.rand(2, 128, 33, 33),
        torch.rand(2, 320, 17, 17),
        torch.rand(2, 512, 9, 9),
    ]
    start_time = time.time()
    out = model(x)
    elapsed = time.time() - start_time
    # 输出 shape
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            print(f"fpn_out[{i}] shape: {o.shape}")
    else:
        print("fpn_out shape:", out.shape)
    print(f"Inference time: {elapsed * 1000:.2f} ms")
    print("==== Passed! ====")

if __name__ == "__main__":
    # run standalone test as needed
    test_DWTFPN_dct_fast()
