# =============================================================================
# TRAINING-FLOW / VFIM (Visual-Frequency Integration Module)  (file: pim_v1.py)
# -----------------------------------------------------------------------------
# This file supplies the paper's VFIM (Visual-Frequency Integration Module) for
# TAFE-Net ("Frequency Mining Empowered by Text Aggregation", AAAI 2026). The
# VFIM is realised by the class `PRIM1` (see bottom of file). In the RTM
# backbone (AsymCMNeXt_0524[_convback]) it is instantiated twice, as prim1/prim2:
#
#     prim1 = PRIM1(6, 3)      # low-freq  path: input = Cat(I_v, I_lf) -> I_hat_l
#                              #   (fed to the Transformer/SegFormer branch)
#     prim2 = PRIM1(6, 3)      # high-freq path: input = Cat(I_v, I_hf) -> I_hat_h
#                              #   (fed to the ConvNeXt branch)
#
# i.e. 6 input channels = concat of the RGB visual image I_v (HxWx3) with one
# DCT frequency view (I_hf high or I_lf low, HxWx3), and 3 output channels =
# paper's integrated image I_hat (HxWx3), which is then handed to the MFFE
# backbones (ConvNeXt for I_hat_h, SegFormer/MiT for I_hat_l).
#
# Internal VFIM structure (all in this file):
#   PRIM1  = paper VFIM      : conv1(1x1, 6->16) -> two parallel branches
#            {MOA, MSA1} -> fused by MergePR -> 3ch I_hat.
#   MOA    = paper DSB (Direction-Sensitive Branch) : 1x1 reduce ->
#            AsymmetricConv4 (four asymmetric convs) -> 1x1 up.
#   AsymmetricConv4 = the FOUR asymmetric convs (1x7),(7x1),(1x5),(5x1) whose
#            concatenation aggregates horizontal + vertical text-stroke cues.
#   MSA1   = paper DAB (Direction-Agnostic Branch) : 1x1 reduce +
#            DepthwiseSeparableConv (3x3 DW + 1x1 PW) + MixFeedForward FFN,
#            with residual connections.
#   DepthwiseSeparableConv = paper DAB's DWConv(+BN+ReLU) local mixer.
#   MixFeedForward         = the FFN inside DAB.
#   MergePR = the paper's final ConvBlock: Cat(DSB,DAB) -> Conv3x3 -> I_hat.
#
# PAPER<->CODE NAMING IS NOMINAL ONLY: the code never spells VFIM/DSB/DAB; the
# correspondence below is the mapping used consistently across the RTM configs.
#
# NOTE: many sibling classes in this file (MSA2/MSA3/MSA4, PRIM2/PRIM3/PRIM4,
# EfficientSelfAttention, dilatedComConv4, ...) are inactive variants NOT used by
# the VFIM instantiation and are intentionally left un-annotated here.
# =============================================================================
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum
from math import sqrt
import torch
import math
from torch.nn.functional import upsample
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, \
    PairwiseDistance
from torch.nn import functional as F
from torch.autograd import Variable
from math import sqrt, floor

class MSA3(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA3, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(720, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class MSA2(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA2, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(288, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)


# VFIM / DAB local mixer -- the depthwise-separable convolution used inside the
# paper's DAB (Direction-Agnostic Branch). It plays the role of the paper's
# "DWConv + BN + ReLU" local token mixer (channels in == channels out).
# PAPER DIFF: the paper's DAB spells this DWConv as DWConv + BN + ReLU, but here
# the BatchNorm is commented out (see the two "BatchNorm(dim)" lines below), so
# the active block is only DWConv -> ReLU -> pointwise-1x1 (no internal Norm).
class DepthwiseSeparableConv(nn.Module):
    """
    轻量级的深度可分离卷积块，用于替代 Self-Attention。
    输入和输出通道数相同。
    """
    # English: lightweight depthwise-separable conv block; stands in for
    # self-attention as the DAB's spatial mixer. Input and output channels match.
    def __init__(self, dim, kernel_size=3, padding=1, activation=nn.ReLU):
        super().__init__()
        # 确保 InstanceNorm2d 可以处理传入的 dim
        self.net = nn.Sequential(
            # 深度卷积: 每个通道独立进行空间卷积
            # Depthwise 3x3 conv (groups=dim): per-channel spatial filtering = the
            # spatial part of the paper's DAB DWConv.
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False),
            # BatchNorm(dim), # 可以选择在这里加Norm
            # PAPER DIFF: paper DAB has BN here (DWConv+BN+ReLU); it is disabled.
            activation(inplace=True),
            # 逐点卷积: 1x1卷积混合通道信息
            # Pointwise 1x1 conv: mixes channels (the "separable" half of DWConv).
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            # BatchNorm(dim) # 或者在这里加Norm
            # 注意：原始 MSA 在块外部使用了 LayerNorm(InstanceNorm2d)，这里暂时不加内部Norm
            # Note: the enclosing MSA1 applies LayerNorm(=InstanceNorm2d) outside
            # this block, so no internal Norm is added here.
        )
    def forward(self, x):
        return self.net(x)

    def initialize(self):
        weight_init(self) # 假设 weight_init 能处理 Conv2d

# --- 修改后的 MSA 类 (用 DepthwiseSeparableConv 替换 SelfAttention) ---

class MSA1(nn.Module):
    """VFIM Direction-Agnostic Branch (paper DAB), used by PRIM1 (= VFIM).

    Realises the paper's DAB: a 1x1 dimensionality reduction followed by a
    depthwise-separable local mixer (DepthwiseSeparableConv) and a MixFeedForward
    FFN, each wrapped with a LayerNorm + residual connection. It captures
    orientation-independent context, complementing MOA's direction-sensitive cues.

    PAPER DIFF: on top of the paper's "1x1 + DWConv(+BN+ReLU) + FFN", this code
    adds an nn.Unfold(kernel=3, stride=2) overlap-patch embedding (self-attention
    heritage from SegFormer/MiT) and a bilinear resample back to the input size;
    that overlap-patch/resample stage is not described in the paper's DAB.
    """
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA1, self).__init__()
        # 1x1 dim-reduce: inplanes -> inplanes//reduction1 (paper DAB's 1x1 stage).
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        # PAPER DIFF: overlap-patch unfold (k=3, s=2) -- extra stage not in paper DAB.
        self.unfold_kernel_size = 3
        self.unfold_dilation = 1
        self.unfold_padding = 1
        self.unfold_stride = 2
        self.get_overlap_patches = nn.Unfold(
            kernel_size=self.unfold_kernel_size,
            dilation=self.unfold_dilation,
            stride=self.unfold_stride,
            padding=self.unfold_padding
        )
        # 1x1 conv embeds the unfolded patches back to `planes` channels.
        unfold_channels = (inplanes // reduction1) * (self.unfold_kernel_size ** 2)
        self.overlap_embed = nn.Conv2d(unfold_channels, planes, kernel_size=(1, 1), stride=(1, 1))

        # --- 修改点：用 DepthwiseSeparableConv 替换 EfficientSelfAttention ---
        # self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        # Paper DAB local mixer: depthwise-separable conv replaces self-attention.
        self.SpatialProcessor = DepthwiseSeparableConv(dim=planes, kernel_size=3, padding=1, activation=nn.ReLU)
        # --- 结束修改点 ---

        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)  # paper DAB's FFN
        self.LN = LayerNorm(planes) # 保持 LayerNorm (InstanceNorm2d)

    def forward(self, x):
        x_size = x.size()
        # 1x1 dim-reduce (paper DAB 1x1 stage).
        x = self.reductionLayers(x)
        h_reduced, w_reduced = x.shape[-2:]
        # PAPER DIFF: overlap-patch unfold (stride 2) downsamples by ~2x -- not in
        # the paper's DAB; the spatial size is restored by the interpolate below.
        x_unfolded = self.get_overlap_patches(x)
        h_unfold_out = floor((h_reduced + 2 * self.unfold_padding - self.unfold_dilation * (self.unfold_kernel_size - 1) - 1) / self.unfold_stride + 1)
        w_unfold_out = floor((w_reduced + 2 * self.unfold_padding - self.unfold_dilation * (self.unfold_kernel_size - 1) - 1) / self.unfold_stride + 1)
        x = rearrange(x_unfolded, 'b c (h w) -> b c h w', h=h_unfold_out, w=w_unfold_out)
        x = self.overlap_embed(x)

        # --- 修改点：调用新的 SpatialProcessor ---
        # 应用 LayerNorm -> SpatialProcessor -> 残差连接
        # LayerNorm -> depthwise-separable mixer -> residual (paper DAB DWConv block).
        x = self.SpatialProcessor(self.LN(x)) + x
        # --- 结束修改点 ---

        # 应用 LayerNorm -> FeedForward -> 残差连接 (保持不变)
        # LayerNorm -> FFN -> residual (paper DAB FFN stage).
        x = self.ffd(self.LN(x)) + x

        # Resample back to the input resolution (undoes the unfold downsampling).
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self) # 确保 weight_init 能初始化新模块



class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.reduction_ratio = reduction_ratio

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:] # h:64,w:64
        heads, r = self.heads, self.reduction_ratio # heads:1 r:8
        # to_qkv:Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1) # x:[1,32,64,64]->[1,96,64,64]->3*[1,32,64,64]
        # k, v = map(lambda t: reduce(t, 'b c (h r1) (w r2) -> b c h w', 'mean', r1 = r, r2 = r), (k, v))
        # k,v : [1,32,64,64] -> [1,32,8,8]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))
        # 分为multi-head 此时head数为1
        # q:[1,32,64,64] -> [1,4096,32]
        # k:[1,32,8,8] -> [1,64,32]
        # v:[1,32,8,8] -> [1,64,32]
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # q*k: [1,4096,32]*[1,64,32]->[1,4096,64]
        attn = sim.softmax(dim = -1)
        # attention:[1,4096,64]
        out = einsum('b i j, b j d -> b i d', attn, v)
        # attn*v: [1,4096,64]*[1,64,32]->[1,4096,32]
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        # out: [1,4096,32]->[1,32,64,64]
        # to_out:Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # output:[1,32,64,64]->[1,32,64,64]
        return self.to_out(out)
    
    def initialize(self):
        weight_init(self)


# VFIM / DAB FFN activation -- hand-rolled tanh approximation of GELU.
# Reachable from PRIM1 via MSA1(=paper DAB).ffd = MixFeedForward, whose net
# applies GELU() between its two 1x1 projections (see MixFeedForward.net below).
# Equivalent to torch.nn.GELU(approximate='tanh'); implemented explicitly here.
class GELU(nn.Module):
    """Tanh-approximation GELU activation used by MixFeedForward (paper DAB's FFN).

    forward computes 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))), the standard
    tanh approximation of the GELU non-linearity. Runs on the VFIM forward path
    as the activation inside MSA1's Mix-FFN.
    """
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Unfold):
            pass
        elif isinstance(m, GELU):
            pass
        elif isinstance(m, Softmax):
            pass
        elif isinstance(m, Sigmoid):
            pass
        else:
            m.initialize()


# VFIM / DSB conv builder -- shared bias-free Conv2d factory. Reachable from
# PRIM1 via MOA(=paper DSB) -> AsymmetricConv4, which calls it to build the four
# asymmetric (1,k)/(k,1) direction-sensitive kernels. (Also referenced by the
# dead dilatedComConv4, which PRIM1 never instantiates.)
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class dilatedComConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 4, 4]):
        super(dilatedComConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0],dilation=1)
        self.conv2_2 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1],dilation=2)
        self.conv2_3 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2],dilation=3)
        self.conv2_4 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3],dilation=4)

    def forward(self, x):
        conv2_1 = self.conv2_1(x)
        conv2_2 = self.conv2_2(x)
        conv2_3 = self.conv2_3(x)
        conv2_4 = self.conv2_4(x)
        return torch.cat((conv2_1, conv2_2, conv2_3, conv2_4), dim=1)

    def initialize(self):
        weight_init(self)






# VFIM / DSB core -- the FOUR asymmetric convolutions of the paper's DSB
# (Direction-Sensitive Branch). Four parallel branches with 1-D-shaped kernels:
#   (1,k1) and (k1,1)  +  (1,k2) and (k2,1),  concatenated on the channel axis.
# The horizontal (1,k) kernels respond to horizontal text strokes and the
# vertical (k,1) kernels to vertical strokes, so the concat aggregates both
# horizontal + vertical text cues -- exactly the paper's DSB motivation.
# PAPER DIFF: the paper's DSB uses kernel sizes 7 AND 11 (Conv1x7/7x1 and
# Conv1x11/11x1); this code uses 7 AND 5 (default kernel_k1=7, kernel_k2=5).
class AsymmetricConv4(nn.Module):
    """
    使用长宽不同的卷积核 (Asymmetric Kernels) 的并行卷积模块。
    确保每个分支输出尺寸相同。
    """
    # English: parallel conv module with asymmetric (1-D-shaped) kernels; padding
    # is chosen so every branch keeps the same H,W and they can be concatenated.
    def __init__(self, inplans, planes, kernel_k1=7, kernel_k2=5, stride=1, groups=1):
        """
        初始化函数。
        :param inplans: 输入通道数
        :param planes: 总输出通道数 (会被均分到4个分支)
        :param kernel_k1: 第1和第2个分支使用的非对称卷积核尺寸
        :param kernel_k2: 第3和第4个分支使用的非对称卷积核尺寸
        :param stride: 卷积步长
        :param groups: 分组卷积的组数 (这里简化，每个分支可以独立设置，但通常一起设置)
        """
        super(AsymmetricConv4, self).__init__()
        
        # 确保总输出通道数可以被4整除
        assert planes % 4 == 0, "Total output planes must be divisible by 4"
        branch_planes = planes // 4 # 每个分支的输出通道数

        # --- 分支 1: Kernel (1, k1) ---
        # Branch 1: horizontal kernel (1,7) -> horizontal text-stroke cues.
        padding_k1_w = (kernel_k1 - 1) // 2
        self.conv1 = conv(inplans, branch_planes, kernel_size=(1, kernel_k1), 
                          padding=(0, padding_k1_w), stride=stride, groups=groups, dilation=1)

        # --- 分支 2: Kernel (k1, 1) ---
        # Branch 2: vertical kernel (7,1) -> vertical text-stroke cues.
        padding_k1_h = (kernel_k1 - 1) // 2
        self.conv2 = conv(inplans, branch_planes, kernel_size=(kernel_k1, 1), 
                          padding=(padding_k1_h, 0), stride=stride, groups=groups, dilation=1)

        # --- 分支 3: Kernel (1, k2) ---
        # Branch 3: horizontal kernel (1,5). PAPER DIFF: paper DSB uses 11 here.
        padding_k2_w = (kernel_k2 - 1) // 2
        self.conv3 = conv(inplans, branch_planes, kernel_size=(1, kernel_k2), 
                          padding=(0, padding_k2_w), stride=stride, groups=groups, dilation=1)

        # --- 分支 4: Kernel (k2, 1) ---
        # Branch 4: vertical kernel (5,1). PAPER DIFF: paper DSB uses 11 here.
        padding_k2_h = (kernel_k2 - 1) // 2
        self.conv4 = conv(inplans, branch_planes, kernel_size=(kernel_k2, 1), 
                          padding=(padding_k2_h, 0), stride=stride, groups=groups, dilation=1)

    def forward(self, x):
        # 分别通过四个卷积分支
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        
        # 在通道维度 (dim=1) 上拼接结果
        # 由于 padding 的设置，out1, out2, out3, out4 的 H 和 W 维度应该与输入 x 相同 (如果 stride=1)
        # 或者按照 stride 相应缩小，但彼此之间 H 和 W 维度相同
        return torch.cat((out1, out2, out3, out4), dim=1)

    def initialize(self):
        # 调用权重初始化函数 (需确保 weight_init 已定义)
        weight_init(self)




# VFIM / DSB -- the paper's DSB (Direction-Sensitive Branch), one of PRIM1's two
# branches. Structure: 1x1 conv dim-reduce -> AsymmetricConv4 (the four
# asymmetric convs that pick up horizontal + vertical text-stroke orientations)
# -> 1x1 conv restore. Each conv is followed by BatchNorm + ReLU.
class MOA(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4, kernel_k1=7, kernel_k2=5):
        """
        MOA 模块，使用 AsymmetricConv4 替代原来的 dilatedComConv4。
        添加了 kernel_k1 和 kernel_k2 参数用于控制 AsymmetricConv4。
        """
        # English: paper DSB. Uses AsymmetricConv4 (kernel_k1, kernel_k2) as the
        # direction-sensitive core between a 1x1 reduce and a 1x1 restore.
        super(MOA, self).__init__()
        intermediate_planes = inplanes // reduction1
        
        self.layers = nn.Sequential(
            # 1x1 卷积降维
            # 1x1 dim-reduce: inplanes -> inplanes//reduction1 (paper DSB 1x1 stage).
            nn.Conv2d(inplanes, intermediate_planes, kernel_size=1, bias=False),
            BatchNorm(intermediate_planes),
            nn.ReLU(inplace=True),
            
            # 使用修改后的非对称卷积模块
            # Four asymmetric convs = the direction-sensitive core of paper DSB.
            AsymmetricConv4(intermediate_planes, intermediate_planes, 
                            kernel_k1=kernel_k1, kernel_k2=kernel_k2, stride=1), # stride=1 保证尺寸
            BatchNorm(intermediate_planes),
            nn.ReLU(inplace=True),
            
            # 1x1 卷积升维/调整维度
            nn.Conv2d(intermediate_planes, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)
    
    def initialize(self):
        weight_init(self)


class MixFeedForward(nn.Module):
    """Mix-FFN feed-forward network -- the FFN stage inside the paper's DAB.

    A SegFormer-style Mix-FFN: 1x1 conv expands channels (dim -> dim*expansion),
    a 3x3 conv injects local position information, GELU activation, then a 1x1
    conv projects back to `dim`. Used by MSA1 (= paper DAB) as its FFN.
    (NB: unlike SegFormer's Mix-FFN, this 3x3 conv is dense/groups=1, not depthwise.)
    """
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor  # FFN inner width = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),               # 1x1 expand: dim -> hidden_dim
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),  # 3x3 conv: local position mixing
            GELU(),                                       # GELU non-linearity
            # nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1)                # 1x1 project back: hidden_dim -> dim
        )


    def forward(self, x):
        return self.net(x)
    def initialize(self):
        weight_init(self)

LayerNorm = partial(nn.InstanceNorm2d, affine = True)

class MSA4(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA4, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(1152, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class MergePR(nn.Module):
    """VFIM fusion block -- the paper's final ConvBlock I_hat = ConvBlock(Cat(DSB,DAB)).

    Concatenates the two VFIM branch outputs on the channel axis and fuses them
    with a single Conv3x3 -> BatchNorm -> ReLU. In PRIM1 the arguments are
    (MOA(x), MSA1(x)) = (DSB, DAB), and the `planes` output channels equal 3, i.e.
    the paper's integrated image I_hat (HxWx3).
    """
    def __init__(self, inplanes, planes, BatchNorm):
        super(MergePR, self).__init__()

        # Conv3x3 -> BN -> ReLU over the concatenated (DSB, DAB) features.
        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes,  kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        # local_context = MOA/DSB output, global_context = MSA1/DAB output.
        x = torch.cat((local_context, global_context), dim=1)  # Cat(DSB, DAB) on channels
        x = self.features(x)                                   # ConvBlock -> I_hat
        return x

    def initialize(self):
        weight_init(self)


class PRIM4(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM4, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.MSA = MSA4(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.MOA(x),  self.MSA(x))
        return x

    def initialize(self):
        weight_init(self)

class PRIM3(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM3, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.MSA = MSA3(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.MOA(x),  self.MSA(x))
        return x

    def initialize(self):
        weight_init(self)

class PRIM2(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM2, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.MSA = MSA2(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.MOA(x),  self.MSA(x))
        return x

    def initialize(self):
        weight_init(self)

class PRIM1(nn.Module):
    """VFIM (Visual-Frequency Integration Module) -- the paper's VFIM.

    Instantiated in the RTM backbone as prim1/prim2 = PRIM1(6, 3): the 6 input
    channels are Cat(I_v, freq-view) (I_v = RGB visual image + one DCT view I_hf or
    I_lf), and the 3 output channels are the paper's integrated image I_hat (HxWx3;
    I_hat_h from the high-freq input, I_hat_l from the low-freq input).

    Data flow (= paper VFIM):
        conv1 (1x1, 6 -> 16 ch) -> two parallel branches
            MOA  = DSB (Direction-Sensitive Branch, asymmetric convs),
            MSA1 = DAB (Direction-Agnostic Branch, DWConv + FFN),
        -> MergePR fuses Cat(DSB, DAB) via Conv3x3 -> I_hat (3 ch).
    PAPER<->CODE naming is nominal: `self.MSA` here is an MSA1 instance (= DAB).
    """
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM1, self).__init__()

        # 1x1 stem: project the 6ch Cat(I_v, freq-view) up to 16 working channels.
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, 16, kernel_size=1, bias=False),
            BatchNorm(16),
            nn.ReLU(inplace=True)
        )

        inplanes=16  # branches operate on the 16ch stem output
        out_size_local_context = int(inplanes/4)   # DSB branch out channels (=4)

        out_size_global_max_context = int(inplanes/4)  # DAB branch out channels (=4)

        # DSB branch (Direction-Sensitive): asymmetric convs for text-stroke cues.
        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        # DAB branch (Direction-Agnostic): DWConv + FFN for orientation-free context.
        self.MSA = MSA1(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        # Fuse Cat(DSB, DAB) (4+4 ch) -> `planes` (=3) ch = paper I_hat.
        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):
        x=self.conv1(x)  # 6ch Cat(I_v, freq-view) -> 16ch
        # I_hat = ConvBlock(Cat(DSB(x), DAB(x))): run both branches, fuse to 3ch.
        x = self.merge_context(self.MOA(x),  self.MSA(x))
        return x

    def initialize(self):
        weight_init(self)