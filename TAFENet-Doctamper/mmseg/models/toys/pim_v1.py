"""VFIM (Visual-Frequency Integration Module) building blocks for TAFE-Net (AAAI 2026).

Implements the paper's VFIM module, i.e. eq(3)/eq(4):
    Ih_hat = ConvBlock(Cat(DSB(I_h), DAB(I_h)))   # eq(3)
    Il_hat = ConvBlock(Cat(DSB(I_l), DAB(I_l)))   # eq(4)
where the VFIM input is the 6-channel Cat(Iv, I_hf) / Cat(Iv, I_lf) formed by
eq(1)/eq(2) upstream in the backbone. The active processing chain is:

- ``PRIM1``           : the VFIM unit (eq3/eq4; used as prim1/prim2 in the backbone).
- ``MOA``             : the paper's DSB (Direction-Sensitive Branch) with asymmetric convs.
- ``AsymmetricConv4`` : the DSB asymmetric kernels (1x11/11x1/1x7/7x1 per paper VFIM).
- ``MSA1``            : the paper's DAB (Direction-Agnostic Branch: 1x1 + 3x3 dwsep + FFN).
- ``MergePR`` / ``MergePR1`` : Cat+project of DSB & DAB (eq3/4) / single-branch project.

All other classes (MSA2/MSA3/MSA4, EfficientSelfAttention, dilatedComConv4,
PRIM2/PRIM3/PRIM4, SCSEModule, GELU) are alternative or unused variants kept
around for experiments.
"""
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
from ..backbones.mit import EfficientMultiheadAttention
from math import sqrt, floor

class MSA3(nn.Module):
    """Alternative/unused DAB variant (attention-based); kept for experiments."""
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
    """Alternative/unused DAB variant (attention-based); kept for experiments."""
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


class DepthwiseSeparableConv(nn.Module):
    """Lightweight depthwise-separable conv block, used in place of Self-Attention.

    Input and output channel counts are the same. Building block of the DAB (MSA1).
    """
    def __init__(self, dim, kernel_size=3, padding=1, activation=nn.ReLU):
        super().__init__()
        # Make sure InstanceNorm2d can handle the given dim
        self.net = nn.Sequential(
            # Depthwise conv: spatial convolution applied per channel independently
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False),
            # BatchNorm(dim), # optionally add Norm here
            activation(inplace=True),
            # Pointwise conv: 1x1 conv mixing channel information
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            # BatchNorm(dim) # or add Norm here
            # Note: the original MSA applies LayerNorm(InstanceNorm2d) outside the block, so no inner Norm is added here for now
        )
    def forward(self, x):
        return self.net(x)

    def initialize(self):
        weight_init(self) # assume weight_init can handle Conv2d

# --- Modified MSA class (replacing Self-Attention with DepthwiseSeparableConv) ---

class MSA1(nn.Module):
    """The paper's DAB (Direction-Agnostic Branch): the active DAB used in PRIM1/VFIM (eq3/eq4).

    Paper VFIM DAB = 1x1 ConvBlock + 3x3 depthwise-separable conv + FFN. Here:
    1x1 reduce -> unfold patches -> 1x1 embed -> DepthwiseSeparableConv(3x3) +
    residual -> MixFeedForward (FFN) + residual. (DEVIATION vs paper: an extra
    unfold/embed patch-embedding step wraps the 3x3 dwsep conv.)
    """
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(MSA1, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
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
        unfold_channels = (inplanes // reduction1) * (self.unfold_kernel_size ** 2)
        self.overlap_embed = nn.Conv2d(unfold_channels, planes, kernel_size=(1, 1), stride=(1, 1))

        # self.FU = nn.ModuleList([nn.Sequential(SCSEModule(planes,reduction=1), nn.Conv2d(planes, planes, 3, 1, 1), nn.BatchNorm2d(planes), nn.ReLU(True))])
        # self.FU.append(nn.Conv2d(planes, planes, 1, 1, 0))

        # --- Change point: replace EfficientSelfAttention with DepthwiseSeparableConv ---
        # self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        # self.SelfAttention = EfficientMultiheadAttention(embed_dims=planes, num_heads=4, sr_ratio=8,qkv_bias=True)
        self.SpatialProcessor = DepthwiseSeparableConv(dim=planes, kernel_size=3, padding=1, activation=nn.ReLU)
        # --- End change point ---

        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes) # keep LayerNorm (InstanceNorm2d)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h_reduced, w_reduced = x.shape[-2:]
        x_unfolded = self.get_overlap_patches(x)
        h_unfold_out = floor((h_reduced + 2 * self.unfold_padding - self.unfold_dilation * (self.unfold_kernel_size - 1) - 1) / self.unfold_stride + 1)
        w_unfold_out = floor((w_reduced + 2 * self.unfold_padding - self.unfold_dilation * (self.unfold_kernel_size - 1) - 1) / self.unfold_stride + 1)
        x = rearrange(x_unfolded, 'b c (h w) -> b c h w', h=h_unfold_out, w=w_unfold_out)
        x = self.overlap_embed(x)

        # x = self.SelfAttention(self.LN(x),(x.shape[2], x.shape[3]))




        # --- Change point: call the new SpatialProcessor ---
        # Apply LayerNorm -> SpatialProcessor -> residual connection
        x = self.SpatialProcessor(self.LN(x)) + x
        # x_1=self.FU[0](x)
        # x = self.FU[1](x_1)+x

        # --- End change point ---

        # Apply LayerNorm -> FeedForward -> residual connection (unchanged)
        x = self.ffd(self.LN(x)) + x

        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self) # make sure weight_init can initialize the new module






class EfficientSelfAttention(nn.Module):
    """Alternative/unused attention variant; kept for experiments."""
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
        # split into multi-head; here the number of heads is 1
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


class GELU(nn.Module):
    """Alternative/unused variant: tanh-approximation GELU activation."""
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


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class dilatedComConv4(nn.Module):
    """Alternative/unused variant: four parallel dilated convolutions (DSB precursor)."""

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






class AsymmetricConv4(nn.Module):
    """The DSB asymmetric convolutions (paper VFIM Direction-Sensitive Branch kernels).

    Paper DSB = four asymmetric convs (1x7, 7x1, 1x11, 11x1). Here four parallel
    asymmetric convs (1xk1, k1x1, 1xk2, k2x1; MOA passes k1=11, k2=7 ->
    1x11, 11x1, 1x7, 7x1) concatenated along the channel dimension to aggregate
    slender text along the horizontal/vertical directions. Each branch outputs
    the same spatial size.
    """
    def __init__(self, inplans, planes, kernel_k1=7, kernel_k2=5, stride=1, groups=1):
        """Initializer.

        :param inplans: number of input channels
        :param planes: total number of output channels (split evenly across 4 branches)
        :param kernel_k1: asymmetric kernel size used by branches 1 and 2
        :param kernel_k2: asymmetric kernel size used by branches 3 and 4
        :param stride: convolution stride
        :param groups: number of groups for grouped convolution (simplified here; each branch could be set independently, but they are usually set together)
        """
        super(AsymmetricConv4, self).__init__()

        # Make sure the total output channel count is divisible by 4
        assert planes % 4 == 0, "Total output planes must be divisible by 4"
        branch_planes = planes // 4 # output channels per branch

        # --- Branch 1: Kernel (1, k1) ---
        padding_k1_w = (kernel_k1 - 1) // 2
        self.conv1 = conv(inplans, branch_planes, kernel_size=(1, kernel_k1),
                          padding=(0, padding_k1_w), stride=stride, groups=groups, dilation=1)

        # --- Branch 2: Kernel (k1, 1) ---
        padding_k1_h = (kernel_k1 - 1) // 2
        self.conv2 = conv(inplans, branch_planes, kernel_size=(kernel_k1, 1),
                          padding=(padding_k1_h, 0), stride=stride, groups=groups, dilation=1)

        # --- Branch 3: Kernel (1, k2) ---
        padding_k2_w = (kernel_k2 - 1) // 2
        self.conv3 = conv(inplans, branch_planes, kernel_size=(1, kernel_k2),
                          padding=(0, padding_k2_w), stride=stride, groups=groups, dilation=1)

        # --- Branch 4: Kernel (k2, 1) ---
        padding_k2_h = (kernel_k2 - 1) // 2
        self.conv4 = conv(inplans, branch_planes, kernel_size=(kernel_k2, 1),
                          padding=(padding_k2_h, 0), stride=stride, groups=groups, dilation=1)

    def forward(self, x):
        # Pass through the four convolution branches separately
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)

        # Concatenate the results along the channel dimension (dim=1)
        # Thanks to the padding settings, out1, out2, out3, out4 should have the same H and W as the input x (when stride=1)
        # Or shrink according to stride, but they share the same H and W as each other
        return torch.cat((out1, out2, out3, out4), dim=1)

    def initialize(self):
        # Call the weight initialization function (make sure weight_init is defined)
        weight_init(self)




class MOA(nn.Module):
    """The paper's DSB (Direction-Sensitive Branch): the active DSB used in PRIM1/VFIM (eq3/eq4).

    1x1 reduce -> AsymmetricConv4 (DSB asymmetric kernels 1x11/11x1/1x7/7x1) -> 1x1.
    In the active PRIM1.forward this branch IS called as DSB(...) inside the
    two-input MergePR (the DAB-only ablation that drops it is commented out).
    """
    # def __init__(self, inplanes, planes, BatchNorm, reduction1=4, kernel_k1=7, kernel_k2=5):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4, kernel_k1=11, kernel_k2=7):
        """MOA module using AsymmetricConv4 in place of the original dilatedComConv4.

        Adds kernel_k1 and kernel_k2 parameters to control AsymmetricConv4.
        """
        super(MOA, self).__init__()
        intermediate_planes = inplanes // reduction1

        self.layers = nn.Sequential(
            # 1x1 conv to reduce dimensionality
            nn.Conv2d(inplanes, intermediate_planes, kernel_size=1, bias=False),
            BatchNorm(intermediate_planes),
            nn.ReLU(inplace=True),

            # Use the modified asymmetric convolution module
            AsymmetricConv4(intermediate_planes, intermediate_planes,
                            kernel_k1=kernel_k1, kernel_k2=kernel_k2, stride=1), # stride=1 preserves the spatial size
            BatchNorm(intermediate_planes),
            nn.ReLU(inplace=True),

            # 1x1 conv to increase/adjust dimensionality
            nn.Conv2d(intermediate_planes, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)
    
    def initialize(self):
        weight_init(self)


class MixFeedForward(nn.Module):
    """Mixed feed-forward network (FFN) building block of the DAB (MSA1)."""
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            GELU(),
            # nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1)
        )


    def forward(self, x):
        return self.net(x)
    def initialize(self):
        weight_init(self)

LayerNorm = partial(nn.InstanceNorm2d, affine = True)

class MSA4(nn.Module):
    """Alternative/unused DAB variant (attention-based); kept for experiments."""
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
    """Two-input merge = the ConvBlock(Cat(DSB, DAB)) of eq(3)/eq(4).

    Concatenate DSB and DAB outputs, then project (3x3 conv). This is the active
    merge in the full VFIM (DSB + DAB) -> Ih_hat / Il_hat.
    """
    def __init__(self, inplanes, planes, BatchNorm):
        super(MergePR, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes,  kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        x = torch.cat((local_context, global_context), dim=1)
        x = self.features(x)
        return x

    def initialize(self):
        weight_init(self)


class PRIM4(nn.Module):
    """Alternative/unused VFIM variant (uses MSA4 as the DAB); kept for experiments."""
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
    """Alternative/unused VFIM variant (uses MSA3 as the DAB); kept for experiments."""
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
    """Alternative/unused VFIM variant (uses MSA2 as the DAB); kept for experiments."""
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



class MergePR1(nn.Module):
    """Single-input merge: project a single branch output (3x3 conv), the eq(3)/eq(4)
    ConvBlock reduced to one branch.

    Used only by the DAB-only "without-DSB" ablation (commented out in PRIM1.forward).
    """
    def __init__(self, inplanes, planes, BatchNorm):
        super(MergePR1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes,  kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context):
        # x = torch.cat((local_context, global_context), dim=1)
        x=local_context
        x = self.features(x)
        return x

    def initialize(self):
        weight_init(self)







class PRIM1(nn.Module):
    """The VFIM unit implementing eq(3)/eq(4) (used as prim1/prim2 in the backbone).

    Input is the eq(1)/eq(2) 6-channel Cat(freq-view, RGB) [I_h=Cat(Iv,I_hf) or
    I_l=Cat(Iv,I_lf)], first projected down by conv1. It builds both the DSB
    (self.MOA) and DAB (self.MSA) branches and the two-input MergePR that applies
    ConvBlock(Cat(DSB, DAB)) to yield Ih_hat/Il_hat. In this file's active
    forward(), the full two-branch path ``merge_context(self.MOA(x), self.MSA(x))``
    (DSB + DAB, eq3/eq4) is the line that runs, while the DAB-only ablation line
    ``merge_context(self.MSA(x))`` (uses MergePR1) is commented out.
    """
    def __init__(self, inplanes, planes, BatchNorm):
        super(PRIM1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, 16, kernel_size=1, bias=False),
            BatchNorm(16),
            nn.ReLU(inplace=True)
        )

        inplanes=16
        # out_size_local_context = int(inplanes/2)

        # out_size_global_max_context = int(inplanes/2)

        # self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=1)

        # self.MSA = MSA1(inplanes, out_size_local_context, BatchNorm, reduction1=1)


        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.MOA = MOA(inplanes, out_size_local_context, BatchNorm, reduction1=4)   # DSB branch (eq3/eq4)

        self.MSA = MSA1(inplanes, out_size_local_context, BatchNorm, reduction1=4)  # DAB branch (eq3/eq4)


        self.merge_context = MergePR(out_size_local_context + out_size_global_max_context, planes, BatchNorm)  # ACTIVE: ConvBlock(Cat(DSB, DAB)) eq(3)/eq(4)
        # self.merge_context = MergePR1(out_size_local_context, planes, BatchNorm)   # ablation variant: single-branch (DAB-only) project


    def forward(self, x):
        """VFIM forward = eq(3)/eq(4).

        Project the eq(1)/eq(2) 6-channel Cat(freq-view, RGB) input down with conv1,
        then fuse the two branches. The ACTIVE line runs the full two-branch VFIM
        (DSB self.MOA + DAB self.MSA) through the two-input MergePR, i.e.
        ConvBlock(Cat(DSB, DAB)) -> Ih_hat/Il_hat; the commented-out line below it
        is the DAB-only "without-DSB" ablation (merge_context(self.MSA(x))).
        """
        x=self.conv1(x)                                    # 1x1 project Cat(view,RGB) 6ch -> 16ch
        x = self.merge_context(self.MOA(x),  self.MSA(x))  # ACTIVE eq(3)/eq(4): ConvBlock(Cat(DSB, DAB))
        # x=self.merge_context(self.MSA(x))                # ablation: DAB-only (drops DSB), uses MergePR1
        return x

    def initialize(self):
        weight_init(self)


class SCSEModule(nn.Module):
    """Alternative/unused variant: spatial & channel squeeze-and-excitation module."""
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