#这个是convnext和segformer一起的
# Copyright (c) OpenMMLab. All rights reserved.
# =============================================================================
# TRAINING-FLOW / MFFE backbone  (file: aysm_cmnext_0524.py)
# -----------------------------------------------------------------------------
# MFFE (Multi-Frequency Feature Extractor) - SegFormer variant.
#
# The ACTIVE class on the RTM training path in this file is `AsymCMNeXt_0524`
# (defined below). It IS the paper's MFFE (Multi-Frequency Feature Extractor)
# from TAFE-Net ("Frequency Mining Empowered by Text Aggregation", AAAI 2026),
# instantiated as the SegFormer/MiT-b2 variant ("TAFE-Net" in Table 1).
#
# It is built as backbone.type='AsymCMNeXt_0524' by the top-level segmentor
# MyModelFull (mmseg/models/segmentors/my_model_full.py) and is called with the
# packed inputs x = [I_v, [I_lf, I_hf, F_d]], where:
#   x[0] = x_cam   = RGB image I_v  -> main_branch (SegFormer/MiT-b2 visual enc.)
#   x[1] = x_extra = [I_lf, I_hf, F_d] = low-freq view, high-freq view, and the
#                    DTD-FPH (Frequency Perception Head) DCT tamper feature F_d.
# Inside forward(): PRIM1 (= paper VFIM, Visual-Frequency Integration Module)
# fuses each frequency view with I_v to form the integrated views I_hat_l /
# I_hat_h; these are encoded by the Transformer branch (extra_branch) and the
# CNN/ConvNeXt branch (extra1_branch). Their per-stage features are fused with
# the visual stream via SCSEModule_improve (= paper scSE) to build the
# multi-frequency pyramid F_1..F_4 (channels [64,128,320,512]) that MyModelFull
# then forwards into DFDE + SegFormerHead.
#
# The rest of this file is a library of helper blocks plus MANY inactive /
# commented-out fusion variants; only AsymCMNeXt_0524 is annotated below.
# =============================================================================
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
from torch.distributions import Bernoulli

class SamplingGatedModule(nn.Module):
    def __init__(self, input_feature_channels, decision_hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_feature_channels, input_feature_channels, kernel_size=7, stride=2, padding=3)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.decision_net = nn.Sequential(
            nn.Linear(input_feature_channels, decision_hidden_dim),
            nn.ReLU(),
            nn.Linear(decision_hidden_dim, 2) # 输出一个logit
        )

    def forward(self, feature_input):
        """
        feature_input: (batch_size, channels, height, width) 输入的特征图
        """
        feature_input=self.conv1(feature_input)
        pooled_features = self.global_avg_pool(feature_input)
        flattened_features = torch.flatten(pooled_features, 1)
        logit = self.decision_net(flattened_features) # (batch_size, 1)
        
        # prob_select_one 是选择动作1的概率
        # prob_select_one = torch.sigmoid(logit) # (batch_size, 1)
        # prob_select_one = torch.ones_like(logit)
        # prob_select_one = torch.zeros_like(logit)

        probs = F.softmax(logit, dim=-1)
        prob_select_one = probs[:, 1].unsqueeze(-1)
        # 创建伯努利分布
        # 注意：Bernoulli(probs)期望probs中的每个值都在[0,1]
        m = Bernoulli(probs=prob_select_one)

        # 从分布中采样动作 (0 或 1)
        # sampled_action 的形状将是 (batch_size, 1)
        sampled_action = m.sample() # 这步是不可导的

        # 为了进行策略梯度更新 (如REINFORCE)，我们需要采样动作的对数概率
        # log_prob = m.log_prob(sampled_action) # (batch_size, 1)
        # 这个 log_prob 将在计算损失时与奖励相乘

        # 返回采样得到的动作、用于采样的概率以及原始logit
        # log_prob 通常在训练循环中根据 sampled_action 和 prob_select_one 计算

        sampled_action_reshaped = sampled_action[:, :, None, None]


        return sampled_action_reshaped


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        """
        前向传播：对输入进行四舍五入。
        ctx: 上下文对象，用于存储反向传播所需的信息。
        input_tensor: Sigmoid函数的输出，一个在(0,1)区间的张量。
        """
        # 直接进行四舍五入操作
        rounded_tensor = torch.round(input_tensor)
        # ctx.save_for_backward(input_tensor) # 对于round的STE，通常不需要保存输入
                                            # 因为梯度直接透传，不依赖于输入值本身。
                                            # 但如果STE的变种依赖于输入值（例如，梯度只在某个区间内透传），则需要保存。
        return rounded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：梯度直接透传。
        grad_output: 从上一层传来的梯度，即 d(loss)/d(rounded_tensor)。
        """
        # 直接返回输出的梯度，就好像round操作是恒等函数一样
        # grad_input = grad_output.clone() # 严格来说，应该clone，但通常直接返回也行
        return grad_output

# 创建一个可调用的STE round函数
round_ste = RoundSTE.apply


class GatedModule(nn.Module):
    def __init__(self, initial_alpha_value=0.0):
        super().__init__()
        # 可学习的参数 alpha，Sigmoid 将作用于它
        # 初始化为0.0，使得初始sigmoid(alpha) = 0.5，权重选择不偏向0或1
        self.alpha = nn.Parameter(torch.tensor(initial_alpha_value, dtype=torch.float32))
        # 如果你需要一个向量或矩阵的权重，可以这样初始化：
        # self.alpha = nn.Parameter(torch.randn(num_weights))

    def forward(self,):
        # 将 alpha 通过 sigmoid 转换为 (0,1) 之间的概率值
        p = torch.sigmoid(self.alpha)

        # 使用 STE round 函数得到二元权重 (0 或 1)
        x_weight_in1_0 = round_ste(p)

        # 应用公式
        # x_cam_1 = x_weight_in1_0 * ext + x_cam

        # 返回所有相关值，方便观察
        return x_weight_in1_0



class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class ScoreNetwork0(nn.Module):
    def __init__(self):
        super(ScoreNetwork0, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(128),
                                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(128, 512, kernel_size=1),
                                    nn.Conv2d(512, 128, kernel_size=1),
                                    nn.GELU())
        self.conv2 = nn.Conv2d(128, 2,  kernel_size=7, stride=2, padding=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        short_cut = x
        x = self.invert(x)
        x = short_cut + x
        x = self.conv2(x)
        x = x.float()
        x = self.softmax(x)
        return x


class ScoreNetwork1(nn.Module):
    def __init__(self):
        super(ScoreNetwork1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(256),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(256, 1024, kernel_size=1),
                                    nn.Conv2d(1024, 256, kernel_size=1),
                                    nn.GELU())
        self.conv2 = nn.Conv2d(256, 2,  kernel_size=7, stride=2, padding=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        short_cut = x
        x = self.invert(x)
        x = short_cut + x
        x = self.conv2(x)
        x = x.float()
        x = self.softmax(x)
        return x


class ModifiedScoreNetwork0(nn.Module):
    def __init__(self):
        super(ModifiedScoreNetwork0, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(128), # Verwendung der oben definierten LayerNorm2d
                                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(128, 512, kernel_size=1),
                                    nn.Conv2d(512, 128, kernel_size=1),
                                    nn.GELU())
        # Diese Schicht wird 2 Kanäle ausgeben, einen für x_weight0 und einen für x_weight1
        self.conv2 = nn.Conv2d(128, 2,  kernel_size=7, stride=2, padding=3)
        
        # Global Average Pooling, um räumliche Dimensionen zu reduzieren
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Softmax-Schicht, die über die Dimension der beiden Gewichte operiert (dim=1 für einen Tensor der Form [BatchSize, 2])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Eingangs-Tensor x hat z.B. die Form (BatchSize, 128, H, W)
        x = self.conv1(x) # Form z.B. (BatchSize, 128, H/2, W/2)
        short_cut = x
        x = self.invert(x) # Form bleibt (BatchSize, 128, H/2, W/2)
        x = short_cut + x
        x = self.conv2(x) # Form z.B. (BatchSize, 2, H/4, W/4)
        
        # Global Average Pooling anwenden
        x = self.global_avg_pool(x) # Form (BatchSize, 2, 1, 1)
        
        # Umformen zu (BatchSize, 2)
        x = x.view(x.size(0), -1) 
        
        # Sicherstellen, dass der Datentyp float ist (sollte bereits der Fall sein)
        x = x.float() 
        
        # Softmax auf die beiden Gewichte anwenden
        # x hat hier die Form (BatchSize, 2).
        # self.softmax(x) gibt einen Tensor der Form (BatchSize, 2) zurück,
        # wobei für jede Zeile (jedes Batch-Element) gilt: gewicht_0 + gewicht_1 = 1, und beide Gewichte >= 0.
        x_softmaxed = self.softmax(x)
        
        # Die beiden (softmaxed) Gewichte extrahieren
        # x_weight0 und x_weight1 sind Tensoren der Form (BatchSize,)
        x_weight0 = x_softmaxed[:, 0]
        x_weight1 = x_softmaxed[:, 1]
        
        # Umformen für Broadcasting, wie von Ihnen im vorherigen Code gewünscht
        x_weight0_reshaped = x_weight0[:, None, None, None] # Form (BatchSize, 1, 1, 1)
        x_weight1_reshaped = x_weight1[:, None, None, None] # Form (BatchSize, 1, 1, 1)
        
        return x_weight0_reshaped, x_weight1_reshaped


class ModifiedScoreNetwork1(nn.Module):
    def __init__(self):
        super(ModifiedScoreNetwork1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(256),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(256, 1024, kernel_size=1),
                                    nn.Conv2d(1024, 256, kernel_size=1),
                                    nn.GELU())
        # Diese Schicht wird 2 Kanäle ausgeben, einen für x_weight0 und einen für x_weight1
        self.conv2 = nn.Conv2d(256, 2,  kernel_size=7, stride=2, padding=3)
        
        # Global Average Pooling, um räumliche Dimensionen zu reduzieren
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Softmax-Schicht, die über die Dimension der beiden Gewichte operiert (dim=1 für einen Tensor der Form [BatchSize, 2])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Eingangs-Tensor x hat z.B. die Form (BatchSize, 128, H, W)
        x = self.conv1(x) # Form z.B. (BatchSize, 128, H/2, W/2)
        short_cut = x
        x = self.invert(x) # Form bleibt (BatchSize, 128, H/2, W/2)
        x = short_cut + x
        x = self.conv2(x) # Form z.B. (BatchSize, 2, H/4, W/4)
        
        # Global Average Pooling anwenden
        x = self.global_avg_pool(x) # Form (BatchSize, 2, 1, 1)
        
        # Umformen zu (BatchSize, 2)
        x = x.view(x.size(0), -1) 
        
        # Sicherstellen, dass der Datentyp float ist (sollte bereits der Fall sein)
        x = x.float() 
        
        # Softmax auf die beiden Gewichte anwenden
        # x hat hier die Form (BatchSize, 2).
        # self.softmax(x) gibt einen Tensor der Form (BatchSize, 2) zurück,
        # wobei für jede Zeile (jedes Batch-Element) gilt: gewicht_0 + gewicht_1 = 1, und beide Gewichte >= 0.
        x_softmaxed = self.softmax(x)
        
        # Die beiden (softmaxed) Gewichte extrahieren
        # x_weight0 und x_weight1 sind Tensoren der Form (BatchSize,)
        x_weight0 = x_softmaxed[:, 0]
        x_weight1 = x_softmaxed[:, 1]
        
        # Umformen für Broadcasting, wie von Ihnen im vorherigen Code gewünscht
        x_weight0_reshaped = x_weight0[:, None, None, None] # Form (BatchSize, 1, 1, 1)
        x_weight1_reshaped = x_weight1[:, None, None, None] # Form (BatchSize, 1, 1, 1)
        
        return x_weight0_reshaped, x_weight1_reshaped



class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

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


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, H, W):
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


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



class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


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


class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


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
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]    # 加权
        x_f = functools.reduce(torch.max, x_ext)
        return x_f

@MODELS.register_module()
class ConvNeXt_0524(BaseModule):
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
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
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
class HubVisionTransformer0524(BaseModule):
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
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = DetailedPatchEmbedParallel(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                # padding=patch_sizes[i] // 2,
                # num_modals=self.in_modals[i-1] if (i == self.skip_patch_embed_stage) else self.in_modals[i],
                num_modals=1
            )

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
        if len(x_ext) == 1:
            x_f = x_ext[0]

        else:
            x_scores = module(x_ext)    # [B, num_modals, H, W]
            x_scores = x_scores.unsqueeze(2)    # [B, num_modals, 1, H, W]
            x_scores = x_scores.transpose(0, 1)    # [num_modals, B, 1, H, W]

            # print(x_scores.shape, x_ext[0].shape)

            for i in range(len(x_ext)):
                x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]    # 加权
            # x_f = functools.reduce(torch.max, x_ext)
            x_f = torch.sum(torch.stack(x_ext), dim=0)    # [B, C, H, W]

        x_f = x_f.flatten(2).transpose(1, 2)

        return x_f


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


#这是改进后的spatial attention模块，没有那么简单了
# ===== TRAINING-FLOW / MFFE scSE fusion : ESA (spatial-attention half) =====
# Enhanced Spatial Attention (from RFDN's ESA, AIM-2020 "Residual Feature
# Distillation Network"). Used here as the spatial-attention (sSE) branch of the
# paper's scSE frequency-fusion step (SCSEModule_improve): it predicts a per-pixel
# gate m in (0,1) that re-weights the fused feature spatially.
# Pipeline: 1x1 squeeze to f=C//4 channels -> strided conv + 7x7 maxpool to shrink
# the map (cheaply grows the receptive field) -> a few 3x3 convs -> bilinear
# upsample back to the input size -> add the un-shrunk 1x1 branch (conv_f) ->
# 1x1 expand back to C -> sigmoid mask -> multiply the input. Output has the same
# NCHW shape as the input.
class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        # self.maxPooling = nn.MaxPool2d(kernel_size=5, stride=1)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))          # 1x1 squeeze: C -> f=C//4
        c1 = self.conv2(c1_)               # strided 3x3: downsample the map
        v_max = self.maxPooling(c1)        # 7x7/stride-3 maxpool: enlarge receptive field
        # print('v_max的大小',v_max.size())
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)  # upsample back to input HxW
        cf = self.conv_f(c1_)              # 1x1 skip on the un-downsampled features
        c4 = self.conv4((c3 + cf))         # fuse both scales, 1x1 expand f -> C
        m = self.sigmoid(c4)               # per-pixel spatial gate in (0,1)

        return input * m                   # spatially re-weighted feature


# stdv_channels / mean_channels: global spatial-pooling helpers for CCALayer's
# "contrast-aware" channel attention (from IMDN/RFDN). Given an NCHW tensor they
# collapse H,W to 1x1, returning per-channel statistics of shape (N,C,1,1).
# (Note: the parameter is named F and only shadows the functional import locally.)
def stdv_channels(F):
    # per-channel spatial standard deviation (the "contrast" signal): sqrt of the
    # mean squared deviation from the per-channel mean.
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    # per-channel spatial mean == global average pooling written out explicitly.
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


# ===== TRAINING-FLOW / MFFE scSE fusion : CCALayer (channel-attention half) =====
# Contrast-aware Channel Attention (CCA, from IMDN/RFDN). Used here as the
# channel-attention (cSE) branch of the paper's scSE frequency-fusion step
# (SCSEModule_improve). Unlike plain SE, the descriptor combines the per-channel
# std-dev ("contrast") with the mean, so texture/contrast — not just average
# energy — drives the channel gate; a 1x1 bottleneck (C -> C//reduction -> C) +
# sigmoid then produces the per-channel weights.
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels          # per-channel spatial std (contrast)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # per-channel spatial mean
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)  # (N,C,1,1) contrast + mean descriptor
        y = self.conv_du(y)                      # (N,C,1,1) per-channel gate in (0,1)
        return x * y                             # channel-reweighted feature



# ===== TRAINING-FLOW / MFFE frequency fusion : scSE (SCSEModule_improve) =====
# This IS the paper's "scSE" fusion step used by AsymCMNeXt_0524 (MFFE) inside
# FU / FU1 / FU2 to merge the concatenated multi-branch features before the 3x3
# refine conv. It is an UPGRADED spatial-and-channel Squeeze-&-Excitation:
#   * channel branch cSE = CCALayer (contrast-aware channel attention), and
#   * spatial branch sSE = ESA (enhanced spatial attention),
# instead of the textbook Roy et al. cSE/sSE (that plain SCSEModule also exists
# in this file but is unused by the RTM configs). Both branches keep the input
# channel count, so the sum has the same NCHW shape as the input.
class SCSEModule_improve(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = CCALayer(in_channels,reduction)  # channel attention (contrast-aware)
        self.sSE = ESA(in_channels)                 # spatial attention (enhanced)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)  # additive fusion of channel- and spatial-gated features




@MODELS.register_module()
class AsymCMNeXt_0524(BaseModule):
    """The backbone of CMNeXt but allow asymmetric input.

    This backbone is the Upgrade of `CMNeXt:'
    Delivering Arbitrary-Modal Semantic Segmentation
    modified from SegFormer

    PAPER ROLE: this class IS the paper's MFFE (Multi-Frequency Feature
    Extractor), the SegFormer variant = "TAFE-Net" (TAFE-Net, AAAI 2026). It
    runs three encoders over the RGB image I_v and its frequency views and fuses
    them into the multi-frequency pyramid F_1..F_4 (channels [64,128,320,512]):
      * main_branch   = SegFormer / MiT-b2 visual encoder of I_v (= x_cam).
      * extra_branch  = Transformer branch; encodes the VFIM low-freq integrated
                        view I_hat_l  (forward: x_extra_1 = prim1(Cat(I_lf, I_v))).
      * extra1_branch = CNN(ConvNeXt) branch; encodes the VFIM high-freq
                        integrated view I_hat_h (forward: x_extra_2 = prim2(...)).
    Frequency fusion uses SCSEModule_improve (= paper scSE) inside FU/FU1/FU2,
    and the DTD Frequency-Perception-Head feature F_d enters ONLY at stage 1
    (H/8). See __init__ (branch construction) and forward (fusion) below for the
    exact paper<->code mapping.
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

        # --- Branch 1: main visual encoder = paper SegFormer encoder of I_v ---
        # backbone_main = MixVisionTransformer / MiT-b2. Encodes x_cam = RGB = I_v.
        self.main_branch = MODELS.build(backbone_main)
        # self.main_branch.input1=DetailedPatchEmbedParallel()


        # --- Branch 2: Transformer branch (paper) = extra_branch ---
        # backbone_extra = HubVisionTransformer0521. In forward() it encodes the
        # VFIM low-freq integrated view I_hat_l = prim1(Cat(I_lf, I_v)).
        self.extra_branch = MODELS.build(backbone_extra)



        # --- Branch 3: CNN(ConvNeXt) branch (paper) = extra1_branch ---
        # PAPER DIFF: the configured backbone_extra1 (ConvNeXt_0521) build below is
        # commented out; instead a timm ConvNeXt-V2-tiny (in_chans=3) is used. In
        # forward() it encodes the VFIM high-freq view I_hat_h = prim2(Cat(I_hf,I_v)).
        # self.extra1_branch = MODELS.build(backbone_extra1)


        self.extra1_branch = timm.create_model(
            model_name='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
            features_only=False,
            pretrained=True,
            in_chans=3,
            cache_dir='./ss1',
            checkpoint_path='',
            # pretrained_cfg=dict(file='/media/data2/yzq/code/RTM-shuang-2-doctamper/convnextv2_tiny_22k_384_ema.pt'),
            # pretrained_strict=False
        )
        self.extra1_branch.head = None  # drop the ImageNet classifier; keep stage features






        # A CE loss is built here but the paper's segmentation loss
        # L = L_ce(M_p,M_g) + L_lov(M_p,M_g) is actually applied on the final
        # prediction M_p by the SegformerHead / MyModelFull, not by this backbone.
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        self.loss_decode = build_loss(loss_decode)


        aux_end=dict(
        type='FCNHead',
        in_channels=128,
        # in_channels=320,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ],)

        # self.aux_end=MODELS.build(aux_end)

        # aux_end=dict(
        #         type='SegformerHead',
        #         # in_channels=[256, 256, 256, 256],
        #         in_channels=[64, 128, 320, 512],
        #         in_index=[0, 1, 2, 3],
        #         channels=256,
        #         dropout_ratio=0.1,
        #         num_classes=2,
        #         norm_cfg=dict(type='SyncBN', requires_grad=True),
        #         align_corners=False,
        #         # sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=100000),
        #         loss_decode=[
        #             dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        #             dict(type='LovaszLoss', loss_weight=1.0, per_image=False, reduction='none'),
        #         ],)


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

        # conv11[i]: 1x1 conv aligning the ConvNeXt (high-freq branch F_h) stage
        # widths 96/192/384/768 -> the SegFormer/MiT-b2 widths 64/128/320/512, so
        # F_h can be concatenated/added with the visual and Transformer features.
        self.conv11=[]
        self.conv11.append(nn.Conv2d(96, 64, 1, 1, 0))
        self.conv11.append(nn.Conv2d(192, 128, 1, 1, 0))
        self.conv11.append(nn.Conv2d(384, 320, 1, 1, 0))
        self.conv11.append(nn.Conv2d(768, 512, 1, 1, 0))


        # Frequency-fusion modules. Each is [ scSE-based fuse conv , 1x1 refine ].
        # PAPER DIFF: the paper's "scSE" here is SCSEModule_improve (a custom
        # CCALayer + ESA attention), NOT the textbook Roy et al. scSE. The plain
        # SCSEModule class also exists in this file but is unused by the RTM configs.
        #
        # FU  = stage-1 cross-modal fuse: scSE(Cat(F_hl2, F_d)) at 256ch -> 128ch,
        #       producing F_cf2 (forward i==1). This is the ONLY place F_d enters.
        self.FU = nn.ModuleList([nn.Sequential(SCSEModule_improve(256), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))])
        self.FU.append(nn.Conv2d(128, 128, 1, 1, 0))
        # self.FU[1].weight.data.zero_()
        # self.FU1 = nn.ModuleList([nn.Sequential(SCSEModule(256), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))])  # (inactive variant, not used by the RTM configs)
        # self.FU1.append(nn.Conv2d(128, 128, 1, 1, 0))
        # FU1 = stage-0 frequency fuse: scSE(Cat(F_l, F_h)) at 128ch -> 64ch,
        #       producing F_hl1 (forward i==0).
        self.FU1 = nn.ModuleList([nn.Sequential(SCSEModule_improve(128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))])
        self.FU1.append(nn.Conv2d(64, 64, 1, 1, 0))
        # self.FU1[1].weight.data.zero_()
        # self.FU1 = nn.ModuleList([nn.Sequential(SCSEModule(160), nn.Conv2d(160, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))])  # (inactive variant, not used by the RTM configs)
        # self.FU1.append(nn.Conv2d(64, 64, 1, 1, 0))


        # FU2 = stage-1 frequency fuse: scSE(Cat(F_l, F_h)) at 256ch -> 128ch,
        #       producing F_hl2 (forward i==1).
        self.FU2 = nn.ModuleList([nn.Sequential(SCSEModule_improve(256), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))])
        self.FU2.append(nn.Conv2d(128, 128, 1, 1, 0))
        # feature rectification module

        # self.FU3 = nn.ModuleList([nn.Sequential(SCSEModule_improve(128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))])
        # self.FU3.append(nn.Conv2d(64, 64, 1, 1, 0))
        # self.FU3=nn.Conv2d(64, 64, 1, 1, 0)
        # self.FU3.weight.data.zero_()


        # self.FU4 = nn.ModuleList([nn.Sequential(SCSEModule_improve(256), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))])
        # self.FU4.append(nn.Conv2d(128, 128, 1, 1, 0))
        # self.FU4=nn.Conv2d(128, 128, 1, 1, 0)
        # self.FU4.weight.data.zero_()



        # self.update_gate_0 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.Sigmoid())

        # self.update_gate_1 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())


        # self.update_gate_0=ModifiedScoreNetwork0()
        # self.update_gate_1=ModifiedScoreNetwork1()

        # self.x_weight_in0 = nn.Parameter(torch.tensor([0.0]))
        # self.x_weight_in1 = nn.Parameter(torch.tensor([0.0]))

        # self.x_weight_in_all=GatedModule()

        # self.x_weight_in0 = SamplingGatedModule(128)
        # self.x_weight_in1 = SamplingGatedModule(256)



        self.FFMs = []

        
        # prim1 / prim2 = the paper's VFIM (Visual-Frequency Integration Module):
        # each takes a 6-channel concat Cat(view, I_v) and returns a 3-channel
        # integrated view (DSB direction-sensitive + DAB direction-agnostic ->
        # ConvBlock). prim1: I_l=Cat(I_lf,I_v) -> I_hat_l (low-freq, to Transformer
        # branch); prim2: I_h=Cat(I_hf,I_v) -> I_hat_h (high-freq, to ConvNeXt).
        self.prim1=PRIM1(6, 3, nn.BatchNorm2d)
        self.prim2=PRIM1(6, 3, nn.BatchNorm2d)

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
        cosine_similarity = F.cosine_similarity(x_f, x_f_1, dim=1, eps=1e-6)  # 输出形状为 (N, H, W)

            # 转换为余弦距离
        cosine_distance = 1 - cosine_similarity

            # 恢复成形状 (N, 1, H, W)，以便后续拼接
        channel_anomaly_map = cosine_distance.unsqueeze(1)  # 输出形状为 (N, 1, H, W)

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
        """MFFE forward: fuse the visual stream with the frequency branches.

        Args:
            x (list): packed by MyModelFull as [I_v, [I_lf, I_hf, F_d]], i.e.
                x[0] = x_cam   = RGB image I_v (visual encoder input),
                x[1] = x_extra = [I_lf, I_hf, F_d] = low-freq view, high-freq
                view, and the DTD-FPH DCT tamper feature F_d (already at H/8).

        Returns:
            list[Tensor]: the multi-frequency pyramid [F_1, F_2, F_3, F_4] with
                channels [64,128,320,512] (strides 4/8/16/32). F_1, F_2 are the
                frequency-fused features (F_1=F_v+F_hl1, F_2=F_m+F_cf2); F_3, F_4
                are the pure visual SegFormer features.
        """

        outs = []

        x_cam = x[0]      # x_cam = RGB image I_v -> SegFormer/MiT-b2 visual encoder
        x_extra = x[1]    # x_extra = [I_lf, I_hf, F_d] (frequency streams)

        # align channel: any single-channel frequency map that spatially matches
        # I_v is tiled up to the extra_branch input channel count.
        for i in range(len(x_extra)):
            if x_extra[i].size()[1] != self.extra_branch.in_channels:
                if (x_extra[i].size()[1] == 1) & (x_cam.size()[2:] == x_extra[i].size()[2:]):
                    x_extra[i] = x_extra[i].repeat(1, self.extra_branch.in_channels, 1, 1)

        if self.in_stages is not None:
            if max(self.in_stages) > 0:
                # (inactive branch for the RTM configs, which use in_stages=(0,0)):
                # would inject some extra modals only from a later stage in_stage.
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
                # ACTIVE path (in_stages=(0,0)): build the two VFIM integrated
                # views up front, and keep F_d aside for the stage-1 fusion.
                # x_extra_1 = I_hat_l = prim1(Cat(I_lf, I_v)) [6ch->3ch] -> Transformer branch.
                x_extra_1=x_extra[0]                          # I_lf (low-freq view)
                x_extra_1=torch.cat([x_extra_1,x[0]],dim=1)   # I_l = Cat(I_lf, I_v), 6ch
                x_extra_1=self.prim1(x_extra_1)               # VFIM -> I_hat_l, 3ch

                # x_extra_2 = I_hat_h = prim2(Cat(I_hf, I_v)) [6ch->3ch] -> ConvNeXt branch.
                x_extra_2=x_extra[1]                          # I_hf (high-freq view)
                x_extra_2=torch.cat([x_extra_2,x[0]],dim=1)   # I_h = Cat(I_hf, I_v), 6ch

                x_extra_2=self.prim2(x_extra_2)               # VFIM -> I_hat_h, 3ch
                in_stage = -1
                x_extra_3=x_extra[2]                          # F_d = DTD-FPH DCT feature (H/8), used only at stage 1
        else:
            in_stage = -1

        B = x_cam.shape[0]

        # shift_stage loop: for MiT-b2 shift_stage == 0 (main and extra branches
        # have the same number of stages), so this loop runs 0 times ("not executed").
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
        # cross recalibration: the main 4-stage MFFE loop. Per stage i it advances
        # the visual SegFormer stream (x_cam), the Transformer branch (x_f, from
        # I_hat_l) and the ConvNeXt branch (x_f_1, from I_hat_h), then fuses them.
        for i in range(self.num_stage_extra):
            # import pdb;pdb.set_trace()
            layer = self.main_branch.layers[(i+self.shift_stage)]
            layer_extra = self.extra_branch.layers[i]
            
            # layer_extra_1=self.extra_branch.layers_1[i]



            # Visual SegFormer stage i advances the visual stream x_cam (= I_v at
            # i=0, else the previously fused F_i carried over) -> stage output F_v
            # (i=0), the multimodal F_m (i==1), or a later visual feature (i>=2).
            x_cam, hw_shape = layer[0](x_cam)
            H, W = hw_shape
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)

            if self.in_stages is not None:
                if i == in_stage:
                    # (inactive: in_stage == -1, so i == in_stage is never true here)
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
                    # ACTIVE: Transformer-branch patch-embed / downsample at stage i
                    # of the low-freq integrated view I_hat_l (carried in x_extra_1).
                    # import pdb;pdb.set_trace()
                    
                    # x_extra_2=torch.cat([x_extra_2,x[0]]

                    #20250620进行更改
                    # x_extra_1, _ = layer_extra[0]([x_extra_1])
                    x_extra_1,_= layer_extra[0](x_extra_1)


                    # x_extra_2,_ = layer_extra_1[0]([x_extra_2])

            else:
                x_extra, _ = layer_extra[0](x_extra)

            # if self.no_select:
            #     x_f = torch.stack(x_extra,dim=0).mean(dim=0)
            #     x_f = x_f.flatten(2).transpose(1, 2)
            # else:
            #     x_f = self.extra_branch.tokenselect(x_extra, self.extra_branch.extra_score_predictor[i])
            
            # Transformer branch stage i: run the encoder blocks + norm on the
            # downsampled I_hat_l tokens -> x_f = low-freq feature F_l (NCHW).
            x_f=x_extra_1
            # x_f = x_f.flatten(2).transpose(1, 2)
            for block in layer_extra[1]:
                x_f = block(x_f, hw_shape)
            x_f = layer_extra[2](x_f)
            x_f = nlc_to_nchw(x_f, hw_shape)



            # ConvNeXt branch stage i: stem (once, i==0) + stage on I_hat_h, then
            # conv11[i] aligns channels -> x_f_1 = high-freq feature F_h (NCHW).
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



            # Bundle the two frequency-branch features for the NEXT stage:
            # x_extra_all = [F_l, F_h] (Transformer + ConvNeXt stage outputs).
            x_extra_all=[x_f,x_f_1]
            # (inactive fusion variant below, not used by the RTM configs)
            # if i==0:
                




            #     ext_1 = self.FU1[0](torch.cat((x_f, x_f_1), dim=1))
            #     x_fused = self.FU1[1](ext_1) + x_cam
            #     # x_cam=x_fused

            # elif i==1:

            #     ext1 = self.FU2[0](torch.cat((x_f, x_f_1), dim=1))
            #     ext1_1=self.FU2[1](ext1)


            #     ext = self.FU[0](torch.cat((x_cam, x_extra_3), dim=1))
            #     x_fused = self.FU[1](ext)+ext1_1
            #     # x_cam=x_fused

            #从6月2号开始的
            if i==0:
                '''
                ext_1 = self.FU1[0](torch.cat((x_f, x_f_1), dim=1))
                ext_1=self.FU1[1](ext_1)
                # x_fused = ext_1 + x_cam

                

                # x_weight_in0_0=self.x_weight_in0(torch.cat((ext_1, x_cam), dim=1))

                # print(x_weight_in0_0,x_weight_in0_0.size())
                # x_cam=ext_1+x_cam

                # x_weight_in_all=self.x_weight_in_all()
                # x_cam=x_weight_in_all*ext_1+x_cam


                # ext_2 = self.FU3[0](torch.cat((ext_1, x_cam), dim=1))
                x_fused1 = self.FU3(ext_1)
                x_cam=x_fused1+x_cam
                x_fused=x_cam
                '''
                # STAGE-0 frequency fusion (H/4): F_hl1 = scSE(Cat(F_l, F_h))
                # via FU1, then F_1 = F_hl1 + F_v (=x_cam). F_1 is carried on
                # as the visual feature entering stage 1.
                ext_1 = self.FU1[0](torch.cat((x_f, x_f_1), dim=1))   # scSE fuse -> F_hl1 (pre-refine)
                x_fused = self.FU1[1](ext_1) + x_cam                  # F_1 = F_hl1 + F_v
                x_cam=x_fused

            elif i==1:
                '''
                ext1 = self.FU2[0](torch.cat((x_f, x_f_1), dim=1))
                ext1_1=self.FU2[1](ext1)

                ext = self.FU[0](torch.cat((ext1_1, x_extra_3), dim=1))
                ext=self.FU[1](ext)
                # x_fused = ext+x_cam

                # x_weight_in1_0=self.x_weight_in1(torch.cat((ext, x_cam), dim=1))
                # x_cam=ext+x_cam
                # x_weight_in_all=self.x_weight_in_all()

                # x_cam=x_weight_in_all*ext+x_cam

                # print(x_weight_in_all)

                # ext_2 = self.FU4[0](torch.cat((ext, x_cam), dim=1))
                x_fused1 = self.FU4(ext)

                x_cam=x_fused1+x_cam
                x_fused=x_cam
                '''

                ext1 = self.FU2[0](torch.cat((x_f, x_f_1), dim=1))
                ext1_1=self.FU2[1](ext1)
                

                # STAGE-1 cross-modal fusion (H/8). Above (ext1 -> ext1_1) is
                # F_hl2 = scSE(Cat(F_l, F_h)) via FU2. Here F_d (x_extra_3, the
                # DTD-FPH DCT feature) enters for the ONLY time: F_cf2 =
                # scSE(Cat(F_hl2, F_d)) via FU, then F_2 = F_cf2 + F_m (=x_cam).
                ext = self.FU[0](torch.cat((ext1_1, x_extra_3), dim=1))   # scSE fuse -> F_cf2 (pre-refine)
                x_fused = self.FU[1](ext)+x_cam                           # F_2 = F_cf2 + F_m
                x_cam=x_fused

                #----
                # ext1 = self.FU2[0](torch.cat((x_f, x_f_1), dim=1))
                # ext1_1=self.FU2[1](ext1)
                # ext_2 = self.FU[0](torch.cat((x_extra_3, x_cam), dim=1))
                # x_fused = self.FU[1](ext_2)+x_cam+ext1_1
                # x_cam=x_fused



            else:
                # STAGES 2 & 3 (H/16, H/32): no frequency fusion here, so
                # x_fused = x_cam -> F_3 and F_4 are pure visual SegFormer features.
                x_fused=x_cam

            



            # Collect this stage's output: F_1 (i=0), F_2 (i=1), F_3 (i=2), F_4 (i=3).
            if (i+self.shift_stage) in self.out_indices:
                outs.append(x_fused)


            #在4月4日改变
            # Hand the two frequency-branch features to the next stage:
            # x_extra_1 = F_l (Transformer branch), x_extra_2 = F_h (ConvNeXt branch).
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
                




                ext_1 = self.FU1[0](torch.cat((x_f, x_f_1), dim=1))
                x_fused = self.FU1[1](ext_1) + x_cam
                # x_cam=x_fused

            elif i==1:

                ext1 = self.FU2[0](torch.cat((x_f, x_f_1), dim=1))
                ext1_1=self.FU2[1](ext1)
                aux_input.append(ext1_1)

                ext = self.FU[0](torch.cat((x_cam, x_extra_3), dim=1))
                x_fused = self.FU[1](ext)+ext1_1
                # x_cam=x_fused

            else:
                x_fused=x_cam
            



            if (i+self.shift_stage) in self.out_indices:
                outs.append(x_fused)


            #在4月4日改变
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