import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN

from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.backbones.mit import MixFFN, EfficientMultiheadAttention, TransformerEncoderLayer
from mmseg.models.backbones.nat import NATLayer, NATBlock

from ..segmentors.base import BaseSegmentor
from ..backbones.resnet import BasicBlock

from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from ..toys.fapa import FAPA_v2

from efficientnet_pytorch.utils import *
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import collections

BlockArgs = collections.namedtuple('BlockArgs', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio', 'input_filters',
                                                 'output_filters', 'se_ratio', 'id_skip'])
GlobalParams = collections.namedtuple('GlobalParams',
                                      ['width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
                                       'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon', 'drop_connect_rate',
                                       'depth_divisor', 'min_depth', 'include_top'])
global_params = GlobalParams(width_coefficient=1.8, depth_coefficient=2.6, image_size=528, dropout_rate=0.0,
                             num_classes=1000, batch_norm_momentum=0.99, batch_norm_epsilon=0.001,
                             drop_connect_rate=0.0, depth_divisor=8, min_depth=None, include_top=True)


# =============================================================================
# TRAINING-FLOW / FPH — Frequency Perception Head  (file: frequencers.py)
# -----------------------------------------------------------------------------
# This file collects the frequency "preprocessor_sec" modules used by the DocTamper
# training flow. The ONE module here that is on the active DocTamper training path is
# `Doctamperdct` (below) = the paper TAFE-Net's FPH (Frequency Perception Head,
# borrowed from DTD / DocTamper, Qu et al. 2023). It is the "DTD Frequency
# Perception Head" referenced in my_model_full.forward_encoder: it takes the JPEG
# DCT stream (D, T) and produces the DCT tamper feature F_d for the MFFE
# (Multi-Frequency Feature Extractor) frequency-fusion path.
#
# Paper<->code, FPH only:
#   inputs  : x = D  (clipped JPEG luminance DCT coefficient map, integer levels
#                     0..20 after clipping), qtable = T (8x8 quantization table,
#                     entries indexed 0..63).
#   output  : F_d in H/8 x W/8 x C_d  (C_d = 128 here) — one feature vector per
#             8x8 JPEG block, consumed downstream by the MFFE frequency fusion.
#
# The EfficientNet helpers above (MBConvBlock / global_params) are building
# blocks used inside Doctamperdct.conv0. CATNetDCT / DCTProcessor / PEG (below)
# are separate DCT encoders / helpers and are NOT the FPH the DocTamper configs wire
# in — only their use inside Doctamperdct is on the paper's FPH path.
# =============================================================================


class MBConvBlock(nn.Module):
    """EfficientNet Mobile Inverted Bottleneck (MBConv) block.

    (from EfficientNet, Tan & Le 2019; used here as the FPH `conv0` refinement
    stage — 3 stacked MBConvBlocks that process the per-JPEG-block tokens on the
    way to F_d. Not the paper's own module, just the FPH downsampler backbone.)

    An inverted-residual + squeeze-excite unit:
      * expand   : 1x1 conv widens channels by expand_ratio (6) -> BN -> Swish,
      * depthwise: KxK depthwise conv (groups == channels), stride s -> BN -> Swish,
      * SE       : squeeze-excite channel attention (global-avg-pool -> 1x1
                   reduce -> Swish -> 1x1 expand -> sigmoid gate, se_ratio=0.25),
      * project  : 1x1 conv down to output_filters -> BN (no activation),
      * id_skip  : residual add when id_skip and stride==1 and in==out channels.
    In the FPH every stage keeps 128 channels with id_skip=True, BUT `stride` is
    passed as the list `[1]` (not int 1), so the `self._block_args.stride == 1`
    guard in forward is False (`[1] == 1` is False) and the residual skip is in
    fact NOT taken — each block runs as a plain feed-forward inverted-residual.
    `MemoryEfficientSwish` is the SiLU activation; the
    `get_same_padding_conv2d` / `calculate_output_image_size` / `drop_connect`
    helpers come from `efficientnet_pytorch.utils`.
    """
    def __init__(self, block_args, global_params, image_size=25):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        # expand phase: 1x1 conv widens channels by expand_ratio (skipped when ratio==1)
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        # depthwise phase: per-channel spatial conv over the widened features
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        # squeeze-and-excite: pool to a per-channel descriptor, then use the
        # sigmoid-gated bottleneck to re-weight (attend over) each channel
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        # project phase: 1x1 conv back down to output_filters (linear, no Swish)
        x = self._project_conv(x)
        x = self._bn2(x)
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()



class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        # Only for demo use, more complicated functions are effective too.
    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:] # cls token不参与PEG
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat # 产生PE加上自身
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class AddCoords(nn.Module):
    """CoordConv-style coordinate augmentation, helper for the FPH `conv0` stem.

    Appends normalized x/y coordinate channels (and, with_r=True, a radius
    channel) to the input feature map, so the following stride-8 conv sees the
    absolute position of each token inside the block grid.

    PAPER DIFF: these coordinate channels are an EXTRA in this implementation and
    are NOT part of the paper's FPH description. They widen the FPH input by 3
    channels (32 -> 35: 16 dequantized DCT + 16 conv2 feature + 3 coord).
    """
    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r  # with_r=True -> also append a radial-distance channel

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        # Build a (x_dim x y_dim) coordinate grid matching the feature-map spatial size.
        xx_c, yy_c = torch.meshgrid(torch.arange(x_dim, dtype=input_tensor.dtype, device=input_tensor.device),
                                    torch.arange(y_dim, dtype=input_tensor.dtype, device=input_tensor.device))
        # Normalize each axis to [-1, 1] so position is scale-invariant.
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size, 1, x_dim, y_dim)  # -> (B, 1, x_dim, y_dim)
        yy_c = yy_c.expand(batch_size, 1, x_dim, y_dim)  # -> (B, 1, x_dim, y_dim)
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)  # append the 2 coord channels
        if self.with_r:
            # radius channel (distance from grid centre) as an extra 3rd coord channel
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


@MODELS.register_module()
class Doctamperdct(nn.Module):
    """FPH — Frequency Perception Head of the paper (TAFE-Net / MFFE stage 1).

    Encodes the JPEG DCT stream (D, T) within 8x8 blocks into the DCT tamper
    feature F_d (H/8 x W/8 x 128). Adapted from DTD / DocTamper (Qu et al. 2023).

    forward(x=D, qtable=T) pipeline:
      * obembed : one-hot embed the 21 clipped DCT coefficient levels of D,
      * conv1   : 3x3 dilation-8 conv whose receptive field spans a full 8x8
                  JPEG block  -> 64ch,  conv2 : 1x1 -> 16ch,
      * reshape to 8x8 blocks and multiply by qtembed(T) = per-frequency,
        block-wise DEQUANTIZATION of the DCT coefficients,
      * AddCoords appends x/y/radius coord channels (32 -> 35ch),
      * conv0   : stride-8 8x8 conv (one token per JPEG block) + 3x EfficientNet
                  MBConvBlock  -> F_d (H/8 x W/8 x 128).

    PAPER DIFF: the AddCoords coordinate channels are an extra not present in the
    paper FPH; `out_channles=(256,256,512)` is dead (the conv0 stem is hardcoded
    to 128 channels); and a FAPA post-processing branch (self.fapa_module /
    FAPA_v2) is commented out (inactive variant, not used by the DocTamper configs).
    """

    def __init__(self):
        super(Doctamperdct, self).__init__()
        # obembed: one-hot embed of D's 21 clipped DCT levels. Embedding(21,21)
        # frozen to the identity eye(21) => index k -> one-hot(k). (B,H,W) -> (B,H,W,21)
        self.obembed = nn.Embedding(21, 21).from_pretrained(torch.eye(21))
        # qtembed: per-frequency dequantization lookup. Maps each of the 64 (8x8)
        # quant-table entries of T to a learned 16-d vector used to rescale coeffs.
        self.qtembed = nn.Embedding(64, 16)
        # conv1: 3x3 with dilation=8 (pad 8) -> the 3 taps land on the SAME position
        # in 3 neighbouring 8x8 JPEG blocks, i.e. the receptive field spans a block. 21->64ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=64, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(64, momentum=0.01), nn.ReLU(inplace=True))
        # conv2: 1x1 projection 64->16ch, matching the 16-d qtembed dequantization vectors.
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=0.01), nn.ReLU(inplace=True))
        self.addcoords = AddCoords()  # appends x/y/radius coord channels (32 -> 35ch)
        repeats = (1, 1, 1)  # one MBConvBlock per stage (3 stages)
        in_channles = (128, 128, 128)  # feature width kept at 128 through the MBConv stages
        # in_channles = (256, 256, 256)
        out_channles = (256, 256, 512)  # PAPER DIFF: dead — conv0 stem is hardcoded to 128ch, this tuple is unused
        # conv0: stride-8 8x8 conv collapses each 8x8 JPEG block (35ch) into ONE token
        # at H/8 x W/8 (128ch), then 3 EfficientNet MBConvBlocks refine it -> F_d.
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=35, out_channels=128, kernel_size=8, stride=8, padding=0, bias=False),
            # nn.Conv2d(in_channels=35, out_channels=256, kernel_size=8, stride=8, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.01),
            # nn.BatchNorm2d(256, momentum=0.01),
            
            nn.ReLU(inplace=True),
            MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                                  input_filters=in_channles[0], output_filters=in_channles[1], se_ratio=0.25,
                                  id_skip=True), global_params),
            MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                                  input_filters=in_channles[1], output_filters=in_channles[1], se_ratio=0.25,
                                  id_skip=True), global_params),
            MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                                  input_filters=in_channles[1], output_filters=in_channles[1], se_ratio=0.25,
                                  id_skip=True), global_params), )
        
        
        # PAPER DIFF: optional FAPA post-processing on F_d (inactive variant, not used by the DocTamper configs).
        # self.fapa_module = FAPA_v2(
        #     in_channels=128,
        #     ksize=48, # Choose ksize based on expected spatial dim of doctamper_base output
        #     num_heads=2,
        #     dropout_rate=0.0,
        # )

    def forward(self, x, qtable):
        # x = self.conv2(self.conv1(self.obembed(x).permute(0, 3, 1, 2).contiguous()))
        # import pdb;pdb.set_trace()
        x1=self.obembed(x.long()).permute(0, 3, 1, 2).contiguous()  # D -> one-hot (B,H,W,21) -> (B,21,H,W)
        x2=self.conv1(x1)  # block-spanning dilated conv -> (B,64,H,W)
        x=self.conv2(x2)  # 1x1 -> (B,16,H,W)
        # x = self.conv2(self.conv1(self.obembed(x.long()).permute(0, 3, 1, 2).contiguous()))
        B, C, H, W = x.shape  # C = 16 (one channel per qtembed dequant dim)
        # ss1=x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2,4) 
        # ss2=self.qtembed(qtable.unsqueeze(-1).unsqueeze(-1).long()).transpose(1, 6).squeeze(6).contiguous().permute(0, 1, 4, 2, 5,3)

        # ss_all=ss1*ss2
        # Fused FPH head, read inside-out:
        #   (a) x.reshape(...).permute(...)         : fold (B,16,H,W) into per-block 8x8 grids,
        #   (b) * qtembed(T)                        : block-wise DEQUANTIZATION (multiply each
        #                                             8x8 frequency position by its T embedding),
        #   (c) permute(...).reshape(B,C,H,W)       : unfold back to (B,16,H,W),
        #   (d) cat([dequantized, x], dim=1)        : 16 + 16 = 32ch,
        #   (e) addcoords                           : append x/y/radius -> 35ch,
        #   (f) conv0 (stride-8 conv + 3 MBConv)    : -> F_d in H/8 x W/8 x 128.
        return self.conv0(self.addcoords(torch.cat(((x.reshape(B,C,H//8,8,W//8,8).permute(0,1,3,5,2,4)*self.qtembed(qtable.unsqueeze(-1).unsqueeze(-1).long()).transpose(1,6).squeeze(6).contiguous()).permute(0,1,4,2,5,3).reshape(B,C,H,W),x), dim=1)))

        # x_output=self.conv0(self.addcoords(torch.cat(((x.reshape(B,C,H//8,8,W//8,8).permute(0,1,3,5,2,4)*self.qtembed(qtable.unsqueeze(-1).unsqueeze(-1).long()).transpose(1,6).squeeze(6).contiguous()).permute(0,1,4,2,5,3).reshape(B,C,H,W),x), dim=1)))
        # return self.fapa_module(x_output)

@MODELS.register_module()
class CATNetDCT(nn.Module):
    # Alternative DCT encoder (CAT-Net style), not the FPH the DocTamper configs wire in.
    def __init__(self, in_channels, out_channels=4, channels=64, embed_dim=4, norm_cfg=dict(type='BN'), upsample=False):
        super(CATNetDCT, self).__init__()
        self.upsample = upsample
        self.norm_cfg = norm_cfg
        self.dct_layer0_dil = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            padding=8,
            dilation=8,
            norm_cfg=norm_cfg,
        )
        self.dct_layer1_tail = ConvModule(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=norm_cfg,
        )

        self.dct_layer2 = self._make_layer(BasicBlock, inplanes=embed_dim * 64 * 2, planes=out_channels, blocks=4, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                in_channels=inplanes,
                out_channels=planes * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg = None
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_cfg=self.norm_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_cfg=self.norm_cfg))

        return nn.Sequential(*layers)

    def forward(self, extras):
        x = extras['dct']
        qtable = extras['qtable']
        x = self.dct_layer0_dil(x)
        x = self.dct_layer1_tail(x)
        B, C, H, W = x.shape
        x0 = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8,
                                                                                     W // 8)  # [B, 256, 32, 32]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([x0, x1], dim=1)
        x = self.dct_layer2(x)  # x.shape = torch.Size([1, 96, 64, 64]) [2,96,32,32]

        if self.upsample:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x


@MODELS.register_module()
class DCTProcessor(nn.Module):
    # Alternative DCT encoder (transformer-based), not the FPH the DocTamper configs wire in.
    def __init__(self,
                 in_channels,
                 embed_dims=64,
                 out_channels=128,
                 num_heads=1,
                 patch_size=3,
                 stride=1,
                 mlp_ratio=4,
                 relation=False,
                 quantization=False,
                 reshape=True,
                 band_range=None,
                 sr_ratio=1,
                 reduce_neg=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),):
        super(DCTProcessor, self).__init__()

        self.relation = relation
        self.quantization = quantization
        self.reshape = reshape
        self.band_range = band_range
        self.reduce_neg = reduce_neg

        if band_range is not None:
            if isinstance(band_range, tuple):
                assert band_range[0] < band_range[1]
            elif isinstance(band_range, int):
                band_range = (band_range, band_range + 1)
            else:
                raise ValueError('band should be tuple or int')
            self.band = band_range


        if self.relation:
            in_channels_model = 1


        else:
            if self.band_range is not None:
                in_channels_model = self.band_range[1] - self.band_range[0]
            else:
                in_channels_model = in_channels * 64


        self.local_perception = ConvModule(
            in_channels=in_channels_model,
            out_channels=embed_dims,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.dw_conv = ConvModule(
            in_channels=embed_dims,
            out_channels=embed_dims * 2,
            kernel_size=3,
            padding=1,
            groups=embed_dims,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.band_perception = ConvModule(
            in_channels=embed_dims * 2,
            out_channels=embed_dims,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.patch_embed = PatchEmbed(
            in_channels=embed_dims,
            embed_dims=embed_dims * num_heads,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            norm_cfg=dict(type='LN', eps=1e-6))


        # encoder
        self.spatial_encoder = TransformerEncoderLayer(
            embed_dims=embed_dims * num_heads,
            num_heads=num_heads,
            feedforward_channels = mlp_ratio * embed_dims * num_heads,
            sr_ratio=sr_ratio,
        )

        self.norm = build_norm_layer(dict(type='LN', eps=1e-6), embed_dims * num_heads)[1]


        self.projection = ConvModule(
            in_channels=embed_dims * num_heads,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )



    def _to_relation(self, dct):
        # dct.shape = (B, C, H, W)
        dct = torch.abs(dct)
        dct = torch.mean(dct, dim=1, keepdim=True)
        dct = torch.div(dct, torch.norm(dct, p=2, keepdim=True))

        return dct

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        if self.reduce_neg:
            x = torch.abs(x)

        B, C, H, W = x.shape
        if self.reshape:
            x = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8, W // 8)

        if self.band_range is not None:
            x = x[:, self.band[0]:self.band[1], :, :]

        if self.relation:
            x = self._to_relation(x)

        x = self.local_perception(x)
        x = self.dw_conv(x)
        x = self.band_perception(x)

        x, hw_shape = self.patch_embed(x)

        x = self.spatial_encoder(x, hw_shape)
        x = self.norm(x)

        x = nlc_to_nchw(x, hw_shape)


        x = self.projection(x)

        return x
