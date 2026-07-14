# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import Upsample, resize
from .decode_head import BaseDecodeHead


# =============================================================================
# TRAINING-FLOW / DECODE HEAD  (file: fpn_head.py)
# -----------------------------------------------------------------------------
# `FPNHead` is the decode head used by the ConvNeXt training variant of the paper
# TAFE-Net ("Frequency Mining Empowered by Text Aggregation", AAAI 2026), i.e. the
# variant reported as "TAFE-Net*" (ConvNeXt V2 baseline instead of SegFormer). It
# is wired as model.decode_head in the DocTamper config tafenet_convnext_doctamper.py.
#
# PAPER DIFF: the paper's final prediction is M_p = SegFormerHead(F'_1,F_2,F_3,F'_4)
# (a SegFormer MLP-fusion head). This ConvNeXt variant SWAPS IN FPNHead (Semantic
# FPN, https://arxiv.org/abs/1901.02446) as the decode head in its place. Both
# heads play the same role in the training flow: they consume the four refined
# pyramid features coming out of the DFDE / DWT neck and emit the tamper mask M_p.
#
# WHERE IT SITS IN THE FLOW:
#   MFFE backbone -> F_1,F_2,F_3,F_4  ->  DFDE / DWT neck -> F'_1,F_2,F_3,F'_4
#                                     ->  FPNHead.forward(...)  -> M_p
# The neck features handed to `forward` are the paper's (F'_1, F_2, F_3, F'_4):
# F'_1 and F'_4 are the direction-aware DFDE-refined levels, F_2/F_3 pass through.
#
# WHAT THE HEAD DOES (Semantic FPN):
#   For each input level it runs a small conv "scale head" and, for coarser levels,
#   repeatedly x2-upsamples until every level is at the FINEST feature stride
#   (feature_strides[0]); it SUMS the levels, then `cls_seg` maps the fused feature
#   to per-pixel class logits = the tamper mask M_p (later up-sampled to I_v's HxW).
#
# LOSS: computed by BaseDecodeHead.loss_by_feat over M_p vs. the GT mask M_g. The
# convnext DocTamper config sets loss = L_ce + L_lov (CrossEntropy + Lovasz) with an
# OHEMPixelSampler, matching the paper's L = L_ce + L_lov.
# =============================================================================
@MODELS.register_module()
class FPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks (Semantic FPN) decode head.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    DocTamper training role: this is the decode head of the ConvNeXt training variant of
    the paper TAFE-Net (reported as "TAFE-Net*"). It consumes the four pyramid
    features emitted by the DFDE / DWT neck -- the paper's (F'_1, F_2, F_3, F'_4) --
    and produces the tamper-mask logits M_p.

    PAPER DIFF: the paper predicts M_p with a SegFormerHead; this variant substitutes
    FPNHead. Functionally interchangeable at this point in the flow (four multi-scale
    features in, one dense prediction M_p out).

    The mechanism (Semantic FPN): each input level passes through a small conv
    `scale_head`; coarser levels are progressively x2-upsampled up to the finest
    feature stride (`feature_strides[0]`); all levels are summed; `cls_seg` maps the
    fused feature to per-pixel class logits = M_p. Loss (L_ce + L_lov = paper's L)
    is applied by the inherited BaseDecodeHead.loss_by_feat against the GT mask M_g.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):
        # `inputs`: the 4 pyramid features from the DFDE / DWT neck, i.e. the paper's
        # (F'_1, F_2, F_3, F'_4) ordered finest->coarsest stride. 'multiple_select'
        # input_transform means we just pick out the selected feature levels.
        x = self._transform_inputs(inputs)

        # Level 0 (finest stride = feature_strides[0]) sets the target resolution.
        # Its scale_head is a single conv (no upsampling needed for the finest map).
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            # Coarser level i: run its scale_head (conv(s) + repeated x2 bilinear
            # upsamples), then resize to level-0's HxW and SUM into the fused feature.
            # Summing every level at the finest stride = Semantic FPN aggregation.
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        # Project the fused feature to per-pixel class logits = tamper mask M_p
        # (BaseDecodeHead later up-samples M_p to the input image I_v's HxW and
        # applies the loss L_ce + L_lov against the GT mask M_g).
        output = self.cls_seg(output)
        return output
