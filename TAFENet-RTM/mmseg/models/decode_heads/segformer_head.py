# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize

from pathlib import Path
import numpy as np
import pickle as pkl
import os


# =============================================================================
# TRAINING-FLOW / DECODE HEAD  (file: segformer_head.py)
# -----------------------------------------------------------------------------
# `SegformerHead` is the segmentation decode head of the SegFormer variant of the
# paper's TAFE-Net ("Frequency Mining Empowered by Text Aggregation", AAAI 2026,
# SegFormer/MiT-b2 baseline = Table1 "TAFE-Net"). It is the LAST stage of the
# training flow: it turns the four refined pyramid features into the binary
# tamper logits M_p and (through BaseDecodeHead) drives the loss.
#
# In the RTM training flow it is called as:
#   MyModelFull.forward_encoder -> DFDE neck (DWTFPN_dct_v6) yields the paper's
#   (F'_1, F_2, F_3, F'_4) -> SegformerHead(inputs) -> M_p (2-class tamper logits).
#
# INPUTS (paper symbols): the 4 neck features F'_1, F_2, F_3, F'_4 at strides
#   1/4, 1/8, 1/16, 1/32 of the input, with per-stage MiT channels. This head is
#   the classic SegFormer "all-MLP" decoder: unify channels per level (1x1),
#   upsample every level to the highest (stage-0) resolution, concatenate, fuse
#   with one 1x1 conv, then classify -> M_p.
#
# LOSS: computed by BaseDecodeHead.loss_by_feat over M_p vs ground-truth mask M_g;
#   the RTM configs set decode_head.loss_decode = [CrossEntropy, Lovasz], i.e. the
#   paper's L = L_ce(M_p, M_g) + L_lov(M_p, M_g).
#
# PAPER<->CODE: SegformerHead == paper SegFormerHead. This is the stock MMSeg
#   SegFormer head; the TAFE-Net-specific frequency machinery lives upstream
#   (VFIM/MFFE in the backbone, DFDE in the neck), not here.
# =============================================================================
@MODELS.register_module()
class SegformerHead(BaseDecodeHead):
    """All-MLP SegFormer decode head = paper's SegFormerHead (TAFE-Net SegFormer variant).

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    In the RTM / TAFE-Net training flow it consumes the four DFDE-refined pyramid
    features (paper F'_1, F_2, F_3, F'_4), unifies their channels with a per-level
    1x1 ConvModule, upsamples all of them to the highest (stage-0) resolution,
    concatenates, fuses with a single 1x1 ConvModule, and finally applies cls_seg
    to emit the 2-class tamper logits M_p. The training loss L_ce + L_lov (paper
    Eq. loss) is computed by the parent `BaseDecodeHead` over M_p and mask M_g.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
        save_feat: If True, cache the fused pre-classifier feature to disk during
            inference (visualization only; not on the training path). Default: False.
    """

    def __init__(self, interpolate_mode='bilinear', save_feat=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        self.save_feat = save_feat
        if self.save_feat:
            self.save_dir = 'vis/feat'
        else:
            self.save_dir = None

        assert num_inputs == len(self.in_index)

        # Per-level 1x1 ConvModule: one per input feature (F'_1, F_2, F_3, F'_4),
        # projecting each stage's channels self.in_channels[i] -> a common
        # self.channels so the four levels can be concatenated after resizing.
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # Fusion 1x1 ConvModule: mixes the concatenated 4-level features
        # (self.channels * num_inputs channels) back down to self.channels.
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        """Decode the 4 DFDE-refined pyramid features into tamper logits M_p.

        Args:
            inputs: the four neck features (paper F'_1, F_2, F_3, F'_4) at strides
                1/4, 1/8, 1/16, 1/32.

        Returns:
            Tensor: the 2-class tamper logit map M_p [B, num_classes, H/4, W/4].
        """
        # Select the configured levels (in_index) -> list of 4 features
        # F'_1, F_2, F_3, F'_4 at 1/4, 1/8, 1/16, 1/32 of the input.
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            # Per-level 1x1 conv unifies channels -> self.channels, then resize
            # (bilinear) up to inputs[0]'s spatial size = the highest / stage-0
            # (1/4) resolution, so all four levels share one grid.
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # Concatenate the 4 aligned levels along channels -> fuse with the 1x1
        # fusion conv back to self.channels.
        out = self.fusion_conv(torch.cat(outs, dim=1))

        # Classifier (1x1 conv over self.channels -> num_classes) = the paper's
        # tamper logits M_p; loss L_ce + L_lov is computed by BaseDecodeHead.
        out = self.cls_seg(out)

        return out

    def forward_infer(self, inputs):
        # INFERENCE/VIS ONLY (not on the training path): same forward as above but
        # also returns the fused pre-classifier feature so predict()/save_feat can
        # dump it for visualization; forward() is what training uses.
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

            # feats.append(
            #     self.projs[idx][1](self.projs[idx][0](outs[idx])))

        feats = self.fusion_conv(torch.cat(outs, dim=1))
        # feats = out.detach()

        # mid = self.seg_proj(out)
        out = self.cls_seg(feats)

        return out, feats

    def predict(self, inputs, batch_img_metas, test_cfg):
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        if self.save_feat:
            seg_logits, feats = self.forward_infer(inputs)
            self.save_deep_feature(feats.detach(), self.save_dir, batch_img_metas)
        else:
            seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def save_deep_feature(self, feat, save_dir, batch_img_metas):
        """save intermediate feature map"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # for i in range(len(batch_img_metas)):
        feat = feat[0].detach().permute(1,2,0).cpu().numpy()
        img_meta = batch_img_metas[0]
        img_name = img_meta['img_path']
        img_name = os.path.basename(img_name).split('.')[0]
        # img_name = img_name.split('/')[-1].split('.')[0]
        img_name = img_name + '.npy'
        np.save(os.path.join(save_dir, img_name), feat)
        print('save feature map: ', img_name)
