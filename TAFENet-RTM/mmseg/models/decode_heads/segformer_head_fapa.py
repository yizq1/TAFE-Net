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
from ..toys.fapa import FAPA_v2



@MODELS.register_module()
class SegformerHeadWithFAPA(BaseDecodeHead):
    """
    集成了 FAPA_v2 注意力模块的 Segformer 解码头。

    在多尺度特征融合后，使用 FAPA_v2 对特征进行空间注意力增强。
    """

    def __init__(self,
                 interpolate_mode='bilinear',
                 save_feat=False,
                 fapa_ksize=8,         # FAPA_v2 的 ksize 参数
                 fapa_num_heads=8,     # FAPA_v2 的 num_heads 参数
                 fapa_dropout_rate=0.1,# FAPA_v2 的 dropout_rate 参数
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        self.save_feat = save_feat
        if self.save_feat:
            self.save_dir = 'vis/feat' # 特征保存路径
        else:
            self.save_dir = None

        assert num_inputs == len(self.in_index), \
            f"输入数量 ({num_inputs}) 必须等于索引数量 ({len(self.in_index)})"

        # 1. 各个输入阶段的 1x1 卷积，用于统一通道数
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels, # 统一到 self.channels
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # 2. 融合卷积，用于合并来自不同阶段的特征
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs, # 输入通道为 C * num_inputs
            out_channels=self.channels,             # 输出通道为 C
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        # ------------------- FAPA_v2 集成点 -------------------
        # 3. 实例化 FAPA_v2 模块
        #    它将在 fusion_conv 之后应用
        #    输入通道数应为 fusion_conv 的输出通道数，即 self.channels
        if self.channels % fapa_num_heads != 0:
            # 确保通道数可以被头数整除，否则调整头数或通道数
            # 这里简单地抛出错误，实际应用中可能需要更灵活的处理
            raise ValueError(f"SegformerHead 的 channels ({self.channels}) "
                             f"必须能被 FAPA_v2 的 num_heads ({fapa_num_heads}) 整除。")

        self.fapa_attention = FAPA_v2(
            in_channels=self.channels,
            ksize=fapa_ksize,
            num_heads=fapa_num_heads,
            dropout_rate=fapa_dropout_rate,
            debug=False # 正常运行时关闭 debug
        )
        # 注意：需要确保 FAPA_v2 的权重得到初始化。
        # BaseDecodeHead 或你的训练框架可能会处理初始化，
        # 如果没有，你可能需要手动调用 self.fapa_attention.initialize() 或类似方法。
        # 例如，可以在 SegformerHead 的 init_weights 方法中添加：
        # if hasattr(self, 'fapa_attention'):
        #     self.fapa_attention.initialize()
        # ------------------------------------------------------

        # 4. 最终的分割分类层 (在 BaseDecodeHead 中定义或继承)
        # self.cls_seg = nn.Conv2d(...)

    def forward(self, inputs):
        # 接收来自 backbone 的 4 个阶段特征图: 1/4, 1/8, 1/16, 1/32 分辨率
        inputs = self._transform_inputs(inputs) # 根据 self.in_index 选择输入
        outs = []

        # 处理每个阶段的输入
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            # 应用 1x1 卷积统一通道数，并上采样到第一个阶段（最大分辨率）的大小
            outs.append(
                resize(
                    input=conv(x),             # (B, C, Hi, Wi) -> (B, self.channels, Hi, Wi)
                    size=inputs[0].shape[2:],  # 目标尺寸 H0, W0
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)) # 输出 (B, self.channels, H0, W0)

        # 拼接所有上采样后的特征图 (沿通道维度)
        concat_out = torch.cat(outs, dim=1) # (B, self.channels * num_inputs, H0, W0)

        # 应用融合卷积，将通道数降回 self.channels
        fused_out = self.fusion_conv(concat_out) # (B, self.channels, H0, W0)

        # -------- FAPA_v2 应用点 --------
        # 在融合后的特征图上应用 FAPA_v2 注意力
        attn_out = self.fapa_attention(fused_out) # (B, self.channels, H0, W0)

        #加入残差连接
        attn_out=attn_out+fused_out
        # -------------------------------

        # 应用最终的分类层得到分割 Logits
        out = self.cls_seg(attn_out) # (B, num_classes, H0, W0)

        return out

    def forward_infer(self, inputs):
        """用于推理和特征保存的前向传播"""
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

        # 拼接并融合
        concat_out = torch.cat(outs, dim=1)
        fused_out = self.fusion_conv(concat_out)

        # 保存的特征 feats 定义为 FAPA 注意力之前的融合特征
        # 这与原始代码中 feats = self.fusion_conv(...) 的位置一致
        feats = fused_out

        # 应用 FAPA 注意力
        attn_out = self.fapa_attention(fused_out)

        # 应用分类层
        out = self.cls_seg(attn_out)

        # 返回分割结果和用于保存的特征图
        return out, feats

    def predict(self, inputs, batch_img_metas, test_cfg):
        """用于预测的前向函数"""
        if self.save_feat:
            # 如果需要保存特征，调用 forward_infer 获取 logits 和 feats
            seg_logits, feats = self.forward_infer(inputs)
            # 保存特征图 (feats 是 FAPA 应用之前的融合特征)
            self.save_deep_feature(feats.detach(), self.save_dir, batch_img_metas)
        else:
            # 否则，只进行标准的前向传播获取 logits
            seg_logits = self.forward(inputs)

        # 使用基类或自定义的方法根据 logits 预测最终分割结果
        return self.predict_by_feat(seg_logits, batch_img_metas)

    def save_deep_feature(self, feat, save_dir, batch_img_metas):
        """保存中间特征图 (与原始代码逻辑保持一致)"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 假设 batch size 为 1 进行保存，或者需要循环处理 batch
        # 注意：这里只处理了 batch 中的第一个样本
        if feat.shape[0] > 0: # 确保 batch 不为空
            feat_to_save = feat[0].detach().permute(1, 2, 0).cpu().numpy() # (H, W, C)
            img_meta = batch_img_metas[0]
            img_path = img_meta.get('img_path', img_meta.get('filename', 'unknown_image')) # 兼容不同版本的 mmseg meta key
            img_name = os.path.basename(img_path)
            img_name = os.path.splitext(img_name)[0] # 获取不带扩展名的文件名
            save_path = os.path.join(save_dir, img_name + '.npy')
            try:
                np.save(save_path, feat_to_save)
                print(f'成功保存特征图到: {save_path}')
            except Exception as e:
                print(f"错误：无法保存特征图到 {save_path}. 原因: {e}")
        else:
            print("警告：输入的特征图 batch size 为 0，无法保存。")
