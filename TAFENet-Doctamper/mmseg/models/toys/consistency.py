import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
# from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS


class ToSimiVolume(nn.Module):
    def __init__(self, in_channels, mode='dot', norm=False, flatten=True):
        '''
        in_channels: number of channels of input
        mode: 'dot' or 'cosine'
        norm: if norm, consistency map ~ [-1,1], otherwise ~ [0,1]
        '''
        super(ToSimiVolume, self).__init__()

        self.in_channels = in_channels
        self.mode = mode
        self.norm = norm
        self.flatten = flatten

        assert self.mode in ['dot', 'cosine']

        self.proj1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))
        self.proj2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))

        # self.sig = nn.Sigmoid()


    def forward(self, x):
        '''
             x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        '''
        B = x.size()[0]

        x_1 = self.proj1(x).view(B, self.in_channels, -1)   # (N, C, T*H*W)
        x_2 = self.proj1(x).view(B, self.in_channels, -1)   # (N, C, T*H*W)
        # x_1 = x_1.permute(0, 2, 1)                          # (N, T*H*W, C)

        if self.mode == 'dot':
            x_1 = x_1.permute(0, 2, 1)                      # (N, T*H*W, C)
            attn = torch.matmul(x_1, x_2)
            attn = attn / math.sqrt(self.in_channels)

            attn = F.sigmoid(attn)
            if self.norm:
                attn = (attn - 0.5) * 2
        else:
            attn = (F.normalize(x_1, dim=1).transpose(-2, -1) @ F.normalize(x_2, dim=1))
            if self.norm:
                pass
            else:
                attn = (attn + 1) / 2

        # contiguous here just allocates contiguous chunk of memory
        y = attn.permute(0, 2, 1).contiguous()

        out = y.view(B, *x.size()[2:], *x.size()[2:])
        if self.flatten:
            out = out.view(B, -1, *x.size()[2:])

        return out
    

class CrossAttnBlock(nn.Module):
    """
    一个多头自注意力的简化版，用做模态间交互。
    你也可以直接用 torch.nn.MultiheadAttention 来替代。
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        # Q, K, V 的可学习投影
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_value):
        """
        query: [B, N_q, C]
        key_value: [B, N_kv, C]
        返回: [B, N_q, C]
        """
        B, N_q, _ = query.shape
        B, N_kv, _ = key_value.shape

        # 1) 线性投影
        Q = self.W_q(query)      # [B, N_q, C]
        K = self.W_k(key_value)  # [B, N_kv, C]
        V = self.W_v(key_value)  # [B, N_kv, C]

        # 2) 分多头
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3) 注意力分数: Q @ K^T / sqrt(dim)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, heads, N_q, N_kv]
        attn_probs = F.softmax(attn_scores, dim=-1)                                 # softmax

        # 4) 计算加权 V
        attn_out = torch.matmul(attn_probs, V)  # [B, num_heads, N_q, head_dim]

        # 5) 合并多头
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()  # [B, N_q, num_heads, head_dim]
        attn_out = attn_out.view(B, N_q, self.embed_dim)      # [B, N_q, C]

        # 6) 输出投影
        out = self.proj_out(attn_out)  # [B, N_q, C]

        return out


class ToSimiVolumeEx(nn.Module):
    """
    与原先 ToSimiVolume 类似，但增加可学习 scale 来调节相似度分布。
    你也可自由改动投影方式, 这里仅作演示。
    """
    def __init__(self, in_channels, 
                 mode='cosine', norm=True, flatten=True):
        super().__init__()
        self.in_channels = in_channels
        self.mode = mode
        self.norm = norm
        self.flatten = flatten

        # 投影层, 仅1个conv演示 (你也可以用2个Q,K投影)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 在做相似度前或后,可学习的scale
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1) 投影
        feat = self.proj(x)  # [B, C, H, W]
        
        # 2) flatten (B, C, H, W) -> (B, C, N)
        feat = feat.view(B, C, -1)  # N = H*W

        # 3) normalize  (cosine 情况)
        if self.mode == 'cosine':
            feat = F.normalize(feat, dim=1)  # 沿通道维度normalize

        # 4) 相似度 => [B, N, N]
        #    dot/cosine都可以用 feat^T feat 的方式:
        simi = torch.bmm(feat.transpose(1, 2), feat)  # [B, N, N]

        # 5) scale + norm
        simi = simi * self.scale  # learnable scale
        
        if self.mode == 'cosine' and self.norm:
            # 将[-1,1]映射到[0,1], 具体看需求
            simi = (simi + 1.0) / 2.0
        
        elif self.mode == 'dot':
            # 如果需要,也可以加softmax或sigmoid
            simi = F.sigmoid(simi)

        # 6) reshape回 [B, N, H, W] or [B, channels, H, W]
        #    这里只示范把 N*N -> N,H,W 需保证 N=H*W, 做 reshape:
        simi = simi.view(B, H*W, H, W)
        
        if self.flatten:
            # flatten成通道维度
            simi = simi.permute(0, 2, 3, 1).contiguous()  # => [B, H, W, N]
            simi = simi.view(B, -1, H, W)  # => [B, N, H, W]
        
        return simi