# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .typing_utils import SampleList


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


# Stock mmseg collator (image + gt_sem_seg only, no `extras`). It is NOT on the
# DocTamper/TAFE-Net path in this repo — see `stack_batch_with_extra` below, which is
# the variant `SegDataPreProcessorWithExtra` actually calls.
def stack_batch(inputs: List[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            gt_sem_seg = data_sample.gt_sem_seg.data
            del data_sample.gt_sem_seg.data
            data_sample.gt_sem_seg.data = F.pad(
                gt_sem_seg, padding_size, value=seg_pad_val)
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': data_sample.gt_sem_seg.shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples


# =============================================================================
# TRAINING-FLOW / batch collation with extras: pads I_v + (D,T,img1) together  (file: misc.py)
# -----------------------------------------------------------------------------
# Active entry point of this file. `SegDataPreProcessorWithExtra.forward` calls
# this once per train/test iteration to turn a list of per-sample CHW image
# tensors (the paper's visual image I_v) PLUS a parallel `extras` dict of
# auxiliary tensors into batched, right-bottom-padded 4D tensors, so that the RGB
# image and its frequency companions all end up on ONE common HxW canvas and stay
# spatially aligned before the segmentor sees them.
#
# `extras` is a Dict-of-Lists (one entry per sample), e.g.
#   extras['dct']    = D  (JPEG luminance DCT coefficient map; consumed by FPH)
#   extras['qtable'] = T  (8x8 JPEG quantization table)
#   extras['img1']   = a JPEG-recompressed second RGB view (source of I_lf/I_hf)
# Each key is padded/stacked by a rule chosen from how its tensor relates to the
# image resolution (per-key branch comments below): resolution-independent side
# info is passed through untouched, the DCT map is padded to the padded image's
# H,W, and full-resolution image companions reuse the image's own padding.
#
# The padding geometry (max_size / size_divisor / right-bottom `padding_size`)
# and the gt_sem_seg handling are identical to the stock `stack_batch` above; the
# only addition here is the `extras` handling.
# =============================================================================
def stack_batch_with_extra(inputs: List[torch.Tensor],
                           extras = None,
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255,
                # dct_pad_val: Union[int, float] = 0,
                extra_pad_val: Union[int, float] = 0,) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        extras: The input of extra information in dictionary format, each element is a List[Tensor]
            with the same HW as input
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs = []
    padded_samples = []
    padded_extras = dict()


    # Mirror the extras dict as a Dict-of-empty-Lists; we fill one padded entry
    # per sample below, then stack each list into a batched tensor at the end.
    if extras is not None:
        for key in extras:
            padded_extras[key] = []

    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)

        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            gt_sem_seg = data_sample.gt_sem_seg.data
            del data_sample.gt_sem_seg.data
            data_sample.gt_sem_seg.data = F.pad(
                gt_sem_seg, padding_size, value=seg_pad_val)
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': data_sample.gt_sem_seg.shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))

        # pad extra input information
        # For this same sample i, pad each auxiliary tensor onto the SAME canvas
        # as the image so every stream stays aligned. The rule per key differs:
        if extras is not None:
            for key in extras:
                extra_tensor = extras[key][i]
                if key == 'qtable' or key == 'edge' or key == 'dis_map':
                    # Resolution-independent side info (8x8 quant table T, an edge
                    # flag, a distance map kept at native size): must NOT be
                    # spatially padded, so append verbatim and let torch.stack
                    # below merely batch it.
                    padded_extras[key].append(extra_tensor)
                else:
                    if key == 'dct':
                        # D is spatially tied to the image but may currently be
                        # smaller than the padded image; pad it to the padded
                        # image's exact H,W, computing the right/bottom margins
                        # from the DCT tensor's OWN last-two dims (not the image's)
                        # so D lines up with I_v after right-bottom padding.
                        pad_img_h, pad_img_w = pad_img.shape[-2], pad_img.shape[-1]
                        padding_size_dct = (0, pad_img_w - extra_tensor.shape[-1], 0, pad_img_h - extra_tensor.shape[-2])
                        pad_extra = F.pad(extra_tensor, padding_size_dct, value=extra_pad_val)
                    else:
                        # Full-resolution image companion (e.g. img1 second RGB
                        # view, ori_img copy): reuse the IMAGE's own padding_size
                        # so it stays pixel-aligned with I_v.
                        pad_extra = F.pad(extra_tensor, padding_size, value=extra_pad_val)
                    padded_extras[key].append(pad_extra)

    # Collapse each per-sample list into one batched N-dim tensor. Because every
    # sample was padded to the same canvas above, all entries of a key share a
    # shape and stack cleanly; a shape mismatch here surfaces as a per-key error.
    if extras is not None:
        for key in padded_extras:
            try:
                padded_info = torch.stack(padded_extras[key], dim=0)
                padded_extras[key] = padded_info
            except:
                print("error in [{}]".format(key))
                raise Exception

    # Batched image (I_v), the batched extras dict (D, T, img1, ...), and the
    # padded data_samples, matching the 3-tuple SegDataPreProcessorWithExtra unpacks.
    return torch.stack(padded_inputs, dim=0), padded_extras, padded_samples