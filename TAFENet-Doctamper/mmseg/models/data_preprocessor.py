# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch, stack_batch_with_extra


# =============================================================================
# TRAINING-FLOW / data preprocessor: builds inputs(I_v)+extras(D,T,img1) for MyModelFull  (file: data_preprocessor.py)
# -----------------------------------------------------------------------------
# This file provides the batch preprocessor that MMEngine's training loop runs
# right before it calls the segmentor. The DocTamper configs use
# `SegDataPreProcessorWithExtra` (below) as `model.data_preprocessor`, so every
# training iteration flows:
#     Runner.train_step -> SegDataPreProcessorWithExtra.forward(data, training)
#                       -> MyModelFull.forward(inputs, extras, data_samples)
#
# Paper <-> code correspondence (paper = TAFE-Net, "Frequency Mining Empowered
# by Text Aggregation", AAAI 2026):
#   * `inputs`  = the normalized RGB image = the paper's visual image I_v
#                 (BGR->RGB, then mean=[123.675,116.28,103.53],
#                  std=[58.395,57.12,57.375]).
#   * `extras`  = an auxiliary dict carried straight from the dataset pipeline,
#                 holding the frequency tensors MyModelFull's MFFE needs:
#                   extras['dct']    = D  (JPEG luminance DCT coefficient map)
#                   extras['qtable'] = T  (8x8 JPEG quantization table)
#                   extras['img1']   = a JPEG-recompressed second RGB view.
#                 (D,T) are consumed by the FPH (Frequency Perception Head), and
#                 I_v / img1 are the source of the I_lf / I_hf views downstream.
#
# `SegDataPreProcessor` (immediately below) is the stock mmseg preprocessor with
# no extras support; it is NOT on the DocTamper/TAFE-Net training path in this repo.
# =============================================================================
@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            assert len(inputs) == 1, (
                'Batch inference is not support currently, '
                'as the image size might be different in a batch')
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)

@MODELS.register_module()
class SegDataPreProcessorWithExtra(BaseDataPreprocessor):
    """Batch preprocessor for MyModelFull (= the paper's TAFE-Net).

    Extends the stock segmentation preprocessor so that, alongside the RGB
    image, it also collates / device-casts / pads the auxiliary frequency
    tensors that the DocTamper dataset pipeline attaches under ``data['extra']``.

    For each batch it produces:
      * ``inputs`` : the normalized RGB image = the paper's visual image I_v
        (optionally BGR->RGB, then ``(x - mean) / std`` with the ImageNet stats
        mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]).
      * ``extras`` : a dict of auxiliary tensors passed through untouched except
        for device/float casting (and BGR->RGB for the second view):
            extras['dct']    = D  (JPEG luminance DCT coefficient map)
            extras['qtable'] = T  (8x8 JPEG quantization table)
            extras['img1']   = a JPEG-recompressed second RGB view
        These become the (D,T) fed to the FPH and the source of I_lf/I_hf in
        MyModelFull's MFFE (Multi-Frequency Feature Extractor).

    When ``copy_img`` is set, the un-normalized RGB image is also stashed as
    ``extras['ori_img']``. During training, image and extras are padded together
    via :func:`stack_batch_with_extra` so they stay spatially aligned.
    """

    def __init__(
            self,
            mean: Sequence[Number] = None,
            std: Sequence[Number] = None,
            size: Optional[tuple] = None,
            size_divisor: Optional[int] = None,
            pad_val: Number = 0,
            seg_pad_val: Number = 255,
            dct_pad_val: Number = 0,
            extra_pad_val: Number = 0,
            copy_img: bool = False,
            bgr_to_rgb: bool = False,
            rgb_to_bgr: bool = False,
            batch_augments: Optional[List[dict]] = None,
            test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.dct_pad_val = dct_pad_val
        self.extra_pad_val = extra_pad_val
        self.seg_pad_val = seg_pad_val
        self.copy_img = copy_img

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        # Collate the sampled batch and move every tensor to the compute device.
        data = self.cast_data(data)  # type: ignore
        # Per-sample RGB images (each [3, H, W], still BGR + integer range here).
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # `extras` is the auxiliary bundle produced by the DocTamper pipeline: a
        # Dict-of-Lists carrying the frequency tensors MyModelFull needs, i.e.
        #   extras['dct']    = D  (quantized JPEG luminance DCT coefficient map)
        #   extras['qtable'] = T  (8x8 JPEG quantization table)
        #   extras['img1']   = a JPEG-recompressed second RGB view.
        # (D,T) feed the FPH; I_v / img1 are the source of I_lf / I_hf downstream.
        extras = data.get('extra', None)    # extra infos are Dict(List)

        # TODO: whether normalize should be after stack_batch
        # BGR -> RGB so the image matches the ImageNet-order mean/std below; the
        # normalized RGB result is the paper's visual image I_v.
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        # Cast image pixels to float before normalization.
        inputs = [_input.float() for _input in inputs]

        # Carry + cast every auxiliary tensor. The 'img1' second RGB view is also
        # BGR -> RGB converted so it lines up with I_v; D and T (extras['dct'] /
        # extras['qtable']) are simply cast to float (no channel swap).
        if extras is not None:
            for key in extras:
                if key=='img1':
                    if self.channel_conversion and inputs[0].size(0) == 3:
                        extras[key] = [_input[[2, 1, 0], ...] for _input in extras[key]]
                extras[key] = [_temp.float() for _temp in extras[key]]

        # copy_img=True stashes the un-normalized RGB image under extras['ori_img']
        # (e.g. for later JPEG re-compression / visualization).
        if self.copy_img:
            if extras is None:
                extras = {}
            extras['ori_img'] = inputs.copy()

        # Normalize with ImageNet stats mean=[123.675,116.28,103.53],
        # std=[58.395,57.12,57.375]. The normalized RGB tensor IS the paper's I_v;
        # the 'img1' second view is normalized the same way.
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]
            if "img1" in extras:
                extras['img1'] = [(_input - self.mean) / self.std for _input in extras[key]]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            # Pad the image AND every extra tensor jointly (so I_v and D/T/img1
            # stay spatially aligned) and stack them into batched tensors.
            inputs, extras, data_samples = stack_batch_with_extra(
                inputs=inputs,
                extras=extras,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val,
                extra_pad_val=self.extra_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)

        else:
            assert len(inputs) == 1, (
                'Batch inference is not support currently, '
                'as the image size might be different in a batch')
            # pad images when testing
            # Same joint image+extra padding as training, but for the single sample.
            if self.test_cfg:
                inputs, extras, padded_samples = stack_batch_with_extra(
                    inputs=inputs,
                    extras=extras,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val,
                    extra_pad_val=self.extra_pad_val)

                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

                # Stack each extra list into a batched tensor to mirror `inputs`.
                if extras is not None:
                    for key in extras:
                        extras[key] = torch.stack(extras[key], dim=0)

        # Hand MyModelFull the batch: inputs = I_v, extras = {D, T, img1[, ori_img]}.
        return dict(inputs=inputs, extras=extras, data_samples=data_samples)


