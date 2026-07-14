# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


# =============================================================================
# TRAINING-FLOW / DATA-PACKING TRANSFORM  (file: formatting.py)
# -----------------------------------------------------------------------------
# This file holds the FINAL pipeline transform that PACKS one preprocessed
# dataset sample into the dict MMEngine hands to the model each iteration. Two
# variants live here:
#   * PackSegInputs          -> stock MMSeg packer (image + GT only). NOT on the
#                               DocTamper / TAFE-Net training path; see its docstring.
#   * PackSegInputsWithExtra -> the DocTamper / TAFE-Net packer (registered as
#                               type='PackSegInputsWithExtra' in the DocTamper configs).
#
# In paper terms (TAFE-Net, "Frequency Mining Empowered by Text Aggregation",
# AAAI 2026) PackSegInputsWithExtra is the bridge from the DATA PIPELINE to the
# MODEL. It emits three things:
#     inputs        = visual RGB image tensor I_v            [3, H, W]
#     data_samples  = SegDataSample carrying the GT mask M_g (as gt_sem_seg)
#     extra         = dict of the requested `extra_keys`, e.g. the JPEG
#                     luminance DCT coefficient map D ('dct'), the 8x8
#                     quantization table T ('qtable'), and a second image
#                     view ('img1').
# Downstream, SegDataPreProcessorWithExtra forwards `extra` to
# MyModelFull.forward_encoder, which turns (I_v, D, T, img1) into the frequency
# streams I_lf / I_hf (DCT low/high views) and F_d (FPH output) that feed the
# MFFE (Multi-Frequency Feature Extractor). Without this transform delivering
# D, T and img1 ALONGSIDE I_v, the frequency branches of VFIM / MFFE / FPH could
# not be built.
# =============================================================================


# NOTE: base stock MMSeg packer -- image + GT only, no `extra` dict. It is NOT
# used by the DocTamper configs; the paper's frequency cues are packed by
# PackSegInputsWithExtra below. Kept here only for reference / non-DocTamper pipelines.
@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


''' Customized '''
@TRANSFORMS.register_module()
class PackSegInputsWithExtra(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
        extra_keys (Sequence[str] | str): extra ``results`` keys to copy into the
            emitted ``extra`` dict (default: empty tuple). The special value
            ``'dct'`` additionally pulls in ``'qtable'`` (the coefficient map D
            and its table T travel together); any OTHER key is treated as an
            HxWxC image-like array and transposed to CxHxW like the main image.

    DocTamper / TAFE-Net role:
        Same packing contract as ``PackSegInputs`` (image -> ``inputs``, GT
        mask -> ``data_samples.gt_sem_seg`` = M_g) but it ALSO emits an
        ``extra`` dict of frequency cues. For the DocTamper configs ``extra`` carries
        the quantized JPEG luminance DCT coefficient map D (``'dct'``), the 8x8
        quantization table T (``'qtable'``), and a second re-compressed image
        view (``'img1'``). ``SegDataPreProcessorWithExtra`` forwards ``extra``
        to ``MyModelFull.forward_encoder``, which turns (I_v, D, T) into the
        frequency streams I_lf / I_hf and F_d (via the FPH) that feed the MFFE.
        Thus this transform is the bridge that makes D, T and the extra view
        available to the paper's VFIM / MFFE / FPH branches alongside I_v.
    """

    def __init__(self, extra_keys=(),
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction')):
        self.meta_keys = meta_keys
        self.extra_keys = extra_keys
        if isinstance(extra_keys, str):
            self.extra_keys = (extra_keys, )

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # inputs = the visual RGB image I_v: HxWxC numpy -> contiguous CxHxW
            # tensor [3, H, W]. This is the main visual stream fed to the paper's
            # MFFE backbone (and the source of the DCT low/high views I_lf/I_hf
            # derived downstream by SegDataPreProcessorWithExtra/forward_encoder).
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        # print(results.keys(),results)
        # data_samples = SegDataSample holding the annotations.
        data_sample = SegDataSample()
        # gt_seg_map [H, W] -> [1, H, W] int64 -> gt_sem_seg = the paper's GT
        # tamper mask M_g, used later in L = L_ce(M_p, M_g) + L_lov(M_p, M_g).
        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                data=to_tensor(results['gt_seg_map'][None,
                                                     ...].astype(np.int64)))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        # add extra information
        # --- extra: auxiliary frequency cues for the paper's frequency branches.
        # Everything named in `extra_keys` is packed into a SEPARATE `extra` dict
        # (kept out of `inputs` / `data_samples`). SegDataPreProcessorWithExtra
        # later forwards this dict to MyModelFull.forward_encoder so the frequency
        # streams (I_lf / I_hf and F_d) can be built for the MFFE.
        extra_info = dict()
        for key in self.extra_keys:
            if key == 'dct':
                # 'dct' is the DCT tamper cue and always travels with its table:
                #   extra['dct'] = quantized luminance DCT coefficient map D
                #   (HxW, clipped), passed through AS-IS (no CxHxW transpose --
                #   it is 8x8-block coefficient data, not an RGB image).
                if 'dct' in results:
                    dct = results['dct']
                    dct = np.ascontiguousarray(dct)
                    extra_info['dct'] = to_tensor(dct)
                #   extra['qtable'] = 8x8 JPEG quantization table T. Together the
                #   pair (D, T) drives the FPH (Frequency Perception Head, from
                #   DTD) -> F_d inside the MFFE.
                if 'qtable' in results:
                    qtable = results['qtable']
                    qtable = np.ascontiguousarray(qtable)
                    extra_info['qtable'] = to_tensor(qtable)
            else:
                # Any other extra key (e.g. 'img1', a second JPEG-recompressed RGB
                # view) is treated like an image: HxWxC -> contiguous CxHxW tensor
                # [C, H, W], carried in `extra` as an auxiliary RGB view.
                # (NB: the paper's low/high-freq DCT views I_lf / I_hf are NOT split
                # from this key -- forward_encoder feeds I_v (`inputs`) to the Low/
                # High DCT extractors to build I_lf / I_hf.)
                if key in results:
                    extra_data = results[key]
                    if len(extra_data.shape) < 3:
                        extra_data = np.expand_dims(extra_data, -1)
                    extra_data = np.ascontiguousarray(extra_data.transpose(2, 0, 1))
                    extra_info[key] = to_tensor(extra_data)

        # Attach the collected cues; consumed downstream by the *WithExtra
        # preprocessor/model, NOT by the stock MMSeg pipeline.
        packed_results['extra'] = extra_info

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
