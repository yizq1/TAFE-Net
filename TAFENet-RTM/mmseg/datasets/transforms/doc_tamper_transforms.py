import os
import pickle
import random
import tempfile
from pathlib import Path

import jpegio
import mmcv
import numpy as np
import six
from mmcv.transforms import BaseTransform, TRANSFORMS
from PIL import Image
import copy


# =============================================================================
# TRAINING-FLOW / DATASET TRANSFORMS  (file: doc_tamper_transforms.py)
# dataset transforms: produce paper inputs D (dct), T (qtable), img1
# -----------------------------------------------------------------------------
# This module holds the MMCV dataset-pipeline transform that turns a document
# image (and its on-disk JPEG) into the auxiliary frequency inputs consumed by
# TAFE-Net ("Frequency Mining Empowered by Text Aggregation", AAAI 2026). It
# runs inside the RTM (RealTextManipulation) training pipeline, right before
# PackSegInputs(WithExtra) packs the results into the model's `extras` dict.
#
# Paper <-> code correspondence for the tensors produced here:
#   results['dct']    = D   : JPEG DCT coefficient map of the LUMINANCE (Y)
#                             channel, |coef_arrays[0]| clipped to [0, 20]  (H x W)
#   results['qtable'] = T   : the 8x8 JPEG quantization table quant_tables[0]
#                             (Y channel), clipped to [0, 63]           (1 x 8 x 8)
#   results['img']    = I_v : the RGB image view fed to the visual backbone.
#
# Downstream these feed the MFFE (Multi-Frequency Feature Extractor): (D, T) are
# encoded by the FPH (Frequency Perception Head, from DTD/Qu2023) into the DCT
# tamper feature F_d, while I_v feeds the visual path and, via a DCT high/low
# split, the I_hf / I_lf frequency views. This transform is the code source of
# the paper step "compute the JPEG DCT coefficient map D and the quantization
# table T from its Y channel". The second RGB view carried as extras['img1'] is
# built downstream (by PackSegInputsWithExtra) from results['img'] produced here.
# =============================================================================


@TRANSFORMS.register_module()
class RandomJpegCompressAndLoadInfo(BaseTransform):
    """Random JPEG (re)compression + JPEG-info loader (dataset transform).

    Role in the TAFE-Net / RTM training flow
    ----------------------------------------
    Given a document image (results['img']) and its JPEG path
    (results['img_path']), this transform emits the paper's auxiliary inputs:
      * results['dct']    = D   : DCT coefficient map of the Y (luminance)
                                  channel, |coef_arrays[0]| clipped to [0, 20].
      * results['qtable'] = T   : 8x8 luminance quantization table
                                  quant_tables[0], clipped to [0, 63].
      * results['img']    = I_v : the RGB image view for the visual backbone.

    Conceptually the transform re-encodes the image at a RANDOM JPEG quality and
    then reloads D and T, so the model trains under varied compression and stays
    robust to it (D, T -> FPH -> F_d in the MFFE).

    # PAPER DIFF: in the ACTIVE path the random re-compression loop is commented
    #   out (see `transform` below); D and T are read directly from the ORIGINAL
    #   on-disk JPEG at results['img_path'], so the random-quality re-encode that
    #   the class name / constructor args describe does not actually run here.
    """

    def __init__(self, jpeg_compress_time=(1, 2, 3), course=False, quality_lower=75, compress_pk=None, load_info=True,
                 return_rgb=False):
        """Configure the (optional) random JPEG re-compression and info loading.

        Args:
            jpeg_compress_time: candidate counts of successive JPEG compressions
                (used only by the currently commented-out re-compression loop).
            course: unsupported here (raises NotImplementedError if True).
            quality_lower: lowest random JPEG quality factor for re-compression.
            compress_pk: optional pickle of precomputed [q2, q1, q] quality lists
                keyed by image index (used only by the commented-out loop).
            load_info: if True, load D (dct) and T (qtable) from the JPEG.
            return_rgb: if True keep RGB; else collapse to luminance (L) before
                re-expanding to RGB for results['img'] (= I_v).
        """
        super().__init__()
        self.jpeg_compress_time = jpeg_compress_time
        self.course = course
        self.quality_lower = quality_lower
        self.compress_pk = compress_pk
        if self.compress_pk is not None:
            assert os.path.exists(self.compress_pk), f"{self.compress_pk} not exists"
            self.compress_pk = pickle.load(open(self.compress_pk, 'rb'))
            # List of [q2, q1, q]

        self.load_info = load_info
        if course:
            raise NotImplementedError
        self.return_rgb = return_rgb

    def transform(self, results: dict) -> dict:
        """Produce D (dct), T (qtable) and the RGB view I_v (results['img']).

        Reads the source RGB array, (optionally) re-compresses it as JPEG, loads
        the luminance DCT coefficients + quantization table, and writes them back
        into `results` for PackSegInputs(WithExtra) to carry as model `extras`.
        """
        img: np.ndarray = results['img']              # source RGB image [H, W, 3]

        if self.course:
            raise NotImplementedError                 # 'course' schedule unsupported here
        else:
            quality_lower = self.quality_lower        # lowest random JPEG quality factor

        # ---- (inactive variant, not used by the RTM configs) -----------------------
        # Pick a random number of successive JPEG compressions and their quality
        # factors (or look them up per image); kept for reference, NOT executed.
        # if self.compress_pk is None:
        #     jpeg_compress_time = random.choice(self.jpeg_compress_time)
        #     compress_quality = np.random.randint(quality_lower, 101, jpeg_compress_time)
        # else:
        #     image_path = results['img_path']
        #     index = int(Path(image_path).stem)
        #     compress_quality = self.compress_pk[index]

        im = Image.fromarray(img)                     # PIL view of the source image

        if self.return_rgb:
            im_ = im.copy()                           # keep RGB
        else:
            im_ = im.convert("L")                     # else collapse to luminance (Y)

        # ---- (inactive variant, not used by the RTM configs) -----------------------
        # The paper re-encodes at the random quality then reloads D, T for
        # compression robustness; that in-memory re-compress + jpegio.read loop is
        # commented out below and NOT executed here.
        # compress_quality = [compress_quality[0]]  # keep the same as the author,
        # compress_quality=[100]
        # for q in compress_quality:
        #     buffer = six.BytesIO()
        #     im_.save(buffer, format="JPEG", quality=int(q))
        #     im_ = Image.open(buffer)

        # if self.load_info:
        #     # jpg = jpegio.read(buffer.getvalue())
        #     buffer.seek(0)
        #     with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        #         f.write(buffer.getvalue())
        #         f.flush()
        #         jpg = jpegio.read(f.name)
        # ---- ACTIVE path: load D and T from the on-disk JPEG's Y channel -----------
        if self.load_info:
            jpg=jpegio.read(results['img_path'])       # read the ORIGINAL JPEG structure


            # coef_arrays[0] / quant_tables[0] = the LUMINANCE (Y) channel:
            dct = copy.deepcopy(jpg.coef_arrays[0])                        # raw Y DCT coeffs [H, W]
            use_qtb = copy.deepcopy(jpg.quant_tables[0]).astype(np.uint8)  # 8x8 Y quant table

            # D = |DCT| clipped to [0, 20]                        -> results['dct']  [H, W]
            results['dct'] = np.clip(np.abs(dct), 0, 20)
            # T = quant table clipped to [0, 63], shaped [1, 8, 8] int32 -> results['qtable']
            # PAPER DIFF: D, T come from the original on-disk JPEG (img_path), not from a
            #   freshly random-re-compressed buffer -- the re-compress loop above is disabled.
            results['qtable'] = np.expand_dims(np.clip(use_qtb, 0, 63).astype(np.int32), 0)
            

        # I_v: RGB view fed to the visual backbone (from im_, so grayscale->RGB when
        # return_rgb is False; a genuine RGB copy when return_rgb is True).
        im = im_.convert('RGB')
        results['img'] = np.array(im)                 # results['img'] = I_v  [H, W, 3]

        return results
