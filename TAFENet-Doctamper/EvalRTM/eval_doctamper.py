#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DocTamper offline mask evaluator.

Standalone toolkit (run AFTER inference) that scores predicted tampering
masks against ground-truth masks on the DocTamper dataset. Computes
pixel-level, micro-averaged binary segmentation metrics
(IoU / Precision / Recall / F1) from a confusion matrix.

Usage:
    python eval_binary.py --pred_dir path/to/preds --gt_dir path/to/gts
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import csv


# def parse_args():
#     parser = argparse.ArgumentParser(description="Binary-mask metrics")
#     parser.add_argument("--pred_dir", help="directory with predicted PNG masks",default='/data3/yzq/code/RTM-shuang-2/work_dirs1/ascformer_rtm_img_img1_highlowdct_convnext_dwtfpn_MultiScaleMultiLowFreqExtractor_doctamper_new_0511/result/pred_mask')
#     parser.add_argument("--gt_dir",   help="directory with GT PNG masks",default='/data3/yzq/data/DocTamperV1/DocTamperV1-SCD/masks-dan')
#     parser.add_argument("--save_csv", default=None,  help="optional csv file for per-image metrics")
#     parser.add_argument("--th",       type=float, default=0.5,
#                         help="binarisation threshold if predictions are probability maps")
#     return parser.parse_args()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary segmentation evaluation with multiprocessing
Metrics: IoU / Precision / Recall / F1  (micro-averaged)
----------------------------------------------------------------
python eval_binary_mp.py \
       --pred_dir path/to/preds \
       --gt_dir   path/to/gts \
       --save_csv metrics.csv \
       --workers  8
"""

import argparse
import glob
import os
import csv
from pathlib import Path
from typing import Tuple, Dict, List

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed


# ──────────────────────────────── Utils ────────────────────────────────────
def binarise(img: np.ndarray, th: float) -> np.ndarray:
    """Convert gray / prob map to {0,1} uint8 mask."""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return (img >= int(th * 255)).astype(np.uint8)


def metrics_from_confusion(tp, fp, fn) -> Dict[str, float]:
    """Derive IoU / Precision / Recall / F1 from confusion-matrix counts."""
    iou       = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * tp / (2 * tp + fp + fn + 1e-7)
    return dict(iou=iou, precision=precision, recall=recall, f1=f1)


def evaluate_pair(args: Tuple[str, str, float]) -> Tuple[str, int, int, int, int, Dict[str, float]]:
    """
    Worker function: compute metrics for a single image.
    Returns: (filename, tp, fp, fn, tn, metrics_dict)
    """
    pred_path, gt_path, th = args
    name = os.path.basename(pred_path)

    gt   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        raise FileNotFoundError(f"Missing file: {gt_path} or {pred_path}")

    # If sizes differ, crop to the overlapping region
    if gt.shape != pred.shape:
        h = min(gt.shape[0], pred.shape[0])
        w = min(gt.shape[1], pred.shape[1])
        gt, pred = gt[:h, :w], pred[:h, :w]

    gt   = (gt > 0).astype(np.uint8)
    pred = binarise(pred, th)

    tn, fp, fn, tp = confusion_matrix(
        gt.flatten(), pred.flatten(), labels=[0, 1]
    ).ravel()

    return name, int(tp), int(fp), int(fn), int(tn), metrics_from_confusion(tp, fp, fn)


# ──────────────────────────────── Main ─────────────────────────────────────
def parse_args():
    """Parse command-line arguments (pred/gt dirs, threshold, workers, CSV)."""
    parser = argparse.ArgumentParser(description="Binary-mask metrics (multiprocessing)")
    parser.add_argument("--pred_dir", help="directory with predicted PNG masks",default='/data3/yzq/code/RTM-shuang-2/work_dirs_xiaorong_on_a60004/work_dirs_xiaorong/0714_with_stim_fph_highlow_without_dfde/result/pred_mask_testing')
    parser.add_argument("--gt_dir",   help="directory with GT PNG masks",default='/data3/yzq/data/DocTamperV1/DocTamperV1-TestingSet/masks-dan')
    parser.add_argument("--save_csv", default=None,  help="optional csv file for per-image metrics")
    parser.add_argument("--th",       type=float, default=0.5,
                        help="binarisation threshold if predictions are probability maps")
    parser.add_argument("--workers",  type=int, default=4,
                        help="number of worker processes (default: cpu count)")
    return parser.parse_args()


def main():
    """Evaluate all pred/gt pairs in parallel, aggregate, print, and optionally save CSV."""
    args = parse_args()

    pred_list = sorted(glob.glob(os.path.join(args.pred_dir, "*.png")))
    if not pred_list:
        raise RuntimeError("No PNGs found in pred_dir")

    # Build the list of (pred_path, gt_path, th) arguments for parallel processing
    tasks: List[Tuple[str, str, float]] = []
    for pred_path in pred_list:
        name    = os.path.basename(pred_path)
        gt_path = os.path.join(args.gt_dir, name)
        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found for {name}, skipped.")
            continue
        tasks.append((pred_path, gt_path, args.th))

    if not tasks:
        raise RuntimeError("No valid image pairs to evaluate.")

    # ── Multiprocessing ──────────────────────────────────────────────────
    per_image = []
    total = dict(tp=0, fp=0, fn=0, tn=0)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_pair, t) for t in tasks]
        for fut in as_completed(futures):
            name, tp, fp, fn, tn, m = fut.result()
            per_image.append((name, m["iou"], m["precision"], m["recall"], m["f1"]))

            total["tp"] += tp
            total["fp"] += fp
            total["fn"] += fn
            total["tn"] += tn

    # ── Aggregation ──────────────────────────────────────────────────────
    print(total["tp"],total["fp"],total["fn"])
    overall = metrics_from_confusion(total["tp"], total["fp"], total["fn"])

    print("=================================================")
    print(f"Images evaluated : {len(per_image)}")
    print("Overall metrics (micro, all pixels):")
    print(f"  IoU       : {overall['iou']*100:6.2f}%")
    print(f"  Precision : {overall['precision']*100:6.2f}%")
    print(f"  Recall    : {overall['recall']*100:6.2f}%")
    print(f"  F1 / Dice : {overall['f1']*100:6.2f}%")
    print("=================================================")

    # ── Optionally save CSV ──────────────────────────────────────────────
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["filename", "iou", "precision", "recall", "f1"])
            for row in sorted(per_image):  # write rows sorted by filename
                writer.writerow(row)
        print(f"Per-image metrics saved to {args.save_csv}")


if __name__ == "__main__":
    main()