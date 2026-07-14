#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DocTamper offline mask evaluator (DocTamperMetric-compatible variant).

Standalone toolkit (run AFTER inference) that scores predicted tampering
masks against ground-truth masks on the DocTamper dataset, using a
multiprocessing worker pool. Reports IoU / Precision / Recall / F1, with
overall metrics computed to match the original DocTamperMetric logic
(global confusion-matrix histogram + per-image-mean precision/recall).
----------------------------------------------------------------
python eval_binary_mp_modified.py \
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
from typing import Tuple, Dict, List, Any

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


# ──────────────────────────────── Utils ────────────────────────────────────
def binarise(img: np.ndarray, th: float) -> np.ndarray:
    """Convert gray / prob map to {0,1} uint8 mask."""
    if img.dtype != np.uint8: # Assuming input can be float probability map
        if np.max(img) <= 1.0 and np.min(img) >=0.0: # Probability map [0,1]
             img = (img * 255).astype(np.uint8)
        else: # Grayscale image already in 0-255 range but not uint8
            img = img.astype(np.uint8)

    # Binarise to 0 or 1
    binary_mask = (img >= int(th * 255)).astype(np.uint8)
    return binary_mask


def fast_hist(label_pred, label_true, num_classes=2):
    """
    Replicate DocTamperMetric's _fast_hist method
    """
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        (num_classes * label_true[mask].astype(int) + label_pred[mask]).astype(int),
        minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def per_image_metrics_calculator(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculates IoU, Precision, Recall, F1 for a single image's TP, FP, FN."""
    iou       = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * tp / (2 * tp + fp + fn + 1e-7)
    return dict(iou=iou, precision=precision, recall=recall, f1=f1)


def evaluate_pair(args_tuple: Tuple[str, str, float]) -> Tuple[str, np.ndarray, float, float, float, float]:
    """
    Worker function: Calculates metrics for a single image pair.
    Returns: (filename, hist_img, precision_img, recall_img, iou_img_for_csv, f1_img_for_csv)
    """
    pred_path, gt_path, th = args_tuple
    name = os.path.basename(pred_path)

    gt_img   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt_img is None:
        # Try adding common image extensions if gt_path doesn't have one and is not found
        found_gt = False
        if '.' not in os.path.basename(gt_path):
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                temp_gt_path = gt_path + ext
                if os.path.exists(temp_gt_path):
                    gt_img = cv2.imread(temp_gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt_img is not None:
                        found_gt = True
                        break
        if not found_gt:
             raise FileNotFoundError(f"GT file not found or unreadable: {gt_path} (for pred: {name})")

    if pred_img is None:
        raise FileNotFoundError(f"Prediction file not found or unreadable: {pred_path}")

    # If dimensions differ, crop to the overlapping region
    if gt_img.shape != pred_img.shape:
        h = min(gt_img.shape[0], pred_img.shape[0])
        w = min(gt_img.shape[1], pred_img.shape[1])
        gt_img, pred_img = gt_img[:h, :w], pred_img[:h, :w]

    # Ensure GT is binary (0 or 1)
    gt_binary = (gt_img > 0).astype(np.uint8) # Assuming any pixel > 0 in GT is foreground
    # Binarise prediction
    pred_binary = binarise(pred_img, th)

    # Use DocTamperMetric's fast_hist method
    hist_img = fast_hist(pred_binary.flatten(), gt_binary.flatten(), num_classes=2)

    # Per-image precision and recall using DocTamperMetric logic exactly
    matched = (pred_binary * gt_binary).sum()
    pred_sum = pred_binary.sum()
    target_sum = gt_binary.sum()
    
    # DocTamperMetric uses .mean().item() but for single image, mean of scalar is itself
    precision_img = (matched / (pred_sum + 1e-8)) if pred_sum > 0 else 0.0
    recall_img = (matched / target_sum) if target_sum > 0 else 0.0

    # Per-image metrics for CSV (using confusion matrix values)
    TP = hist_img[1, 1]
    FP = hist_img[0, 1] 
    FN = hist_img[1, 0]
    
    csv_metrics = per_image_metrics_calculator(int(TP), int(FP), int(FN))
    iou_img_for_csv = csv_metrics['iou']
    f1_img_for_csv  = csv_metrics['f1']

    return name, hist_img, precision_img, recall_img, iou_img_for_csv, f1_img_for_csv


# ──────────────────────────────── Main ─────────────────────────────────────
def parse_args():
    """Parse command-line arguments (pred/gt dirs, threshold, workers, CSV)."""
    parser = argparse.ArgumentParser(description="Binary-mask metrics (multiprocessing)")
    # parser.add_argument("--pred_dir", help="directory with predicted PNG masks",default='/data1/yzq/code/IMDLBenCo-main-0713/eval_dir_mesorch_f1/pred_test_yashuo')
    parser.add_argument("--pred_dir", help="directory with predicted PNG masks",default='/data1/yzq/code/IMDLBenCo-main-0713/save_img_dir_mvss/pred_test')
    # parser.add_argument("--gt_dir",   help="directory with GT PNG masks",default='/data1/yzq/code/visual_on_doctamper/result_mesorch_f1/pred_mask_test')
    # parser.add_argument("--gt_dir",   help="directory with GT PNG masks",default='/data3/yzq/data/DocTamperV1/DocTamperV1-SCD/mask_v2')
    parser.add_argument("--gt_dir",   help="directory with GT PNG masks",default='/data3/yzq/data/DocTamperV1/DocTamperV1-TestingSet/mask_v2')
    parser.add_argument("--save_csv", default=None,  help="optional csv file for per-image metrics")
    parser.add_argument("--th",       type=float, default=0.5,
                        help="binarisation threshold if predictions are probability maps")
    parser.add_argument("--workers",  type=int, default=12,
                        help="number of worker processes (default: cpu count)")
    return parser.parse_args()


def main():
    """Evaluate all pred/gt pairs in parallel, accumulate the global histogram,
    compute DocTamperMetric-style overall metrics, print, and optionally save CSV."""
    args = parse_args()

    pred_list = sorted(glob.glob(os.path.join(args.pred_dir, "*.png")))
    if not pred_list:
        # Try other common extensions if no PNGs found
        for ext in ["*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            pred_list.extend(glob.glob(os.path.join(args.pred_dir, ext)))
        pred_list = sorted(list(set(pred_list))) # Remove duplicates and sort

    if not pred_list:
        raise RuntimeError(f"No compatible image files (PNG, JPG, BMP, TIF) found in pred_dir: {args.pred_dir}")

    tasks: List[Tuple[str, str, float]] = []
    for pred_path in pred_list:
        name_with_ext = os.path.basename(pred_path)
        name_no_ext = Path(name_with_ext).stem # Get filename without extension

        # Attempt to find GT mask by matching filename without extension
        # Common GT extensions can be different from pred extensions
        gt_path_found = None
        for gt_ext in ["*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]: # Add more if needed
            potential_gt_name = name_no_ext + Path(gt_ext).suffix # e.g., image1.png
            gt_path_candidate = os.path.join(args.gt_dir, potential_gt_name)
            if os.path.exists(gt_path_candidate):
                gt_path_found = gt_path_candidate
                break
        
        if not gt_path_found: # Fallback to exact name match if stem matching fails
             gt_path_candidate = os.path.join(args.gt_dir, name_with_ext)
             if os.path.exists(gt_path_candidate):
                 gt_path_found = gt_path_candidate

        if not gt_path_found:
            print(f"[WARN] GT not found for prediction '{name_with_ext}' (searched for '{name_no_ext}.*'), skipped.")
            continue
        tasks.append((pred_path, gt_path_found, args.th))

    if not tasks:
        raise RuntimeError("No valid image pairs to evaluate.")

    # Initialize accumulators
    global_hist = np.zeros((2, 2), dtype=np.int64)
    list_per_image_precision: List[float] = []
    list_per_image_recall: List[float] = []
    per_image_metrics_for_csv: List[Tuple[str, float, float, float, float]] = [] # name, iou, precision, recall, f1

    # Multiprocessing
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_pair, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            try:
                name, hist_img, precision_img, recall_img, iou_img_csv, f1_img_csv = fut.result()
                
                global_hist += hist_img
                list_per_image_precision.append(precision_img)
                list_per_image_recall.append(recall_img)
                per_image_metrics_for_csv.append((name, iou_img_csv, precision_img, recall_img, f1_img_csv))
                
                print(f"Processed {i+1}/{len(tasks)}: {name}", end='\r')

            except FileNotFoundError as e:
                print(f"\n[ERROR] Skipping a pair due to file error: {e}")
            except Exception as e:
                print(f"\n[ERROR] Error processing a pair: {e}")
    print("\nProcessing complete.")

    if not per_image_metrics_for_csv:
        print("No images were successfully processed. Exiting.")
        return

    # Calculate overall metrics based on DocTamperMetric logic exactly
    # DocTamperMetric uses: acc = np.diag(self.hist).sum() / self.hist.sum()
    # But the returned 'acc' is actually acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
    
    # acc_cls: Class-balanced accuracy
    acc_cls = np.nanmean(np.diag(global_hist) / global_hist.sum(axis=1))
    
    # iou: IoU calculation - iu[1] where iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
    iu = np.diag(global_hist) / (global_hist.sum(axis=1) + global_hist.sum(axis=0) - np.diag(global_hist))
    iou_class1 = iu[1]  # IoU for class 1 (foreground)

    # precision & recall: Mean of per-image values (same as DocTamperMetric)
    mean_precision_overall = np.array(list_per_image_precision).mean() if list_per_image_precision else 0.0
    mean_recall_overall = np.array(list_per_image_recall).mean() if list_per_image_recall else 0.0

    # f1: Calculated from mean_precision_overall and mean_recall_overall (same as DocTamperMetric)
    f1_overall = (2 * mean_precision_overall * mean_recall_overall) / \
                 (mean_precision_overall + mean_recall_overall + 1e-8)

    # Extract TP, TN, FP, FN for display
    TP = int(global_hist[1, 1])
    TN = int(global_hist[0, 0]) 
    FP = int(global_hist[0, 1])
    FN = int(global_hist[1, 0])

    print("\n=================================================")
    print(f"Images evaluated : {len(per_image_metrics_for_csv)}")
    print("Overall metrics (DocTamperMetric style):")
    print(f"  acc (Class-balanced) : {acc_cls*100:6.2f}%")
    print(f"  iou (Class 1)        : {iou_class1*100:6.2f}%")
    print(f"  precision (Mean)     : {mean_precision_overall*100:6.2f}%")
    print(f"  recall (Mean)        : {mean_recall_overall*100:6.2f}%")
    print(f"  f1 (Mean)            : {f1_overall*100:6.2f}%")
    print("-------------------------------------------------")
    print(f"  TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print("=================================================")

    # Save per-image metrics to CSV
    if args.save_csv:
        # Sort by filename for consistent output
        per_image_metrics_for_csv.sort(key=lambda x: x[0])
        with open(args.save_csv, "w", newline="") as fp_csv:
            writer = csv.writer(fp_csv)
            writer.writerow(["filename", "iou_img", "precision_img", "recall_img", "f1_img"])
            for row in per_image_metrics_for_csv:
                writer.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])
        print(f"Per-image metrics saved to {args.save_csv}")


if __name__ == "__main__":
    main()