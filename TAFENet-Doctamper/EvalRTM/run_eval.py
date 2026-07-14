"""
CLI entry point for the standalone offline RTM evaluation toolkit.

Scores predicted binary tampering masks against ground-truth masks after
inference: builds a ``BinaryMetirc`` evaluator, runs the full evaluation,
dumps a JSON result, and prints per-manipulation-type metric tables.

Usage:
    python run_eval.py --pred_dir <PredMask_dir> --gt_dir <gt_dir> [--save_dir <out_dir>]

The prediction directory must be named ``{MethodName}_mask``; the method name
is derived from that folder name.
"""
from eval_rtm import BinaryMetirc
from os import path as osp

from pathlib import Path
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

from prettytable import PrettyTable
import json
import argparse


def parse_args():
    """Parse CLI args: prediction dir, ground-truth dir, and optional output dir."""
    parser = argparse.ArgumentParser(
        description='Full evaluation Metircs between binary masks')
    parser.add_argument('--pred_dir', help='pred mask dir')
    parser.add_argument('--gt_dir', help='save result in dir')
    parser.add_argument('--save_dir', help='save result in dir', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    pred_dir = args.pred_dir

    if pred_dir[-1] == '/':
        pred_dir = pred_dir[:-1]

    mode = ['iou', 'f1']
    use_post = False

    gt_dir = args.gt_dir
    save_dir = args.save_dir


    # Derive the method name from the prediction folder named "{MethodName}_mask".
    method_name = osp.basename(pred_dir)
    method_name = method_name.split('_')[0]

    evaluation = BinaryMetirc(method_name, gt_dir, pred_dir, save_dir=save_dir, mode=mode)
    print("Evaluating [ {} ]".format(evaluation.method))


    # eval_full: compute per-image confusion matrices/metrics in parallel.
    evaluation.eval_full()
    # save_result: dump per-image results and aggregated matrices to JSON.
    evaluation.save_result()

    print("{} evaluation finished".format(evaluation.method))

    # evaluation.load_json(osp.join(evaluation.save_dir, 'result_{}.json'.format(method_name)))
    # show_result: aggregate per-manipulation-type metrics and print tables.
    evaluation.show_result(mode=['iou', 'f1'])

    print("The result of [ {} ]".format(evaluation.method))

