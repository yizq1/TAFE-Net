"""
Evaluating RTM, save statistics, plot results
"""
import os
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

import concurrent.futures
from pqdm.processes import pqdm
import time

cate_mapper = {
    'cpmv': 'cpmv',
    'insert': 'insertion',
    'splice': 'splicing',
    'edit': 'editing',
    'inpaint': 'inpainting',
    'cover': 'cover',
    'good': 'good',
    'manual': 'manual',
    'all': 'all'
}

cate_spliter = {
    'cpmv': 'manual',
    'insert': 'manual',
    'splice': 'manual',
    'edit': 'manual',
    'inpaint': 'manual',
    'cover': 'manual',
    'good': 'good',
}

tamper_classes = ['cpmv', 'insertion', 'splicing', 'editing', 'inpainting', 'cover']

NJOBS = 12
protocols_pixel = ['iou', 'f1', 'cf1', 'mae']
protocols_image = ['cf1']
protocols_region = ['dqp', 'dqg', 'mq']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Full evaluation Metircs between binary masks')
    parser.add_argument('--pred_dir', help='pred mask dir')
    parser.add_argument('--save_dir', help='save result in dir')
    parser.add_argument('--text', type=str, default=None, help='restrict regions with extra text location')
    args = parser.parse_args()
    return args


def harmony(x, y):
    return 2 * x * y / (x + y + 1e-7)

def iou_meaure(tp, fp, fn):
    return tp / (tp + fp + fn + 1e-12)

def f_measure(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn + 1e-6)

def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)

def evaluate_matrix(matrix):
    tp, fp, fn, tn = matrix['tp'], matrix['fp'], matrix['fn'], matrix['tn']
    iou = iou_meaure(tp, fp, fn)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = f_measure(tp, fp, fn)

    results = dict(iou=iou, precision=precision, recall=recall, f1=f1)
    return results

def sum_dict(a, b):
    """sum dict1 and dict for all values, if only in one dict, merge keys"""
    temp = dict()
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])

    return temp

def sum_confusion_matrix(a, b):
    for key in a:
        a[key] = sum_dict(a[key], b[key])
    return a



def load_result(path):
    with open(path, 'r') as fp:
        result = json.load(fp)

    return result


class BinaryMetirc(object):
    """Pixel-level evaluator for binary tampering masks on the RTM dataset.

    Compares predicted masks against ground truth, computing per-image
    confusion matrices and IoU/Precision/Recall/F1, then aggregates the
    results by manipulation type for reporting.
    """
    def __init__(self, method, gt_dir, pred_dir, save_dir=None, mode=['iou', 'f1'], threshold=0.5):
        """Store paths/config and collect the sorted list of prediction PNGs.

        Args:
            method: Method name used in output filenames.
            gt_dir: Directory holding ground-truth masks (matched by filename).
            pred_dir: Directory holding predicted masks (``*.png``).
            save_dir: Output directory; defaults to the parent of ``pred_dir``.
            mode: Metrics to compute per image (e.g. ``['iou', 'f1']``).
            threshold: Binarization threshold for predictions.
        """

        self.method = method
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.threshold = threshold
        self.mode = mode

        if save_dir is None:
            self.save_dir = osp.split(pred_dir)[0]
        else:
            self.save_dir = save_dir

        self.result = []
        self.all_confusion_matrix = dict(tamp=dict(tp=0, fp=0, fn=0, tn=0),
                                         all=dict(tp=0, fp=0, fn=0, tn=0))

        self.pred_list = list(glob(osp.join(pred_dir, '*.png')))
        self.pred_list.sort()
        print("Test image number: ", self.pred_list.__len__())


    def eval_full(self):
        """Evaluate every prediction in parallel and aggregate the matrices.

        Dispatches ``eval_single`` over all predictions via ``pqdm``, collects
        per-image results, sums their confusion matrices, then reports the
        global tampered-only ('tamp') and all-pixels ('all') metrics.
        """
        self.result = []
        self.all_confusion_matrix = dict(tamp=dict(tp=0, fp=0, fn=0, tn=0),
                                         all=dict(tp=0, fp=0, fn=0, tn=0))

        args_pqdm = []
        for i in self.pred_list:
            args_pqdm.append([i, self.mode, self.gt_dir, self.threshold])
        print("Begin evaluation with multiple Processes")

        results = pqdm(args_pqdm, self.eval_single, n_jobs=NJOBS, argument_type='args')
        print("Valid result number: ", results.__len__())
        # res = [per_image_result_dict, per_image_confusion_matrix]; accumulate both.
        for res in results:
            try:
                self.result.append(res[0])
                self.all_confusion_matrix = sum_confusion_matrix(self.all_confusion_matrix, res[1])
            except:
                print(res)

        self.metric_all_images()
        self.metric_all_images(key='all')


    def metric_all_images(self, key='tamp', metric=['iou', 'f1']):
        """Compute and print micro-averaged IoU/F1 from the summed confusion matrix.

        Uses pixel counts pooled over all images (``key`` selects 'tamp' or
        'all'), so it reflects dataset-wide pixels rather than a per-image mean.
        """
        result_all = dict()
        for m in metric:
            if m == 'iou':
                ans_all = self.all_confusion_matrix[key]['tp'] / \
                          (self.all_confusion_matrix[key]['tp'] + self.all_confusion_matrix[key]['fp'] + self.all_confusion_matrix[key]['fn'] +1e-6)
                # result_all['m' + m] = ans_all
            if m == 'f1':
                ans_all = (2 * self.all_confusion_matrix[key]['tp']) / \
                          (2 * self.all_confusion_matrix[key]['tp'] + self.all_confusion_matrix[key]['fp'] + self.all_confusion_matrix[key]['fn'] +1e-6)
            result_all['m' + m] = ans_all

        print(result_all)


    def load_masks(self, pred_path):
        """Load a pred/gt mask pair (matched by filename) and binarize to {0, 1}."""
        file_name = osp.basename(pred_path)
        gt_path = osp.join(self.gt_dir, file_name)

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        assert pred.shape == gt.shape

        pred = pred // 255
        gt = gt // 255

        return gt, pred


    # @classmethod
    def eval_single(self, pred_path, mode, gt_dir, threshold):
        """Evaluate one prediction against its ground truth.

        Loads and binarizes the pred/gt pair (cropping pred to gt shape on
        mismatch), builds the pixel confusion matrix, and returns
        ``[result_dict, confusion_matrix]``. Authentic ('good') images carry no
        tampered pixels, so their localization metrics are forced to 0 and only
        the image-level ``score`` is meaningful.
        """

        file_name = osp.basename(pred_path)
        gt_path = osp.join(gt_dir, file_name)

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        try:
            assert pred.shape == gt.shape
        except:
            h, w = gt.shape
            pred = pred[:h, :w]
            assert pred.shape == gt.shape

        gt[gt>0] = 255

        pred = pred // 255

        gt = gt // 255

        result_single = dict()
        all_confusion_matrix = dict(tamp=dict(tp=0, fp=0, fn=0, tn=0),
                                    all=dict(tp=0, fp=0, fn=0, tn=0))

        result_single['filename'] = file_name
        # Flatten masks to 1-D and tally pixel-wise TN/FP/FN/TP.
        tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten(), labels=[0, 1]).ravel()

        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

        all_confusion_matrix['all']['tp'] = tp
        all_confusion_matrix['all']['fp'] = fp
        all_confusion_matrix['all']['fn'] = fn
        all_confusion_matrix['all']['tn'] = tn

        result_single.update(dict(tp=tp, fp=fp, fn=fn, tn=tn))

        if osp.splitext(file_name)[0].split('_')[0] == 'good':
            if 'iou' in mode:
                result_single.update({'iou': 0})
            if 'f1' in mode:
                result_single.update({'f1': 0})
            if 'dq' in mode:
                result_single.update({'num_gt': 0, 'hit_gt': 0, 'hit_pred': 0})
                retval, _, _, _ = cv2.connectedComponentsWithStats(pred)
                result_single.update({'num_pred': retval-1})

            result_single.update({'score': int(pred.max())})
            return [result_single, all_confusion_matrix]

        else:
            # print(file_name)
            # tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten(), labels=[0, 1]).ravel()
            all_confusion_matrix['tamp']['tp'] = tp
            all_confusion_matrix['tamp']['fp'] = fp
            all_confusion_matrix['tamp']['fn'] = fn
            all_confusion_matrix['tamp']['tn'] = tn

            # result_single.update(dict(tp=tp, fp=fp, fn=fn, tn=tn))

            if 'iou' in mode:

                # ans = self.cal_iou(gt.flatten(), pred.flatten())
                # IoU = TP / (TP + FP + FN); eps avoids divide-by-zero.
                ans = tp / (tp + fp + fn + 1e-6)
                result_single.update({'iou':ans})
                # result_single[mode] = iou

            if 'f1' in mode:
                # ans = self.cal_f1(gt.flatten(), pred.flatten())
                # F1 = 2*TP / (2*TP + FP + FN); eps avoids divide-by-zero.
                ans = (2 * tp) / ( 2 * tp + fp + fn + 1e-6)
                result_single.update({'f1': ans})

            if 'mae' in mode:
                ans = (fp + fn) / (tp + tn + fp + fn)
                result_single.update({'mae': ans})


            result_single.update({'score': int(pred.max())})

            return [result_single, all_confusion_matrix]


    def cal_iou(self, gt, pred):
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0,1]).ravel()
        iou = tp / (tp + fp + fn + 1e-6)
        return {'iou':iou}

    def cal_f1(self, gt, pred):
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0,1]).ravel()
        f1 = (2 * tp) / ( 2 * tp + fp + fn + 1e-6)

        return {'f1':f1}

    def cal_pixel(self, gt, pred):
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0,1]).ravel()
        iou = tp / (tp + fp + fn + 1e-6)
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        ans = {'iou':iou, 'f1':f1}

        return ans

    def sort(self, mode='iou', reverse=True):
        self.result.sort(key=lambda x:x[mode], reverse=reverse)

    def save_result(self):
        """Dump per-image results and the aggregated confusion matrix to JSON."""

        print("tamperd image num: ", self.result.__len__())
        print("save result in [{}]".format(self.save_dir))
        with open(osp.join(self.save_dir, "result_{}.json".format(self.method)), mode='w') as fp:
            json.dump([self.result, self.all_confusion_matrix], fp, indent=2, separators=(',',':'))


    def show_result(self, mode):
        """Aggregate per-image results by manipulation type and print/save tables.

        Buckets each sample by its manipulation category (via ``cate_mapper``),
        accumulating both per-class pixel confusion matrices and per-class
        localization scores, plus rolled-up 'manual'/'tamp'/'all' groups. Emits
        three PrettyTables (main localization, pixel-level, image-level accuracy)
        and writes them to ``ans_{method}{ocr_suffix}.txt``.
        """

        print('==============================')
        print('Metrics: ', mode)
        print('==============================')

        # initial meter
        loc_meter = dict()  # pixel and instance
        cls_meter = dict()  # classification
        pix_meter = dict()  # pixel level metric following semantic segmentation
        for k in tamper_classes:
            pix_meter[k] = dict(tp=0, tn=0, fp=0, fn=0)
        for k in ['manual', 'script', 'tamp', 'all', 'good']:
            pix_meter[k] = dict(tp=0, tn=0, fp=0, fn=0)

        for k in cate_mapper:
            loc_meter[cate_mapper[k]] = dict()
            cls_meter[cate_mapper[k]] = dict(t=0, tp=0, fp=0, fn=0, tn=0, num=0, f1=0, cf1=0)

            for m in mode:
                if m == 'iou' or m == 'f1':
                    loc_meter[cate_mapper[k]][m] = []
                    loc_meter[cate_mapper[k]].update({'avg_{}'.format(m): 0})

        # start analyze
        for sample in self.result:
            # Manipulation type is the filename prefix, normalized via cate_mapper.
            name = sample['filename'].split('_')[0]
            # cate = cate_spliter[name]
            name = cate_mapper[name]

            # update pixel meter
            for key in pix_meter[name]:

                pix_meter[name][key] += sample[key]
                pix_meter['all'][key] += sample[key]
                if name != 'good':
                    pix_meter['tamp'][key] += sample[key]
                    if name != 'script':
                        pix_meter['manual'][key] += sample[key]

            for m in mode:
                if m == 'iou' or m == 'f1':
                    loc_meter[name][m].append(sample[m])

                    if name == 'good':
                        pass
                    else:
                        if name == 'script':
                            pass
                        else:
                            loc_meter['manual'][m].append(sample[m])
                        loc_meter['all'][m].append(sample[m])


            cls_meter[name]['num'] += 1

            if name == 'good':
                if sample['score'] == 0:
                    cls_meter[name]['t'] += 1

            else:
                if sample['score'] > 0:
                    cls_meter[name]['t'] += 1


        for name in loc_meter:
            for m in mode:
                if m == 'iou' or m == 'f1':
                    loc_meter[name]['avg_{}'.format(m)] = np.mean(loc_meter[name][m])

        for name in cls_meter:
            if name == 'all' or name == 'manual':
                continue

            cls_meter['all']['num'] += cls_meter[name]['num']
            cls_meter['all']['t'] += cls_meter[name]['t']
            if name != 'script' and name != 'good':
                cls_meter['manual']['num'] += cls_meter[name]['num']
                cls_meter['manual']['t'] += cls_meter[name]['t']

            if name == 'good':
                cls_meter['all']['tn'] = cls_meter[name]['t']
                cls_meter['all']['fp'] = cls_meter[name]['num'] - cls_meter[name]['t']
            else:
                cls_meter['all']['tp'] += cls_meter[name]['t']
                cls_meter['all']['fn'] += cls_meter[name]['num'] - cls_meter[name]['t']


        cls_meter['all']['f1'] = f_measure(cls_meter['all']['tp'], cls_meter['all']['fp'], cls_meter['all']['fn'])


        # initial output
        proto = ['Cate']
        if 'iou' in mode:
            proto.append('iou')
        if 'f1' in mode:
            proto.append('f1')

        proto.extend(['Image f1'])


        # show evaluation results
        tb = PrettyTable(proto)
        tb_pix = PrettyTable(['Cate', 'IoU', 'Precision', 'Recall', 'F1'])
        tb_img = PrettyTable(['Cate', 'Max Hit', 'nRate'])


        tb.title = 'Main Metric'
        tb_img.title = 'Image-level Acc'
        tb_pix.title = 'Pixel-level'

        for name in loc_meter:
            tb.add_row([
                name,
                '{:.2f}%'.format(loc_meter[name]['avg_iou'] * 100), '{:.2f}%'.format(loc_meter[name]['avg_f1'] * 100),
                '{:.2f}%'.format(cls_meter[name]['f1'] * 100) if name == 'all' else '-',])

            tb_img.add_row([name,
                            '{}/{}'.format(cls_meter[name]['t'], cls_meter[name]['num']),
                            '{:.2f}%'.format(cls_meter[name]['t'] / cls_meter[name]['num'] * 100)])


        # additional metric
        tb.add_row(['m_tamp',
                    '{:.2f}%'.format(
                        iou_meaure(self.all_confusion_matrix['tamp']['tp'], self.all_confusion_matrix['tamp']['fp'],
                                   self.all_confusion_matrix['tamp']['fn']) * 100),
                    '{:.2f}%'.format(
                        f_measure(self.all_confusion_matrix['tamp']['tp'], self.all_confusion_matrix['tamp']['fp'],
                                  self.all_confusion_matrix['tamp']['fn']) * 100),
                    '-'])
        tb.add_row(['m_all',
                    '{:.2f}%'.format(
                        iou_meaure(self.all_confusion_matrix['all']['tp'], self.all_confusion_matrix['all']['fp'],
                                   self.all_confusion_matrix['all']['fn']) * 100),
                    '{:.2f}%'.format(
                        f_measure(self.all_confusion_matrix['all']['tp'], self.all_confusion_matrix['all']['fp'],
                                  self.all_confusion_matrix['all']['fn']) * 100),
                    '-'])

        tb_img.add_row(['P&R',
                        '{:.2f}%'.format(
                            cls_meter['all']['tp'] / (cls_meter['all']['tp'] + cls_meter['all']['fp'] + 1e-12) * 100),
                        '{:.2f}%'.format(
                            cls_meter['all']['tp'] / (cls_meter['all']['tp'] + cls_meter['all']['fn'] + 1e-12) * 100)])

        tb_img.add_row(['Acc',
                        '-',
                        '{:.2f}%'.format(
                            accuracy(cls_meter['all']['tp'], cls_meter['all']['fp'], cls_meter['all']['fn'],
                                     cls_meter['all']['tn']) * 100)])

        for name in pix_meter:
            pix_results = evaluate_matrix(pix_meter[name])
            tb_pix.add_row([name,
                            '{:.2f}%'.format(pix_results['iou'] * 100),
                            '{:.2f}%'.format(pix_results['precision'] * 100),
                            '{:.2f}%'.format(pix_results['recall'] * 100),
                            '{:.2f}%'.format(pix_results['f1'] * 100)])

        print(tb)
        print(tb_pix)
        print(tb_img)

        with open(osp.join(self.save_dir, 'ans_{}{}.txt'.format(self.method, self.ocr_suffix)), 'w') as fp:
            fp.write(str(tb))
            fp.writelines('\n\n')
            fp.write(str(tb_pix))
            fp.writelines('\n\n')
            fp.write(str(tb_img))
            fp.writelines('\n\n')

        print('=================================')

    def load_json(self, path):
        """Restore cached results/matrices from a saved JSON, re-deriving the method name."""
        # self.result, self.all_confusion_matrix = load_result(path)
        self.result, self.all_confusion_matrix = load_result(path)
        filename = osp.basename(path)
        filename = osp.splitext(filename)[0]
        segments = filename.split('_')

        # print(segments)
        self.method = segments[1]


        print('text images num: ', self.result.__len__())






if __name__ == "__main__":


    mode = ['iou', 'f1', 'mae']


    # pred_dir = './work_dirs/ascformer_rtm_qwen2_1/result/pred_mask'
    pred_dir='/data1/yzq/code/IMDLBenCo-main-0713/eval_dir_mesorch_f1_rtm/pred'
    gt_dir = '/data3/yzq/data/LLM_datasets/RealTextManipulation/SegmentationClass'

    if pred_dir[-1] == '/':
        pred_dir = pred_dir[:-1]
    method_name = osp.split(pred_dir)[-1]
    method_name = method_name.split('_')[0]


    evaluation = BinaryMetirc(method_name, gt_dir, pred_dir, mode=mode)

    print("Evaluating [ {} ]".format(evaluation.method))
    evaluation.eval_full()
    evaluation.save_result()
    print("{} evaluation finished".format(evaluation.method))
    evaluation.show_result(mode=['iou', 'f1'])





