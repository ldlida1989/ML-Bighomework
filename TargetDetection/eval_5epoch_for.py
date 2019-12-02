#! -*- coding:utf-8 -*-
"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import pickle
import shutil
import sys
import warnings
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data

from data import BaseTransform
from data import SIXray_CLASSES as labelmap
from data import SIXray_ROOT, SIXrayAnnotationTransform, SIXrayDetection
from ssd import build_ssd

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


warnings.filterwarnings("ignore")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")


EPOCH = 5
GPUID = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = GPUID


# 解析命令行参数
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default="weights", type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default="res", type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.2, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--SIXray_root', default=SIXray_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--imagesetfile',
                    default="data/sub_test_core_coreless.txt", type=str,
                    help='imageset file path to open')
args = parser.parse_args()

# 使用cuda
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# 图片和标签路径
annopath = os.path.join(
    args.SIXray_root, 'Annotation', '%s.txt')
imgpath = os.path.join(
    args.SIXray_root, 'Image', '%s.jpg')

devkit_path = args.save_folder
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename, imgpath):
    """ 解析标注文件，返回字典列表 """
    objects = []

    # 打开图像，读入图像大小
    img = cv2.imread(imgpath)
    height, width, channels = img.shape

    with open(filename.encode('utf-8'), "r", encoding='utf-8') as f1:
        dataread = f1.readlines()
        for annotation in dataread:
            obj_struct = {}
            temp = annotation.split()
            name = temp[1]
            if name != '带电芯充电宝' and name != '不带电芯充电宝':
                continue
            xmin = int(temp[2])
            # 只读取V视角的
            if int(xmin) > width:
                continue
            if xmin < 0:
                xmin = 1
            ymin = int(temp[3])
            if ymin < 0:
                ymin = 1
            xmax = int(temp[4])
            if xmax > width:
                xmax = width - 1
            ymax = int(temp[5])
            if ymax > height:
                ymax = height - 1
            # name
            obj_struct['name'] = name
            obj_struct['pose'] = 'Unspecified'
            obj_struct['truncated'] = 0
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [float(xmin) - 1,
                                  float(ymin) - 1,
                                  float(xmax) - 1,
                                  float(ymax) - 1]
            objects.append(obj_struct)
    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


# 返回结果的文件名，格式如示例
def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


# 将预测结果写入到文件中，txt格式
def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename.encode('utf-8'), 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects a1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


# 评估结果：计算mAP
def do_python_eval(output_dir='output', use_07=False):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i, cls in enumerate(labelmap):
        # 读取之前保存的结果文件
        filename = get_voc_results_file_template(set_type, cls)
        # 计算AP
        rec, prec, ap = voc_eval(
            filename, annopath, imgpath, args.imagesetfile, cls, cachedir,
            ovthresh=0.2, use_07_metric=use_07_metric)
        aps += [ap]
        with open(os.path.join(output_dir, str(cls.encode('utf-8')) + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print("EPOCH, {:d}, mAP, {:.4f}, core_AP, {:.4f}, coreless_AP, {:.4f}".format(
        EPOCH, np.mean(aps), aps[0], aps[1]))


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# 评估结果
def voc_eval(detpath,
             annopath,
             imgpath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
    detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
    annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # 标注读取
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % imagename, imgpath % imagename)
        # save
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]

        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    
    # 预测结果读取
    detfile = detpath.format(classname)
    with open(detfile.encode('utf-8'), 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            BBGT = R['bbox'].astype(float)
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)

            if ovmax > ovthresh:
                tp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


# 评估网络
def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all_boxes shape： (num_images+1, num_images)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    print('output_dir', output_dir)
    det_file = os.path.join(output_dir, 'detections.pkl')

    # 预测每一个测试图片
    for i in range(num_images):
        im, gt, h, w, og_im = dataset.pull_item(i)
        # 这里im的颜色偏暗，因为BaseTransform减去了一个mean

        # print(im_det)
        x = torch.tensor(im.unsqueeze(0))

        if args.cuda:
            x = x.cuda()
        
        # 预测，预测结果为置信度和坐标
        detections = net(x).data

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            # 缩放回原始图像尺寸
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            
            scores = dets[:, 0].cpu().numpy()
            boxes_np = boxes.cpu().numpy()
            sc_ind = np.argsort(-scores, axis=0)
            scores = np.array(scores[sc_ind[0]])
            cls_dets = np.hstack((boxes_np[sc_ind[0], :], scores)).astype(np.float32, copy=False).reshape(-1, 5)
            all_boxes[j][i] = cls_dets

        if i % 10 == 0:
            print('评估{:4}/{}完成'.format(i+1, num_images))

    # 将预测结果保存到文件中
    with open(det_file.encode('utf-8'), 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    # with open(det_file, 'rb') as f:
    #     all_boxes = pickle.load(f)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


# 评估预测结果
def evaluate_detections(box_list, output_dir, dataset):
    # 将结果写入到文件
    write_voc_results_file(box_list, dataset)
    # 评估结果
    do_python_eval(output_dir)


# 设置模型路径和保存路径
def reset_args(EPOCH):
    global args
    args.trained_model = "weights/ssd300_Xray20190723_{:d}.pth".format(
        EPOCH)
    saver_root = 'saver/'
    if not os.path.exists(saver_root):
        os.mkdir(saver_root)
    args.save_folder = saver_root + '{:d}epoch_500/'.format(EPOCH)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    else:
        shutil.rmtree(args.save_folder)
        os.mkdir(args.save_folder)

    global devkit_path
    devkit_path = args.save_folder


if __name__ == '__main__':
    EPOCHS = [45700]
    print(EPOCHS)

    for EPOCH in EPOCHS:
        reset_args(EPOCH)

        # 加载网络
        num_classes = len(labelmap) + 1  # +a1 for background
        net = build_ssd('test', 300, num_classes)  # initialize SSD
        # 加载模型参数
        net.load_state_dict(torch.load(args.trained_model))

        # 加载数据
        dataset = SIXrayDetection(args.SIXray_root, args.imagesetfile,
                                  BaseTransform(300, dataset_mean),
                                  SIXrayAnnotationTransform())
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True

        # 评估
        net.eval()
        test_net(args.save_folder, net, args.cuda, dataset,
                 BaseTransform(net.size, dataset_mean), args.top_k, 300,
                 thresh=args.confidence_threshold)
