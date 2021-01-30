from __future__ import division
import argparse
import os.path as osp
import shutil
import time

import mmcv
from mmcv import Config

from mmdet.datasets import build_dataset
from app import *
import torch

import json
from mmdet.core.post_processing import bbox_iou

def parse_args():
    parser = argparse.ArgumentParser(description='Test anchor match')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')

    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is None:
        args.work_dir = cfg.work_dir
    
    if not osp.exists(osp.join(args.work_dir, 'anchors.json')):
        base_sizes = [4, 8, 16, 32]
        anchors = torch.empty([0,2])
        for base_size in base_sizes:
            anchors = torch.cat((anchors, gen_base_anchors(base_size)))
    else:
        with open(osp.join(args.work_dir, 'anchors.json'), 'r') as f:
            anchors = torch.Tensor(json.load(f)['base_anchors'])
    
    cfg.data.train['quiet'] = True
    dataset = build_dataset(cfg.data.train)
    gt_boxes_all = dataset.load_transformed_gt_info()
    boxes_all = []
    for key in gt_boxes_all:
        boxes_all += gt_boxes_all[key]
    zeros = torch.zeros(anchors.size())
    anchors_f = torch.cat((zeros, anchors), dim=1).cuda(0)
    # import pdb;
    # pdb.set_trace()
    zeros = torch.zeros((len(boxes_all), 2))
    boxes_f = torch.cat((zeros, torch.Tensor(boxes_all)), dim=1).cuda(0)
    distance = torch.min(1 - bbox_iou(boxes_f, anchors_f), dim=1).values
    D = torch.mean(distance).tolist()
    V = torch.var(distance).tolist()
    M = 1 / (0.6 * V + 0.4 * D)
    print("D: %f, V: %f, M: %f" % (D, V, M))

def gen_base_anchors(base_size):
    w = base_size
    h = base_size
    scales = torch.Tensor([8])
    ratios = torch.Tensor([0.5, 1.0, 2.0])
    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :])
    hs = (h * h_ratios[:, None] * scales[None, :])
    return torch.cat((ws, hs), dim=1)

if __name__ == "__main__":
    main()




