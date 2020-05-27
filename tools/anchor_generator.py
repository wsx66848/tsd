from __future__ import division
import argparse
import os.path as osp
import shutil
import time

import mmcv
from mmcv import Config

from mmdet.datasets import build_dataset

from app import *
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Generate anchors by k-means')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--timestamp', help='the timestamp when starting training')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # check the type of the detector
    if isinstance(cfg.model, dict) and cfg.model.get('type', 'FasterRCNN') != 'MyFasterRCNN':
        return

    # set cudnn_benchmark
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    cur_timestamp = int(time.time()) if args.timestamp is None else int(args.timestamp)
    # read meta info, init config and log file name
    meta_info_path = osp.join('metas', 'meta_{}.json'.format(cur_timestamp))
    if osp.exists(meta_info_path):
        with open(meta_info_path, 'r') as f:
            meta_info = json.load(f)
            data_root = meta_info['data_root']
            cfg.work_dir = meta_info['work_dir']
            # voc data format
            cfg.data_root = data_root + '/'
            old_prefix = cfg.data.train.img_prefix
            cfg.data.train.img_prefix = cfg.data_root
            cfg.data.val.img_prefix = cfg.data_root
            cfg.data.test.img_prefix = cfg.data_root
            cfg.data.train.ann_file = cfg.data.train.ann_file.replace(old_prefix, cfg.data_root) 
            cfg.data.val.ann_file = cfg.data.val.ann_file.replace(old_prefix, cfg.data_root) 
            cfg.data.test.ann_file = cfg.data.test.ann_file.replace(old_prefix, cfg.data_root) 
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if 'rpn_head' in cfg.model and isinstance(cfg.data.train, dict):
        cfg.data.train['anchor_nums'] = cfg.model['rpn_head']['anchor_nums'] * len(cfg.model['rpn_head']['anchor_strides'])
    datasets = [build_dataset(cfg.data.train)]

    # write anchor into file
    anchors = dict()
    anchors['ori_size'] = getattr(datasets[0], 'ori_size', (1333,800))
    anchors['base_anchors'] = getattr(datasets[0], 'base_anchors', [])
    assert mmcv.is_list_of(anchors['base_anchors'], list)
    with open(osp.join(cfg.work_dir, "anchors.json"), 'w+') as f:
        json.dump(anchors, f)

if __name__ == '__main__':
    print("anchor generator start.....")
    main()
    print("anchor generator end.....")
    exit(0)