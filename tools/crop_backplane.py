import argparse
import copy
import os
import os.path as osp
import shutil
import time

import mmcv
import torch
from mmcv import Config
from mmdet.apis import init_detector, inference_detector

from app import *
import json
import xml.etree.ElementTree as ET
import numpy as np
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

backplane_name = BackplaneEasyDataset.CLASSES[0]

def load_annotations(ann_file, img_prefix):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = osp.join(img_prefix, 'JPEGImages/{}.jpg'.format(img_id))
            xml_path = osp.join(img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            number = 0
            bboxes = dict(backplane=[], intern=[])
            for obj in root.findall('object'):
                name = obj.find('name').text
                difficult = 0
                if obj.find('difficult').text is not None:
                    difficult = int(obj.find('difficult').text)
                bnd_box = obj.find('bndbox')
                bbox = dict(
                    xmin = float(bnd_box.find('xmin').text),
                    ymin = float(bnd_box.find('ymin').text),
                    xmax = float(bnd_box.find('xmax').text),
                    ymax = float(bnd_box.find('ymax').text),
                    name = name)
                if not difficult:
                    if name == backplane_name:
                        number += 1
                        bboxes[backplane_name].append(bbox)
                    else:
                        bboxes['intern'].append(bbox)

            # if without backplane, then skip
            if number == 0:
                continue

            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height, bboxes=bboxes))

        return img_infos

def newObject(name, xmin, ymin, xmax, ymax):
    obj = ET.Element('object')
    name_tag = ET.Element('name')
    name_tag.text = name
    obj.append(name_tag) 
    pose = ET.Element('pose')
    pose.text = None
    obj.append(pose)
    difficult = ET.Element('difficult')
    difficult.text = None
    obj.append(difficult)
    truncated = ET.Element('truncated')
    truncated.text = None
    obj.append(truncated)

    bndbox = ET.Element('bndbox')
    xmin_tag = ET.Element('xmin')
    xmin_tag.text = xmin
    ymin_tag = ET.Element('ymin')
    ymin_tag.text = ymin
    xmax_tag = ET.Element('xmax')
    xmax_tag.text = xmax
    ymax_tag = ET.Element('ymax')
    ymax_tag.text = ymax
    bndbox.append(xmin_tag)
    bndbox.append(ymin_tag)
    bndbox.append(xmax_tag)
    bndbox.append(ymax_tag)
    obj.append(bndbox)

    return obj

def crop_single_image(img_info, dst, mode='train'):
    img_prefix = osp.dirname(osp.dirname(img_info['filename']))
    img_id = osp.splitext(osp.basename(img_info['filename']))[0]
    xml_file = ET.parse(osp.join(img_prefix, 'Annotations', '{}.xml'.format(img_id)))
    backplanes = img_info['bboxes'][backplane_name]
    image = imageio.imread(img_info['filename'])
    bbs = BoundingBoxesOnImage([BoundingBox(x1=max(box['xmin'], 0), y1=max(box['ymin'], 0), x2=max(box['xmax'], 0), y2=max(box['ymax'], 0), label=box['name']) for box in img_info['bboxes']['intern']], shape=image.shape)
    height, width, _ = image.shape
    for i, backplane in enumerate(backplanes):
        crop_top, crop_bottom, crop_left, crop_right = -1 * max(backplane['ymin'], 0), max(backplane['ymax'], 0) - height, -1 * max(backplane['xmin'], 0), max(backplane['xmax'], 0) - width
        if crop_top > 0 or crop_bottom > 0 or crop_left > 0 or crop_right > 0:
            import pdb;pdb.set_trace()
        assert crop_top <= 0 and crop_bottom <= 0 and crop_left <=0 and crop_right <= 0
        img_crop, bbx_crop= iaa.CropAndPad(px=(int(crop_top), int(crop_right), int(crop_bottom), int(crop_left)), keep_size=False)(image=image, bounding_boxes=bbs)
        bbx_crop = bbx_crop.clip_out_of_image()
        img_name = img_id
        if i != 0:
            img_name = img_name + '_{}'.format(i)
        mmcv.imwrite(img_crop, osp.join(dst, 'JPEGImages', '{}.jpg'.format(img_name)))
        with open(osp.join(dst, 'ImageSets', 'Main', 'trainval.txt'), 'a+') as f:
            f.write(img_name + '\n')
        extra_txt = 'train.txt' if mode == 'train' else 'val.txt'
        with open(osp.join(dst, 'ImageSets', 'Main', extra_txt), 'a+') as f:
            f.write(img_name + '\n')
        # change filename 
        new_xml = copy.deepcopy(xml_file)
        new_root = new_xml.getroot()
        fn_tag = new_root.find('filename')
        fn_tag.text = '{}.jpg'.format(img_name)

        # change size
        size_tag = new_root.find('size')
        new_height, new_width, new_depth = img_crop.shape
        height_tag = size_tag.find('height')
        height_tag.text = str(new_height)
        width_tag = size_tag.find('width')
        width_tag.text = str(new_width)
        depth_tag = size_tag.find('depth')
        depth_tag.text = str(new_depth)

        # change object
        for obj in new_root.findall('object'):
            new_root.remove(obj)
        for box in bbx_crop.items:
                new_root.append(newObject(box.label, str(box.x1.item()), str(box.y1.item()), str(box.x2.item()), str(box.y2.item())))
        new_xml.write(osp.join(dst, 'Annotations', '{}.xml'.format(img_name)))


def detect_backplane(config_path, trainlog_path, device='cuda:0'):
    assert osp.exists(config_path) and osp.exists(trainlog_path)

    cfg = Config.fromfile(config_path)

    # get max ap checkpoint
    cp_base_dir = osp.dirname(trainlog_path)
    lines = []
    with open(trainlog_path, 'r') as f:
        lines = f.readlines()
    max_ap = -1
    cp_name = ''
    for line in lines:
        info = json.loads(line)
        if info.get('mode', 'train') == 'val':
            ap = info.get('mAP', 0)
            if ap > max_ap:
                max_ap = ap
                cp_name = 'epoch_{}.pth'.format(str(info.get('epoch')))
    checkpoint_path = osp.join(cp_base_dir, cp_name)
    print("max_ap: %f, checkpoint file: %s\n" % (max_ap, checkpoint_path))

    # get train dataset img list, crop by annotation
    print("load train dataset...")
    train_imgs = load_annotations(cfg.data.train.ann_file, cfg.data.train.img_prefix)

    # get val/test dataset img list, crop by model result
    print("load val dataset...")
    val_imgs = load_annotations(cfg.data.val.ann_file, cfg.data.val.img_prefix)

    print("init detect model...")
    model = init_detector(cfg, checkpoint_path, device=device)
    for i, img in enumerate(val_imgs):
        img = mmcv.imread(img['filename']) 
        result = inference_detector(model, img)
        print("\r detecting... {}/{}".format(i, len(val_imgs)), end="")
        assert len(result) == 1 # only backplane
        backplanes = []
        for backplane in result[0]:
            if backplane[4] > 0.9:
                backplanes.append(dict(xmin=backplane[0], ymin=backplane[1], xmax=backplane[2], ymax=backplane[3], name=backplane_name)) 
        val_imgs[i]['bboxes'][backplane_name] = backplanes

    print("\ndetect done...")
    
    # generate new voc dataset
    old_data_root = osp.abspath(cfg.data_root)
    real_data_root = old_data_root
    if osp.islink(old_data_root):
        real_data_root = osp.abspath(os.readlink(old_data_root))
    data_dir = osp.dirname(real_data_root)
    new_data_root = osp.join(data_dir, osp.basename(real_data_root) + '_' + str(int(time.time())))
    os.makedirs(new_data_root)
    os.makedirs(osp.join(new_data_root, 'Annotations'))
    os.makedirs(osp.join(new_data_root, 'JPEGImages'))
    os.makedirs(osp.join(new_data_root, 'ImageSets', 'Main'))
    total_imgs = len(train_imgs) + len(val_imgs)

    # train crop
    print("start crop...")
    count = 0
    # import pdb;pdb.set_trace()
    for train_img in train_imgs:
        crop_single_image(train_img, new_data_root, 'train')
        count += 1
        print("\r cropping... {}/{}".format(count, total_imgs), end="")
    for val_img in val_imgs:
        crop_single_image(val_img, new_data_root, 'val')
        count += 1
        print("\r cropping... {}/{}".format(count, total_imgs), end="")

    print("\ncrop finished...")
    # import pdb;pdb.set_trace()
    data_root = new_data_root
    if osp.islink(old_data_root):
        data_root = osp.join(osp.dirname(old_data_root), osp.basename(new_data_root))
        os.symlink(new_data_root, data_root)
        print("link created...")
    return data_root

def parse_args():
    parser = argparse.ArgumentParser(description='crop backplane image according to backplane detector')
    parser.add_argument('--timestamp', help='the timestamp when starting training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    timestamp = int(time.time()) if args.timestamp is None else int(args.timestamp)
    meta_info_path = osp.join('metas', 'meta_{}.json'.format(cur_timestamp))
    assert osp.exists(meta_info_path)
    with open(meta_info_path, 'r') as f:
        meta_info = json.load(f)
        config_path = meta_info['config_path']
        log_path = meta_info['log_path']

    data_root = detect_backplane(config_path, log_path)
    with open(meta_info_path, 'w+') as f:
        meta_info = json.load(f)
        meta_info['data_root'] = data_root
        json.dump(meta_info, f)

if __name__ == '__main__':
    main()