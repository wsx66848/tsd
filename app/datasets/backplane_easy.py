from mmdet.datasets import DATASETS
from .xml_style import MyXMLDataset
from mmdet.datasets.pipelines import Compose
import numpy as np
from .kmeans import AnchorKmeans


@DATASETS.register_module
class BackplaneEasyDataset(MyXMLDataset):
    #easy
    """
    CLASSES = ('netport', 'optical_netport','backplane','manufacturer','indicatorlight', 'usb',)
    """
    #batch
    """
    CLASSES = ('netport','two_netport','four_netport','optical_netport','two_optical_netport','four_optical_netport','backplane',
            'manufacturer','indicatorlight', 'usb','usb_indicator')
    """
    #batch_full
    # CLASSES = ('netport','two_netport','four_netport','optical_netport','two_optical_netport','four_optical_netport','backplane',
    CLASSES = ('netport','two_netport','four_netport','optical_netport','two_optical_netport','four_optical_netport','backplane',
             'manufacturer','indicatorlight', 'usb')
    # CLASSES = ('backplane',)

    def __init__(self, min_size = None, anchor_nums=None, cluster_nums=None, quiet=False, **kwargs):
        super(BackplaneEasyDataset, self).__init__(min_size, **kwargs)
        if self.test_mode == False and quiet == False:
            self.anchor_nums = anchor_nums
            self.cluster_nums = cluster_nums
            self._resize = (1333,800)
            for key in range(len(kwargs.get('pipeline', []))):
                transform = kwargs.get('pipeline')[key]
                if transform['type'] == 'Resize' and 'img_scale' in transform and len(transform['img_scale']) >= 1:
                    self._resize = transform['img_scale'][0]
                    break
            self.anchors = self.get_anchor_size()
    
    @property
    def base_anchors(self):
        return self.anchors

    @property
    def ori_size(self):
        return self._resize

    def get_anchor_size(self):
        CLASSES = self.CLASSES
        img_infos = self.img_infos
        transfroms = Compose([dict(type='LoadImageFromFile'), 
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='Resize', img_scale=[self._resize], keep_ratio=True)])
        gt_boxes_all = dict()
        for idx in range(len(img_infos)):
            img_info = img_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results)
            transfroms(results)
            gt_boxes = results['gt_bboxes']
            labels = ann_info['labels']
            assert len(gt_boxes) == len(labels)
            for i, label in enumerate(labels):
                w = gt_boxes[i][2] - gt_boxes[i][0]
                h = gt_boxes[i][3] - gt_boxes[i][1]
                classname = CLASSES[label - 1]
                if classname not in gt_boxes_all:
                    gt_boxes_all[classname] = []
                gt_boxes_all[classname].append([w, h])

        # default k-cluster config
        cluster_k = {
            'netport': 5,
            'two_netport': 5,
            'four_netport': 6,
            'optical_netport': 5,
            'two_optical_netport': 5,
            'four_optical_netport': 4,
            'backplane': 6,
            'manufacturer': 12,
            'indicatorlight': 6,
            'usb': 9
        }
        if self.cluster_nums is not None and isinstance(self.cluster_nums, list):
            cluster_k = dict()
            for i, classname in enumerate(CLASSES):
                cluster_k[classname] = self.cluster_nums[i]
        gen_anchor_num = sum(cluster_k.values())

        need_anchor_num = 45
        if self.anchor_nums is not None:
            need_anchor_num = self.anchor_nums
        
        assert gen_anchor_num >= need_anchor_num

        base_anchors = dict()
        for classname in CLASSES:
            if classname in gt_boxes_all:
                class_boxes = gt_boxes_all[classname]
                k = cluster_k[classname] if classname in cluster_k else 3
                iter_num = 0
                model = AnchorKmeans(k, 800)
                while True:
                    res = model.fit(class_boxes)
                    iter_num += 1
                    if res is True and model.avg_iou() > 0.75:
                        base_anchors[classname] = model.anchors
                        break
                    assert iter_num < 300
                print("%s iou: %f\n" % (classname, model.avg_iou()))

        anchor_all = []
        for key in base_anchors:
            anchor_all += base_anchors[key].tolist()
        anchor_iou = AnchorKmeans.iou(np.array(anchor_all), np.array(anchor_all))
        sorting_iou = []
        for row in range(len(anchor_iou)):
            for col in range(len(anchor_iou)):
                if row == col:
                    continue
                item = (anchor_iou[row][col], min(row,col), max(row,col))
                if item not in sorting_iou:
                    sorting_iou.append(item)
        sorted_iou = np.sort(np.array(sorting_iou, dtype=[('iou', float), ('small', int), ('big', int)]), order='iou')
        sorted_iou = sorted_iou[::-1]
        delete_num = gen_anchor_num - need_anchor_num
        deleted = []
        for index in range(len(sorted_iou)):
            if len(deleted) == delete_num:
                break
            if sorted_iou[index][1] not in deleted:
                deleted.append(sorted_iou[index][1])
        final_anchor_all = np.delete(np.array(anchor_all), deleted, axis=0)
        anchor_area = [(anchor[0] * anchor[1], [anchor[0], anchor[1]]) for anchor in final_anchor_all]
        sorted_anchor_area = np.sort(np.array(anchor_area, dtype=[('area', float), ('scale', list)]), order='area')

        anchor_sequence = [sorted_anchor_area[index][1] for index in range(len(sorted_anchor_area))]
        print(anchor_sequence)

        # import pdb;pdb.set_trace()
        return anchor_sequence