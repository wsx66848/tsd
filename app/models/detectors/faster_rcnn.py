from mmdet.models.registry import DETECTORS
from mmdet.models import FasterRCNN
import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import builder

import pdb

@DETECTORS.register_module
class MyFasterRCNN(FasterRCNN):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 anchors = dict()):
        if rpn_head is not None and isinstance(rpn_head, dict):
            rpn_head['ori_size'] = anchors.get('ori_size', (1333, 800))
            rpn_head['base_anchors'] = anchors.get('base_anchors', [])
        super(MyFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)