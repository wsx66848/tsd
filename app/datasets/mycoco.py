from mmdet.datasets import DATASETS
from mmdet.datasets import CocoDataset

@DATASETS.register_module
class MyCocoDataset(CocoDataset):

    CLASSES = ('car', 'person', 'cat', 'tvmonitor', 'aeroplane', 'dog', 'sofa',
               'horse', 'motorbike','bird', 'boat', 'bicycle', 'train', 'bottle',
               'bus', 'sheep', 'cow', 'chair', 'diningtable', 'pottedplant')