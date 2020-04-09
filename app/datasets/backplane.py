from mmdet.datasets import DATASETS
from .xml_style import MyXMLDataset


@DATASETS.register_module
class BackplaneDataset(MyXMLDataset):

    CLASSES = ('cable', 'netport', 'netport_occluded', 'backplane',
            'power_switch', 'power_outlet', 'manufacturer',
               'indicatorlight', 'usb', 'rst', 'ground_terminal',)
