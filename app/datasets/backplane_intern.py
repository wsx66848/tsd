from .backplane_easy import BackplaneEasyDataset
from mmdet.datasets import DATASETS


@DATASETS.register_module
class BackplaneInternDataset(BackplaneEasyDataset):

    CLASSES = ('netport','two_netport','four_netport','optical_netport','two_optical_netport','four_optical_netport',
             'manufacturer','indicatorlight', 'usb')