from mmdet.datasets import DATASETS
from .xml_style import MyXMLDataset


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
    CLASSES = ('netport','two_netport','four_netport','optical_netport','two_optical_netport','four_optical_netport','backplane',
            'manufacturer','indicatorlight','light_indicator', 'usb','usb_indicator')
