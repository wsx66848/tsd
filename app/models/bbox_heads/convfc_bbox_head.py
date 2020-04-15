from mmdet.models.registry import HEADS
from mmdet.models.bbox_heads import SharedFCBBoxHead
import numpy as np
import pdb
from .relation_network import RelationModule
from ..utils import *

@HEADS.register_module
class MySharedFCBBoxHead(SharedFCBBoxHead):

    def __init__(self, n_relations = 0, distance_weight = True, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        super(MySharedFCBBoxHead, self).__init__(
            num_fcs=num_fcs,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.n_relations = n_relations
        self.distance_weight = distance_weight
        fc_features = self.fc_out_channels
        if(n_relations>0):
            self.dim_g = int(fc_features/n_relations)
            self.relation= RelationModule(n_relations = n_relations, appearance_feature_dim=fc_features,
                                        key_feature_dim = self.dim_g, geo_feature_dim = self.dim_g)

    def forward(self, x, rois):
        """
        x 1024 (number of roi = train_cfg.rcnn.sampler.number * image_per_gpu) * 256 (channel) * 7 * 7(roi feat size) apperance feature
        rois 1024 * 5(index, x1, y1, x2, y2) geo feature
        
        return 
            cls_score 1024 (number of roi) * num_classes
            bbox_pred()  1024 (number of roi) * [num_classes * 4 (x,y,w,h)]
        """
        #pdb.set_trace()
        if self.n_relations > 0:
            position_embedding = PositionalEmbedding(rois[:, 1:],dim_g = self.dim_g)
            distance_weights = None
            if self.distance_weight:
                distance_weights = DistanceWeight(rois[:, 1:])
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                #这有两个fc 在两个fc中间加入Relation Networks  x就是输入  RN在构造器中的输入其实就一个特征维度fc_out_channels
                x = self.relu(fc(x))
                if self.n_relations > 0:
                    x = self.relation((x, position_embedding), distance_weight = distance_weights)
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred