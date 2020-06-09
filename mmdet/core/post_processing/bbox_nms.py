import torch
from torch.nn import functional as F

from mmdet.ops.nms import nms_wrapper
from myutils import my_config
from .nms_tool import *
import numpy as np


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], 1), **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
    
    # nms optimization
    classes = my_config.get('classes', [])
    global cat2label 
    cat2label = {cat: i for i, cat in enumerate(classes)}
    backplane_indic = (labels == cat2label['backplane'])
    backplane_bboxes, intern_bboxes = bboxes[backplane_indic], bboxes[~backplane_indic]
    backplane_scores, intern_scores = scores[backplane_indic], scores[~backplane_indic]
    backplane_labels, intern_labels = labels[backplane_indic], labels[~backplane_indic]

    cuda_device = bboxes.get_device()
    # remove duplicate backplane
    backplane_ios = bbox_ios(backplane_bboxes, backplane_bboxes, cuda_device)
    backplane_delete = []
    for indic in (backplane_ios > 0.9).nonzero().tolist():
        if indic[0] == indic[1]:
            continue
        backplane_delete.append(indic[0])
        
    backplane_delete = torch.BoolTensor([False if i in backplane_delete else True for i in range(backplane_bboxes.size(0))]).cuda(cuda_device)
    backplane_bboxes = backplane_bboxes[backplane_delete]
    backplane_scores = backplane_scores[backplane_delete]
    backplane_labels = backplane_labels[backplane_delete]

    intersection = bbox_intersection(intern_bboxes, backplane_bboxes, cuda_device)
    intern_backplane_indic = torch.argmax(intersection, dim=1)
    backplane_size = backplane_bboxes.size(0)
    for backplane_index in range(backplane_size):
        intern_bboxes_index = intern_bboxes[intern_backplane_indic == backplane_index]
        intern_scores_index = intern_scores[intern_backplane_indic == backplane_index]
        intern_labels_index = intern_labels[intern_backplane_indic == backplane_index]

        # extract different class bboxes and scores, labels
        class_specific_bboxes = dict()
        class_specific_scores = dict()
        class_specific_labels = dict()
        for key in cat2label:
            if key == 'backplane':
                continue
            class_specific_indic = intern_labels_index == cat2label[key]
            class_specific_bboxes[key] = intern_bboxes_index[class_specific_indic]
            class_specific_scores[key] = intern_scores_index[class_specific_indic]
            class_specific_labels[key] = intern_labels_index[class_specific_indic]

        ignore_keys = ['backplane']

        manu_scores = class_specific_scores['manufacturer']
        manu_bboxes = class_specific_bboxes['manufacturer']
        manu_labels = class_specific_labels['manufacturer']
        if manu_bboxes.size(0) > 1:
            _, indic = manu_scores.sort(descending=True)
            indic = indic[:2]
            manu_bboxes = manu_bboxes[indic]
            manu_labels = manu_labels[indic]
            manu_scores = manu_scores[indic]
        class_specific_scores['manufacturer'] = manu_scores
        class_specific_bboxes['manufacturer'] = manu_bboxes 
        class_specific_labels['manufacturer'] = manu_labels 


        netport_metas, two_netport_metas, four_netport_metas = split_and_merge_netport(class_specific_bboxes, class_specific_scores, class_specific_labels, 'netport', cuda_device)
        class_specific_bboxes['netport'], class_specific_scores['netport'], class_specific_labels['netport'] = netport_metas
        class_specific_bboxes['two_netport'], class_specific_scores['two_netport'], class_specific_labels['two_netport'] = two_netport_metas
        class_specific_bboxes['four_netport'], class_specific_scores['four_netport'], class_specific_labels['four_netport'] = four_netport_metas
        for key in cat2label:
            if key in ignore_keys:
                continue
            backplane_bboxes = torch.cat((backplane_bboxes, class_specific_bboxes[key]))
            backplane_scores = torch.cat((backplane_scores, class_specific_scores[key]))
            backplane_labels = torch.cat((backplane_labels, class_specific_labels[key]))
    
    bboxes = backplane_bboxes
    scores = backplane_scores
    labels = backplane_labels
    
    return torch.cat([bboxes, scores[:, None]], 1), labels

def split_and_merge_netport(class_specific_bboxes, class_specific_scores, class_specific_labels, key='netport', cuda_device='0'):
    four_key = 'four_' + key
    two_key = 'two_' + key
    four_netport_bboxes, four_netport_scores, four_netport_labels = class_specific_bboxes[four_key], class_specific_scores[four_key], class_specific_labels[four_key]
    two_netport_bboxes, two_netport_scores, two_netport_labels = class_specific_bboxes[two_key], class_specific_scores[two_key], class_specific_labels[two_key]
    netport_bboxes, netport_scores, netport_labels = class_specific_bboxes[key], class_specific_scores[key], class_specific_labels[key]

    direction = 0
    if four_netport_bboxes.size(0) > 0 or two_netport_bboxes.size(0) > 0:
        if four_netport_bboxes.size(0) > 0:
            direction = get_direction(four_netport_bboxes)
        else:
            direction = get_direction(two_netport_bboxes)

        if four_netport_bboxes.size(0) > 0:
            four_split_bboxes, four_split_scores = split_multinetport(four_netport_bboxes, four_netport_scores, 4, direction, cuda_device)
            four_split_labels = torch.full((four_netport_bboxes.size(0) * 4,), cat2label['netport']).long().cuda(cuda_device)
            netport_bboxes = torch.cat((netport_bboxes, four_split_bboxes))
            netport_scores = torch.cat((netport_scores, four_split_scores))
            netport_labels = torch.cat((netport_labels, four_split_labels))
        if two_netport_bboxes.size(0) > 0:
            two_split_bboxes, two_split_scores = split_multinetport(two_netport_bboxes, two_netport_scores, 2, direction, cuda_device)
            two_split_labels = torch.full((two_netport_bboxes.size(0) * 2,), cat2label['netport']).long().cuda(cuda_device)
            netport_bboxes = torch.cat((netport_bboxes, two_split_bboxes))
            netport_scores = torch.cat((netport_scores, two_split_scores))
            netport_labels = torch.cat((netport_labels, two_split_labels))
    
    if netport_bboxes.size(0) > 0:

        # remove outlier netport by k neighbor disantance
        if direction == 0:
            netport_scale = netport_bboxes[:, 2] - netport_bboxes[:, 0] 
        else:
            netport_scale = netport_bboxes[:, 3] - netport_bboxes[:, 1]
        netport_outlier = get_outlier(netport_scale, distance_thr=0.5, cuda_device=cuda_device) 
        for scale in netport_outlier.tolist():
            netport_bboxes = netport_bboxes[netport_scale != scale]
            netport_scores = netport_scores[netport_scale != scale]
            netport_labels = netport_labels[netport_scale != scale]
            netport_scale = netport_scale[netport_scale != scale]

        # find nested netport
        netport_ios = bbox_ios(netport_bboxes, netport_bboxes, cuda_device)
        netport_delete = []
        nested_thr = 0.8
        for indic in (netport_ios > nested_thr).nonzero().tolist():
            if indic[0] == indic[1]:
                continue
            if netport_scores[indic[0]] < netport_scores[indic[1]]:
                netport_delete.append(indic[0])
        netport_delete = torch.BoolTensor([False if i in netport_delete else True for i in range(netport_bboxes.size(0))]).cuda(cuda_device)
        netport_bboxes = netport_bboxes[netport_delete]
        netport_scores = netport_scores[netport_delete]
        netport_labels = netport_labels[netport_delete]

        # use second difference to set score thr
        netport_score_thr = get_diff_scorethr(netport_scores)
        if netport_score_thr < 0.3:
            netport_bboxes = netport_bboxes[netport_scores > netport_score_thr]
            netport_labels = netport_labels[netport_scores > netport_score_thr]
            netport_scores = netport_scores[netport_scores > netport_score_thr]

        # netport union
        new_netport_bboxes, new_netport_scores, new_netport_labels, netport_delete = get_netport_union(netport_bboxes, netport_scores, cat2label[key], thr=0.6, cuda_device=cuda_device)
        netport_delete = torch.BoolTensor([False if i in netport_delete else True for i in range(netport_bboxes.size(0))]).cuda(cuda_device)
        netport_bboxes = netport_bboxes[netport_delete]
        netport_scores = netport_scores[netport_delete]
        netport_labels = netport_labels[netport_delete]
        netport_bboxes = torch.cat((netport_bboxes, torch.Tensor(new_netport_bboxes).cuda(cuda_device)))
        netport_scores = torch.cat((netport_scores, torch.Tensor(new_netport_scores).cuda(cuda_device)))
        netport_labels = torch.cat((netport_labels, torch.Tensor(new_netport_labels).long().cuda(cuda_device)))

        # sort
        # import pdb;pdb.set_trace()
        netports = torch.cat((netport_bboxes, netport_scores[:, None]), 1)
        if direction == 0:
            order = ['x1', 'y1', 'x2', 'y2', 'score']
        else:
            order = ['y1', 'x1', 'y2', 'x2', 'score']
        sorted_netports = np.sort(np.array([tuple(val) for val in netports.tolist()], 
            dtype=[('x1', float), ('y1', float), ('x2', float), ('y2', float), ('score', float)]), order=order)
        merge_netports = []
        merged_netports = []
        merged_four_netports = []
        merged_two_netports = []
        for index in range(sorted_netports.shape[0]):
            netport = sorted_netports[index]
            flag = False
            pop_index = -1
            for merge_index in range(len(merge_netports)):
                if can_merge(merge_netports[merge_index][-1], netport, direction):
                    merge_netports[merge_index].append(netport)
                    if len(merge_netports[merge_index]) == 4:
                        merged_four_netports.append(merge(merge_netports[merge_index], direction))
                        pop_index = merge_index
                    flag = True
                    break
            if pop_index != -1:
                merge_netports.pop(pop_index)
            if not flag:
                merge_netports.append([netport])
        for merge_netport in merge_netports:
            if len(merge_netport) == 1:
                merged_netports.append(list(merge_netport[0]))
            if len(merge_netport) == 2:
                merged_two_netports.append(merge(merge_netport, direction))
            if len(merge_netport) == 3:
                merged_two_netports.append(merge(merge_netport[:2], direction))
                merged_netports.append(list(merge_netport[2]))
        
        # merged_netports = torch.cat((netport_bboxes, netport_scores[:, None]), 1).tolist()
        # merged_two_netports = []
        # merged_four_netports = []
        netport_bboxes, netport_scores, netport_labels = torch.Tensor(0, 4).float().cuda(cuda_device), torch.Tensor(0).float().cuda(cuda_device), torch.Tensor(0).long().cuda(cuda_device)
        two_netport_bboxes, two_netport_scores, two_netport_labels = torch.Tensor(0, 4).float().cuda(cuda_device), torch.Tensor(0).float().cuda(cuda_device), torch.Tensor(0).long().cuda(cuda_device)
        four_netport_bboxes, four_netport_scores, four_netport_labels = torch.Tensor(0, 4).float().cuda(cuda_device), torch.Tensor(0).float().cuda(cuda_device), torch.Tensor(0).long().cuda(cuda_device)

        netport_tensor, two_netport_tensor, four_netport_tensor= torch.Tensor(merged_netports).cuda(cuda_device), torch.Tensor(merged_two_netports).cuda(cuda_device), torch.Tensor(merged_four_netports).cuda(cuda_device)
        if netport_tensor.size(0) > 0:
            netport_bboxes, netport_scores, netport_labels = netport_tensor[:, :4], netport_tensor[:, 4], torch.full((netport_tensor.size(0), ), cat2label['netport']).long().cuda(cuda_device)

        if two_netport_tensor.size(0) > 0:
            two_netport_bboxes, two_netport_scores, two_netport_labels = two_netport_tensor[:, :4], two_netport_tensor[:, 4], torch.full((two_netport_tensor.size(0), ), cat2label['two_netport']).long().cuda(cuda_device)

        if four_netport_tensor.size(0) > 0:
            four_netport_bboxes, four_netport_scores, four_netport_labels = four_netport_tensor[:, :4], four_netport_tensor[:, 4], torch.full((four_netport_tensor.size(0), ), cat2label['four_netport']).long().cuda(cuda_device)

        # remove duplicate by ios
        diff_score_thr = 0.8
        ios_thr = 0.9
        netport_delete = []
        two_netport_delete = []
        four_netport_delete = []
        if netport_bboxes.size(0) > 0:
            if two_netport_bboxes.size(0) > 0: 
                netport_ios = bbox_ios(netport_bboxes,  two_netport_bboxes, cuda_device)
                for indic in (netport_ios > ios_thr).nonzero().tolist():
                    if netport_scores[indic[0]] < two_netport_scores[indic[1]]:
                        netport_delete.append(indic[0])
                    elif (netport_scores[indic[0]] - two_netport_scores[indic[1]]) / netport_scores > diff_score_thr:
                        two_netport_delete.append(indic[1])
            if four_netport_bboxes.size(0) > 0: 
                netport_ios = bbox_ios(netport_bboxes,  four_netport_bboxes, cuda_device)
                for indic in (netport_ios > ios_thr).nonzero().tolist():
                    if netport_scores[indic[0]] < four_netport_scores[indic[1]]:
                        netport_delete.append(indic[0])
                    elif (netport_scores[indic[0]] - four_netport_scores[indic[1]]) / netport_scores > diff_score_thr:
                        four_netport_delete.append(indic[1])
        netport_delete = torch.BoolTensor([False if i in netport_delete else True for i in range(netport_bboxes.size(0))]).cuda(cuda_device)
        two_netport_delete = torch.BoolTensor([False if i in two_netport_delete else True for i in range(two_netport_bboxes.size(0))]).cuda(cuda_device)
        four_netport_delete = torch.BoolTensor([False if i in four_netport_delete else True for i in range(four_netport_bboxes.size(0))]).cuda(cuda_device)
        netport_bboxes, netport_scores, netport_labels = netport_bboxes[netport_delete], netport_scores[netport_delete], netport_labels[netport_delete]
        two_netport_bboxes, two_netport_scores, two_netport_labels = two_netport_bboxes[two_netport_delete], two_netport_scores[two_netport_delete], two_netport_labels[two_netport_delete]
        four_netport_bboxes, four_netport_scores, four_netport_labels = four_netport_bboxes[four_netport_delete], four_netport_scores[four_netport_delete], four_netport_labels[four_netport_delete]

        two_netport_delete = []
        four_netport_delete = []
        if two_netport_bboxes.size(0) > 0 and four_netport_bboxes.size(0) > 0: 
                two_netport_ios = bbox_ios(two_netport_bboxes,  four_netport_bboxes, cuda_device)
                for indic in (two_netport_ios > ios_thr).nonzero().tolist():
                    if two_netport_scores[indic[0]] < four_netport_scores[indic[1]]:
                        two_netport_delete.append(indic[0])
                    elif (two_netport_scores[indic[0]] - four_netport_scores[indic[1]]) / two_netport_scores > diff_score_thr:
                        four_netport_delete.append(indic[1])
        two_netport_delete = torch.BoolTensor([False if i in two_netport_delete else True for i in range(two_netport_bboxes.size(0))]).cuda(cuda_device)
        four_netport_delete = torch.BoolTensor([False if i in four_netport_delete else True for i in range(four_netport_bboxes.size(0))]).cuda(cuda_device)
        two_netport_bboxes, two_netport_scores, two_netport_labels = two_netport_bboxes[two_netport_delete], two_netport_scores[two_netport_delete], two_netport_labels[two_netport_delete]
        four_netport_bboxes, four_netport_scores, four_netport_labels = four_netport_bboxes[four_netport_delete], four_netport_scores[four_netport_delete], four_netport_labels[four_netport_delete]

    return [netport_bboxes, netport_scores, netport_labels], [two_netport_bboxes, two_netport_scores, two_netport_labels], [four_netport_bboxes, four_netport_scores, four_netport_labels]





