import torch
from torch.nn import functional as F

def get_direction(bboxes):
    """get direction of netport
    
    Args:
        bboxes (Tensor): shape(n, 4)
    
    Returns:
        direction int(0 | 1)
    """
    w = bboxes[:, 2] - bboxes[:, 0] + 1
    h = bboxes[:, 3] - bboxes[:, 1] + 1
    if (w >= h).nonzero().size(0) >= bboxes.size(0) / 2:
        return 0
    else:
        return 1

def can_merge(bbox1, bbox2, direction):
    """whether the two bbox can merge

    Args:
        bbox1 tuple (x1, y1, x2, y2, score)
        bbox2 tuple(x1, y1, x2, y2, score)

    Returns:
        Bool 
    """
    first_factor, second_factor = 0.2, 0.2
    if direction == 0:
        h = bbox1[3] - bbox1[1] + 1
        w = bbox1[2] - bbox1[0] + 1
        y1_min, y1_max, y2_min, y2_max = bbox1[1] - h * first_factor, bbox1[1] + h * first_factor, bbox1[3] - h * first_factor, bbox1[3] + h * first_factor 
        x1_min, x1_max = bbox1[2] - w * (second_factor + 0.1), bbox1[2] + w * (second_factor - 0.05)
        if bbox2[1] >= y1_min and bbox2[1] <= y1_max and bbox2[3] >= y2_min and bbox2[3] <= y2_max and bbox2[0] >= x1_min and bbox2[0] <= x1_max:
            return True
    else:
        h = bbox1[3] - bbox1[1] + 1
        w = bbox1[2] - bbox1[0] + 1
        x1_min, x1_max, x2_min, x2_max = bbox1[1] - w * first_factor, bbox1[1] + w * first_factor, bbox1[2] - w * first_factor, bbox1[2] + w * first_factor 
        y1_min, y1_max = bbox1[3] - h * (second_factor + 0.1), bbox1[3] + (second_factor  - 0.05)
        if bbox2[0] >= x1_min and bbox2[0] <= x1_max and bbox2[2] >= x2_min and bbox2[2] <= x2_max and bbox2[1] >= y1_min and bbox2[1] <= y1_max:
            return True
    return False

def merge(bboxes, direction):
    """merge netports 

    Args:
        bboxes (list): [tuple(x1, y1, x2, y2, score)]
        direction (int): vertical or horizontal
    
    Returns:
        list: (x1, y1, x2, y2, score)
    """
    num = len(bboxes)
    assert num > 0
    if direction == 0:
        x1, x2 = bboxes[0][0], bboxes[-1][2]
        y1, y2 = bboxes[0][1], bboxes[0][3]
        scores, max_score = 0, bboxes[0][4]
        for index in range(num):
            scores += bboxes[index][4] 
            if bboxes[index][4] > max_score:
                y1, y2 = bboxes[index][1], bboxes[index][3]
                max_score = bboxes[index][4]
        score = scores / num
    else:
        y1, y2 = bboxes[0][1], bboxes[-1][3]
        x1, x2 = bboxes[0][0], bboxes[0][2]
        scores, max_score = 0, bboxes[0][4]
        for index in range(num):
            scores += bboxes[index][4] 
            if bboxes[index][4] > max_score:
                x1, x2 = bboxes[index][0], bboxes[index][2]
                max_score = bboxes[index][4]
        score = scores / num
    
    return [x1, y1, x2, y2, score]

def split_multinetport(bboxes, scores, num=4, direction=0, cuda_device=0):
    """split multi netports into single
    
    Args:
        bboxes (Tensor): shape(n, 4)
        scores (Tensor): shape(n)
        labels (Tensor): shape(n)
        num    (Int):    num of single netport

    Returns:
        Tensor: shape(n * num, 4)
        Tensor: shape(n * num)
        Tensor: shape(n * num)
    """
    if direction == 0:
        w = bboxes[:, 2] - bboxes[:, 0]
        ratios = torch.arange(num).cuda(cuda_device)
        x1_stride = (w / num)[:, None] * ratios[None, :]
        x2_stride = (w / num)[:, None] * (ratios - num + 1)[None, :]
        y1_stride = torch.zeros(bboxes.size(0), num).cuda(cuda_device)
        y2_stride = y1_stride
    else:
        h = bboxes[:, 3] - bboxes[:, 1]
        ratios = torch.arange(num).cuda(cuda_device)
        y1_stride = (h / num)[:, None] * ratios[None, :]
        y2_stride = (h / num)[:, None] * (ratios - num + 1)[None, :]
        x1_stride = torch.zeros(bboxes.size(0), num).cuda(cuda_device)
        x2_stride = x1_stride

    stride = torch.stack([x1_stride, y1_stride, x2_stride, y2_stride], dim=-1)
    split_bboxes = (bboxes[:, None, :] + stride).view(-1, 4)
    split_scores = scores.view(-1, 1).repeat(1,num).view(-1)
    
    return split_bboxes, split_scores

def bbox_intersection(bbox1, bbox2, cuda_device=0):
    """calculate the intersection between bbox1 and bbox2.

    Args:
        bbox1 (Tensor): shape (n, 4)
        bbox2 (Tensor): shape (k, 4)

    Returns:
        Tensor: shape (n, k)
    """
    zeros = torch.zeros(bbox1.size(0), bbox2.size(0)).cuda(cuda_device)
    x_min = torch.max(bbox1[:, 0, None], bbox2[None, :, 0])
    x_max = torch.min(bbox1[:, 2, None], bbox2[None, :, 2])
    y_min = torch.max(bbox1[:, 1, None], bbox2[None, :, 1])
    y_max = torch.min(bbox1[:, 3, None], bbox2[None, :, 3])
    delta_x = torch.max(x_max - x_min + 1, zeros)
    delta_y = torch.max(y_max - y_min + 1, zeros)
    intersection = delta_x * delta_y

    return intersection

def bbox_iou(bbox1, bbox2, cuda_device=0):
    """calculate iou between bbox1 and bbox2.
    
    Args:
        bbox1 (Tensor): shape(n, 4)
        bbox2 (Tensor): shape(k, 4)
    
    Returns:
        Tensor: shape (n, k)
    """
    intersection = bbox_intersection(bbox1, bbox2, cuda_device)
    area1 = (bbox1[:, 2] - bbox1[:, 0] + 1) * (bbox1[:, 3] - bbox1[:, 1] + 1)
    area2 = (bbox2[:, 2] - bbox2[:, 0] + 1) * (bbox2[:, 3] - bbox2[:, 1] + 1)
    return intersection / (area1[:, None] + area2[None, :] - intersection)

def bbox_ios(bbox1, bbox2, cuda_device=0):
    """calculate ios(intersection over self) between bbox1 and bbox2

    Args:
        bbox1 (Tensor): shape(n, 4)
        bbox2 (Tensor): shape(k, 4)
    
    Returns:
        Tensor: shape (n, k)
    """
    intersection = bbox_intersection(bbox1, bbox2, cuda_device)
    w = bbox1[:, 2] - bbox1[:, 0] + 1
    h = bbox1[:, 3] - bbox1[:, 1] + 1
    area_bbox1 = w * h 
    return intersection / area_bbox1[:, None]

def get_outlier(netport_scale, distance_thr=0.5, cuda_device='0'):
    """ get outlier scale
    
    Args:
        netport (Tensor): shape(n) n scales
        distance_thr (int)
    
    Returns:
       Tensor: shape(k) 
    """
    netport_scale_unique = torch.unique(netport_scale)
    netport_outlier = torch.FloatTensor(0).cuda(cuda_device)
    if netport_scale_unique.size(0) > 3:
        netport_scale_distance = torch.abs(netport_scale_unique[:, None] - netport_scale_unique[None, :])
        netport_scale_sorted, _ = torch.sort(netport_scale_distance, dim=1)
        netport_neighbor_distance  = torch.sum(netport_scale_sorted[:, :4], dim=1) / 3 
        netport_neighbor_relative = netport_neighbor_distance / netport_scale_unique
        netport_outlier = netport_scale_unique[netport_neighbor_relative > distance_thr]
    return netport_outlier

def get_diff_scorethr(netport_scores):
    """get score threshold by diff

    Args:
        netport_scores (Tensor): shape(n) n scores

    Returns:
       score_thr (float)
    """
    netport_unique_scores = torch.unique(netport_scores)
    netport_sort, _ = torch.sort(netport_unique_scores)
    netport_pad = F.pad(netport_sort, (1,1), 'constant')
    netport_pad[0] = netport_pad[1]
    netport_pad[-1] = netport_pad[-2]
    netport_difference  = netport_pad[1:] - netport_pad[:-1]
    netport_difference = netport_difference[1:] - netport_difference[:-1] / netport_unique_scores
    netport_score_thr = netport_unique_scores[torch.argmax(netport_difference)]
    return netport_score_thr

def get_netport_union(netport_bboxes, netport_scores, label, thr=0.6, cuda_device='0'):
    """get netport union by iou

    Args:
        netport_bboxes (Tensor): shape(n, 4) 
        netport_scores (Tensor): shape(n)
        label (int)
        threshold (int)

    Returns:
        netport_bboxes (list)
        netport_scores (list)
        netport_labels (list)
        netport_delete (list)
    """

    netport_iou = bbox_iou(netport_bboxes, netport_bboxes, cuda_device) 
    netport_delete = []
    new_netport_bboxes = []
    new_netport_scores = []
    new_netport_labels = []
    for indic in (netport_iou > thr).nonzero().tolist():
        if indic[0] == indic[1]:
            continue
        if not (indic[0] in netport_delete and indic[1] in netport_delete):
            netport_delete.append(indic[0])
            netport_delete.append(indic[1])
            new_netport_bboxes.append([min(netport_bboxes[indic[0]][0].item(), netport_bboxes[indic[1]][0].item()),
                min(netport_bboxes[indic[0]][1].item(), netport_bboxes[indic[1]][1].item()),
                max(netport_bboxes[indic[0]][2].item(), netport_bboxes[indic[1]][2].item()),
                max(netport_bboxes[indic[0]][3].item(), netport_bboxes[indic[1]][3].item())])
            new_netport_scores.append((netport_scores[indic[0]].item() + netport_scores[indic[1]].item()) / 2)
            new_netport_labels.append(label)
    return new_netport_bboxes, new_netport_scores, new_netport_labels, netport_delete