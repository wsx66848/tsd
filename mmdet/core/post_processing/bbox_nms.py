import torch

from mmdet.ops.nms import nms_wrapper
from myutils import my_config


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
    
    """
    # target backplane
    # import pdb;pdb.set_trace()
    # find which label is backplane
    bbox_size = bboxes.size(0)
    cat2label = {cat: i + 1 for i, cat in enumerate(my_config.get('classes'))}
    backplane_indic = (labels == cat2label['backplane'] - 1).nonzero()
    backplane_indic = backplane_indic[:, 0].tolist()

    # calculate area of bboxes
    self_bboxes = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1) 

    # calculate intersection of bboxes
    zeros = torch.zeros(bbox_size, bbox_size).cuda()
    x_min = torch.max(bboxes[:, 0, None], bboxes[None, :, 0])
    x_max = torch.min(bboxes[:, 2, None], bboxes[None, :, 2])
    y_min = torch.max(bboxes[:, 1, None], bboxes[None, :, 1])
    y_max = torch.min(bboxes[:, 3, None], bboxes[None, :, 3])
    delta_x = torch.max(x_max - x_min + 1, zeros)
    delta_y = torch.max(y_max - y_min + 1, zeros)
    intersection = delta_x * delta_y
    # calculate ios of bboxes
    ios = intersection / self_bboxes[:, None]
    find = (ios > 0.9).nonzero().tolist()
    deleted = torch.full([bbox_size], True).bool().cuda()
    for inter in find:
        if inter[0] == inter[1]:
            continue
        if inter[0] not in backplane_indic and inter[1] in backplane_indic:
            continue
        if scores[inter[0]] < scores[inter[1]]:
            deleted[inter[0]] = False

    bboxes = bboxes[deleted]
    scores = scores[deleted]
    labels = labels[deleted]
    """

    return torch.cat([bboxes, scores[:, None]], 1), labels



