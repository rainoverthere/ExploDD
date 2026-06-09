# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
torch box API
"""
import torch
import numpy as np


def xyxy2xywh(boxes):
    """
    Changes box order from (xmin,ymin,xmax,ymax) to (xcenter,ycenter,width,height).

    Args:
        boxes (tensor): bounding boxes, sized [N,4].

    Returns:
        boxes (tensor): converted bounding boxes, sized [N,4].
    """
    a = boxes[..., :2]
    b = boxes[..., 2:]
    return torch.cat([(a + b) / 2, b - a], -1)


def xywh2xyxy(boxes):
    """Changes box order from (xcenter, ycenter, width, height) to (xmin,ymin,xmax,ymax).

    Args:
        boxes (tensor): bounding boxes, sized [N,4].

    Returns:
        boxes (tensor) : converted bounding boxes, sized [N,4].
    """
    a = boxes[..., :2]
    b = boxes[..., 2:]
    return torch.cat([a - b / 2, a + b / 2], -1)


def box_clamp(boxes, xmin, ymin, xmax, ymax):
    """Clamps boxes.

    Args:
        boxes (tensor): bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
        xmin (number): min value of x.
        ymin (number): min value of y.
        xmax (number): max value of x.
        ymax (number): max value of y.

    Returns:
      (tensor) clamped boxes.
    """
    boxes[:, 0].clamp_(min=xmin, max=xmax)
    boxes[:, 1].clamp_(min=ymin, max=ymax)
    boxes[:, 2].clamp_(min=xmin, max=xmax)
    boxes[:, 3].clamp_(min=ymin, max=ymax)
    return boxes


def box_select(boxes, xmin, ymin, xmax, ymax):
    """Selects boxes in range (xmin,ymin,xmax,ymax).

    Args:
        boxes (tensor): bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
        xmin (number): min value of x.
        ymin (number): min value of y.
        xmax (number): max value of x.
        ymax (number): max value of y.

    Returns:
      (tensor) selected boxes, sized [M,4].
      (tensor) selected mask, sized [N,].
    """
    mask = (boxes[:, 0] >= xmin) & (boxes[:, 1] >= ymin) \
        & (boxes[:, 2] <= xmax) & (boxes[:, 3] <= ymax)
    boxes = boxes[mask, :]
    return boxes, mask


def box_iou(box1, box2):
    """Computes the intersection over union of two sets of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (tensor) bounding boxes, sized [N,4].
        box2: (tensor) bounding boxes, sized [M,4].

    Returns:
        (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)      # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def batch_box_iou(box1, box2):
    """Computes the intersection over union of two sets of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (tensor) bounding boxes, sized [N,4].
        box2: (tensor) bounding boxes, sized [B,M,4].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    lt = torch.max(box1[None, :, None, :2], box2[:, None, :, :2])  # [B,N,M,2] broadcast_max( (_,N,_,2), (B,_,M,2) )
    rb = torch.min(box1[None, :, None, 2:], box2[:, None, :, 2:])  # [B,N,M,2]

    wh = (rb - lt).clamp(min=0)      # [B,N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [B,N,M]

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])  # [N,]
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])  # [B,M,]
    iou = inter / (area1[None, :, None] + area2[:, None, :] - inter)  # [B,N,M]
    return iou


def box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes: (tensor) bounding boxes, sized [N,4].
        scores: (tensor) confidence scores, sized [N,].
        threshold: (float) overlap threshold.

    Returns:
        keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.dim() == 0:
            i = order.item()
        else:
            i = order[0]

        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.tensor(keep, dtype=torch.long)


def bbox_to_deltas(boxes, default_boxes, variances=[0.1, 0.2], max_width=10000):
    """
    converts boxes expressed in absolute coordinate to anchor-relative coordinates.

    Args:
        boxes: xyxy boxes Nx4 tensor
        default_boxes: cxcywh anchor boxes Nx4 tensor
        variances: variances according to SSD paper.
        max_width: additional clamping to avoid infinite gt boxes.

    Returns:
        deltas: boxes expressed relatively to the center and size of anchor boxes.
    """
    boxes = xyxy2xywh(boxes)
    boxes[..., 2:] = boxes[..., 2:].clamp_(2, max_width)
    loc_xy = (boxes[..., :2] - default_boxes[..., :2]) / default_boxes[..., 2:] / variances[0]
    loc_wh = torch.log(boxes[..., 2:] / default_boxes[..., 2:]) / variances[1]
    deltas = torch.cat([loc_xy, loc_wh], -1)
    return deltas


def deltas_to_bbox(loc_preds, default_boxes, variances=[0.1, 0.2]):
    """
    converts boxes expressed in anchor-relative coordinates to absolute coordinates.

    Args:
        loc_preds: deltas boxes Nx4 tensor
        default_boxes: cxcywh anchor boxes Nx4 tensor
        variances: variances according to SSD paper.

    Returns:
        box_preds: boxes expressed in absolute coordinates.
    """
    xy = loc_preds[..., :2] * variances[0] * default_boxes[..., 2:] + default_boxes[..., :2]
    wh = torch.exp(loc_preds[..., 2:] * variances[1]) * default_boxes[..., 2:]
    box_preds = torch.cat([xy - wh / 2, xy + wh / 2], -1)
    return box_preds


def pack_boxes_list(targets):
    """Packs targets altogether

    Because numpy array have variable-length,
    We pad each group.

    Args:
        targets: list of np.ndarray in struct BBOX_dtype
    Returns:
        packed targets in shape [x1,y1,x2,y2,label]
        num_boxes: list of number of boxes per frame
    """
    max_size = max([len(frame) for frame in targets])
    max_size = max(2, max_size)
    gt_padded = torch.ones((len(targets), max_size, 5), dtype=torch.float32) * -2
    num_boxes = []
    for t in range(len(targets)):
        boxes = targets[t]
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        gt_padded[t, :len(boxes)] = boxes[:, :5]
        num_boxes.append(len(boxes))
    return gt_padded, num_boxes


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  fg_iou_threshold, bg_iou_threshold, allow_low_quality_matches=True):
    """
    Assigns ground truth boxes as targets to priors (also called anchor boxes).

    Args:
        gt_boxes (tensor): ground truth boxes tensor of shape (num_targets, 4).
        gt_labels (tensor): int tensor of size (num_target) containing the class labels of targets.
        corner_form_priors (tensor): tensor of shape (num_priors, 4), contains the priors boxes in format
            (xmin, ymin, xmax, ymax).
        fg_iou_threshold (float): minimal iou with a prior box to be considered a positive match
            (to be assigned a ground truth box)
        bg_iou_threshold (float): below this iou threshold a prior box is considered to match the background
            (the prior box doesn't match any ground truth box)
        allow_low_quality_matches (boolean): allow bad matches to be considered anyway.

    Returns:
        boxes (tensor): of shape coordinate values of the ground truth box assigned to each prior box
            in the format (xmin, ymin, xmax, ymax)
        labels (tensor):  of shape (num_priors)  containing class label of the ground truth box assigned to each prior.
    """
    # size: num_priors x num_targets
    ious = box_iou(corner_form_priors, gt_boxes)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    if allow_low_quality_matches:
        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
            best_target_per_prior[prior_index] = 2

    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]

    mask = (best_target_per_prior > bg_iou_threshold) * (best_target_per_prior < fg_iou_threshold)

    labels[mask] = -1
    labels[best_target_per_prior < bg_iou_threshold] = 0  # the background id
    boxes = gt_boxes[best_target_per_prior_index]

    return boxes, labels
