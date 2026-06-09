# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This script instantiates a jittable single-shot detector.
It is called by export.py.
It allows to exports to the C++ Detection & Tracking application.
"""
import os
import json
import torch
import torch.nn as nn

from typing import List
from . import anchors
from ..preprocessing import PREPROCESSING_DICT


class BoxDecoder(nn.Module):
    """
    Jittable Box Decoding

    This reuses the anchor module that is jittable for its forward,
    but not all its other functions like encode or decode.

    It decodes 2 prediction tensors (regression & box classification)
    that are N,Nanchors,4 and N,Nanchors,C
    (with Nanchor: number of anchor & C: number of classes)

    into a nested list of torch.tensors [x1,y1,w,h,score,class_id]
    """

    def __init__(self):
        super(BoxDecoder, self).__init__()

    def forward(
            self, anchors, loc_preds, cls_preds, varx: float, vary: float, score_thresh: float,
            batch_size: int):
        """
        for Torch.Jit
        this forward does not perform nms
        """
        xy = loc_preds[..., :2] * varx * anchors[..., 2:] + anchors[..., :2]
        wh = torch.exp(loc_preds[..., 2:] * vary) * anchors[..., 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], xy.dim()-1)

        box_tensor, batch_index = self._box_filtering(box_preds, cls_preds, score_thresh)

        num_tbins = len(cls_preds) // batch_size

        box_tensor = box_tensor.data
        targets = [[torch.zeros((0,), dtype=torch.float32) for _ in range(batch_size)]
                   for _ in range(num_tbins)]
        for t in range(num_tbins):
            for i in range(batch_size):
                idx = t * batch_size + i
                sel = batch_index == idx
                boxes = box_tensor[sel]
                targets[t][i] = boxes
        return targets

    def _box_filtering(self, boxes, scores, score_thresh: float):
        batchsize, num_anchors = boxes.size()[:2]
        num_classes = scores.size(-1)

        scores, idxs = self._box_indices(scores, batchsize, num_anchors, num_classes)

        # filtering boxes by score
        boxes = boxes.unsqueeze(2).expand(-1, num_anchors, num_classes, 4).contiguous()
        scores, score_indices = self._score_filtering(scores, score_thresh)
        boxes = boxes.view(-1, 4)[score_indices].contiguous()
        idxs = idxs.view(-1)[score_indices].contiguous()

        labels = (idxs % num_classes).float() + 1
        batch_index = idxs // num_classes
        box_tensor = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1)], dim=1)
        return box_tensor, batch_index

    def _box_indices(self, scores, batchsize: int, num_anchors: int, num_classes: int):
        """gives the boxes indices whether we give the best scoring boxes by classes or considered altogether
        """
        rows = torch.arange(batchsize, dtype=torch.long, device=scores.device)[:, None]
        cols = torch.arange(num_classes, dtype=torch.long, device=scores.device)[None, :]
        idxs = rows * num_classes + cols
        idxs = idxs.unsqueeze(1).expand(batchsize, num_anchors, num_classes).contiguous()
        return scores, idxs

    def _score_filtering(self, scores, score_threshold: float):
        """returns the sorted scores and their indices

        Args:
            scores (Tensor): batchsize x num_anchors x num_classes
            score_threshold (float): minimal score to be retained
        """
        scores = scores.view(-1)
        score_indices = torch.arange(scores.shape[0]).to(scores.device)
        mask = (scores >= score_threshold).to(scores.device)
        score_indices = score_indices[mask].contiguous()
        scores = scores[score_indices]
        return scores, score_indices


class SSD(torch.nn.Module):
    """ This module can be exported to C++.
        Given input tensor it outputs a nested list of filtered boxes.
        Args:
            net: neural network model
            anchor_list: anchor configuration
    """

    def __init__(self, net, anchor_list="PSEE_ANCHORS"):
        super(SSD, self).__init__()
        self.feature_extractor = net.feature_extractor
        self.rpn = net.rpn
        self.anchor_generator = anchors.Anchors(anchor_list=anchor_list, num_levels=self.feature_extractor.levels)
        self.box_decoder = BoxDecoder()

    @torch.jit.export
    def forward_network_without_box_decoding(self, x) -> List[torch.Tensor]:
        xs = self.feature_extractor(x)
        loc, cls = self.rpn(xs)
        return [loc, cls]

    def forward(self, x, score_thresh: float = 0.4) -> List[List[torch.Tensor]]:
        xs = self.feature_extractor(x)
        loc, cls = self.rpn(xs)
        cls = self.rpn.get_scores(cls)
        anchors = self.anchor_generator(xs, x)
        targets = self.box_decoder(anchors, loc, cls, 0.1, 0.2, score_thresh, x.size(1))
        return targets

    @torch.jit.export
    def reset_all(self):
        self.feature_extractor.reset_all()


def export_lightning_model(lightning_model, out_directory, nms_thresh, score_thresh, precision = 32):
    """
    Exports lightning model to Torch JIT & json parameter files read by the C++ D&T application.

    Args:
        lightning_model: pytorch lightning class
        out_directory: output directory
        nms_thresh:
        score_thresh:
        precision (int): save the model in float 16 or 32 precision
    """
    params = lightning_model.hparams
    ssd = SSD(lightning_model.detector.cpu(), anchor_list=params['anchor_list'])
    params['preprocessing_name'] = params.get('preprocess', 'none') # for images, e.g. toy problem
    params['delta_t'] = params.get('delta_t',0)
    if params['delta_t'] == 0:
        assert params['preprocessing_name'] == 'none'
    params['max_incr_per_pixel'] = params.get('max_incr_per_pixel', 1)
    params['shift'] = params.get('shift', 0)
    label_map = ['background'] + params['classes']
    params['label_map'] = label_map
    ssd.eval()
    export_ssd(ssd, params, out_directory, nms_thresh, score_thresh, precision)


def export_ssd(ssd, params, out_directory, nms_thresh, score_thresh, precision = 32):
    """Exports Jitted class SSD
    & json parameter files read by the C++
    D&T application

    Args:
        ssd: jitted class
        params: hyper parameters
        out_directory: output directory
        nms_thresh:
        score_thresh:
        precision (int): save the model in float 16 or 32 precision
    """
    assert precision in (16,32), "only 16 and 32 precision (float) are supported"

    script = torch.jit.script(ssd)
    if precision == 16:
        script.half()
    script.save(os.path.join(out_directory, "model.ptjit"))

    dic_json = {}
    dic_json["label_map"] = params["label_map"]
    dic_json["delta_t"] = int(params["delta_t"])
    dic_json["num_channels"] = int(params['in_channels'])
    dic_json["in_channels"] = int(params['in_channels'])
    # num_classes contains background because of softmax prediction
    dic_json["num_classes"] = len(params['label_map'])
    preproc_name = params.get('preprocess', 'none')
    dic_json["preprocessing_name"] = preproc_name
    assert preproc_name == 'none' or preproc_name in PREPROCESSING_DICT
    if preproc_name in PREPROCESSING_DICT:
        cin = PREPROCESSING_DICT[preproc_name]["cin"]
        assert cin == params["in_channels"], "cin: {}  params['in_channels']: {}".format(cin, params["in_channels"])
        if 'timesurface' not in preproc_name:
            assert "max_incr_per_pixel" in PREPROCESSING_DICT[preproc_name]["kwargs"]
            dic_json["max_incr_per_pixel"] = PREPROCESSING_DICT[preproc_name]["kwargs"]["max_incr_per_pixel"]
        dic_json["clip_value_after_normalization"] = 1.0

    if preproc_name == "event_cube_paper" or preproc_name == "event_cube":
        dic_json["preprocessing_name"] = "event_cube"
        split_polarity = PREPROCESSING_DICT[preproc_name]["kwargs"]["split_polarity"]
        dic_json["split_polarity"] = split_polarity
        if split_polarity:
            assert cin % 2 == 0
            num_utbins = cin // 2
        else:
            num_utbins = cin
        dic_json["num_utbins"] = num_utbins

    dic_json["iou_threshold"] = nms_thresh
    dic_json["confidence_threshold"] = score_thresh
    dic_json["num_anchor_boxes"] = 0
    filename_json = os.path.join(out_directory, "info_ssd_jit.json")
    json.dump(dic_json, open(filename_json, "w"), indent=4)
