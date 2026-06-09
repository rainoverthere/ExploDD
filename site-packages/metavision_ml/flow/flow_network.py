# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This file defines the training step (forward pass + loss computation) and validation stages
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from metavision_core_ml.core.temporal_modules import time_to_batch

from . import feature_extractor as archs
from ..core.warp_modules import Warping
from ..core.pyramid import Pyramid


class FlowNetwork(nn.Module):
    """
    Torch Module comprised of a feature extractor that computes flow but
    also modules that are useful for training such as Warpers and loss modules.

    Attributes:
        in_channels (int): number of channels in input.
        feature_extractor (nn.Module): neural network predicts flow pyramid.
        pyramid (object): stores resized inputs.
         warp modules per level.warping_head (nn.ModuleList): warp modules per level.
        criterion (nn.Module): Module computing the loss.

    Args:
        array_dim (int List): input shape (num_tbins, number of channels, height, width).
        flow_loss_weight (dict): dictionary, whose keys are name of flow losses and the values are float
            weight factors.
        feature_extractor_name (string): name of the feature extractor architecture to instantiate.
    """

    def __init__(self, array_dim, flow_loss_weights, feature_extractor_name="eminet", **kwargs):
        super(FlowNetwork, self).__init__()

        feature_extractor = getattr(archs, feature_extractor_name)
        self.in_channels = array_dim[1]

        self.feature_extractor = feature_extractor(self.in_channels, **kwargs)
        self.pyramid = Pyramid(self.feature_extractor.scales)
        self.warping_head = [Warping() for _ in range(self.feature_extractor.scales)]

    def get_nb_scales(self):
        return self.feature_extractor.scales

    def get_warping_head(self):
        return self.warping_head

    def forward(self, inputs):
        return self.feature_extractor(inputs)

    def interpolate_all_inputs_using_pyramid(self, inputs):
        interpolated_inputs = self.pyramid.compute_all_levels(inputs)
        return interpolated_inputs

    def sharpen(self, inputs, micro_tbin=0, depth_level=-1):
        """
        Rescales the input at a given depth level and uses the flow to sharpen the input by moving
        time bins according to the forward optical flow.

        Args:
            inputs (torch.Tensor): batch (T,B,C,H,W)
            micro_tbin (int): micro tbin to warp individual flow to
            depth_level (int): at which scale we run sharpen
        """
        interpolated_input = self.pyramid.compute_level(inputs, depth_level)

        flow = self.feature_extractor(inputs)[depth_level]
        if flow.dim() == 5:
            flow = time_to_batch(flow)[0]

        sharp_img = self.warping_head[depth_level].sharpen_micro_tbin(
            interpolated_input, flow, micro_tbin)
        return flow, sharp_img

    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        for _, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.feature_extractor.modules():
            if hasattr(module, "reset_all"):
                module.reset_all()
