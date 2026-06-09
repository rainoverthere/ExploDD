# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Pyramid of inputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from metavision_core_ml.core.temporal_modules import seq_wise


class Pyramid(object):
    """
    Holds a Pyramid of Resized Tensors

    Args:
        num_levels: number of scales
        mode: mode of interpolation
        align_corners: see Pytorch Documentation of interpolate
    """

    def __init__(self, num_levels, mode="bilinear", align_corners=True):
        self.scale_factors = []
        for depth in reversed(list(range(num_levels))):
            scale_factor = 2 ** (-depth)
            self.scale_factors.append(scale_factor)
        self.mode = mode
        self.align_corners = align_corners

    def compute_level(self, inputs, level):
        """
        Computes One Level of Resize

        Args:
            x (torch.Tensor): batch (num_tbins, batch_size, channels, height, width)
            level (int): level of pyramid
        """
        scale_factor = self.scale_factors[level]
        return seq_wise(F.interpolate)(inputs, scale_factor=scale_factor, mode=self.mode,
                                       align_corners=self.align_corners)

    def compute_all_levels(self, inputs):
        """
        Computes all levels of Pyramid

        Args:
            x (torch.Tensor): batch (num_tbins, batch_size, channels, height, width)
        """
        num_levels = len(self.scale_factors)
        levels = []
        for level in range(num_levels):
            levels.append(self.compute_level(inputs, level))
        return levels
