# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Module implementing functional Unet style networks with a regression map.
"""
import torch
import torch.nn as nn

from metavision_core_ml.core.temporal_modules import SequenceWise
from .unet import Unet, interpolate
from typing import List, Tuple


class RegressorHead(nn.Module):
    """
    Performs a dense regression after a feature computation.
    """

    def __init__(self, block, in_channels, out_channels, n_output_channels, kernel=3, stride=1, padding=1):
        super(RegressorHead, self).__init__()
        self.block = block(in_channels, out_channels)
        self.regressor = SequenceWise(
            nn.Conv2d(out_channels, n_output_channels, kernel, stride=stride, padding=padding))
        self.out_channels = out_channels + n_output_channels

    def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.block(inp)
        value = self.regressor(feature_map)
        return feature_map, value


class UnetRegressor(Unet):
    """Generic Unet implementation computing a regression map at different scales.

    Schematics::

        ENCODER ->   DECODER
        D       ->          U  ->
        \ D     ->       U / ->
         \ D    ->     U / ->
          \ D   ->    U / ->
           \ MIDDLE   /

    This model works with either (B, C H W) tensors or (T, B, C, H, W) tensors.

    Args:
        down_block (function): function `(input_channels x output_channels) -> torch.nn.Module` used to create each layer
            of the encoder. The eventual spatial downsampling is supposed to be part of the Module (i.e. Either
            the module needs to have pooling or a strided convolution if the unet needs to have a hourglass shape).
        middle_block (function):  function `(input_channels x output_channels) -> torch.nn.Module` used to instantiate
            the middle part of the unet architecture. For instance, this is a good part to put a recurrent layer.
            If downsampling is used in the encoder, this part has the lowest spatial resolution.
        up_block (function): function `(input_channels x output_channels) -> torch.nn.Module` used to create each layer
            of the decoder. Its module will be run after eventual upsampling, so that its spatial resolution is the
            same as the corresponding decoding layer.
        n_input_channels (int): Number of channels in the input of the Unet model.
        n_output_channels (int): Number of channels in the output feature map of the Unet model.
        down_channel_counts (int List): Number of filters in the "down" layers of the encoder.
        middle_channel_counts (int List): Number of filters in the middle layer.
        up_channel_counts (int Lists): Number of filters in the "up" layers of the decoder.
    """

    def __init__(self, down_block, middle_block, up_block, n_input_channels=5, n_output_channels=2,
                 down_channel_counts=[32, 64], middle_channel_count=128, up_channel_counts=[64, 32, 8]):

        def regressor_up_block(in_channels, out_channels):
            return RegressorHead(up_block, in_channels, out_channels, n_output_channels)

        super(UnetRegressor, self).__init__(
            down_block, middle_block, regressor_up_block, n_input_channels=n_input_channels,
            down_channel_counts=down_channel_counts, middle_channel_count=middle_channel_count,
            up_channel_counts=up_channel_counts)

    def forward_return_values_and_features(
            self, x, return_features: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # down or encoding
        encoded_features = [x]
        for down_layer in self.encoders:
            x = down_layer(x)
            encoded_features.append(x)

        # middle layer
        decoded_features = [self.decoders[0](encoded_features[-1])]

        # up or decoding
        upscale_sizes = [e.shape[-2:] for e in encoded_features[:-1]]
        upscale_sizes.reverse()
        encoded_features.reverse()

        values = []

        nb_layers = min([len(self.decoders[1:]), len(encoded_features), len(upscale_sizes)])
        for idx_layer in range(nb_layers):
            skip = encoded_features[idx_layer]
            upscale_size = upscale_sizes[idx_layer]

            merged = self._merge_interpolate(decoded_features[-1], skip, upscale_size)

            # reuse previous regression value as a feature as well
            if values:
                upscaled_values = interpolate(values[-1], merged.shape[-2:])
                merged = torch.cat((merged, upscaled_values), merged.dim() - 3)

            feature = torch.empty(0)
            value = torch.empty(0)
            for idx_dummy, up_layer in enumerate(self.decoders[1:]):
                # hack to enable torch.jit.script
                if idx_dummy == idx_layer:
                    feature, value = up_layer.forward(merged)

            decoded_features.append(feature)
            values.append(value)

        return decoded_features, values

    def forward(self, x) -> List[torch.Tensor]:
        decoded_features, values = self.forward_return_values_and_features(x)
        return values
