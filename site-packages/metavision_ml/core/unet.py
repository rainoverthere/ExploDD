# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Module implementing functional Unet style networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from metavision_core_ml.core.temporal_modules import time_to_batch, batch_to_time


class Unet(nn.Module):
    """Generic Unet implementation.

    D        ->           U
     \D      ->         U /
      \D      ->      U /
       \D    ->     U /
         \ MIDDLE /

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

        down_channel_counts (int List): Number of filters in the "down" layers of the encoder.
        middle_channel_counts (int List): Number of filters in the middle layer.
        up_channel_counts (int Lists): Number of filters in the "up" layers of the decoder.
    """

    def __init__(self, down_block, middle_block, up_block, n_input_channels=5,
                 down_channel_counts=[32, 64], middle_channel_count=128, up_channel_counts=[64, 32, 8]):
        super(Unet, self).__init__()
        assert len(down_channel_counts) <= len(up_channel_counts)
        self.scales = len(up_channel_counts)

        self.encoders = nn.ModuleList()
        last_output_channels = n_input_channels

        for out_channels in down_channel_counts:
            encoded_features = down_block(last_output_channels, out_channels)
            self.encoders.append(encoded_features)
            last_output_channels = out_channels

        middle = middle_block(last_output_channels, middle_channel_count)
        last_output_channels = middle_channel_count

        self.decoders = nn.ModuleList([middle])

        for out_channels, skip in zip(up_channel_counts, reversed(self.encoders)):

            in_channels = self.decoders[-1].out_channels + skip.out_channels
            self.decoders.append(up_block(in_channels, out_channels))

    def _merge_interpolate(
            self,
            previous_features: torch.Tensor,
            skip_connection: torch.Tensor,
            interpolate_size: List[int]) -> torch.Tensor:
        """Merges previous features with the skip connection then rescales."""
        # fuse previous layer and skip connection
        merged = torch.cat((previous_features, skip_connection), dim=previous_features.dim() - 3)

        # upsampling of previous features
        return interpolate(merged, interpolate_size)

    def forward(self, x):
        # down or encoding
        encoded_features = [x]

        for down_layer in self.encoders:
            x = down_layer(x)
            encoded_features.append(x)

        # middle layer
        decoded_features = [self.decoders[0](encoded_features[-1])]

        # up or decoding
        upscale_sizes = reversed([e.shape[-2:] for e in encoded_features[:-1]])

        for up_layer, skip, upscale_size in zip(self.decoders[1:], reversed(encoded_features), upscale_sizes):

            merged = self._merge_interpolate(decoded_features[-1], skip, upscale_size)

            decoded_features.append(up_layer(merged))

        return decoded_features


def interpolate(tensor: torch.Tensor, size: List[int]) -> torch.Tensor:
    """Generic interpolation for TNCHW and NCHW Tensors."""
    if tensor.dim() == 5:
        x4, batch_size = time_to_batch(tensor)
        y4 = F.interpolate(x4, size=size, mode='nearest')
        return batch_to_time(y4, batch_size)
    else:
        return F.interpolate(tensor, size=size, mode='nearest')
