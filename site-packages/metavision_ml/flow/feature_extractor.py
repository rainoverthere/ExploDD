# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Different macro level neural network architectures for flow regression in pytorch
"""
import torch
import torch.nn as nn

from metavision_core_ml.core.modules import ConvLayer, PreActBlock
from metavision_core_ml.core.temporal_modules import SequenceWise, ConvRNN

from ..core.unet_variants import UnetRegressor


AVAILABLES_ARCHS = ("eminet", "eminet_non_sep", "midinet", "midinet2")


def eminet(n_input_channels, base=16, scales=3, separable=True, rnn_cell="lstm", **kwargs):
    """
    Unet Regressor model with depthwise separable recurrent convolution for minimal footprint.

    Args:
        n_input_channels (int): number of channels in input features.
        scales (int): number of convolutional layers in the encoder (also in the decoder).
        base (int): base multiplier for the number of channels in each layer. For instance with scales = 2 and base = 4
            there will be [4, 8, 16, 8, 4] channels in the network.
        separable (bool): whether convolutions in the encoder and the decoders are depthwise separable convolutions.
            This saves a lot of parameters but makes the network harder to train.
        rnn_cell (string): type of cell used for the rnn, either 'lstm' or 'gru'.
    """
    down_channel_counts = [base * (2**factor) for factor in range(scales)]
    up_channel_counts = list(reversed(down_channel_counts))
    middle_channel_count = 2 * down_channel_counts[-1]

    def down(in_channels, out_channels):
        return SequenceWise(
            ConvLayer(in_channels, out_channels, 3, separable=separable, depth_multiplier=4, stride=2,
                      norm="none", activation="LeakyReLU"))

    def middle(in_channels, out_channels):
        return ConvRNN(in_channels, out_channels, 3, 1, cell=rnn_cell, separable=True)

    def up(in_channels, out_channels): return SequenceWise(ConvLayer(
        in_channels, out_channels, 3, separable=separable, stride=1, activation="LeakyReLU"))

    return UnetRegressor(down, middle, up, n_input_channels=n_input_channels, up_channel_counts=up_channel_counts,
                         down_channel_counts=down_channel_counts, middle_channel_count=middle_channel_count)


def eminet_non_sep(n_input_channels, base=16, scales=3, rnn_cell="lstm", **kwargs):
    """
    Constructor for Unet without depthwise separable convolutions.

    Args:
        n_input_channels (int): number of channels in input features.
        scales (int): number of convolutional layers in the encoder (also in the decoder).
        base (int): base multiplier for the number of channels in each layer. For instance with scales = 2 and base = 4
            there will be [4, 8, 16, 8, 4] channels in the network.
        rnn_cell (string): type of cell used for the rnn, either 'lstm' or 'gru'.
    """
    down_channel_counts = [base * (2**factor) for factor in range(scales)]
    up_channel_counts = list(reversed(down_channel_counts))
    middle_channel_count = 2 * down_channel_counts[-1]

    def down(in_channels, out_channels):
        return SequenceWise(
            ConvLayer(in_channels, out_channels, 3, separable=False, depth_multiplier=4, stride=2,
                      norm="none", activation="LeakyReLU"))

    def middle(in_channels, out_channels):
        return ConvRNN(in_channels, out_channels, 3, 1, separable=False, cell=rnn_cell)

    def up(in_channels, out_channels): return SequenceWise(ConvLayer(
        in_channels, out_channels, 3, stride=1, norm="BatchNorm2d", activation="LeakyReLU", separable=False))

    return UnetRegressor(down, middle, up, n_input_channels=n_input_channels, up_channel_counts=up_channel_counts,
                         down_channel_counts=down_channel_counts, middle_channel_count=middle_channel_count)


def midinet(n_input_channels, base=16, scales=3, separable=True, rnn_cell="lstm", **kwargs):
    """
    Unet Regressor model with Squeeze excitation layers.

    Args:
        n_input_channels (int): number of channels in input features.
        scales (int): number of convolutional layers in the encoder (also in the decoder).
        base (int): base multiplier for the number of channels in each layer. For instance with scales = 2 and base = 4
            there will be [4, 8, 16, 8, 4] channels in the network.
        separable (boolean): if True, uses depthwise separable convolutions for the forward convolutional layer.
        rnn_cell (string): type of cell used for the rnn, either 'lstm' or 'gru'.
    """
    down_channel_counts = [base * (2**factor) for factor in range(scales)]
    up_channel_counts = list(reversed(down_channel_counts))
    middle_channel_count = 2 * down_channel_counts[-1]

    def down(in_channels, out_channels): return SequenceWise(PreActBlock(
        in_channels, out_channels, stride=2))

    def middle(in_channels, out_channels):
        return ConvRNN(in_channels, out_channels, 3, 1, separable=separable, cell=rnn_cell)

    def up(in_channels, out_channels): return SequenceWise(ConvLayer(
        in_channels, out_channels, 3, separable=separable, stride=1, activation="LeakyReLU"))

    return UnetRegressor(down, middle, up, n_input_channels=n_input_channels, up_channel_counts=up_channel_counts,
                         down_channel_counts=down_channel_counts, middle_channel_count=middle_channel_count)


def midinet2(n_input_channels, base=16, scales=3, separable=True, rnn_cell="lstm", depth=1):
    """
    midinet with a fine-tuned middle block (convRNN with tunable depth + residual connection)

    Args:
        n_input_channels (int): number of channels in input features.
        scales (int): number of convolutional layers in the encoder (also in the decoder).
        base (int): base multiplier for the number of channels in each layer. For instance with scales = 2 and base = 4
            there will be [4, 8, 16, 8, 4] channels in the network.
        separable (boolean): if True, uses depthwise separable convolutions for the forward convolutional layer.
        depth (int): number of convRNN layers in the middle part of the Unet. Must be one or above.
        rnn_cell (string): type of cell used for the rnn, either 'lstm' or 'gru'.
    """
    down_channel_counts = [base * (2**factor) for factor in range(scales)]
    up_channel_counts = list(reversed(down_channel_counts))
    middle_channel_count = 2 * down_channel_counts[-1]

    def down(in_channels, out_channels): return SequenceWise(PreActBlock(
        in_channels, out_channels, stride=2))

    class middle(nn.Module):
        def __init__(self, in_channels, out_channels, depth=depth):
            super(middle, self).__init__()
            self.conv1 = SequenceWise(ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                                norm="none", activation='ReLU', separable=separable))
            self.convs = nn.ModuleList([ConvRNN(out_channels, out_channels, 3, 1,
                                                separable=separable, norm="none", cell=rnn_cell)
                                        for i in range(depth)])

            self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
            self.out_channels = out_channels

        def forward(self, x):
            x = self.conv1(x)
            for i, conv in enumerate(self.convs):
                x = x + self.resweight[i] * torch.relu(conv(x))
            return x

    def up(in_channels, out_channels): return SequenceWise(ConvLayer(
        in_channels, out_channels, 3, separable=separable, stride=1, activation="LeakyReLU"))

    return UnetRegressor(down, middle, up, n_input_channels=n_input_channels, up_channel_counts=up_channel_counts,
                         down_channel_counts=down_channel_counts, middle_channel_count=middle_channel_count)
