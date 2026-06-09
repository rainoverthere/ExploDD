# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Models for classification
"""
import torch
import torch.nn as nn
from metavision_core_ml.core.modules import ConvLayer
from metavision_core_ml.core.temporal_modules import SequenceWise, ConvRNN
from torchvision import models


class ConvRNNClassifier(nn.Module):
    """
    ConvRNN Classifier

    Feed-Forward + RNN light model

    Args:
        cin (int): aaa
        base (int): bbb
        cout (int): ccc
        num_classes (int): ddd
    """

    def __init__(self, cin=1, base=16, cout=256, num_classes=2):
        super(ConvRNNClassifier, self).__init__()
        self.cin = cin
        self.base = base
        self.conv1 = SequenceWise(nn.Sequential(
            ConvLayer(cin, self.base, kernel_size=3, stride=1, padding=1, norm='BatchNorm2d'),
            ConvLayer(self.base, self.base * 2, kernel_size=3, stride=2, padding=1, norm='BatchNorm2d'),
            ConvLayer(self.base * 2, self.base * 2, kernel_size=3, stride=1, padding=1, norm='BatchNorm2d'),
            ConvLayer(self.base * 2, self.base * 4, kernel_size=3, stride=2, padding=1, norm='BatchNorm2d'),
            ConvLayer(self.base * 4, self.base * 4, kernel_size=3, stride=1, padding=1, norm='BatchNorm2d'),
        ))
        self.levels = 2
        self.conv2 = nn.Sequential(ConvRNN(self.base * 4, self.base * 8, stride=2, separable=False),
                                   ConvRNN(self.base * 8, self.base * 16, stride=2, separable=False))

        self.head = SequenceWise(nn.Sequential(
            nn.Conv2d(self.base * 16, cout, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(cout, num_classes, kernel_size=3, stride=1, padding=1)))

    def forward(self, x):
        x = self.conv1(x)
        features = self.conv2(x)
        ndim = len(features.shape)
        logits = torch.mean(torch.mean(self.head(features), ndim - 1), ndim - 2)
        return logits

    @torch.jit.export
    def get_probas(self, x):
        logits = self(x)
        return nn.functional.softmax(logits, dim=-1)

    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2.modules():
            if hasattr(module, "reset_all"):
                module.reset_all()


class LeNetClassifier(nn.Module):
    """
    LeNet RNN
    """

    def __init__(self, cin=1, base=6, cout=256, num_classes=2):
        super().__init__()
        self.cin = cin
        self.base = base
        self.lenet = SequenceWise(nn.Sequential(
            ConvLayer(cin, 6, 5, 1, 0, bias=False),
            nn.AvgPool2d(2),
            ConvLayer(6, 16, 5, 1, 0, bias=False),
            nn.AvgPool2d(2),
            ConvLayer(16, 120, 5, 1, 0, bias=False)
        ))
        self.bottleneck_rnn = nn.Sequential(
            SequenceWise(ConvLayer(120, 32, 1, 1, 0)),
            ConvRNN(32, 32),
            SequenceWise(ConvLayer(32, 120, 1, 1, 0))
        )
        self.head = SequenceWise(nn.Sequential(
            nn.Conv2d(120, cout, 3, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(cout, num_classes, 3, 1, 0)))

    def forward(self, x):
        features = self.lenet(x)
        features = features + self.bottleneck_rnn(features)
        ndim = len(features.shape)
        logits = torch.mean(torch.mean(self.head(features), ndim - 1), ndim - 2)
        return logits

    @torch.jit.export
    def get_probas(self, x):
        logits = self(x)
        return nn.functional.softmax(logits, dim=-1)

    @torch.jit.export
    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        for name, module in self.bottleneck_rnn.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)


def _make_divisible(v, divisor, min_value=None):
    """
    :param v: float
    :param divisor: integer
    :param min_value: integer
    :return: integer
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Mobilenetv2Classifier(nn.Module):
    """
    Mobilenetv2

    Modified Feed-Forward architecture
    """

    def __init__(self, cin=2, width_mul=1., num_classes=2, round_nearest=8, **kwargs):
        super(Mobilenetv2Classifier, self).__init__()
        self.cin = cin
        self.model = models.MobileNetV2(num_classes=num_classes, width_mult=width_mul, round_nearest=round_nearest)
        layer_out_1 = _make_divisible(32 * width_mul, round_nearest)
        self.model.features[0][0] = nn.Conv2d(cin, layer_out_1, kernel_size=(3, 3), stride=(2, 2),
                                              padding=(1, 1), bias=False)

    def forward(self, x):
        logits = self.model(x)
        return logits

    @torch.jit.export
    def get_probas(self, x):
        logits = self(x)
        return nn.functional.softmax(logits, dim=-1)


class SqueezenetClassifier(nn.Module):
    """
    Mobilenetv2

    Modified Feed-Forward architecture
    """

    def __init__(self, cin=2, num_classes=2, **kwargs):
        super(SqueezenetClassifier, self).__init__()
        self.cin = cin
        self.model = models.squeezenet1_1(num_classes=num_classes)
        self.model.features[0] = nn.Conv2d(cin, 64, kernel_size=3, stride=2)

    def forward(self, x):
        logits = self.model(x)
        return logits

    @torch.jit.export
    def get_probas(self, x):
        logits = self(x)
        return nn.functional.softmax(logits, dim=-1)
