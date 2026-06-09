# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

from .fnn_lightning_model import FNNClassificationModel
from .lightning_model import ClassificationModel
from .fnn_data_module import FNNClassificationDataModule
from .data_module import ClassificationDataModule

RNN_MODELS = ["ConvRNNClassifier", "LeNetClassifier"]
FNN_MODELS = ["Mobilenetv2Classifier", "SqueezenetClassifier"]


def get_model_names():
    return RNN_MODELS + FNN_MODELS


def is_rnn(model_name):
    assert model_name in get_model_names(), "The model architecture is not implemented!"
    if model_name in RNN_MODELS:
        return True
    else:
        return False


def get_model_class(model_name):
    if is_rnn(model_name):
        return ClassificationModel
    else:
        return FNNClassificationModel


def get_data_module(model_name):
    if is_rnn(model_name):
        return ClassificationDataModule
    else:
        return FNNClassificationDataModule


def get_model_inchannels(model_name, num_tbins, preprocess_channels):
    if model_name in RNN_MODELS:
        return preprocess_channels
    elif model_name in FNN_MODELS:
        return num_tbins * preprocess_channels
    else:
        raise NotImplementedError("The model architecture is not implemented!")
