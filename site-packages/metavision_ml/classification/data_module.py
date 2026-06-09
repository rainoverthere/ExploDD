# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This data module is a wrapper around SequentialDataLoader for
the classification module.
"""

import os
import glob
from functools import partial

import numpy as np

import torch
from torchvision import transforms
import pytorch_lightning as pl

from metavision_ml.data.sequential_dataset import SequentialDataLoader
from metavision_ml.data.box_processing import load_box_events, create_class_lookup


def load_classes(metadata, batch_start_time, duration, tensor, **kwargs):
    """
    Function to fetch boxes and preprocess them. Should be passed to a SequentialDataLoader.

    Examples:
        >>> from functools import partial
        >>> n_classes = 21
        >>> class_lookup = np.arange(n_classes)  # each class is mapped to itself
        >>> load_boxes_function = partial(load_boxes, class_lookup=class_lookup)

    Args:
        metadata (object): Record details.
        batch_start_time (int): (us) Where to seek in the file to load corresponding bounding boxes
        duration (int): (us) How long to load events from bounding box file
        tensor (np.ndarray): Current preprocessed input, can be used for data dependent preprocessing,
            for instance remove boxes without any features in them.
        class_lookup (np.array): Look up array for class indices.
        labelling_delta_t (int): Indicates the period of labelling in order to only consider time bins
            with actual labels when computing the loss.


    Returns:

        labels (List[np.ndarray]): List of structured array of dtype EventBbox corresponding to each time bins.
        frame_is_labeled (np.ndarray): This boolean mask array of *length* num_tbins indicates whether the frame
        contains a label. It is used to differentiate between time bins that actually contain an empty label
        (for instance no bounding boxes) from time bins that weren't labeled due to cost constraints. The latter time
        bins shouldn't contribute to supervised losses used during training.

    """

    assert 'class_lookup' in kwargs, "you need to provide a class_lookup array corresponding to your labels!"
    class_lookup = kwargs['class_lookup']
    label_delta_t = kwargs.get('label_delta_t', 10000)
    use_label_freq = kwargs.get('use_label_freq', True)

    # size of the feature tensor
    num_tbins = tensor.shape[0]
    delta_t = duration // num_tbins

    box_events = load_box_events(metadata, batch_start_time, duration)

    if len(box_events) > 1 and use_label_freq:
        dt = box_events['t'][1:] - box_events['t'][:-1]
        test = np.unique(dt)
        assert len(test) == 1, "label durations not unique"
        test_case = test[0]
        assert test_case == label_delta_t, f"label delta_t current sequence ({test_case}) != expected label_delta_t ({label_delta_t})"

    tbin_indices, indices = np.unique((box_events['t'] - batch_start_time - 1) // delta_t, return_index=True)
    class_ids = class_lookup[box_events[indices]['class_id']]
    labels = torch.zeros(num_tbins, dtype=torch.long)
    labels[torch.LongTensor(tbin_indices)] = torch.LongTensor(class_ids)
    if use_label_freq:
        frame_is_labeled = torch.arange(batch_start_time, batch_start_time + duration, delta_t) % label_delta_t == 0

    # if label == -1 we want to ignore this frame altogether
    frame_is_labeled &= labels >= 0
    assert len(labels) == len(frame_is_labeled)
    return labels, frame_is_labeled


class ClassificationDataModule(pl.LightningDataModule):
    """
    Data Module for classification
    Applies some data augmentation on top.
    """

    def __init__(self, hparams, data_dir: str = ""):
        super().__init__()
        self.data_dir = data_dir
        self.save_hyperparameters(hparams)
        self._extensions = ['h5', 'dat']
        self.train_files, self.test_files, self.val_files = [], [], []

    def train_dataloader(self):
        if not self.train_files:
            self.train_files = self._from_folder("train")
            if self.hparams.train_plus_val:
                self.train_files += self._from_folder("val")
        return self._get_dataloader(
            self.train_files, mode="train", data_aug=self.hparams.use_data_augmentation, shuffle=self.hparams.shuffle)

    def test_dataloader(self):
        if not self.test_files:
            self.test_files = self._from_folder("test")
        return self._get_dataloader(self.test_files, mode="test", data_aug=False, shuffle=self.hparams.shuffle)

    def val_dataloader(self):
        if not self.val_files:
            if self.hparams.train_plus_val:
                self.val_files = self._from_folder("test")
            else:
                self.val_files = self._from_folder("val")
        return self._get_dataloader(self.val_files, mode="val", data_aug=False, shuffle=self.hparams.shuffle)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch['inputs'] = batch['inputs'].to(device=device)
        batch['mask_keep_memory'] = batch['mask_keep_memory'].to(device=device)
        return batch

    def _from_folder(self, folder):
        files = []
        for ext in self._extensions:
            files += glob.glob(os.path.join(self.data_dir, folder, '*.' + ext))
        return files

    def _get_dataloader(self, files, mode, data_aug=False, shuffle=False):
        """Helper functions used to create a dataloader used by the public functions."""
        data_augs = None
        if data_aug:
            data_augs = transforms.Compose([transforms.RandomApply(
                [transforms.RandomAffine(
                    degrees=(-20, 20),
                    translate=(0.2, 0.2),
                    scale=(0.75, 1.5)),
                 transforms.RandomResizedCrop(
                    (self.hparams.height, self.hparams.width),
                    scale=(0.8, 1.2))],
                p=0.75),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)])

        # We remove 'empty' from the classes
        self.hparams.classes = [label for label in self.hparams.classes if label != 'empty']

        label_map_path = os.path.join(self.data_dir, 'label_map_dictionary.json')
        class_lookup = create_class_lookup(label_map_path, self.hparams.classes)

        load_labels_fn = partial(load_classes, class_lookup=class_lookup, label_delta_t=self.hparams.label_delta_t,
                                 use_label_freq=self.hparams.use_label_freq)

        array_dim = (self.hparams.num_tbins, self.hparams.preprocess_channels, self.hparams.height, self.hparams.width)

        dataloader = SequentialDataLoader(
            files, self.hparams.delta_t, self.hparams.preprocess, array_dim, load_labels=load_labels_fn,
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=shuffle, padding=mode !=
            "train", transforms=data_augs)

        return dataloader
