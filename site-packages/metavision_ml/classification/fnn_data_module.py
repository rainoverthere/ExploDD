# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This data module is a wrapper around NonSequentialDataLoader for
the classification module.
"""

import os
import glob
from functools import partial

import torch
from torchvision import transforms
import pytorch_lightning as pl

from metavision_ml.data.nonsequential_dataset import NonSequentialDataset
from metavision_ml.data.label_loading import load_classes


class FNNClassificationDataModule(pl.LightningDataModule):
    """
    FNN Data Module for classification
    Applies some data augmentation on top.
    """

    def __init__(self, hparams, data_dir: str = ""):
        super().__init__()
        self.data_dir = data_dir
        self.save_hyperparameters(hparams)
        self._extensions = ['h5', 'dat']
        self.train_files, self.test_files, self.val_files = [], [], []
        self.label_delta_t = hparams.label_delta_t
        self.allow_labels_interpolation = hparams.allow_labels_interpolation

    def train_dataloader(self):
        if not self.train_files:
            self.train_files = self._from_folder("train")
            if self.hparams.train_plus_val:
                self.train_files += self._from_folder("val")
        base_seed = self.trainer.current_epoch
        return self._get_dataloader(
            self.train_files, mode="train", data_aug=self.hparams.use_data_augmentation,
            shuffle=True, base_seed=base_seed)

    def test_dataloader(self):
        if not self.test_files:
            self.test_files = self._from_folder("test")
        return self._get_dataloader(self.test_files, mode="test", data_aug=False, shuffle=False)

    def val_dataloader(self):
        if not self.val_files:
            if self.hparams.train_plus_val:
                self.val_files = self._from_folder("test")
            else:
                self.val_files = self._from_folder("val")
        return self._get_dataloader(self.val_files, mode="val", data_aug=False, shuffle=False)

    def _from_folder(self, folder):
        files = []
        for ext in self._extensions:
            files += glob.glob(os.path.join(self.data_dir, folder, '*.' + ext))
        return files

    def _get_dataloader(self, files, mode, data_aug=False, shuffle=False, base_seed=0):
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

        label_map_path = os.path.join(self.data_dir, 'label_map_dictionary_fnn.json')

        load_labels = partial(load_classes, label_map_path=label_map_path)
        array_dim = (self.hparams.num_ev_reps, self.hparams.preprocess_channels,
                     self.hparams.height, self.hparams.width)
        dataset = NonSequentialDataset(file_paths=files, array_dim=array_dim,
                                       transforms=data_augs, base_seed=base_seed, load_labels=load_labels,
                                       label_delta_t=self.label_delta_t,
                                       allow_labels_interpolation=self.allow_labels_interpolation)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
                                                 num_workers=self.hparams.num_workers, pin_memory=False)

        return dataloader
