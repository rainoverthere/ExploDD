# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This module creates data loader to load event based data for self supervised flow.
"""
import os
import glob

import h5py
import numpy as np

import torch
from torchvision import transforms
import pytorch_lightning as pl


from ..data import SequentialDataLoader
from ..data.sequential_dataset import load_labels_stub


def load_labels_flow(metadata, start_time, duration, tensor):
    """Loads flow labels from an HDF5 file

    Args:
        metadata (FileMetadata): This class contains information about the sequence that is being read.
            Ideally the path for the labels should be deducible from `metadata.path`.
        start_time (int): Time in us in the file at which we start reading.
        duration (int): Duration in us of the data we need to read from said file.
        tensor (torch.tensor): Torch tensor of the feature for which labels are loaded.
            It can be used for instance to filter out the labels in area where there is no events.

    Returns:
        labels should be indexable by time bin (to differentiate the labels of each time bin). It could
            therefore be a list of length *num_tbins*.
        (boolean nd array): This boolean mask array of *length* num_tbins indicates
            whether the frame contains a label. It is used to differentiate between time_bins that actually
            contain an empty label (for instance no bounding boxes) from time bins that weren't labeled due
            to cost constraints. The latter timebins shouldn't contribute to supervised losses used during
            training.
    """
    # initialize empty values
    timestamps = np.arange(start_time, start_time + duration, metadata.delta_t)
    labels = [torch.empty(0) for _ in timestamps]
    frame_is_labeled = np.zeros((len(labels)), dtype=bool)

    label_file_path = metadata.path.replace('_td.dat', '_flow.h5').replace(".raw", '_flow.h5')
    with h5py.File(label_file_path, 'r') as f:
        dataset = f['flow']
        dataset_delta_t = dataset.attrs["delta_t"]
        frame_is_labeled = ((timestamps / dataset_delta_t % 1) == 0) * (timestamps > 0)
        labeled_indices = np.where(frame_is_labeled)[0]
        data = torch.from_numpy(dataset[start_time // dataset_delta_t + labeled_indices - 1])
        for ind, d in zip(labeled_indices, data):
            labels[ind] = d

    return labels, frame_is_labeled


class FlowDataModule(pl.LightningDataModule):
    """This data module handles unlabeled event based data as well as labeled validation data.

    The data_dir is meant to contain two directories `train`and  `val` containing HDF5 files of preprocessed
    event features.

    If a test_data_dir is provided it means labeled flow (usually synthetic data) is provided.
    In this case it is *_td.dat files along with *_flow.h5 dense flow labels.

    Args:
        hparams: hyperparameters from the corresponding FlowModel.
        data_dir (str): path towards the directory containing `train` `val` and `test` folder of HDF5 files.
        test_data_dir (str): optional path towards a directory containing DAT event files and HDF5 dense flow
            annotations.
    """

    def __init__(self, hparams, data_dir: str = "", test_data_dir: str = ""):
        super().__init__()
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.save_hyperparameters(hparams)
        self._extensions = ['h5', 'dat']
        if self.test_data_dir:
            files = glob.glob(os.path.join(test_data_dir, "*dat"))
            self.val_dataloader = lambda: self.get_dataloader(files, data_aug=False, labels=True)
            self.test_dataloader = lambda: self.get_dataloader(files, data_aug=False, labels=True)

    def setup(self, stage=None):
        """During this stage the data module parses the train and val folders."""
        self.train_files = self._from_folder('train')
        self.test_files = self._from_folder('val')

    def train_dataloader(self):
        return self._get_dataloader(self.train_files, data_aug=self.hparams.data_aug)

    def val_dataloader(self):
        return self._get_dataloader(self.test_files, data_aug=self.hparams.data_aug)

    def test_dataloader(self):
        return self._get_dataloader(self.test_files, data_aug=False)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch['inputs'] = batch['inputs'].to(device=device)
        batch['mask_keep_memory'] = batch['mask_keep_memory'].to(device=device)

        return batch

    def _from_folder(self, folder):
        files = []
        for ext in self._extensions:
            files += glob.glob(os.path.join(self.data_dir, folder, '*.' + ext))
        return files

    def _get_dataloader(self, files, data_aug=False, labels=False):
        """Helper functions used to create a dataloader used by the public functions."""
        data_augs = None
        if data_aug:
            data_augs = transforms.Compose(
                [transforms.RandomApply([transforms.RandomRotation([-90, 90])], p=0.75),
                 transforms.RandomHorizontalFlip(p=0.5)])
        load_labels = load_labels_flow if labels else load_labels_stub

        dataloader = SequentialDataLoader(
            files, self.hparams.delta_t, self.hparams.preprocess, self.hparams.array_dim,
            transforms=data_augs, batch_size=self.hparams.batch_size, load_labels=load_labels,
            num_workers=self.hparams.num_workers, preprocess_kwargs=self.hparams.preprocess_kwargs)

        return dataloader
