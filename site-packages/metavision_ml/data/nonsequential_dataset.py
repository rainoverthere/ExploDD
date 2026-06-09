# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Nonsequential dataset to load events from h5 files and corresponding labels.
"""

import os
import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data
from metavision_ml.data.transformations import transform_ev_tensor


def get_num_frames(path):
    """
    Args:
        path(string): The path of the h5 file containing event frames.

    Returns:
        num_frames(int): The number of event frames in the h5 file.
    """
    file_extension = os.path.splitext(path)[1].lower()
    assert file_extension in ('.hdf5', ".h5"), "The h5 ext is the only supported format!"
    with h5py.File(path, "r") as f:
        num_frames = f['data'].shape[0]
    if num_frames is None:
        raise ValueError("Can't measure the number of frames. Please check the path!", os.path.basename(path))
    return num_frames


def get_frame_delta_t(path):
    """
    Args:
        path(string): The path of the h5 file of event frames.

    Returns:
        frame_delta_t(int): The duration of each events frame.
    """
    file_extension = os.path.splitext(path)[1].lower()
    assert file_extension in ('.hdf5', ".h5"), "The h5 ext is the only supported format!"
    with h5py.File(path, "r") as f:
        assert f["data"].attrs.get("mode", "delta_t") == "delta_t", "only delta_t mode supported"
        assert f["data"].attrs["delta_t"] > 0
        frame_delta_t = f['data'].attrs["delta_t"]
    return frame_delta_t


def get_precomputed_dataset_mode(path):
    """
    Args:
        path(string): The path of the h5 file of event frames.

    Returns:
        mode(str): The mode used to generate the dataset ("delta_t" or "n_events").
    """
    file_extension = os.path.splitext(path)[1].lower()
    assert file_extension in ('.hdf5', ".h5"), "The h5 ext is the only supported format!"
    with h5py.File(path, "r") as f:
        mode = f["data"].attrs.get("mode", "delta_t")
        assert mode in ["delta_t", "n_events"], "only n_events and delta_t mode are supported."
    return mode


class NonSequentialDataset(data.Dataset):
    def __init__(self, file_paths, array_dim, transforms=None, base_seed=0, load_labels=None,
                 label_delta_t=None, allow_labels_interpolation=False):
        """
        Args:
            file_paths(list): The paths of h5 files containing event frames.
            array_dim(tuple of int): (num_ev_reps, channels, height, width)
                                     num_ev_reps is a number indicates how many event frames are used to form a tensor.
                                     (channels, height, width) will be used to resize the loaded event frames. 
            transforms (torchvision Transforms): transformations applied to each frame and channel of an event tensor.
            base_seed(int): A seed to control the randomness of the transformations
            load_labels(fun): The function to load the labels
            label_delta_t(int):
            allow_labels_interpolation(bool):
        """
        self.file_paths = file_paths
        self.array_dim = array_dim
        self.num_ev_reps = array_dim[0]
        self.transforms = transforms
        self.base_seed = base_seed
        self.load_labels = load_labels
        self.label_delta_t = label_delta_t
        self.allow_labels_interpolation = allow_labels_interpolation
        # Build a counted frame table to sample
        self.num_frames = np.array([get_num_frames(file_path) for file_path in self.file_paths])
        # To avoid sampling event frames in two different files,
        # (num_ev_reps - 1) are reduced from num_frames of each file

        # remove files with too few frames
        ok_idxs = self.num_frames >= self.num_ev_reps
        self.num_frames = self.num_frames[ok_idxs]
        assert len(self.num_frames) > 0, "There are less event frames in all files than one plans to read at a single time!!!"
        self.file_paths = [ok_path for indx, ok_path in enumerate(self.file_paths) if ok_idxs[indx]]
        assert np.all(self.num_frames >= self.num_ev_reps)
        self.valid_num_frames = self.num_frames - (self.num_ev_reps - 1)
        self.cumulative_num_frames = np.cumsum(self.valid_num_frames)

        # Get mode for each file
        all_modes = np.array([get_precomputed_dataset_mode(file_path) for file_path in self.file_paths])
        assert np.all(all_modes == all_modes[0]), "precomputed frames should all have same mode"
        self.precomputed_dataset_mode = all_modes[0]
        if self.precomputed_dataset_mode == "delta_t":
            # Get the duration of each event frame
            all_dts = np.array([get_frame_delta_t(file_path) for file_path in self.file_paths])
            assert np.all(all_dts == all_dts[0])
            self.frame_delta_t = all_dts[0]
            assert self.frame_delta_t > 0, "precomputed_dataset_mode=='delta_t', but delta_t==0. Maybe you are using a dataset with precomputed_dataset_mode=='n_events'?"
        elif self.precomputed_dataset_mode == "n_events":
            self.frame_delta_t = 0

    def _load_frames_from_hdf5(self, file_path, start_frame, num_ev_reps):
        frame_timestamps = None
        with h5py.File(file_path, "r") as f:
            array = f['data'][start_frame:start_frame + num_ev_reps]
            if f['data'].attrs.get('store_as_uint8', False):
                array = array.astype(np.float32) / 255
            if self.precomputed_dataset_mode == "n_events":
                # we will need frame interval, so we add in front the time of at start_frame -1
                start_ts_idx = start_frame - 1 if start_frame >= 1 else 0
                frame_timestamps = f['last_timestamps'][start_ts_idx:start_frame + num_ev_reps]
                # if first index, we assume start_ts is 0 and add it in front (n_events with start_ts>0 not supported yet)
                if start_frame == 0:
                    start_ts = 0
                    frame_timestamps = np.insert(frame_timestamps, 0, int(start_ts))
        return array, frame_timestamps

    def _load_dummy_labels(self, num_ev_reps):
        # The number of labels is equal to the number of event frames
        return [np.empty(0) for _ in range(num_ev_reps)]

    def _resize_frames(self, frames, array_dim):
        """
        Args:
            frames(np.array): the orginal frames loaded from the h5 files.
            array_dim(tuple of int): the dimension to resize the frame
        Return:
            resized_frames(np.array): the resized frames
        """
        resized_frames = np.zeros(array_dim, dtype=frames.dtype)
        h_src, w_src = frames.shape[-2:]
        h_dst, w_dst = array_dim[-2:]

        assert len(frames) == self.num_ev_reps, f"number of event representation incorrect {len(frames)}," \
            "should be {self.num_ev_reps}"
        if h_src != h_dst or w_src != w_dst:
            for frame_idx in range(self.num_ev_reps):
                for channel_idx in range(array_dim[1]):
                    resized_frames[frame_idx, channel_idx] = cv2.resize(frames[frame_idx, channel_idx],
                                                                        (w_dst, h_dst),
                                                                        resized_frames[frame_idx, channel_idx],
                                                                        interpolation=cv2.INTER_AREA)
        else:
            resized_frames[...] = frames  # frames is exactly what we want
        return resized_frames

    def __getitem__(self, index):
        # Here we locate which file to sample
        # If the cumulative_num_frames is [37, 84] and the index is 37, the returned file_idx is 1 not 0
        file_idx = np.searchsorted(self.cumulative_num_frames, index, side='right')
        current_file_path = self.file_paths[file_idx]

        # Here we locate the frame in the file from which we start to sample
        if file_idx == 0:
            start_frame = index
        else:
            start_frame = index - self.cumulative_num_frames[file_idx - 1]

        # Get frames
        precomputed_frames, frame_timestamps = self._load_frames_from_hdf5(
            current_file_path, start_frame, self.num_ev_reps)
        resized_frames = self._resize_frames(precomputed_frames, self.array_dim)
        ev_reps_tensor = torch.from_numpy(resized_frames)
        if self.transforms is not None:
            ev_reps_tensor = transform_ev_tensor(ev_reps_tensor, current_file_path,
                                                 self.transforms, base_seed=self.base_seed + index)

        # Get labels
        if self.load_labels is None:
            labels = self._load_dummy_labels(self.num_ev_reps)
        else:
            labels = self.load_labels(
                current_file_path, start_frame, self.num_ev_reps, self.frame_delta_t,
                frame_timestamps=frame_timestamps, label_delta_t=self.label_delta_t,
                allow_labels_interpolation=self.allow_labels_interpolation)
        labels_tensor = torch.Tensor(labels)

        return (ev_reps_tensor, labels_tensor)

    def __len__(self):
        return self.cumulative_num_frames[-1]
