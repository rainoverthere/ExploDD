# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
This class allows to stream a dataset of .raw or .dat

This is yet another example how to use data.multistream_dataloader
Here we go further and integrate with the same interface as SequentialDataLoader
"""
import os

from functools import partial
import numpy as np
import torch

from metavision_ml.data.cd_processor_iterator import CDProcessorIterator, HDF5Iterator
from metavision_ml.data.scheduler import FileMetadata, get_duration, _get_original_size_file
from metavision_ml.preprocessing import get_preprocess_dict
from metavision_ml.data.sequential_dataset_common import load_labels_stub, collate_fn
from metavision_ml.data.sequential_dataset_common import show_dataloader
from metavision_core_ml.data.stream_dataloader import split_batch_size, split_dataset_sizes
from metavision_core_ml.data.stream_dataloader import StreamDataset, StreamDataLoader
from functools import partial


class CDProcessorDatasetIterator(object):
    """This iterator reads events or preprocessed tensors, computes tensors, load labels and retrieves them
    difference with sequential_dataset_v1 is that load_labels cannot be a pure function
    it has to be a class
    """

    def __init__(
            self, path, height_out, width_out, load_labels, mode, n_events, delta_t, num_tbins,
            preprocess_function_name, preprocess_kwargs={}, start_ts=0, max_duration=None, transforms=None,
            base_seed=None):
        if mode != "delta_t":
            raise NotImplementedError("CDProcessorDataLoader only works in delta_t mode for the time being")
        self.path = path
        ext = os.path.splitext(path)[1]
        if ext in {'.dat', '.raw'}:
            self._iterator = CDProcessorIterator(path, preprocess_function_name, mode, start_ts, max_duration,
                                                 delta_t, n_events, num_tbins, preprocess_kwargs,
                                                 transforms=transforms, base_seed=base_seed)
        elif ext == '.h5':
            self._iterator = HDF5Iterator(path, num_tbins, preprocess_kwargs, transforms=transforms,
                                          base_seed=base_seed)
            assert self._iterator.preprocess == preprocess_function_name
            assert self._iterator.delta_t == delta_t

        self.height = height_out
        self.width = width_out

        delta_t = delta_t if mode == "delta_t" else 50000
        self.num_tbins = num_tbins
        duration = get_duration(path)
        self.metadata = FileMetadata(path, duration, delta_t, self.num_tbins)
        self.load_labels = [load_labels_stub, load_labels][load_labels is not None]

    def resize(self, tensor):
        height, width = tensor.shape[-2:]
        if height != self.height or width != self.width:
            tensor = torch.nn.functional.interpolate(tensor, size=(
                self.height, self.width), mode='bilinear', align_corners=True)
        return tensor

    def __iter__(self):
        keep_memory = 0
        start_ts = self._iterator.get_time()
        for tensor in self._iterator:
            assert tensor.ndim == 4, 'tensor format should be T,C,H,W'
            if len(tensor) != self.num_tbins:
                raise StopIteration
            tensor = self.resize(tensor)
            end_ts = self._iterator.get_time()

            duration = end_ts - start_ts
            self.metadata.delta_t = int(np.ceil(duration / self.num_tbins))
            duration = int(self.num_tbins * self.metadata.delta_t)
            labels, frame_is_labeled = self.load_labels(self.metadata, start_ts, duration, tensor)

            video_infos = (self.metadata, start_ts, duration)
            yield (tensor, labels, keep_memory, frame_is_labeled, video_infos)
            keep_memory = 1
            start_ts = end_ts


class CDProcessorDataLoader(StreamDataLoader):
    """Attempt at doing the same interface than SequentialDataloader
    but using multistream_dataloader implementation.
    """

    def __init__(self, files, mode, delta_t, n_events, max_duration, preprocess_function_name, num_tbins, batch_size,
                 num_workers=2, height=None, width=None, preprocess_kwargs={}, load_labels=None, padding_mode='zeros',
                 transforms=None):

        assert len(files), "list of files is empty"
        if mode != "delta_t":
            raise NotImplementedError("CDProcessorDataLoader only works in delta_t mode for the time being")
        self.preprocess_dict = get_preprocess_dict(preprocess_function_name)
        self.height, self.width = height, width
        if self.height is None or self.width is None:
            self.height, self.width = _get_original_size_file(files[0])

        iterator_fun = partial(CDProcessorDatasetIterator, height_out=self.height, width_out=self.width,
                               load_labels=load_labels, mode=mode, n_events=n_events,
                               preprocess_function_name=preprocess_function_name, delta_t=delta_t, num_tbins=num_tbins,
                               max_duration=max_duration, preprocess_kwargs=preprocess_kwargs, transforms=transforms,
                               base_seed=None)
        fill_value = CDProcessorDataLoader._init_fill_value(preprocess_function_name, self.height, self.width, delta_t,
                                                            num_tbins, padding_mode)

        dataset = StreamDataset(files, iterator_fun, batch_size, padding_mode, fill_value)
        super().__init__(dataset, num_workers, collate_fn)

    def get_vis_func(self):
        """Returns the visualization function corresponding to the preprocessing being used."""
        return self.preprocess_dict['viz']

    def show(self, viz_labels=None):
        return show_dataloader(self, self.height, self.width, self.get_vis_func(), viz_labels)

    @staticmethod
    def _init_fill_value(preprocess_function_name, height, width, delta_t, num_tbins, padding_mode='zeros'):

        preprocess_dict = get_preprocess_dict(preprocess_function_name)
        c = preprocess_dict['cin']

        zeros_tensor = torch.zeros((num_tbins, c, height, width), dtype=torch.float32)
        labels = [torch.empty(0) for _ in range(num_tbins)]
        frame_is_labeled = np.zeros((len(labels)), dtype=bool)
        keep_memory = 0
        video_infos = (FileMetadata("", 0, delta_t, num_tbins), 0, delta_t)

        fill_value = (zeros_tensor, labels, keep_memory, frame_is_labeled, video_infos)
        fill_value = fill_value if padding_mode == 'zeros' else None

        return fill_value
