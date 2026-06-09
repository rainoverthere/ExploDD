# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Iterator of feature tensor for a source of input events.

"""
import time
import math
import cv2
import h5py
import numpy as np
import torch

from metavision_core.event_io import EventsIterator
from ..preprocessing import CDProcessor, get_preprocess_dict
from .transformations import transform_sequence
from ..utils.main_tools import check_input_power_2


class ProcessorIterator(object):
    def __init__(self):
        raise NotImplementedError("Abstract class")

    def cuda(self, device=torch.device('cuda')):
        """Sets the Preprocessor to copy tensors to GPU memory before returning them.

        Args:
            device (torch.device): The destination GPU device. Defaults to the current CUDA device.
        """
        assert torch.cuda.is_available()
        self.device = device
        return self

    def cpu(self):
        """Sets the Preprocessor to leave tensors on CPU."""
        self.device = torch.device('cpu')
        return self

    def to(self, device):
        """Sets the Preprocessor to copy tensors to the given device before returning them.

        Args:
            device (torch.device): The destination GPU device. For instance `torch.device('cpu')`
                or torch.device('cuda').
        """
        assert isinstance(device, torch.device)
        self.device = device
        return self

    def get_time(self):
        """Cut Inner Reader Time"""
        raise NotImplementedError()

    def set_base_seed(self, base_seed):
        """
        Attributes a base seed that is added to transforms when applicable.

        Changing this seed after an epoch is ended allow to differentiate between epochs while maintaning
        temporal coherency of the spatial transformation.

        Args:
            base_seed (int): seed to be added to the hash before drawing a random transformation. if None, will use the
                time instead.
        """
        self.base_seed = int(base_seed) if base_seed is not None else int(time.time() * 1e6)


class CDProcessorIterator(ProcessorIterator):
    """
    Provides feature tensors (torch.Tensor) at regular intervals.

    Relies on the EventsIterator class. The different behaviours of EventsIterator can be leveraged.

    Args:
        path (string): Path to the file to read, or empty for a camera.
        preprocess_function_name (string): Name of the preprocessing function used to turn events into
            features. Can be any of the functions present in metavision_ml.preprocessing or one registered
            by the user.
        mode (str): Mode of Streaming (n_event, delta_t, mixed)
        start_ts (int): Start of EventIterator
        max_duration (int): Total Duration of EventIterator
        delta_t (int): Duration of used events slice in us.
        num_tbins (int): Number of TimeBins
        preprocess_kwargs: dictionary of optional arguments to the preprocessing function. This can be used
            to override the default value of `max_incr_per_pixel` For instance. `{"max_incr_per_pixel": 20}` to clip and
            normalize tensors by 20 at full resolution.
        device (torch.device): Torch device (defaults to cpu).
        height (int): if None the features are not downsampled, however features are downsampled to *height*
            which must be the sensor's height divided by a power of 2.
        width (int): if None the features are not downsampled, however features are downsampled to *width*
            which must be the sensor's width divided by a power of 2.
        transforms (torchvision Transforms): Transformations to be applied to each frame of a sequence.
        base_seed (int): seed to change the random transformation when applicable, if None use time as seed.
        **kwargs: Arbitrary keyword arguments passed to the underlying EventsIterator.

    Attributes:
        mv_it (EventsIterator): object used to read from the file or the camera.
        array_dim (tuple): shape of the tensor (channel, height, width).
        cd_proc (CDProcessor): class computing features from events into a preallocated memory array.
        step (int): counter of iterations.
        event_input_height (int): original height of the sensor in pixels.
        event_input_width (int): original width of the sensor in pixels.
        base_seed (int): seed to change the random transformation when applicable.

    Examples:
        >>> path = "example.raw"
        >>> for tensor in Preprocessor(path, "event_cube", delta_t=10000):
        >>>     # Returns a torch Tensor.
        >>>     print(tensor.shape)

    """

    def __init__(self, path, preprocess_function_name, mode='delta_t', start_ts=0, max_duration=None, delta_t=50000,
                 n_events=10000, num_tbins=1, preprocess_kwargs={}, device=torch.device('cpu'), height=None,
                 width=None, transforms=None, base_seed=0, **kwargs):

        total_n_events = n_events * num_tbins
        total_delta_t = delta_t * num_tbins
        self.mode = mode
        if isinstance(path, type("")):
            self.mv_it = EventsIterator(path, start_ts=start_ts, max_duration=max_duration, mode=mode,
                                        n_events=total_n_events, delta_t=total_delta_t, relative_timestamps=True,
                                        **kwargs)
        else:
            self.mv_it = path

        self.event_input_height, self.event_input_width = self.mv_it.get_size()
        height = height if height is not None else self.event_input_height
        width = width if width is not None else self.event_input_width
        check_input_power_2(self.event_input_height, self.event_input_width, height=height, width=width)

        self.cd_proc = CDProcessor(height, width, num_tbins=num_tbins, preprocessing=preprocess_function_name,
                                   downsampling_factor=int(math.log2(self.event_input_height // height)),
                                   preprocess_kwargs=preprocess_kwargs)

        self.array_dim = self.cd_proc.array.shape
        self.delta_t = delta_t
        self.max_incr_per_pixel = self.cd_proc.preprocess_kwargs.get('max_incr_per_pixel', None)
        self.transforms = transforms
        self.set_base_seed(base_seed)

        self.device = device

    @classmethod
    def from_iterator(cls, iterator, preprocess_function_name, num_tbins=1, preprocess_kwargs={},
                      device=torch.device('cpu'), height=None, width=None, transforms=None):
        mode = iterator.mode
        iterator.relative_timestamps = True
        # Time attributes
        start_ts = iterator.start_ts
        delta_t = iterator.delta_t
        n_events = iterator.n_events
        assert mode != "delta_t" or delta_t % num_tbins == 0
        assert mode != "n_events" or n_events % num_tbins == 0

        return cls(iterator, preprocess_function_name, mode=mode, start_ts=start_ts,
                   delta_t=delta_t // num_tbins, n_events=n_events // num_tbins, num_tbins=num_tbins,
                   preprocess_kwargs=preprocess_kwargs, device=device, height=height, width=width,
                   transforms=transforms)

    def get_time(self):
        return self.mv_it.get_current_time()

    def get_vis_func(self):
        """Returns the visualization function corresponding to the preprocessing being used."""
        return self.cd_proc.preprocess_dict["viz"]

    def show(self, time_bin=0):
        return self.cd_proc.show(self.cd_proc.array[time_bin])

    def __iter__(self):
        self.mv_it_ = iter(self.mv_it)
        self.step = 0
        return self

    def __next__(self):
        self.step += 1
        events = next(self.mv_it_)

        if self.mode == 'n_events' and not len(events):
            raise StopIteration

        # if mode is n_events delta_t=last_time since we use relative_timestamps
        delta_t = self.delta_t * self.array_dim[0] if self.mode == 'delta_t' else events['t'][-1]
        tensor = torch.from_numpy(self.cd_proc(events, delta_t=delta_t).copy())
        tensor = tensor.to(device=self.device)
        tensor = tensor.float()

        if self.transforms is not None:
            tensor = transform_sequence(tensor, 4, self.transforms, base_seed=self.base_seed)

        return tensor


def _read_height_width(h5_dataset, height=None, width=None):
    """reads event_input height and width and resized values from the HDF5 attributes"""

    if "event_input_width" in h5_dataset.attrs:
        event_input_height = h5_dataset.attrs["event_input_height"]
        event_input_width = h5_dataset.attrs["event_input_width"]
    else:
        precomp_downsampling_factor = h5_dataset.attrs.get('downsampling_factor', 0)

        event_input_height = h5_dataset.shape[-2] << precomp_downsampling_factor
        event_input_width = h5_dataset.shape[-1] << precomp_downsampling_factor

    check_input_power_2(event_input_height, event_input_width, height=height, width=width)
    if height is not None:
        assert height <= h5_dataset.shape[-2]
    if width is not None:
        assert width <= h5_dataset.shape[-1]
    return event_input_height, event_input_width


class HDF5Iterator(ProcessorIterator):
    """
    Provides feature tensors (torch.Tensor) at regular intervals from a precomputed HDF5 file.

    Args:
        path (string): Path to the HDF5 file containing precomputed features.
        height (int): if None the features are not downsampled, however features are downsampled to *height*
            which must be the sensor's height divided by a power of 2.
        width (int): if None the features are not downsampled, however features are downsampled to *width*
            which must be the sensor's width divided by a power of 2.
        device (torch.device): Torch device (defaults to cpu).
        start_ts (int): First timestamp to consider in us. (Must be a multiple of the HDF5 file delta_t)
        transforms (torchvision Transforms): Transformations to be applied to each frame of a sequence.
        base_seed (int): seed to change the random transformation when applicable, if None use time as seed.

    Attributes:
        dataset (h5py.Dataset): hDF5 dataset containg the precomputed features.
        array_dim (tuple): shape of the tensor (channel, height, width).
        preprocess_dict (dictionary): dictionary of the parameters used.
        step (int): counter of iterations.
        event_input_height (int): original height of the sensor in pixels.
        event_input_width (int): original width of the sensor in pixels.
        base_seed (int): seed to change the random transformation when applicable.

    Examples:
        >>> path = "example.h5"
        >>> for tensor in HDF5Iterator(path, num_tbins=4):
        >>>     # Returns a torch Tensor.
        >>>     print(tensor.shape)

    """

    def __init__(self, path, num_tbins=1, preprocess_kwargs={}, start_ts=0,
                 device=torch.device('cpu'), height=None, width=None, transforms=None, base_seed=0):

        self.path = path
        self.h5file = h5py.File(path, 'r')
        self.dataset = self.h5file['data']
        self.store_as_uint8 = self.dataset.attrs.get('store_as_uint8', False)

        self.event_input_height, self.event_input_width = _read_height_width(self.dataset, height=height, width=width)

        height = height if height is not None else self.event_input_height
        width = width if width is not None else self.event_input_width

        self._downsample = (height != self.dataset.shape[-2]) or (width != self.dataset.shape[-1])
        self.preprocess = self.dataset.attrs['events_to_tensor'].decode('utf8')
        self.preprocess_dict = get_preprocess_dict(self.preprocess)

        self.array_dim = [num_tbins, self.dataset.shape[1], height, width]
        self.array = np.empty(self.array_dim, dtype=self.dataset.dtype)
        self.mode = self.dataset.attrs.get("mode", "delta_t")
        assert self.mode in ["delta_t", "n_events"], f"unsupported mode {self.mode}. Only delta_t and n_events mode supported"
        self.delta_t = self.dataset.attrs["delta_t"]
        self.n_events = self.dataset.attrs.get("n_events",0)
        self.max_incr_per_pixel = preprocess_kwargs.get('max_incr_per_pixel', None)
        self.transforms = transforms
        self.set_base_seed(base_seed)
        self.start_ts = start_ts
        self.num_tbins = num_tbins
        if self.mode == "n_events":
            self.delta_t = 0
            assert self.n_events > 0
            assert start_ts == 0, "start_ts must be 0 for mode=n_events" # we can not guarantee exact timing as for delta_t case
            self.step = 0
            self.tensor_last_timesteps = self.h5file['last_timestamps']
        else:
            assert self.mode == "delta_t", "only n_events and delta_t modes supported"
            assert self.delta_t > 0 
            if start_ts:
                assert start_ts % self.delta_t == 0, f"start_ts ({start_ts}) must be a multiple of delta_t {self.delta_t}"
            self.step = int(self.start_ts // self.delta_t)
        self.device = device
        

    def get_time(self):
        if self.mode == "n_events":
            return self.tensor_last_timesteps[self.num_tbins * self.step]
        else:
            return self.delta_t * self.num_tbins * self.step

    def checks(self, preprocess_function_name, delta_t, mode="delta_t", n_events = 0):
        """Convenience function to assert precomputed parameters

        Args:
            preprocess_function_name (string): Name of the preprocessing function used to turn events into
                features. Can be any of the functions present in metavision_ml.preprocessing or one registered
                by the user.
            delta_t (int): Duration of used events slice in us.
        """

        assert self.preprocess == preprocess_function_name, f"Incompatible preprocessing features! " \
                                                            f"{self.path} contains {self.preprocess} features, " \
                                                            f"while the current processing function " \
                                                            f"is {preprocess_function_name}."

        assert self.mode == mode, f"{self.path} was computed in {self.mode} mode ! "
        if self.mode == "delta_t":
            assert self.delta_t == delta_t, f"{self.path} contains features computed every {self.delta_t} us ! "
        elif self.mode == "n_events":
            assert self.n_events == n_events, f"{self.path} was computed using {self.n_events} n_events ! "
            
    def get_vis_func(self):
        """Returns the visualization function corresponding to the preprocessing being used."""
        return self.preprocess_dict["viz"]

    def show(self, time_bin=0):
        return self.preprocess_dict["viz"](self.array[time_bin])

    def __iter__(self):
        if self.mode == "delta_t":
            self.step = int(self.start_ts // self.delta_t)
        else:
            self.step = 0
        return self

    def __next__(self):
        if (self.step + 1)* self.array_dim[0] >= len(self.dataset):
            raise StopIteration
        if self._downsample:
            array = self.dataset[self.step * self.array_dim[0]:(self.step + 1) * self.array_dim[0]]
            if self.store_as_uint8:
                array = array.astype(np.float32) / 255
            for tbin in range(self.array_dim[0]):
                for channel in range(self.array_dim[1]):
                    self.array[tbin, channel] = cv2.resize(array[tbin, channel], self.array.shape[-2:][::-1],
                                                           self.array[tbin, channel], interpolation=cv2.INTER_AREA)
        else:
            if self.store_as_uint8:
                array = self.dataset[self.step * self.array_dim[0]:(self.step + 1) * self.array_dim[0]]
                self.array = array.astype(np.float32) / 255
            else:
                self.array = self.dataset[self.step * self.array_dim[0]:(self.step + 1) * self.array_dim[0]]

        tensor = torch.from_numpy(self.array.copy())
        tensor = tensor.to(device=self.device)
        tensor = tensor.float()

        if self.transforms is not None:
            tensor = transform_sequence(tensor, 0, self.transforms, base_seed=self.base_seed)
        self.step += 1
        return tensor

    def __len__(self):
        return len(self.dataset) // self.array_dim[0]
