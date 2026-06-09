# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Label loading functions for different tasks.
"""

import os
import json
import numpy as np
from scipy import interpolate


def get_label_forward_map_dict(label_map_path):
    """
    Args:
        label_map_path(string): The path of the .json file containing label maps.

    Returns:
        label_dic(dictionary): A dictionary has a forward map from key to label.
    """
    assert os.path.isfile(label_map_path), f"The file {label_map_path} doesn't exist."
    with open(label_map_path, "r") as read_file:
        label_dic = json.load(read_file)
        label_dic = {int(str(key)): str(value) for key, value in label_dic.items()}
    return label_dic


def get_label_backward_map_dict(label_map_path):
    """
    Args:
        label_map_path(string): The path of the .json file containing label maps.

    Returns:
        label_dic(dictionary): A dictionary has a backward map from label to key.
    """
    assert os.path.isfile(label_map_path), f"The file {label_map_path} doesn't exist."
    with open(label_map_path, "r") as read_file:
        label_dic = json.load(read_file)
        label_dic = {str(value): int(str(key)) for key, value in label_dic.items()}
    return label_dic


def interpolate_labels(label_ts, label_ids, label_delta_t, frame_delta_t, frame_timestamps=None):
    """
    interpolate discrete labels label_ids from timestamps label_ts to obtain higher frequency labels
    the interpolation is done at regular timestamps so that the new frequency is higher than the fequency of the 
    precomputed event frames
    Args:
        label_ts(array): The path from which the event frame is loaded. 
                                    It will be used to infer the path of the npz file which contains labels
        label_ids(array): from where the event frames are sampled
        label_delta_t(int): delta_t of labels
        frame_delta_t(int): the duration of each event frame (if constant, otherwise you should use frame_timestamps)
        frame_timestamps(array): contains timestamps of event frames we want label for
                                if None it uses frame_delta_t  

    Return: 
        label_ts(array): times at which labels have been interpolated
        label_ids(array): interpolated labels
        label_delta_t(int): maximum delta_t between interpolated labels times
    """

    if frame_timestamps is None:
        min_frame_delta_t = frame_delta_t
    else:
        min_frame_delta_t = min(frame_timestamps[1:] - frame_timestamps[:-1])
    if min_frame_delta_t < label_delta_t:
        f = interpolate.interp1d(label_ts, label_ids, kind='nearest', fill_value=0, assume_sorted=True)
        num_steps = int((label_ts[-1] - label_ts[0]) // (min_frame_delta_t)) + 2
        label_ts = np.linspace(label_ts[0], label_ts[-1], num_steps).astype(int)
        label_ids = f(label_ts).astype(int)
        label_delta_t = max(label_ts[1:] - label_ts[:-1])
        assert label_delta_t <= min_frame_delta_t, "interpolated timestamps delta_t should be smaller than min_frame_delta_t, something is wrong! "
    return label_ts, label_ids, label_delta_t


def load_classes(frame_file_path, start_frame, num_ev_reps, frame_delta_t, label_map_path=None, frame_timestamps=None,
                 label_delta_t=None, allow_labels_interpolation=False):
    """
    Args:
        frame_file_path(string): The path from which the event frame is loaded. 
                                    It will be used to infer the path of the npz file which contains labels
        start_frame(int): from where the event frames are sampled
        num_ev_reps(int): how many event frames are sampled
        frame_delta_t(int): the duration of each event frame
        label_map_path(string): path to the .json file which contains the label map
        frame_timestamps (array): for datasets computed with fixed n_events, contains timestamps for each frame:
                                [start_ts, last_time_frame_0, last_time_frame_1, ...]
                                NB frame_delta_t must be 0 in this case
        label_delta_t(int): delta_t of labels
        allow_labels_interpolation(bool): if true, and if label_delta_t > frames_delta_t, interpolate labels to avoid having many frames with no labels
                                            NB set to true only if you know that there is continuity in time of the labels 
    Return: 
        all_labels(list), a list containing num_ev_reps labels
    """
    if frame_timestamps is not None:
        assert frame_delta_t == 0, "both frame_timestamps and frame_delta_t provided. This is ambiguous!"
    # Try to get the label_map
    assert label_map_path is not None, "The label_map should be provided!!!"
    assert label_map_path.endswith(".json"), f"{label_map_path}) doesn't refer to a .json file!!!"
    # Get the keys of the label "background" and "ignore"
    backward_label_dict = get_label_backward_map_dict(label_map_path)
    assert "background" in backward_label_dict.keys(), "The labels should include 'background'!!!"
    assert "ignore" in backward_label_dict.keys(), "The labels should include 'ignore'!!!"
    background_key = backward_label_dict["background"]
    ignore_key = backward_label_dict["ignore"]

    # Try to get the coresponding label file
    label_file_path = '_labels.npz'.join(frame_file_path.rsplit(".h5", 1))
    assert os.path.isfile(
        label_file_path), f"label file {label_file_path} not found, for frame file {frame_file_path}"
    label_file = np.load(label_file_path)

    label_ts = label_file['ts']
    label_ids = label_file['labels']

    # if label frequency is too large compared to delta_t frames, and we assume continuity in the labels, we can interpolate the classes
    if allow_labels_interpolation:
        label_ts, label_ids, label_delta_t = interpolate_labels(
            label_ts, label_ids, label_delta_t, frame_delta_t, frame_timestamps)

    all_labels = []
    for i in range(num_ev_reps):
        if frame_timestamps is not None:
            frame_begin_time = frame_timestamps[i]
            frame_end_time = frame_timestamps[i + 1]
        else:
            frame_begin_time = (start_frame + i) * frame_delta_t
            frame_end_time = frame_begin_time + frame_delta_t
        if frame_end_time-frame_begin_time < label_delta_t:
            print(
                f"Warning: label_delta_t {label_delta_t} bigger than frame slice duration {frame_end_time-frame_begin_time } ")
            print("Consider using option --allow_labels_interpolation")
            label = [ignore_key]
        labels = label_ids[(label_ts >= frame_begin_time) &
                           (label_ts < frame_end_time)]
        if len(labels) == 0:
            # This indicates the frames are not labeled,
            # so the label is set to the key of "background"
            label = background_key
        else:
            # If we have a unique label in the period of frame_delta_t, we use it
            unique_labels = np.unique(labels)
            if len(unique_labels) == 1:
                label = unique_labels[0]
            else:
                # If we don't have a unique label in the period of frame_delta_t,
                # we mark it also as "ignore"
                label = ignore_key
        all_labels.append(label)
    return all_labels
