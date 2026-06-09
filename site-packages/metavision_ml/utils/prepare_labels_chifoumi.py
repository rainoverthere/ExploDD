# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Extract the labels in _bbox.npy files, map them according to the new convention
and store them in the _labels.npz files
NB: This file should be ONLY used for converting original Chifoumi dataset!!!
"""

import os
import glob
import argparse
import numpy as np


def remap_labels_chifoumi(labels):
    """
        Args: 
        labels(np.array, unit8), the orginial labels in the bbox file

        Returns: 
        labels(np.array, unit8), the new labels corresponding to the label_map_dictionary_fnn
    """
    # The order is important, you don't want the 0s finally be mapped to 3s
    # The ignore label is kept as 255
    labels[labels == 2] = 3
    labels[labels == 1] = 2
    labels[labels == 0] = 1
    return labels


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Root path of the dataset.')

    return parser


def prepare_labels(params):
    # root path
    data_dir = params.dataset_path
    for folder_name in ["train", "val", "test"]:
        bbox_files = glob.glob(os.path.join(data_dir, folder_name, '*.npy'))
        assert len(bbox_files) > 0, f"There are no .npy files in the directory {os.path.join(data_dir, folder_name)}"
        for bbox_file in bbox_files:
            box_events = np.load(bbox_file)
            if "t" in box_events.dtype.names:
                timestamps = box_events['t']
            elif "ts" in box_events.dtype.names:
                timestamps = box_events['ts']
            labels = box_events['class_id']
            labels = remap_labels_chifoumi(labels)
            assert len(timestamps) == len(labels), "For each timestamp, there should be a corresponding label!"
            assert bbox_file.endswith("_bbox.npy")
            output_file = bbox_file.replace("_bbox.npy", "_labels.npz")
            np.savez(output_file, ts=timestamps, labels=labels)


if __name__ == '__main__':
    params_parser = args_parser()
    params, _ = params_parser.parse_known_args()
    prepare_labels(params)
