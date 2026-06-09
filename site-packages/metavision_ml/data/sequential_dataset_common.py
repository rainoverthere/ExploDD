# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Utils for sequential datasets, works for sequential_dataset_map_style and sequential_dataset_iterable_style
"""

import os
import math
import numpy as np
import cv2
import torch


def load_labels_stub(metadata, start_time, duration, tensor):
    """This is a stub implementation of a function to load label data.

    This function doesn't actually load anything and should be passed to the SequentialDataset for
        self-supervised training when no actual labelling is required.

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
    labels = [torch.empty(0) for _ in np.arange(0, duration, metadata.delta_t)]
    frame_is_labeled = np.zeros((len(labels)), dtype=bool)

    return labels, frame_is_labeled


def collate_fn(data_list):
    """
    Builds a batch from the result of the different `__getitem__` calls of the Dataset. This function
    helps define the DataLoader behaviour.

    By doing so it puts the temporal dimensions (each time bin) as the first dimension and
    the batch dimension becomes second.

    Args:
        data_list (tuple list): List where each item is a tuple composed of a tensor, the labels,
            the keep memory mask and the frame_is_labeled mask.

    Returns:
        dictionary: see SequentialDataLoader
    """
    tensors, labels, mask_keep_memory, frame_is_labeled, file_metadata = zip(*data_list)
    tensors = torch.stack(tensors, 1)

    # if some of the file metadata have a valid labelling_delta_t and some don't, this stack operation will fail
    frame_is_labeled = np.stack(frame_is_labeled, 1)
    frame_is_labeled = torch.from_numpy(frame_is_labeled)
    mask_keep_memory = torch.FloatTensor(mask_keep_memory).view(-1, 1, 1, 1)

    t, n = tensors.shape[:2]

    tn_labels = [[labels[i][t] for i in range(n)] for t in range(t)]

    return {"inputs": tensors, "labels": tn_labels, "mask_keep_memory": mask_keep_memory,
            "frame_is_labeled": frame_is_labeled, "video_infos": file_metadata}


def show_dataloader(dataloader, height, width, vis_func, viz_labels=None):
    """
    Visualizes batches of the DataLoader in parallel with open cv.

    This returns a generator that draws the input and also the labels if a "viz_labels" function is provided.

    Args:
        dataloader (DataLoader): iterable of batch of sequential features.
        height (int): height of the feature maps provided by the dataloader.
        width (int): width of the feature maps provided by the dataloader
        viz_func (function): the visualization function corresponding to the preprocessing being used. Takes a tensor
        of shape channels x height x width and turns it into a RGB height width x 3 uint8 image.
        viz_labels (function): Optionally take a visualization function for labels. Its signature is
            - img (np.ndarray) a image of size (height, width, 3) and of dtype np.uint8
            - labels as defined in your load_labels function.
    """
    height_scaled, width_scaled = height, width
    batch_size = dataloader.batch_size
    size_x = int(math.ceil(math.sqrt(batch_size)))
    size_y = int(math.ceil(float(batch_size) / size_x))

    frame = np.zeros((size_y * height_scaled, width_scaled * size_x, 3), dtype=np.uint8)

    for ind, data_dic in enumerate(dataloader):
        batch = data_dic["inputs"]
        mask = data_dic["mask_keep_memory"]
        metadata = data_dic['video_infos']

        for t in range(batch.shape[0]):
            for index, im in enumerate(batch[t]):
                im = im.cpu().numpy()

                y, x = divmod(index, size_x)
                img = vis_func(im)

                # if a function was provided display the labels
                if viz_labels is not None:
                    labels = data_dic["labels"][t][index]
                    img = viz_labels(img, labels)

                if t <= 1 and not mask[index]:
                    # mark the beginning of a sequence with a red square
                    img[:10, :10, 0] = 222
                # put the name of the file
                name = metadata[index][0].path.split('/')[-1]
                cv2.putText(img, name, (int(0.05 * (width_scaled)), int(0.94 * (height_scaled))),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (50, 240, 12))

                frame[y * (height_scaled):(y + 1) * (height_scaled),
                      x * (width_scaled): (x + 1) * (width_scaled)] = img

            yield frame
