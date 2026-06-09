# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Toy Problem Dataset that serves as an example of our streamer dataloader.

This displays moving digits from MNIST database.
The digit varies in size and position.
To use this class, download MNIST.ZIP at
https://kdrive.infomaniak.com/app/share/975517/3c529307-3cec-4fc6-bbb3-87a95c6ef6cf

The dataset both generates chained video clips and provides bounding box with correct class id.

The dataset procedurally generates the video clips, so it is an "Iterable" kind of dataset
"""
from __future__ import absolute_import
import os
import cv2
import torch
import numpy as np
import uuid
from urllib.request import urlretrieve
from zipfile import ZipFile

import metavision_ml.data.moving_box as toy
from metavision_ml.data.scheduler import FileMetadata
from metavision_core_ml.data.stream_dataloader import *

from torchvision import datasets, transforms


DATA_CACHING_PATH = "."


class MovingMnist(toy.Animation):
    """Moving Mnist Animation

    Args:
        idx: unique id
        tbins: number of steps delivered at once
        height: frame height (must be at least 64 pix)
        width: frame width (must be at least 64 pix)
        max_stop: random pause in animation
        max_objects: maximum number of objects per animation
        train: use training/ validation part of MNIST
        max_frames_per_video: maximum frames per video before reset
        drop_labels_p: probability to drop the annotation of certain frames (in which case it is marked in the mask)
        data_caching_path: where to store the MNIST dataset
    """

    def __init__(
            self, idx, tbins, height, width, train, max_frames_per_video, channels=3, max_stop=15, max_objects=2,
            drop_labels_p=0, data_caching_path=DATA_CACHING_PATH):
        assert width >= 64, "width must be at least 60 pix  (current value: {})".format(width)
        assert height >= 64, "height must be at least 60 pix (current value: {})".format(height)
        if not os.path.exists(os.path.join(data_caching_path, "MNIST")):
            os.makedirs(data_caching_path, exist_ok=True)
            get_MNIST(folder=data_caching_path)
        self.dataset_ = datasets.MNIST(data_caching_path, train=train, download=False,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))
        self.label_offset = 1
        self.channels = channels
        np.random.seed(idx)
        self.steps = 0
        self.tbins = tbins
        self.drop_labels_p = drop_labels_p
        self.max_frames_per_video = max_frames_per_video
        max_classes = 10
        # We use a random name here for the FileMetadata
        # As the validation packs boxes by recording name.
        # It is important to associate one MovingMnist to one Metadata for the validation
        # Here there is not really a recording name (since this is procedurally generated)
        # So instead we generate a random one.
        random_name = str(uuid.uuid4())
        self.video_info = FileMetadata(random_name, 50000 * max_frames_per_video, 50000, tbins)
        super(MovingMnist, self).__init__(height, width, channels, max_stop, max_classes, max_objects)

    def reset(self):
        super(MovingMnist, self).reset()
        self.steps = 0
        for i in range(len(self.objects)):
            idx = np.random.randint(0, len(self.dataset_))
            x, y = self.dataset_[idx]
            self.objects[i].class_id = y
            self.objects[i].idx = idx
            img = x.numpy()[0]
            img = (img - img.min()) / (img.max() - img.min())
            abs_img = np.abs(img)
            y, x = np.where(abs_img > 0.45)
            x1, x2 = np.min(x), np.max(x)
            y1, y2 = np.min(y), np.max(y)
            self.objects[i].img = np.repeat(img[y1:y2, x1:x2][..., None], self.channels, 2)

    def step(self):
        self.img[...] = 0
        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, digit in enumerate(self.objects):
            x1, y1, x2, y2 = next(digit)
            boxes[i] = np.array([x1, y1, x2, y2, digit.class_id + self.label_offset])
            thumbnail = cv2.resize(digit.img, (x2 - x1, y2 - y1), cv2.INTER_LINEAR)
            self.img[y1:y2, x1:x2] = np.maximum(self.img[y1:y2, x1:x2], thumbnail)
        output = self.img
        self.steps += 1
        return (output, boxes)

    def __iter__(self):
        for r in range(self.max_frames_per_video // self.tbins):
            reset = self.steps > 0
            imgs, targets, frame_is_labeled = [], [], []
            for t in range(self.tbins):
                img, target = self.step()
                d = np.random.uniform(0, 1) > self.drop_labels_p
                frame_is_labeled.append(d)
                imgs.append(img[None].copy())
                targets.append(target)

            imgs = np.concatenate(imgs, axis=0)

            video_info = (self.video_info, self.steps * self.video_info.delta_t, self.tbins * self.video_info.delta_t)
            yield torch.from_numpy(imgs), targets, reset, video_info, frame_is_labeled


def collate_fn(data_list):
    """
    this collates batch parts to a single dictionary

    Args:
        data_list: batch parts
    """
    batch, boxes, resets, video_infos, frame_is_labeled = zip(*data_list)
    batch = torch.cat([item[:, None] for item in batch], dim=1)
    batch = batch.permute(0, 1, 4, 2, 3).contiguous()
    t, n = batch.shape[:2]
    boxes = [[boxes[i][t] for i in range(n)] for t in range(t)]
    resets = torch.FloatTensor(resets)[:, None, None, None]

    frame_is_labeled = [[frame_is_labeled[i][t] for i in range(n)] for t in range(t)]
    frame_is_labeled = torch.FloatTensor(frame_is_labeled)
    return {'inputs': batch, 'labels': boxes, "frame_is_labeled": frame_is_labeled,
            'mask_keep_memory': resets, "video_infos": video_infos}


def get_MNIST(folder="."):
    sample_path = os.path.join(folder, "MNIST.zip")
    if not os.path.exists(sample_path):
        print("downloading dataset: {}".format(sample_path))
        # we are not using an official share URL from KDrive. We found it by analyzing the http requests.
        # we hope it will continue to work in the future
        urlretrieve("https://kdrive.infomaniak.com/2/app/975517/share/3c529307-3cec-4fc6-bbb3-87a95c6ef6cf/files/12275/download", filename=sample_path)
    with ZipFile(sample_path, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall()


class MovingMNISTDataset(StreamDataLoader):
    """Creates the dataloader for moving mnist

    Args:
        tbins: number of steps per batch
        num_workers: number of parallel workers
        batch_size: number of animations
        height: animation height
        width: animation width
        max_frames_per_video: maximum frames per animation (must be greater than tbins)
        max_frames_per_epoch: maximum frames per epoch
        train: use training part of MNIST dataset.
        dataset_dir: directory where MNIST dataset is stored
    """

    def __init__(self, tbins, num_workers, batch_size, height, width,
                 max_frames_per_video, max_frames_per_epoch, train, dataset_dir=DATA_CACHING_PATH):

        assert max_frames_per_video >= tbins
        height, width, cin = height, width, 3
        n = max_frames_per_epoch // max_frames_per_video
        stream_list = list(range(n))

        def iterator_fun(idx): return MovingMnist(idx, tbins, height, width, train, max_frames_per_video,
                                                  data_caching_path=dataset_dir)

        dataset = StreamDataset(stream_list, iterator_fun, batch_size, "data", None)
        super().__init__(dataset, num_workers, collate_fn)
        self.vis_func = lambda img: (np.moveaxis(img, 0, 2).copy() * 255).astype(np.int32)

    def get_vis_func(self):
        return self.vis_func
