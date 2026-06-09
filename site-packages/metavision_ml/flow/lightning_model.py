# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This file defines the training loop and validation stages for self supervised flow regression with a neural network.
"""
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from skvideo.io import FFmpegWriter

import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from metavision_core_ml.core.temporal_modules import batch_to_time, time_to_batch

from .viz import draw_flow_on_grid
from .flow_network import FlowNetwork
from .losses import FlowLoss
from .losses import charbonnier_loss

pl.seed_everything(123)
LOSS_WEIGHTS = {"data": 1, "smoothness": 0.4, "smoothness2": 4 * 0.15, "l1": 2 * 0.15, 'time_consistency': 1,
                "bw_deblur": 1}


class FlowModel(pl.LightningModule):
    """
    Pytorch lightning module to learn self supervised flow.

    Args:
        delta_t (int): Timeslice delta_t in us.
        preprocess (string): Name of the preprocessing function used to turn events into
            features. Can be any of the functions present in metavision_ml.preprocessing or one registered
            by the user.
        learning_rate (float): factor by which weights update are multiplied during training. A large learning rate
            means the network is updated faster but the convergence might be harder.
        array_dim (int list): Dimensions of feature tensors:
            (num_tbins, channels, sensor_height * 2^-k, sensor_width * 2^-k).
        loss_weights (dict): dictionary, whose keys are names of flow losses and the values are float
            weight factors.
        feature_extractor_name (str): Name of the feature extractor architecture.
        network_kwargs (dict): kwargs of parameters for the feature extractor.
    """

    def __init__(self, delta_t=50000, preprocess="event_cube", learning_rate=1e-5, array_dim=(1, 6, 240, 320),
                 loss_weights=None, feature_extractor_name="eminet", network_kwargs=None,
                 draw_dense=False):
        super(FlowModel, self).__init__()
        self.save_hyperparameters()
        # set network_kwargs
        if network_kwargs is None:
            self.hparams.network_kwargs = {}
            self.set_defaults()
        else:
            # make a copy of the mutable dict
            self.hparams.network_kwargs = dict(network_kwargs)

        # loss_weights
        if self.hparams.loss_weights is None:
            self.hparams.loss_weights = {key: self.hparams.get(key, LOSS_WEIGHTS[key]) for key in LOSS_WEIGHTS.keys()}

        if self.hparams.preprocess not in ('event_cube', "multi_channel_timesurface"):
            self.hparams.loss_weights['bw_deblur'] = 0
            print("This loss doesn't support backward deblurring, setting its loss to 0.")

        self.hparams.num_tbins, self.hparams.cin, self.hparams.height, self.hparams.width = self.hparams.array_dim
        self.network = FlowNetwork(self.hparams.array_dim, self.hparams.loss_weights,
                                   feature_extractor_name=self.hparams.feature_extractor_name,
                                   **self.hparams.network_kwargs)
        self.criterion = FlowLoss(self.hparams.loss_weights, self.network.get_warping_head())

    def set_defaults(self):
        """Sets default for backward compatibility purposes."""
        self.hparams.network_kwargs.setdefault('rnn_cell', 'lstm')
        self.hparams.network_kwargs.setdefault('base', 32)
        self.hparams.network_kwargs.setdefault('separable', True)

    def forward(self, x):
        return self.network.forward(x)

    def compute_loss(self, inputs):
        """
        Computes loss for a given input.

        Args:
            inputs (Tensor): batch of input features. Must be of the shape (T, N, C, H,W).

        Returns:
            a dictionary, where keys are losses names and the Values are Torch float Tensors.
        """
        flows = self.network(inputs)
        flows = [time_to_batch(flow)[0] for flow in flows]
        interpolated_inputs = self.network.interpolate_all_inputs_using_pyramid(inputs)
        loss_dict = self.criterion(flows, interpolated_inputs, inputs.shape[1])
        return loss_dict

    def loss_step(self, batch, batch_idx, mode="train"):
        """Performs a supervised loss if labels are available, otherwise applies a self supervised criterion."""
        self.network.reset(batch["mask_keep_memory"])

        if batch['frame_is_labeled'].any():

            flows = self.network(batch["inputs"])[-1]
            flows_gt = batch['labels']

            metrics = defaultdict(list)

            # we evaluate by recording
            for b, _ in enumerate(batch['video_infos']):

                labeled_mask = batch['frame_is_labeled'][:, b]
                if not labeled_mask.sum():
                    continue
                flow = flows[:, b][labeled_mask]
                gt = torch.stack([f[b] for f, l in zip(flows_gt, labeled_mask) if l]).to(batch['inputs'].device)
                gt = F.interpolate(gt, flow.shape[2:])
                loss = charbonnier_loss(flow - gt)
                self.log(mode + '_charb_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                return loss
        else:
            loss_dict = self.compute_loss(batch["inputs"])
            loss = sum([value for key, value in loss_dict.items()])

            self.log(mode + '_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            for loss_name, loss_value in loss_dict.items():
                self.log(f'{mode}_loss_{loss_name}', loss_value, on_step=True, on_epoch=False, prog_bar=True)

            return loss

    def training_step(self, batch, batch_idx):
        return self.loss_step(batch, batch_idx, "train")

    def test_step(self, batch, batch_nb):
        return self._inference_step(batch, batch_nb)

    def validation_step(self, batch, batch_nb):
        return self.loss_step(batch, batch_nb, "val")

    def test_epoch_end(self, validation_step_outputs):
        return self._inference_epoch_end(validation_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.mean(torch.tensor([elt for elt in validation_step_outputs]))
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999))

    @torch.no_grad()
    def demo_video(self, test_data, log_dir=".", epoch=0, num_batches=100, write_video=True, show_video=False,
                   mask_by_input=True):
        """
        This runs our detector on several videos of the testing dataset.

        Args:
            test_data(object): dataloader for demo
                the validation set corresponding to the dataset_path.
            log_dir (str): directory where to create the video folder containing the result video file.
            epoch (int): index of the epoch. Used to name the video.
            num_batches (int): Number of batches used to create the video.
            write_video (boolean): whether to save a video file in the log_dir/videos/video#{epoch}.mp4
            show_video (boolean): whether to display the result in an opencv window.
            mask_by_input (boolean): if True only display flow arrows on pixel with non null input.
        """
        test_data.to(self.device)

        test_data.dataset.reschedule(max_consecutive_batch=num_batches // 10)
        batch_size = test_data.dataset.batch_size

        nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
        ncols = int(np.ceil(batch_size / nrows))

        grid = np.zeros((self.hparams.num_tbins, nrows * self.hparams.height, ncols * self.hparams.width, 3),
                        dtype=np.uint8)

        if write_video:
            video_dir = os.path.join(log_dir, "videos")
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            video_name = os.path.join(video_dir, f"video#{epoch:d}.mp4")
            print('video name: ', video_name)

            video_writer = FFmpegWriter(video_name, outputdict={'-r': str(int(1e6 / self.hparams.delta_t)),
                                                                '-crf': '20', '-preset': 'veryslow'})
        if show_video:
            window_name = "test epoch {:d}".format(epoch)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        self.network.eval()

        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(test_data), total=num_batches):
                if batch_idx == num_batches:
                    break

                self.network.reset(data['mask_keep_memory'])

                flows = self.network(data['inputs'])
                if flows[-1].shape == 4:
                    flows = [batch_to_time(flow, batch_size) for flow in flows]

                images = data['inputs'].cpu().clone().numpy()

                draw_flow_on_grid(images, flows, grid, test_data.get_vis_func(), scale=-1,
                                  mask_by_input=mask_by_input, draw_dense=self.hparams.draw_dense)

                if write_video:
                    for frame in grid:
                        video_writer.writeFrame(frame)
                if show_video:
                    for frame in grid:
                        cv2.imshow(window_name, frame[..., ::-1])
                    key = cv2.waitKey(1)
                    if key == 27:
                        cv2.destroyWindow(window_name)
                        show_video = False
                        if not write_video:
                            return

        # clean up
        if write_video:
            video_writer.close()
        if show_video:
            cv2.destroyWindow(window_name)

    def _inference_step(self, batch, batch_nb):
        self.network.reset(batch["mask_keep_memory"])

        flows = self.network(batch["inputs"])[-1]
        flows_gt = batch['labels']

        metrics = defaultdict(list)

        # we evaluate by recording
        for b, video_infos in enumerate(batch['video_infos']):

            labeled_mask = batch['frame_is_labeled'][:, b]
            flow = flows[:, b][labeled_mask]
            if not labeled_mask.sum():
                continue
            gt = torch.stack([f[b] for f, l in zip(flows_gt, labeled_mask) if l]).to(batch['inputs'].device)
            gt = F.interpolate(gt, flow.shape[2:])
            end_point_error = np.linalg.norm(flow.cpu() - gt.cpu(), axis=1)
            mean = np.mean(end_point_error, axis=(1, 2))
            median = np.median(end_point_error, axis=(1, 2))
            # an inlier is at 5 % of a full size (which is 2 in relative flow values)
            inlier = np.mean((end_point_error < 2 * 0.05).astype('float'), axis=(1, 2))
            metrics[video_infos[0].path].append((mean, median, inlier))
        return metrics

    def _inference_epoch_end(self, validation_step_outputs):
        total_dict = defaultdict(list)
        for pred in validation_step_outputs:
            for key in pred:
                total_dict[key].append(pred[key])
        if total_dict.values():
            metrics = np.concatenate(
                [np.concatenate(v, axis=2)[0] for v in total_dict.values()], axis=1).mean(axis=1)
            self.log("val/mean_relative", metrics[0], on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/median_relative", metrics[1], on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/global_inlier_average", metrics[2], on_step=False, on_epoch=True, prog_bar=False)
            print(metrics)
        else:
            print("No validation occurred !")
