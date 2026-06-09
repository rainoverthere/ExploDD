# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Pytorch Lightning Module for training a fnn classifier
"""
import argparse
import os
import cv2
import numpy as np
from itertools import islice
from skvideo.io import FFmpegWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, MetricCollection
import pytorch_lightning as pl
from metavision_ml.preprocessing import get_preprocess_dict
from metavision_ml.classification import models, utils_metrics


class FNNClassificationModel(pl.LightningModule):
    """ Pytorch Lightning model for neural network to predict class of scene.

        Args:
            hparams (argparse.Namespace): argparse from train.py application
    """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)

        # params regarding to dimensions
        self.batch_size = hparams.batch_size
        self.num_ev_reps = hparams.num_ev_reps
        self.channels = hparams.preprocess_channels
        self.height = hparams.height
        self.width = hparams.width

        # preprocess name and params
        self.preprocess = hparams.preprocess
        self.preprocess_dict = get_preprocess_dict(self.preprocess)

        # label maps
        self.forward_label_dict = hparams.forward_label_dict
        self.backward_label_dict = hparams.backward_label_dict

        self.ignore_key = self.backward_label_dict["ignore"]
        self.label_map = list(self.backward_label_dict.keys())
        self.label_map.remove('ignore')
        self.num_classes = len(self.label_map)

        # define networks
        model_fn = getattr(models, hparams.models)
        in_channels = self.num_ev_reps * self.channels
        if model_fn == models.SqueezenetClassifier:
            self.net = model_fn(in_channels, num_classes=self.num_classes)
        elif model_fn == models.Mobilenetv2Classifier:
            feature_channels_out = hparams.feature_channels_out
            self.net = model_fn(in_channels, width_mul=hparams.width_mul if hasattr(hparams, 'width_mul') else 1,
                                cout=feature_channels_out, num_classes=self.num_classes)
        else:
            raise NotImplementedError("The model architecture is not implemented!")

        # define metrics for training, validation and test
        metrics = MetricCollection([
            Accuracy(),
            Precision(num_classes=self.num_classes, average="none"),
            Recall(num_classes=self.num_classes, average="none")
        ])
        self.train_metric = metrics.clone(prefix="train_")
        self.valid_metric = metrics.clone(prefix="val_")
        self.test_metric = metrics.clone(prefix="test_")

    def load_pretrained(self, checkpoint_path):
        """ Loads a pretrained detector (of this class)
            and transfer the weights to this module for fine-tuning.

            Args:
                checkpoint_path (str): path to checkpoint of pretrained detector.
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
        module_src = FNNClassificationModel(hparams)
        module_src.load_state_dict(checkpoint['state_dict'])

        # Copy Buffers
        dict_dst = dict(self.net.named_buffers())
        for name, buf in module_src.net.named_buffers():
            if name in dict_dst:
                if dict_dst[name].data.shape == buf.data.shape:
                    dict_dst[name].data.copy_(buf.data)

        # Copy Parameters
        dict_dst = dict(self.named_parameters())
        for name, param in module_src.named_parameters():
            if name in dict_dst:
                if dict_dst[name].data.shape == param.data.shape:
                    dict_dst[name].data.copy_(param.data)

    def forward(self, input):
        """
            Input in the dimension of (batch_size, channels, height, width)
            Output in the dimension of (batch_size, num_classes)
        """
        output = self.net(input)
        return output

    def preprocess_inputs(self, batch):
        """
            Input: 
                event_frames, labels = batch
                event_frames in the shape of (batch_size, num_ev_reps, channels, height, width)
                labels in the shape of (batch_size, num_ev_reps)
            Output:
                we use the last label to represent the label in the group of num_ev_reps event frames
                event_frames in the shape of (batch_size, num_ev_reps * channels, height, width)
                labels in the shape of (batch_size)
        """
        event_frames, labels = batch
        event_frames = event_frames.view(-1, self.num_ev_reps * self.channels, self.height, self.width)
        labels = labels[:, -1].long()

        return event_frames, labels

    def training_step(self, batch, batch_nb):
        # x shape: [B, N*C, H, W], y shape: [B]
        x, y = self.preprocess_inputs(batch)
        # y_hat shape: [B, K]
        y_hat = self.forward(x)
        weight_class = None
        if self.hparams['weights']:
            weight_class = torch.tensor(self.hparams['weights']).to(y_hat.device)
        loss = F.cross_entropy(y_hat, y, weight=weight_class, ignore_index=self.ignore_key, reduction='mean')

        if torch.isnan(loss):
            return None
        else:
            # yprob shape: [B, K]
            yprob = nn.functional.softmax(y_hat, dim=-1)
            assert yprob.shape[-1] == self.num_classes

            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
            train_dict = {"loss": loss, "preds": yprob, "labels": y}

            return train_dict

    def validation_step(self, batch, batch_idx):
        return self._inference_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._inference_step(batch, batch_idx, mode="test")

    def _inference_step(self, batch, batch_idx, mode="val"):
        # x shape: [B, N*C, H, W], y shape: [B]
        x, y = self.preprocess_inputs(batch)
        # y_hat shape: [B, K]
        y_hat = self.forward(x)

        weight_class = None
        if self.hparams['weights']:
            weight_class = torch.tensor(self.hparams['weights']).to(y_hat.device)
        loss = F.cross_entropy(y_hat, y, weight=weight_class, ignore_index=self.ignore_key)

        if torch.isnan(loss):
            return None
        else:
            # yprob shape: [B, K]
            yprob = nn.functional.softmax(y_hat, dim=-1)
            assert yprob.shape[-1] == self.num_classes

            self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True)
            inf_dict = {"preds": yprob, "labels": y}

            return inf_dict

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'test')

    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'train')

    def _epoch_end(self, outputs, mode='val'):
        """
        Runs Metrics

        Args:
            outputs: accumulated outputs
            mode: 'train', 'val' or 'test'
        """
        preds_all, labels_all = [], []
        for dic in outputs:
            preds = dic["preds"]
            labels = dic["labels"]
            # Exclude the cases of ignore
            mask = dic["labels"] != self.ignore_key
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            preds_all.append(valid_preds)
            labels_all.append(valid_labels)

        preds_all, labels_all = torch.cat(preds_all, dim=0), torch.cat(labels_all, dim=0)

        if mode == "train":
            res = self.train_metric(preds_all, labels_all)
        elif mode == "val":
            res = self.valid_metric(preds_all, labels_all)
        elif mode == "test":
            res = self.test_metric(preds_all, labels_all)

            if self.hparams.show_plots:
                for c in list(self.hparams.show_plots):
                    assert c in utils_metrics.PLOTS_DICT, "{} is not in {}".format(c, utils_metrics.PLOTS_DICT)
                    utils_metrics.PLOTS_DICT[c](preds_all.cpu(), labels_all.cpu(), self.label_map)

        else:
            raise TypeError("The mode is unknown!")

        # pack the metrics by class category
        res = utils_metrics.unpack_metrics_dict(res, self.label_map)
        self.log_dict(res, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        if self.hparams.lr_scheduler_step_gamma is None:
            print("No Learning Rate Scheduler")
            return optimizer
        print("Using Learning Rate Scheduler: ", self.hparams.lr_scheduler_step_gamma)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.lr_scheduler_step_gamma)
        return [optimizer], [scheduler]

    def demo_video(self, test_data, epoch=0, num_batches=100, show_video=False, show_pred=True, fps=30):
        """
        This runs our classifier on several videos of the test dataset

        Args:
            test_data (object): Dataloader
            epoch (int, optional): Index of the epoch. Used to name the video
            num_batches (int, optional): Number of batches used to create the video
            show_video (boolean, optional): Whether to display the demo
            show_pred (boolean, optional): Whether to show the prediction results as well.
                                            Set it to "False" to only inspect the input data
            fps (int, optional): Video output frame rate
        """

        nrows = 2 ** ((self.batch_size.bit_length() - 1) // 2)
        ncols = int(np.ceil(self.batch_size / nrows))
        grid = np.zeros((nrows * self.height, ncols * self.width, 3), dtype=np.uint8)

        if not show_pred:
            video_name = os.path.join(self.hparams.root_dir, 'check_input', f'video#{num_batches:d}batches.mp4')
        else:
            video_name = os.path.join(self.hparams.root_dir, 'videos', f'video#{epoch:d}.mp4')

        dir = os.path.dirname(video_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)

        video_writer = FFmpegWriter(video_name, inputdict={'-r': str(fps)}, outputdict={'-crf': '20', '-r': str(fps),
                                                                                        '-preset': 'veryslow'})

        self.net.eval()

        if show_video:
            window_name = "test epoch {:d}".format(epoch)
            cv2.namedWindow(window_name)

        for batch in islice(test_data, num_batches):
            # ev_frames shape: [B, N, C, H, W], labels shape: [B, N]
            ev_frames, labels = batch
            # input_x shape: [B, N*C, H, W]
            input_x, _ = self.preprocess_inputs(batch)
            images = ev_frames.cpu().clone().data.numpy()

            if show_pred:
                with torch.no_grad():
                    # input_x shape: [B, K]
                    predictions = self.net.get_probas(input_x.to(self.device))

            for batch_idx in range(len(images)):

                # Only show the last image
                frame = self.preprocess_dict['viz'](images[batch_idx, -1, ...])
                target_id = int(labels[batch_idx, -1].numpy())
                target_class = self.forward_label_dict[target_id]

                # draw class + score
                color = (0, 255, 0)
                if show_pred:
                    preds = predictions[batch_idx].cpu().numpy()
                    pred_class_id = np.argmax(preds)
                    pred_score = preds[pred_class_id]
                    pred_class = self.forward_label_dict[pred_class_id]
                    text = "wrong" if target_id != pred_class_id else "correct"
                    text = "correct" if target_id == self.ignore_key else text
                    cv2.putText(frame, text, (10, frame.shape[0] - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color
                                if text == "correct" else (255, 0, 0))
                    text = "pred: " + pred_class
                    cv2.putText(frame, text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                    text = "score: " + str(pred_score)
                    cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                text = "gt: " + target_class
                cv2.putText(frame, text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                y = batch_idx // ncols
                x = batch_idx % ncols
                y1, y2 = y * self.height, (y + 1) * self.height
                x1, x2 = x * self.width,  (x + 1) * self.width
                grid[y1:y2, x1:x2] = frame

            if show_video:
                gridup = cv2.pyrUp(grid)
                cv2.imshow(window_name, gridup[..., ::-1])
                cv2.waitKey(50)

            video_writer.writeFrame(grid)

        video_writer.close()
        if show_video:
            cv2.destroyAllWindows()
