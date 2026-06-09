# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Pytorch Lightning Module for training a classifier
"""
import argparse
import os
import cv2
import numpy as np
from skvideo.io import FFmpegWriter

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import islice
from collections import defaultdict
from metavision_ml.classification import models
from torchmetrics import Accuracy, Precision, Recall, MetricCollection
import torch.nn as nn
from torchmetrics.functional import accuracy
from metavision_ml.classification import utils_metrics


class ClassificationModel(pl.LightningModule):
    """ Pytorch Lightning model for neural network to predict class of scene.

        Args:
            hparams (argparse.Namespace): argparse from train.py application
    """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.automatic_optimization = False
        self.save_hyperparameters(hparams)
        in_channels = hparams.preprocess_channels
        feature_base = hparams.feature_base
        feature_channels_out = hparams.feature_channels_out

        model_fn = getattr(models, hparams.models)

        self.num_classes = len(hparams.classes) + 1
        if model_fn == models.ConvRNNClassifier:
            self.net = model_fn(in_channels, base=feature_base, cout=feature_channels_out,
                                num_classes=self.num_classes)
        else:
            raise NotImplementedError("The model architecture is not implemented!")

        self.label_map = ['background'] + hparams.classes
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

            In addition, it may remap the old classification weights if
            some overlap exists between old and new list of classes.

            Args:
                checkpoint_path (str): path to checkpoint of pretrained detector.
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
        module_src = ClassificationModel(hparams)
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

    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        for name, module in self._modules.items():
            if hasattr(module, "reset") and module.__class__ != MetricCollection:
                module.reset(mask)

    def forward(self, batch):
        self.reset(batch["mask_keep_memory"])
        xs = self.net(batch['inputs'])
        return xs

    def _select_valid_frames(self, xs, targets, frame_is_labeled):
        tbins, batch_size, num_classes = xs.shape
        targets = torch.tensor(targets).to(xs.device).view(tbins, batch_size)
        frame_is_labeled = torch.squeeze(frame_is_labeled)
        filter_cls = targets == targets  # exclude nan labels introduced from batching
        filter_cls *= frame_is_labeled.to(xs.device)
        return xs, targets, filter_cls

    def training_step(self, batch, batch_nb):

        y_hat = self.forward(batch)
        if not batch['frame_is_labeled'].sum().item():
            return None
        y_hat, y, filter_cls = self._select_valid_frames(y_hat, batch['labels'], batch['frame_is_labeled'])
        weight_class = None
        if self.hparams['weights']:
            weight_class = torch.tensor(self.hparams['weights']).to(y_hat.device)
        loss = F.cross_entropy(
            y_hat[filter_cls].view(-1, self.num_classes),
            y[filter_cls].long().view(-1),
            weight=weight_class)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.trainer.fit_loop.running_loss.append(loss)

        yprob = nn.functional.softmax(y_hat.view(-1, self.num_classes), dim=-1)
        # yprob shape: [TxB,K], y_hat [T, B, K], y [T, B]

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        train_dict = {'preds': yprob.view(y.shape[0], -1, self.num_classes).permute(1, 0, 2),  # [B, T, K]
                      'labels': y.permute(1, 0).to(torch.int),  # [B, T]
                      "info": batch['video_infos'],
                      "filter": filter_cls.permute(1, 0)}  # [B, T]

        return train_dict

    def validation_step(self, batch, batch_idx):
        return self._inference_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._inference_step(batch, batch_idx, mode="test")

    def _inference_step(self, batch, batch_idx, mode="val"):
        y_hat = self.forward(batch)
        if not batch['frame_is_labeled'].sum().item():
            return None

        y_hat, y, filter_cls = self._select_valid_frames(y_hat, batch['labels'], batch['frame_is_labeled'])
        weight_class = None
        if self.hparams['weights']:
            weight_class = torch.tensor(self.hparams['weights']).to(y_hat.device)
        loss = F.cross_entropy(y_hat[filter_cls].view(-1, self.num_classes), y[filter_cls].long().view(-1),
                               weight=weight_class)

        yprob = nn.functional.softmax(y_hat.view(-1, self.num_classes), dim=-1)
        assert yprob.shape[-1] == self.num_classes  # yprob shape: [TxB,K], y_hat [T, B, K], y [T, B]

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True)
        inf_dict = {"preds": yprob.view(y.shape[0], -1, self.num_classes).permute(1, 0, 2),  # [B, T, K]
                    "labels": y.permute(1, 0).to(torch.int),  # [B, T]
                    "info": batch['video_infos'],
                    "filter": filter_cls.permute(1, 0)}  # [B, T]

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
        # ----------------------------------
        # sort the metrics by file/recording
        # ----------------------------------
        res_per_recording = defaultdict(list)
        # organize the output by filename
        for dic in outputs:
            for i, (metadata, start_time, duration) in enumerate(dic["info"]):
                if metadata.is_padding():
                    continue
                res_per_recording[os.path.basename(metadata.path)].append((start_time,
                                                                           dic["preds"][i][dic["filter"][i]].view(-1, self.num_classes),
                                                                           dic["labels"][i][dic["filter"][i]].view(-1)))

        # ------------------------------------------------------------
        # calculate global metrics and the average sample-wise metrics
        # ------------------------------------------------------------
        preds_seq, labels_seq, preds_all, labels_all = [], [], [], []
        for name in res_per_recording:
            res_per_recording[name] = sorted(res_per_recording[name], key=lambda s: s[0])
            p, l = [], []
            for item in res_per_recording[name]:
                p.append(item[1])
                l.append(item[2])
                preds_all.append(item[1])
                labels_all.append(item[2])
            preds_seq.append(torch.cat(p, 0))
            labels_seq.append(torch.cat(l, 0))

        preds_all, labels_all = torch.cat(preds_all, dim=0), torch.cat(labels_all, dim=0)

        if mode == "train":
            res = self.train_metric(preds_all, labels_all)
        elif mode == "val":
            res = self.valid_metric(preds_all, labels_all)
        elif mode == "test":
            res = self.test_metric(preds_all, labels_all)
            # sample-wise metrics
            acc_seq = [accuracy(preds_seq[i], labels_seq[i]) for i in range(len(preds_seq))]
            print("\nAverage accuracy over all test samples: {:.2f}".format(torch.mean(torch.stack(acc_seq))))

            # calculate time to prediction KPI
            latency_seq, t2p = utils_metrics.calculate_time_to_prediction(preds_seq, labels_seq,
                                                                          self.hparams['delta_t'])
            self.log_dict(t2p, on_epoch=True)

            if self.hparams.show_plots:
                for c in list(self.hparams.show_plots):
                    assert c in utils_metrics.PLOTS_DICT, "{} is not in {}".format(c, utils_metrics.PLOTS_DICT)
                    utils_metrics.PLOTS_DICT[c](preds_all.cpu(), labels_all.cpu(), self.label_map)

            if self.hparams.inspect_result:
                utils_metrics.evaluate_preds_seq(preds_seq, labels_seq, res_per_recording, self.hparams, latency_seq)

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
        test_data.to(self.device)
        hparams = self.hparams

        height, width = hparams.height, hparams.width
        batch_size = hparams.batch_size
        nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
        ncols = int(np.ceil(hparams.batch_size / nrows))

        grid = np.zeros((nrows * hparams.height, ncols * hparams.width, 3), dtype=np.uint8)
        if not show_pred:
            video_name = os.path.join(hparams.root_dir, 'check_input', f'video#{num_batches:d}batches.mp4')
        else:
            video_name = os.path.join(hparams.root_dir, 'videos', f'video#{epoch:d}.mp4')

        dir = os.path.dirname(video_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)

        video_writer = FFmpegWriter(video_name, inputdict={'-r': str(fps)}, outputdict={'-crf': '20', '-r': str(fps),
                                                                                        '-preset': 'veryslow'})

        self.net.eval()

        if show_video:
            window_name = "test epoch {:d}".format(epoch)
            cv2.namedWindow(window_name)

        for batch_nb, batch in enumerate(islice(test_data, num_batches)):
            images = batch["inputs"].cpu().clone().data.numpy()
            batch["inputs"] = batch["inputs"].to(self.device)
            batch["mask_keep_memory"] = batch["mask_keep_memory"].to(self.device)
            files = batch['video_infos']

            if show_pred:
                with torch.no_grad():
                    self.net.reset(batch["mask_keep_memory"])
                    predictions = self.net.get_probas(batch["inputs"])

            for t in range(len(images)):
                for i in range(len(images[0])):

                    if files[i][0].padding:
                        continue
                    frame = test_data.get_vis_func()(images[t][i])
                    filename = os.path.basename(files[i][0].path)
                    target_id = batch["labels"][t][i].numpy()
                    if target_id == -1:
                        target_class = 'ignore'
                    else:
                        target_class = self.label_map[target_id]

                    # draw class + score
                    color = (0, 255, 0)
                    if show_pred:
                        preds = predictions[t][i].cpu().numpy()
                        pred_class_id = np.argmax(preds)
                        pred_score = preds[pred_class_id]
                        pred_class = self.label_map[pred_class_id]
                        text = "wrong" if target_id != pred_class_id else "correct"
                        text = "correct" if target_id == -1 else text
                        cv2.putText(frame, text, (10, frame.shape[0] - 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color
                                    if text == "correct" else (255, 0, 0))
                        text = "pred: " + pred_class
                        cv2.putText(frame, text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                        text = "score: " + str(pred_score)
                        cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                    text = "gt: " + target_class
                    cv2.putText(frame, text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                    cv2.putText(frame, os.path.basename(filename), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                    y = i // ncols
                    x = i % ncols
                    y1, y2 = y * height, (y + 1) * height
                    x1, x2 = x * width, (x + 1) * width
                    grid[y1:y2, x1:x2] = frame

                if show_video:
                    gridup = cv2.pyrUp(grid)
                    cv2.imshow(window_name, gridup[..., ::-1])
                    cv2.waitKey(50)

                video_writer.writeFrame(grid)

        video_writer.close()
        if show_video:
            cv2.destroyAllWindows()
