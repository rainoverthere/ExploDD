# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Detection Training Script

Instanciates and fits the detection Pytorch Lightning Module.
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from metavision_ml.detection.lightning_model import LightningDetectionModel
from metavision_ml.detection.data_factory import get_classes_from_label_map_rnn
from metavision_ml.utils.main_tools import search_latest_checkpoint, infer_preprocessing

import os
import numpy as np
import argparse
from types import SimpleNamespace

torch.manual_seed(0)
np.random.seed(0)


def autocomplete_params(params: argparse.Namespace):
    """Fills in the blank of the configuration

    Args:
        params: argparse.Namespace

    Returns:
        params: argparse.Namespace (completed)
    """
    params = SimpleNamespace(**vars(params))
    if params.dataset_path == 'toy_problem':
        params.in_channels = 3
        params.classes = [str(i) for i in range(10)]
        if params.height_width is None:
            params.height = 128
            params.width = 128
        else:
            params.height, params.width = params.height_width
        params.max_frames_per_epoch = 3000
        params.feature_base = 8
        params.feature_channels_out = 128
        params.lr = 1e-3
        params.anchor_list = 'MNIST_ANCHORS'
    else:
        all_classes = get_classes_from_label_map_rnn(params.dataset_path) #["pedestrian", "two wheeler", "car", ...]
        if not params.classes:
            params.classes = all_classes
        else:
            c1 = set(params.classes)#set是转化为数学集合的意思
            c2 = set(all_classes)
            inter = c1.difference(c2) #集合做差运算
            assert len(inter) == 0, "some classes are not part of this dataset: " + str(inter) + " from: " + str(c2)

        preprocess_dim, preprocess, delta_t, mode, n_events, preprocess_kwargs = infer_preprocessing(params)
        assert mode == "delta_t", "detection training does only supports delta_t mode"
        params.in_channels = preprocess_dim[0]#10, 和h5文件的channels保持一致
        if params.height_width is None:#一般来说都不再做调整，就和预处理的h5文件保持一致就行
            params.height = preprocess_dim[-2] #720
            params.width = preprocess_dim[-1] #1280
        else:
            assert len(params.height_width) == 2
            params.height, params.width = params.height_width
        params.delta_t = delta_t #100us 和预处理保持一致
        params.preprocess = preprocess #'event_cube_paper'

    params = argparse.Namespace(**params.__dict__)
    if params.cpu and params.precision != 32:
        print('Warning: in cpu mode precision must be 32, overriding')
        params.precision = 32

    if params.just_demo or params.just_test or params.just_val:
        params.resume = True

    return params


class TrainCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.train_dataloader is None:
            return

        if hasattr(trainer.train_dataloader.loaders, "dataset"):#居然没报错
            print("Shuffling train dataset for a new epoch !!!")
            trainer.train_dataloader.loaders.dataset.shuffle()
    
    # def on_train_epoch_start(self, trainer, pl_module):
    #     if trainer.train_dataloader is None:
    #         print("ErrorWarning: The train_dataloader is None !!!")
    #         return

    #     if hasattr(trainer.train_dataloader, "dataset"):
    #         print("Shuffling train dataset for a new epoch !")
    #         trainer.train_dataloader.dataset.shuffle()


def train_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dir params
    parser.add_argument('root_dir', type=str, default='train_log', help='logging directory')
    parser.add_argument('dataset_path', type=str, default='toy_problem',
                        help='path to folder containing train, val, test sub-folders with H5 dataset and NPY labels')

    # train params
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_scheduler_step_gamma', type=float, default=None,
                        help='learning rate scheduler step param (disabled if None)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--val_every', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--demo_every', type=int, default=2, help='run demo every X epoch')
    parser.add_argument('--save_every', type=int, default=1, help='save checkpoint every X epochs')
    parser.add_argument('--max_epochs', type=int, default=20, help='train for X epochs')
    parser.add_argument('--just_val', action='store_true', help='run validation on val set')
    parser.add_argument('--just_test', action='store_true', help='run validation on test set')
    parser.add_argument('--just_demo', action='store_true', help='run demo video with trained network')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default='', help='resume from specific checkpoint')
    parser.add_argument('--precision', type=int, default=16, help='mixed precision training default to float16')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='limit train batches to fraction of dataset')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='limit val batches to fraction of dataset')
    parser.add_argument('--limit_test_batches', type=float, default=1.0,
                        help='limit test batches to fraction of dataset')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='accumulate gradient for more than a single batch')
    parser.add_argument('--finetune', action='store_true', help='finetune from checkpoint')

    # data params
    parser.add_argument('--height_width', nargs=2, default=None, type=int,
                        help="if set, downsize the feature tensor to the corresponding resolution using interpolation")
    parser.add_argument('--num_tbins', type=int, default=12, help="timesteps per batch for truncated backprop")
    parser.add_argument('--classes', nargs='+', default=[], help='subset of classes to use')
    parser.add_argument('--num_workers', type=int, default=2, help='number of threads')
    parser.add_argument('--skip_us', type=int, default=0, help='skip this amount of microseconds')
    parser.add_argument(
        '--min_box_diag_network', type=int, default=0,
        help='minimum size for a box to be considered in the GT. WARNING: this is in the resolution of the network, '
             'after rescaling the boxes to match the training resolution.')

    # model params
    parser.add_argument('--detector_config', type=str, default='psee_rnn', help='detector type')
    parser.add_argument('--feature_extractor', type=str, default='Vanilla', help='feature extractor type')
    parser.add_argument('--feature_base', type=int, default=16, help='growth factor of feature extractor')
    parser.add_argument('--feature_channels_out', type=int, default=256, help='number of channels per feature-map')
    parser.add_argument('--anchor_list', type=str, default='PSEE_ANCHORS',
                        choices=['PSEE_ANCHORS', 'COCO_ANCHORS', 'MNIST_ANCHORS'],
                        help='anchor configuration')
    parser.add_argument(
        '--max_boxes_per_input', type=int, default=500,
        help='during evaluation, maximum number of boxes selected per input, before score threshold or NMS,'
        'more can increase recall and precision but it significantly increase the memory cost of performing evaluation')

    # display params
    parser.add_argument("--no_window", action="store_true",
                        help="Disable output window during demo (only write a video)")
    parser.add_argument("--verbose", action="store_true", help="Display CocoKPIs per category")

    return parser


def train(params: argparse.Namespace):
    """Using Pytorch Lightning to train our model

    This trainer trains a Recurrent SSD on your data.
    Every X epochs it creates a demonstration video and computes COCO metrics.

    By Default, you can train on the moving mnist toy problem by setting the dataset_path to "toy_problem".

    Otherwise, you need to indicate a valid dataset path with format compatible
    with metavision_ml/data/sequential_dataset.py (.h5 or .dat files)

    You can visualize logs with tensorboard::

        %tensorboard --log_dir=.
    """
    model = LightningDetectionModel(params)
    gpu_nums = 0 if params.cpu else 1 #1

    if params.finetune:#fine_tune
        assert params.checkpoint is not None
        model.load_pretrained(params.checkpoint)
        ckpt = None
    elif params.checkpoint != "":#提供指定的ckpt
        ckpt = params.checkpoint
    elif params.resume:#断点续训
        ckpt = search_latest_checkpoint(params.root_dir)
    else: #从头开始训练
        ckpt = None

    tmpdir = os.path.join(params.root_dir, 'checkpoints')
    
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, every_n_epochs=params.save_every)#只有在微调的时候关闭这一行
    # checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, every_n_train_steps=5) #只有在微调的时候启用这一行
    # # tmpdir会在训练过程中自动创建; save_top_k=-1代表保存所有的ckpt; 默认情况下params.save_every=1 #有个问题：save_top_k的标准或者说依据是如何具体设置的，根据什么来保存topk？ 根据固定步数保留ckpt似乎是可行的：设置every_n_train_steps=5，训练2个epoch，差不多接近50个ckpt，足够描绘细致的趋势图了.

    logger = TensorBoardLogger(
        save_dir=os.path.join(params.root_dir, 'logs'),
        version=1) #root_dir/logs/version_1

    if ckpt is not None and (params.just_demo or params.just_val or params.just_test):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu') if params.cpu else torch.device("cuda"))
        model.load_state_dict(checkpoint['state_dict'])

    if params.just_demo:
        model.detector.eval()
        if not params.cpu:
            model = model.cuda()
        model.demo_video(-1, num_batches=10, show_video=not params.no_window)#out_dir/videos/video#-1.mp4 默认情况下是val_split
    elif params.just_val:
        pl.Trainer(gpus=gpu_nums, limit_test_batches=params.limit_val_batches).test(
            model, dataloaders=model.val_dataloader())
    elif params.just_test:
        pl.Trainer(gpus=gpu_nums, limit_test_batches=params.limit_test_batches).test(
            model, dataloaders=model.test_dataloader())
    else:
        trainer = pl.Trainer(
            default_root_dir=params.root_dir, logger=logger,
            callbacks=[checkpoint_callback, TrainCallback()],
            gpus=gpu_nums,
            precision=params.precision,
            max_epochs=params.max_epochs, check_val_every_n_epoch=params.val_every,
            resume_from_checkpoint=ckpt,
            accumulate_grad_batches=params.accumulate_grad_batches,
            log_every_n_steps=5,

            # Fast Run
            limit_train_batches=params.limit_train_batches,
            limit_val_batches=params.limit_val_batches,
            limit_test_batches=params.limit_test_batches)
        trainer.fit(model) #开启训练模式


if __name__ == '__main__':
    parser = train_parser()
    params = parser.parse_args()
    params = autocomplete_params(params)
    train(params)
