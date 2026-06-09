# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This file defines callbacks to be added to the pytorch lightning interface during Flow training.
"""
import pytorch_lightning as pl


class FlowCallback(pl.callbacks.Callback):
    """
    Determines the number of max_consecutive batch in the training dataloader before each epoch.

    Args:
        data_module (object): pytorch lightning data module
        curriculum_param (float): parameter used to compute the number of consecutive batches to extract from a video.
        It relies on the following law : max_consecutive_batch = int(p^(epoch-1))
        video_result_every_n_epochs (int): every n epoch a video file is generated showing the result on a validation subset.
        mask_flow_by_input (boolean): if True only display non-zero flows.
    """

    def __init__(self, data_module, curriculum_param=1.414,
                 video_result_every_n_epochs=2, mask_flow_by_input=True):
        super().__init__()
        self.curriculum_param = min(1.0, curriculum_param)
        self.video_every = int(video_result_every_n_epochs)
        self.mask_by_input = mask_flow_by_input
        self.data_module = data_module

    def on_train_epoch_start(self, trainer, pl_module):
        """When epoch starts the callback reschedules the dataset.

        Args:
            trainer (object): trainer object
            pl_module (object): pytorch lightning model
        """
        tensorboard = pl_module.logger.experiment
        max_consecutive_batch = int(max(1, self.curriculum_param**(trainer.current_epoch - 1))
                                    ) if trainer.current_epoch < 20 else int(1e9)

        if trainer.train_dataloader:
            trainer.train_dataloader.loaders.dataset.reschedule(max_consecutive_batch=max_consecutive_batch)

        tensorboard.add_scalar("max_consecutive_batch", max_consecutive_batch, trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        """When epoch ends the callback launches a demo.

        Args:
            trainer (object): trainer object
            pl_module (object): pytorch lightning model
        """
        if not trainer.current_epoch % self.video_every:
            pl_module.demo_video(
                self.data_module.test_dataloader(),
                log_dir=trainer.logger.save_dir,
                epoch=trainer.current_epoch,
                num_batches=50,
                write_video=True,
                show_video=False,
                mask_by_input=self.mask_by_input)
