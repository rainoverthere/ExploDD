# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Set up an ObjectDetector for Inference Pipeline
"""
import json
import os

import metavision_sdk_core
import metavision_sdk_ml

import numpy as np

import torch


class ObjectDetector:
    """Class handling the inference of object detection algorithm.

    Args:
        directory (string): Path to a folder containing a `model.ptjit` network file and a `info_ssd_jit.json`
            containing a few hyperparameters.
        events_input_width (int): Width of the event sensor used, in pixels.
        events_input_height (int): Height of the event sensor used, in pixels.
        runtime (string): Corresponds to the torch.device used for computation ("cpu", "cuda:0", etc.)
        network_input_width (int): Width of the tensor used by the network as input, can be lower than
            events_input_width to reduce computation.
        network_input_height (int): Height of the tensor used by the network as input, can be lower than
            events_input_height to reduce computation.

    Attributes:
        cd_proc (object): Object computing tensor representation from the input events.
        detection_threshold (float): Minimal confidence value for a box to be considered.
        iou_threshold (float): Minimal Intersection Over Union (IOU) value for a box to be discarded by non maximum
            suppression.
        nms_computer_with_rescaling (object): Object handling Non maximum suppression.
    """

    def __init__(self, directory, events_input_width, events_input_height,
                 runtime="cpu", network_input_width=None, network_input_height=None):
        self.device = torch.device(runtime.replace("gpu", "cuda"))
        assert os.path.isdir(directory)
        filename_model = os.path.join(directory, "model.ptjit")
        filename_json = os.path.join(directory, "info_ssd_jit.json")
        assert os.path.isfile(filename_model)
        assert os.path.isfile(filename_json)
        self.model = torch.jit.load(filename_model, map_location=self.device)
        self.model.reset_all()
        self.model.eval()

        self.is_half = list(self.model.parameters())[0].dtype == torch.float16

        if self.is_half:
            assert not runtime == "cpu", "can not run half precision model on cpu"

        with open(filename_json, "r") as file_json:
            self.model_json = json.load(file_json)
        self.events_input_width = events_input_width
        self.events_input_height = events_input_height

        self.network_input_width = events_input_width if network_input_width is None else network_input_width
        self.network_input_height = events_input_height if network_input_height is None else network_input_height

        self.detection_threshold = self.model_json["confidence_threshold"]
        self.iou_threshold = self.model_json["iou_threshold"]

        assert self.model_json["num_classes"] == len(self.model_json["label_map"])
        self.nms_computer_with_rescaling = metavision_sdk_ml.NonMaximumSuppressionWithRescaling(
            network_num_classes=self.model_json["num_classes"], events_input_width=self.events_input_width,
            events_input_height=self.events_input_height, network_input_width=self.network_input_width,
            network_input_height=self.network_input_height, iou_threshold=self.iou_threshold)
        self.nms_buffer = self.nms_computer_with_rescaling.get_empty_output_buffer()

        preprocessing_name = self.model_json["preprocessing_name"]
        if preprocessing_name == "diff" or preprocessing_name == "diff3d":
            self.cd_proc = metavision_sdk_ml.CDProcessing.create_CDProcessingDiff(
                delta_t=self.model_json["delta_t"],
                network_input_width=self.network_input_width, network_input_height=self.network_input_height,
                max_incr_per_pixel=self.model_json["max_incr_per_pixel"],
                clip_value_after_normalization=self.model_json["clip_value_after_normalization"],
                event_input_width=self.events_input_width, event_input_height=self.events_input_height)
        elif preprocessing_name == "histo" or preprocessing_name == "histo3d":
            self.cd_proc = metavision_sdk_ml.CDProcessing.create_CDProcessingHisto(
                delta_t=self.model_json["delta_t"],
                network_input_width=self.network_input_width, network_input_height=self.network_input_height,
                max_incr_per_pixel=self.model_json["max_incr_per_pixel"],
                clip_value_after_normalization=self.model_json["clip_value_after_normalization"],
                event_input_width=self.events_input_width, event_input_height=self.events_input_height, use_CHW=True)
        elif preprocessing_name == "event_cube":
            self.cd_proc = metavision_sdk_ml.CDProcessing.create_CDProcessingEventCube(
                delta_t=self.model_json["delta_t"],
                network_input_width=self.network_input_width, network_input_height=self.network_input_height,
                num_utbins=self.model_json["num_utbins"],
                split_polarity=self.model_json["split_polarity"],
                max_incr_per_pixel=self.model_json["max_incr_per_pixel"],
                clip_value_after_normalization=self.model_json["clip_value_after_normalization"],
                event_input_width=self.events_input_width, event_input_height=self.events_input_height)
        else:
            raise RuntimeError("Invalid processing type: {}".format(preprocessing_name))
        assert self.cd_proc.get_frame_channels() == self.model_json["num_channels"]

    def get_accumulation_time(self):
        return self.model_json["delta_t"]

    def set_detection_threshold(self, thr):
        self.detection_threshold = thr

    def set_iou_threshold(self, thr):
        self.set_iou_threshold = thr

    def get_cd_processor(self):
        return self.cd_proc

    def process(self, ts, frame_buffer_np):
        """Pass the input frame through the object detector and return the obtained box events.

        Args:
            ts (int): Current timestamp in us.
            frame_buffer_np (np.ndarray): Input frame buffer.
        """
        assert len(frame_buffer_np.shape) == 3
        frame_buffer_tensor = torch.from_numpy(
            frame_buffer_np).unsqueeze(0).unsqueeze(0).to(self.device)

        
        if self.is_half:
            frame_buffer_tensor = frame_buffer_tensor.half()
        with torch.no_grad():
            tensorBoxList = self.model(frame_buffer_tensor, self.detection_threshold)

        assert isinstance(tensorBoxList, list)
        assert len(tensorBoxList) == 1
        assert isinstance(tensorBoxList[0], list)
        assert len(tensorBoxList[0]) == 1

        nb_boxes, dim = tensorBoxList[0][0].shape
        assert dim == 6
        detections = np.empty(0, dtype=metavision_sdk_core.EventBbox)
        if nb_boxes > 0:
            tensorBox = tensorBoxList[0][0].cpu()
            detections_before_nms = np.zeros(
                nb_boxes, dtype=metavision_sdk_core.EventBbox)
            detections_before_nms["t"] = ts
            detections_before_nms["x"] = tensorBox[:, 0]
            detections_before_nms["y"] = tensorBox[:, 1]
            detections_before_nms["w"] = tensorBox[:, 2] - tensorBox[:, 0]
            detections_before_nms["h"] = tensorBox[:, 3] - tensorBox[:, 1]
            detections_before_nms["class_confidence"] = tensorBox[:, 4]
            detections_before_nms["class_id"] = tensorBox[:, 5]

            self.nms_computer_with_rescaling.process_events(detections_before_nms, self.nms_buffer)
            detections = self.nms_buffer.numpy()
        return detections

    def reset(self):
        """Resets the memory cells of the neural network
        """
        self.model.reset_all()

    def get_num_classes(self):
        """Gets number of output classes (including background class)"""
        return self.model_json["num_classes"]

    def get_label_map(self):
        """Returns a dict which contains the labels of the classes"""
        return self.model_json["label_map"]
