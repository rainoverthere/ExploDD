# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This script contains test cases for exporting a RNN detector
"""
import torch
import os
import json
import argparse
import numpy as np
from metavision_ml.detection.lightning_model import LightningDetectionModel
from metavision_ml.data.cd_processor_iterator import CDProcessorIterator
from metavision_ml.detection_tracking.object_detector import ObjectDetector
from metavision_core.event_io.events_iterator import EventsIterator

CUDA_NOT_AVAILABLE = not torch.cuda.is_available()


def run_all_tests(checkpoint_path, jit_directory, sequence_raw_filename=None, device="cpu"):
    
    if CUDA_NOT_AVAILABLE and not device == "cpu":
        print(f"no GPU available can not run tests on device: {device}")
        return
    nn_filename = os.path.join(jit_directory, 'model.ptjit')
    testcase_torch_jit_reset(nn_filename, device=device)
    testcase_forward_network_with_and_without_box_decoding(nn_filename, device=device)

    if checkpoint_path != None:
        testcase_compare_ckpt_vs_jit(ckpt_filename=checkpoint_path, jit_directory=jit_directory,
                                     sequence_raw_filename=sequence_raw_filename, device=device)


def get_sizes(nn_filename):
    json_filename = os.path.join(os.path.dirname(nn_filename), "info_ssd_jit.json")
    with open(json_filename) as f:
        dic = json.load(f)
    return dic['in_channels'], dic['num_classes']


def testcase_torch_jit_reset(nn_filename, height=120, width=160, device="cpu"):
    """
    Tests the fact that function reset_all() works properly:
      * memory cell and activations are indeed set to zero
      * when providing the same input tensor to the network, the outputs we obtain from the first propagation following
        a reset should be identical to the outputs we obtain from the first propagation after loading the model
    """
    print("Test the 'reset' function of the model ...")
    assert os.path.isfile(nn_filename)

    device = torch.device(device)

    model = torch.jit.load(nn_filename, map_location=device)
    model.reset_all()
    model.eval()

    in_channels, num_classes = get_sizes(nn_filename)

    T, N, C = 1, 1, in_channels
    x = torch.rand(T, N, C, height, width).to(device)

    #check if model is float16 
    is_half = list(model.parameters())[0].dtype == torch.float16
    if is_half:
        assert not device == "cpu", "can not run half precision model on cpu"
        x = x.half()

    for name, module in model.named_modules():
        if hasattr(module, "prev_c"):
            assert (module.prev_c == 0).all().item()
            assert (module.prev_h == 0).all().item()

    with torch.no_grad():
        # first propagation
        y1 = model.forward_network_without_box_decoding(x)
        assert type(y1) is list
        assert len(y1) == 2
        cls1 = y1[1]
        assert cls1.dim() == 3
        assert cls1.shape[0] == 1
        assert cls1.shape[2] == num_classes
        for name, module in model.named_modules():
            if hasattr(module, "prev_c"):
                assert not (module.prev_c == 0).all().item()
                assert not (module.prev_h == 0).all().item()

        # second propagation (without reset)
        y2 = model.forward_network_without_box_decoding(x)
        cls2 = y2[1]
        assert not (cls1 == cls2).all().item()
        for name, module in model.named_modules():
            if hasattr(module, "prev_c"):
                assert not (module.prev_c == 0).all().item()
                assert not (module.prev_h == 0).all().item()

        # multiple propagations (with reset)
        cls_prev = cls1 
        for _iter in range(10):
            model.reset_all()
            for name, module in model.named_modules():
                if hasattr(module, "prev_c"):
                    assert (module.prev_c == 0).all().item()
                    assert (module.prev_h == 0).all().item()
            y3 = model.forward_network_without_box_decoding(x)
            cls3 = y3[1]
            # for some reason this is not exactly zero at the first iteration
            toll = 1e-7
            if _iter > 0:
                assert (cls_prev - cls3).abs().max().item() < toll, (cls_prev - cls3).abs().max().item()
            for name, module in model.named_modules():
                if hasattr(module, "prev_c"):
                    assert not (module.prev_c == 0).all().item()
                    assert not (module.prev_h == 0).all().item()
            cls_prev = cls3
            

def testcase_forward_network_with_and_without_box_decoding(nn_filename, height=120, width=160, device="cpu"):
    """
    Checks forward() is working on torch.jit model on the device specified by input parameter "device", with and without torch.no_grad()
    """
    print(f'Test forward propagation on {device} ...')
    device = torch.device(device)

    assert os.path.isfile(nn_filename)
    model = torch.jit.load(nn_filename).to(device)
    model.reset_all()
    model.eval()

    in_channels, num_classes = get_sizes(nn_filename)
    T, N, C = 1, 1, in_channels
    x = 0.05 * torch.arange(T*N*C*height*width).reshape(T, N, C, height, width).to(device)

    #check if model is float16 
    is_half = list(model.parameters())[0].dtype == torch.float16
    if is_half:
        assert not device == "cpu", "can not run half precision model on cpu"
        x = x.half()

    model.forward_network_without_box_decoding(x)
    model.forward(x, 0.5)

    with torch.no_grad():
        y1 = model.forward_network_without_box_decoding(x)
        y1 = model(x, 0.5)


def testcase_compare_ckpt_vs_jit(ckpt_filename, jit_directory, height=120, width=160, sequence_raw_filename=None, device="cpu"):
    print("Test if the PyTorch model and Torchjit model have consistent setting, and test if two models produce "
          "the same result when input sequence is provided ...")
    assert os.path.isfile(ckpt_filename)
    assert os.path.isdir(jit_directory)
    checkpoint = torch.load(ckpt_filename, map_location=torch.device(device))
    hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
    lightning_model = LightningDetectionModel(hparams)
    lightning_model.load_state_dict(checkpoint['state_dict'])

    in_channels = lightning_model.hparams["in_channels"]
    assert "background" not in lightning_model.hparams["classes"]
    num_classes = len(lightning_model.hparams["classes"]) + 1  # including the background class

    jit_filename = os.path.join(jit_directory, "model.ptjit")
    assert os.path.isfile(jit_filename)
    jit_json_filename = os.path.join(jit_directory, "info_ssd_jit.json")
    jit_json = json.load(open(jit_json_filename, "r"))

    assert jit_json["in_channels"] == in_channels
    assert jit_json["num_classes"] == num_classes

    detector_ckpt = lightning_model.detector
    detector_ckpt.reset_all()
    detector_ckpt.eval()

    detector_jit = torch.jit.load(jit_filename)
    detector_jit.reset_all()
    detector_jit.eval()

    device = torch.device(device)
    detector_ckpt.to(device)
    detector_jit.to(device)

    #check if model is float16 
    is_half = list(detector_jit.parameters())[0].dtype == torch.float16
    if is_half:
        assert not device == "cpu", "can not run half precision model on cpu"
        detector_ckpt = detector_ckpt.half()
    
    # for test we need to disable jit optimizations 
    # otherwise there will be some difference in the results (especially for half precision)
    # see https://github.com/pytorch/pytorch/issues/74534
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_nvfuser_enabled(False)

    T, N, C = 1, 1, in_channels

    # check propagation with random inputs gives the same results
    with torch.no_grad():
        for _ in range(30):
            x = torch.rand(T, N, C, height, width).to(device)
            if is_half:
                x = x.half()

            loc_ckpt, prob_ckpt = detector_ckpt.forward(x)
            loc_jit, prob_jit = detector_jit.forward_network_without_box_decoding(x)

            assert loc_ckpt.shape == loc_jit.shape
            assert (torch.abs(loc_ckpt - loc_jit) < 1e-5).all(), torch.abs(loc_ckpt - loc_jit).max()

            assert prob_ckpt.shape == prob_jit.shape
            assert (torch.abs(prob_ckpt - prob_jit) < 1e-5).all(), (torch.abs(prob_ckpt - prob_jit)).max()

    detector_ckpt.reset_all()
    detector_ckpt.eval()
    detector_jit.reset_all()
    detector_jit.eval()

    delta_t = lightning_model.hparams.get("delta_t",0)
    assert jit_json["delta_t"] == delta_t
    preproc_name = lightning_model.hparams.get("preprocess",'none')
    assert preproc_name.startswith(jit_json["preprocessing_name"])

    if not sequence_raw_filename:
        return

    # check boxes prediction are the same
    cdproc_iterator = CDProcessorIterator(path=sequence_raw_filename,
                                          preprocess_function_name=preproc_name, delta_t=delta_t)

    tensors_from_cd_proc_iterator = []
    with torch.no_grad():
        for i, x in enumerate(cdproc_iterator):
            if i >= 20:
                break
            if is_half:
                x = x.half()
            x = x.to(device)
            tensors_from_cd_proc_iterator.append(x.clone())
            res_ckpt = detector_ckpt.get_boxes(x[None], score_thresh=0.4, nms_thresh=1.)
            assert len(res_ckpt) == 1
            assert len(res_ckpt[0]) == 1
            res_ckpt = res_ckpt[0][0]
            nb_det_ckpt = 0 if res_ckpt["boxes"] is None else len(res_ckpt["boxes"])

            res_jit = detector_jit.forward(x[None], score_thresh=0.4)
            assert len(res_jit) == 1
            assert len(res_jit[0]) == 1
            res_jit = res_jit[0][0]
            assert nb_det_ckpt == len(res_jit)

            if nb_det_ckpt > 0:
                idxs_ckpt = res_ckpt["scores"].sort()[1]
                idxs_jit = res_jit[:, 4].sort()[1]

                scores_ckpt = res_ckpt["scores"][idxs_ckpt]
                scores_jit = res_jit[idxs_jit, 4]
                assert (torch.abs(scores_ckpt - scores_jit) < 1e-6).all()

                classes_ckpt = res_ckpt["labels"][idxs_ckpt]
                classes_jit = res_jit[idxs_jit, 5]
                assert (classes_ckpt == classes_jit).all()

                boxes_ckpt = res_ckpt["boxes"][idxs_ckpt]
                boxes_jit = res_jit[idxs_jit, :4]
                assert (torch.abs(boxes_ckpt - boxes_jit) < 1e-6).all()
    tensors_from_cd_proc_iterator = torch.cat(tensors_from_cd_proc_iterator, dim=0).cpu().numpy()

    # check CD Processing is the same
    events_iterator = EventsIterator(input_path=sequence_raw_filename, delta_t=delta_t, mode="delta_t")
    ev_height, ev_width = events_iterator.get_size()
    object_detector = ObjectDetector(
        directory=jit_directory, events_input_width=ev_width, events_input_height=ev_height)
    assert object_detector.get_accumulation_time() == delta_t
    cd_proc = object_detector.get_cd_processor()
    input_tensor = cd_proc.init_output_tensor()
    assert input_tensor.shape == (C, ev_height, ev_width)
    ts = 0
    for i, events in enumerate(events_iterator):
        if i >= 20:
            break
        input_tensor.fill(0.)
        cd_proc.process_events(cur_frame_start_ts=ts, events_np=events, frame_tensor_np=input_tensor)
        diff_tensors = np.abs(input_tensor - tensors_from_cd_proc_iterator[i])
        assert abs(tensors_from_cd_proc_iterator[i].min() - input_tensor.min()) < 1e-6

        assert abs(tensors_from_cd_proc_iterator[i].max() - input_tensor.max()) < 0.01
        assert abs(tensors_from_cd_proc_iterator[i].mean(
        ) - input_tensor.mean()) < 1e-4, "{} vs {}".format(tensors_from_cd_proc_iterator[i].mean(), input_tensor.mean())

        assert (np.abs(tensors_from_cd_proc_iterator[i] - input_tensor) < 0.01).all()
        ts += delta_t
