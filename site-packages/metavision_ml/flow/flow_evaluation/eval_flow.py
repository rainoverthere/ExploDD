# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import argparse
import h5py
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2
import tqdm
from skvideo.io import FFmpegWriter
import json

from metavision_sdk_base import EventCD
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io.py_reader import EventDatReader
from metavision_ml.core.warping import compute_iwe
from metavision_ml.preprocessing.viz import normalize, filter_outliers
from metavision_ml.flow.viz import draw_arrows


def events_to_diff_image(events, sensor_size, strict_coord=True):
    """
    Place events into an image using numpy
    """
    xs = events["x"]
    ys = events["y"]
    ps = events["p"] * 2 - 1
    img_size = sensor_size

    mask = np.where(xs > sensor_size[1]-1, 0, 1)*np.where(ys > sensor_size[0]-1,
                                                          0, 1)*np.where(xs < 0, 0, 1)*np.where(ys < 0, 0, 1)
    if strict_coord:
        assert (mask == 1).all()
    coords = np.stack((ys*mask, xs*mask))
    ps *= mask

    try:
        abs_coords = np.ravel_multi_index(coords, sensor_size)
    except ValueError:
        raise ValueError("Issue with input arrays! coords={}, min_x={}, min_y={}, max_x={}, max_y={}, coords.shape={}, sum(coords)={}, sensor_size={}".format(
            coords, min(xs), min(ys), max(xs), max(ys), coords.shape, np.sum(coords), sensor_size))

    img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0]*sensor_size[1])
    img = img.reshape(sensor_size)
    return img


def propagate_flow(flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    C, H, W = flow.shape
    assert C == 2
    x_flow = flow[0]
    y_flow = flow[1]
    flow_x_interp = cv2.remap(x_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    flow_y_interp = cv2.remap(y_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor

    x_mask[x_indices < 0] = False
    x_mask[x_indices >= x_indices.shape[1]] = False

    y_mask[y_indices < 0] = False
    y_mask[y_indices >= y_indices.shape[0]] = False

    return


def compute_flow_kpi_without_gt(flow_est, ev_reader,
                                output_directory=None, save_graphs=False, save_npz=False, save_video=False):
    N, C, H, W = flow_est["flow"].shape
    assert C == 2

    if save_video:
        video_process = FFmpegWriter(os.path.join(output_directory, "kpis_without_gt.mp4"),
                                     inputdict={'-r': "5"},
                                     outputdict={'-vcodec': 'libx264'})

    flow_warp_loss_list = []
    for i in tqdm.tqdm(range(N)):
        start_ts = flow_est["flow_start_ts"][i]
        end_ts = flow_est["flow_end_ts"][i]
        ev_reader.seek_time(start_ts)
        events_i = ev_reader.load_delta_t(end_ts - start_ts)

        flow_i = flow_est["flow"][i].astype(np.float32)

        diff_img = events_to_diff_image(events_i, sensor_size=(H, W))
        var_diff_img = np.var(diff_img)

        iwe_i = compute_iwe(events_i, torch.from_numpy(flow_i), t=events_i["t"][-1])
        var_iwe_i = torch.var(iwe_i).item()

        flow_warp_loss_list.append(var_iwe_i / np.var(diff_img))

        if save_video:
            diff_img_uint8 = np.uint8(255*normalize(filter_outliers(diff_img, 7)))
            iwe_i_uint8 = np.uint8(255*normalize(filter_outliers(iwe_i, 7)))

            timestamps_txt = "ts: {} -> {}".format(start_ts, end_ts)
            frame_diff_i_txt = "var: {:.2f}".format(var_diff_img)
            cv2.putText(diff_img_uint8,
                        timestamps_txt,
                        (int(0.05 * (W)), 40),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 100), thickness=2)
            cv2.putText(diff_img_uint8,
                        frame_diff_i_txt,
                        (int(0.05 * (W)), int(0.92*H)),
                        cv2.FONT_HERSHEY_PLAIN, 3.0, (200, 200, 100), thickness=3)

            fwl_est_txt = "var: {:.2f}   FWL: {:.2f}".format(var_iwe_i, var_iwe_i / var_diff_img)
            cv2.putText(iwe_i_uint8,
                        fwl_est_txt,
                        (int(0.05 * (W)), int(0.92*H)),
                        cv2.FONT_HERSHEY_PLAIN, 3.0, (200, 200, 100), thickness=3)

            img_warp = np.concatenate((diff_img_uint8, iwe_i_uint8), axis=1)
            img_warp[:, W-1:W+1] = 0
            img_warp = cv2.cvtColor(img_warp, cv2.COLOR_GRAY2RGB)

            img_pos = events_to_diff_image(events_i[events_i["p"] == 1], sensor_size=(H, W))
            img_neg = events_to_diff_image(events_i[events_i["p"] == 0], sensor_size=(H, W))
            mask_ev = np.logical_or(img_pos != 0, img_neg != 0)
            img_mask_ev = np.zeros((H, W, 3), dtype=np.uint8)
            img_mask_ev[mask_ev, :] = 255
            flow_i_img = draw_arrows(255*np.ones((H, W, 3), dtype=np.uint8), flow_i)
            img_mask_ev_flow = np.concatenate((img_mask_ev, flow_i_img), axis=1)
            img_mask_ev_flow[:, W-1:W+1, :] = 0

            img_fwl_and_flow = np.concatenate((img_warp, img_mask_ev_flow), axis=0)
            img_fwl_and_flow[H-1:H+1, :, :] = 0
            video_process.writeFrame(img_fwl_and_flow)

    video_process.close()

    FWL_np = np.array(flow_warp_loss_list)
    if save_npz:
        np.savez_compressed(os.path.join(output_directory, "kpis_without_gt.npz"),
                            FWL=FWL_np,
                            flow_start_ts=flow_est["flow_start_ts"][:],
                            flow_end_ts=flow_est["flow_end_ts"][:])

    plt.figure()
    plt.plot(flow_est["flow_start_ts"][:], flow_warp_loss_list)
    plt.axhline(1, c="r", ls="--")
    plt.title("Evolution of Flow Warp Loss (FWL) over time")
    plt.xlabel("time")
    plt.ylabel("FWL")
    if save_graphs:
        plt.savefig(os.path.join(output_directory, "FWL_without_gt.png"))
    else:
        plt.show()

    res = {"FWL": np.mean(FWL_np).item()}
    return res


def estimate_corresponding_gt_flow(start_time, end_time, flow_gt):
    """
    Computes the GT between start_time and end_time

    if start_time and end_time fall within a single flow_gt frame, we simply rescale the flow accordingly
    otherwise we propagate the forward flow from nearby chunk to nearby chunk, discarding invalid positions along the way

    If start_time is before the start_time of the first flow_gt frame,
    or if end_time is after the end_time of the last flow_gt frame,
    or if there are temporal holes in consecutives flow_gt frames between
    then we cannot compute the gt flow for that chunk of events. We return a mask where all pixels are marked as invalid

    Args:
        start_time: start timestamp of event chunk on which we want to compute the GT flow (N,)
        end_time: end timestamp of event chunk on which we want to compute the GT flow (N,)
        flow_gt: h5 file which contains GT frames (N, 2, H, W)
    Returns:
        flow: np.float32 (2, H, W) gt flow between start_time and end_time
        valid_mask: np.bool_ (H, W) valid mask
    """
    N, C, H, W = flow_gt["flow"].shape
    assert C == 2
    gt_start_idx = np.searchsorted(flow_gt["flow_start_ts"], start_time, side="right") - 1
    if gt_start_idx < 0:
        gt_start_idx = 0
    gt_start_ts = flow_gt["flow_start_ts"][gt_start_idx]

    gt_end_idx = np.searchsorted(flow_gt["flow_end_ts"], end_time, side="left")
    if gt_end_idx >= N:
        gt_end_idx = N - 1
    gt_end_ts = flow_gt["flow_end_ts"][gt_end_idx]

    if gt_start_ts > start_time or gt_end_ts < end_time or flow_gt["flow_end_ts"][gt_start_idx] < start_time:
        # chunk of events starts before the gt or ends after the gt. We cannot compute a valid gt
        flow_i = np.zeros(shape=(2, H, W), dtype=np.float32)
        valid_mask = np.zeros(shape=(H, W), dtype=bool)
        return flow_i, valid_mask

    # if we arrive at this point, the following conditions must hold true
    assert gt_start_ts <= start_time
    assert flow_gt["flow_end_ts"][gt_start_idx] >= start_time

    gt_end_idx = np.searchsorted(flow_gt["flow_end_ts"], end_time, side="left")
    assert gt_end_idx < N
    gt_end_ts = flow_gt["flow_end_ts"][gt_end_idx]
    assert gt_end_ts >= end_time
    assert flow_gt["flow_start_ts"][gt_end_idx] < end_time

    if gt_start_idx == gt_end_idx:
        flow_previous_duration = gt_end_ts - gt_start_ts
        flow_new_duration = end_time - start_time
        flow_i = flow_gt["flow"][gt_start_idx].astype(np.float32)
        flow_i[...] *= float(flow_new_duration) / flow_previous_duration
        valid_x = flow_i[0] != np.inf
        valid_y = flow_i[1] != np.inf
        valid_mask = np.logical_and(valid_x, valid_y)
        flow_i[0][~valid_mask] = 0
        flow_i[1][~valid_mask] = 0
        return flow_i, np.ones((H, W), np.bool)

    # check that consecutive gt flow frame don't contain temporal hole (every gt frame starts when the previous one finishes)
    if not((flow_gt["flow_end_ts"][gt_start_idx: gt_end_idx + 1][: -1] -
            flow_gt["flow_start_ts"][gt_start_idx: gt_end_idx + 1][1:]) == 0).all():
        # There are temporal holes in the gt for this chunk of events. We cannot compute a valid gt
        flow_i = np.zeros(shape=(2, H, W), dtype=np.float32)
        valid_mask = np.zeros(shape=(H, W), dtype=np.bool)
        return flow_i, valid_mask

    assert gt_start_idx < gt_end_idx
    assert flow_gt["flow_end_ts"][gt_start_idx] < end_time

    x_indices, y_indices = np.meshgrid(np.arange(W), np.arange(H))
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = float(flow_gt["flow_end_ts"][gt_start_idx] - start_time) / (flow_gt["flow_end_ts"]
                                                                               [gt_start_idx] - flow_gt["flow_start_ts"][gt_start_idx])
    assert scale_factor <= 1.
    propagate_flow(flow_gt["flow"][gt_start_idx], x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor)

    cur_idx = gt_start_idx

    while cur_idx + 1 < gt_end_idx:
        cur_idx += 1
        propagate_flow(flow_gt["flow"][cur_idx], x_indices, y_indices, x_mask, y_mask)

    assert cur_idx == gt_end_idx - 1

    scale_factor = float(end_time - flow_gt["flow_start_ts"][gt_end_idx]
                         ) / (flow_gt["flow_end_ts"][gt_end_idx] - flow_gt["flow_start_ts"][gt_end_idx])
    assert scale_factor <= 1.
    propagate_flow(flow_gt["flow"][gt_end_idx], x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor)
    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0
    flow_i = np.stack((x_shift, y_shift), axis=0)
    assert flow_i.shape == (2, H, W)
    valid_mask = np.logical_and(x_shift, y_shift)
    assert valid_mask.shape == (H, W)
    return flow_i, valid_mask


def compute_flow_kpi_with_gt(flow_est, flow_gt, ev_reader, normal_flow=False,
                             output_directory=None, save_graphs=False, save_npz=False, save_video=False):
    N, C, H, W = flow_est["flow"].shape
    assert C == 2

    if save_video:
        video_process = FFmpegWriter(os.path.join(output_directory, "kpis.mp4"),
                                     inputdict={'-r': "5"},
                                     outputdict={'-vcodec': 'libx264'})

    FWL_est = []
    FWL_gt = []
    nb_pix_list = []
    AEE_list = []
    AEErel_list = []
    FE_list = []
    APEE_list = []
    AAE_list = []
    mask_list = []

    for i in tqdm.tqdm(range(N)):
        start_ts = flow_est["flow_start_ts"][i]
        end_ts = flow_est["flow_end_ts"][i]
        if i == N-1:
            if flow_gt["flow_end_ts"][-1] < end_ts:
                end_ts = flow_gt["flow_end_ts"][-1]
        flow_gt_i, valid_mask_i = estimate_corresponding_gt_flow(start_ts, end_ts, flow_gt)
        flow_i = flow_est["flow"][i].astype(np.float32)
        ev_reader.seek_time(start_ts)
        events_i = ev_reader.load_delta_t(end_ts - start_ts)

        diff_img = events_to_diff_image(events_i, sensor_size=(H, W))
        var_diff_img = np.var(diff_img)

        iwe_i = compute_iwe(events_i, torch.from_numpy(flow_i), t=events_i["t"][-1])
        var_iwe_i = torch.var(iwe_i).item()

        iwe_gt_i = compute_iwe(events_i, torch.from_numpy(flow_gt_i), t=events_i["t"][-1])
        var_iwe_gt_i = torch.var(iwe_gt_i).item()

        est_times.append(start_ts)
        est_times_end.append(end_ts)

        if save_video:
            flow_i_img = draw_arrows(255*np.ones((H, W, 3), dtype=np.uint8), flow_i)
            flow_gt_i_img = draw_arrows(255*np.ones((H, W, 3), dtype=np.uint8), flow_gt_i)

            diff_img_uint8 = np.uint8(255*normalize(filter_outliers(diff_img, 7)))
            iwe_i_uint8 = np.uint8(255*normalize(filter_outliers(iwe_i, 7)))
            iwe_gt_i_uint8 = np.uint8(255*normalize(filter_outliers(iwe_gt_i, 7)))

            timestamps_txt = "ts: {} -> {}".format(start_ts, end_ts)
            frame_diff_i_txt = "var: {:.2f}".format(var_diff_img)
            cv2.putText(diff_img_uint8,
                        timestamps_txt,
                        (int(0.05 * (W)), 40),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 100), thickness=2)
            cv2.putText(diff_img_uint8,
                        frame_diff_i_txt,
                        (int(0.05 * (W)), int(0.92*H)),
                        cv2.FONT_HERSHEY_PLAIN, 3.0, (200, 200, 100), thickness=3)

            fwl_est_txt = "var: {:.2f}   FWL: {:.2f}".format(var_iwe_i, var_iwe_i / var_diff_img)
            cv2.putText(iwe_i_uint8,
                        fwl_est_txt,
                        (int(0.05 * (W)), int(0.92*H)),
                        cv2.FONT_HERSHEY_PLAIN, 3.0, (200, 200, 100), thickness=3)

            fwl_gt_i_txt = "var: {:.2f}  FWL GT: {:.2f}".format(var_iwe_gt_i, var_iwe_gt_i / var_diff_img)
            cv2.putText(iwe_gt_i_uint8,
                        fwl_gt_i_txt,
                        (int(0.05 * (W)), int(0.92*H)),
                        cv2.FONT_HERSHEY_PLAIN, 3.0, (200, 200, 100), thickness=3)

            img_warp = np.concatenate((diff_img_uint8, iwe_i_uint8, iwe_gt_i_uint8), axis=1)
            img_warp[:, W-1:W+1] = 0
            img_warp[:, 2*W-1:2*W+1] = 0
            img_warp = cv2.cvtColor(img_warp, cv2.COLOR_GRAY2RGB)

        FWL_est.append(var_iwe_i / var_diff_img)
        FWL_gt.append(var_iwe_gt_i / var_diff_img)

        img_pos = events_to_diff_image(events_i[events_i["p"] == 1], sensor_size=(H, W))
        img_neg = events_to_diff_image(events_i[events_i["p"] == 0], sensor_size=(H, W))
        mask_ev = np.logical_or(img_pos != 0, img_neg != 0)

        flow_mask = np.logical_and(
            np.logical_and(~np.isinf(flow_gt_i[0]), ~np.isinf(flow_gt_i[1])),
            np.linalg.norm(flow_gt_i, axis=0) > 0)
        total_mask = np.logical_and(flow_mask, np.logical_and(mask_ev, valid_mask_i))

        if total_mask.sum() == 0:
            print("Warning: no valid GT for slice {}, ts: {} -> {}".format(i, start_ts, end_ts))
            nb_pix, AEE, AEErel, FE, APEE, AAE = 0, 0, 0, 0, 0, 0
            if save_video:
                img_mask_diff_flow = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            flow_gt_masked = flow_gt_i[:, total_mask]
            flow_est_masked = flow_i[:, total_mask]

            EE = np.linalg.norm(flow_gt_masked - flow_est_masked, axis=0)
            AEE = EE.mean()

            flow_diff = np.zeros((2, H, W), dtype=np.float32)
            flow_diff[:, total_mask] = flow_gt_masked - flow_est_masked
            EE_mask_img = np.zeros((H, W, 3), dtype=np.uint8)
            EE_mask_img[total_mask, :] = 255

            img_mask_diff_flow = draw_arrows(EE_mask_img, flow_diff)

            EErel = EE / np.linalg.norm(flow_gt_masked, axis=0)
            AEErel = EErel.mean()

            EE_3pix_mask = EE >= 3.
            EE_5pct_mask = EErel >= 0.05
            FE = np.logical_or(EE_3pix_mask, EE_5pct_mask).sum() / EE.size

            PEE = np.abs(
                np.linalg.norm(flow_est_masked, axis=0) - (flow_est_masked * flow_gt_masked).sum(axis=0) /
                (np.linalg.norm(flow_est_masked, axis=0) + 1e-5))
            APEE = PEE.mean()
            nb_pix = EE.size

            mask_est_non_zero = np.linalg.norm(flow_i, axis=0) > 0
            total_mask_non_zero = np.logical_and(total_mask, mask_est_non_zero)

            if total_mask_non_zero.sum() == 0:
                AAE = 0
            else:
                flow_gt_non_zero = flow_gt_i[:, total_mask_non_zero]
                flow_est_non_zero = flow_i[:, total_mask_non_zero]
                AEcos = (flow_gt_non_zero * flow_est_non_zero).sum(axis=0) / (np.linalg.norm(flow_gt_non_zero,
                                                                                             axis=0)*np.linalg.norm(flow_est_non_zero, axis=0))
                AAE = np.degrees(np.arccos(AEcos.clip(-1, 1))).mean()

        if save_video:
            cv2.putText(img_mask_diff_flow,
                        "Mask + Error Flow",
                        (int(0.05 * (W)), 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 100), thickness=2)
            cv2.putText(flow_i_img,
                        "Flow est",
                        (int(0.05 * (W)), 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 100), thickness=2)
            cv2.putText(flow_gt_i_img,
                        "Flow GT",
                        (int(0.05 * (W)), 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 100), thickness=2)
            img_flow = np.concatenate((img_mask_diff_flow, flow_i_img, flow_gt_i_img), axis=1)
            img_flow[:, W-1:W+1, :] = 0
            img_flow[:, 2*W-1:2*W+1, :] = 0

            img_fwl_and_flow = np.concatenate((img_warp, img_flow), axis=0)
            img_fwl_and_flow[H-1:H+1, :, :] = 0
            video_process.writeFrame(img_fwl_and_flow)

        nb_pix_list.append(nb_pix)
        AEE_list.append(AEE)
        AEErel_list.append(AEErel)
        FE_list.append(FE)
        APEE_list.append(APEE)
        AAE_list.append(AAE)
        mask_list.append(total_mask)

    if save_video:
        video_process.close()

    plt.figure()
    plt.plot(flow_est["flow_start_ts"][:], FWL_est, label="FWL est")
    plt.plot(flow_est["flow_start_ts"][:], FWL_gt, label="FWL gt")
    plt.axhline(1, c="r", ls="--")
    plt.title("Evolution of Flow Warp Loss (FWL) over time")
    plt.xlabel("time")
    plt.ylabel("FWL")
    plt.legend()
    if save_graphs:
        plt.savefig(os.path.join(output_directory, "FWL.png"))
    else:
        plt.show()

    fig, axes_array = plt.subplots(nrows=4, ncols=1)
    axes_array[0].plot(flow_est["flow_start_ts"][:], AEE_list, label="AEE")
    axes_array[0].set_title("Average Endpoint Error (AEE)")
    axes_array[1].plot(flow_est["flow_start_ts"][:], AEErel_list, label="AEErel")
    axes_array[1].set_title("Average Endpoint Error Relative to GT (AEErel)")
    axes_array[2].plot(flow_est["flow_start_ts"][:], FE_list, label="FE")
    axes_array[2].set_title("Flow Error (FE)")
    axes_array[3].plot(flow_est["flow_start_ts"][:], AAE_list, label="AEE")
    axes_array[3].set_title("Average Angular Error (AAE)")
    if save_graphs:
        plt.savefig(os.path.join(output_directory, "AEE_AAE.png"))
    else:
        plt.show()

    if normal_flow:
        plt.figure()
        plt.plot(flow_est["flow_start_ts"][:], APEE_list, label="APEE")
        plt.legend()
        if save_graphs:
            plt.savefig(os.path.join(output_directory, "APEE.png"))
        else:
            plt.show()

    AEE_np = np.array(AEE_list)
    AEErel_np = np.array(AEErel_list)
    FE_np = np.array(FE_list)
    APEE_np = np.array(APEE_list)
    AAE_np = np.array(AAE_list)
    FWL_np = np.array(FWL_est)
    FWL_gt_np = np.array(FWL_gt)
    mask_np = np.array(mask_list)
    if save_npz:
        np.savez_compressed(os.path.join(output_directory, "kpis.npz"),
                            AEE=AEE_np, AEErel=AEErel_np, FE=FE_np, APEE=APEE_np,
                            AAE=AAE_np, FWL=FWL_np, FWL_gt=FWL_gt_np,
                            flow_start_ts=flow_est["flow_start_ts"][:],
                            flow_end_ts=flow_est["flow_end_ts"][:],
                            mask=mask_np)

    res = {"AEE": np.mean(AEE_np).item(),
           "AEErel": np.mean(AEErel_np).item(),
           "FE": np.mean(FE_np).item(),
           "APEE": np.mean(APEE_np).item(),
           "AAE": np.mean(AAE_np).item(),
           "FWL": np.mean(FWL_np).item(),
           "FWL_gt": np.mean(FWL_gt_np).item()
           }
    return res


def check_h5py_flow_consistency(flow_h5):
    assert "flow" in flow_h5
    assert "flow_start_ts" in flow_h5
    assert "flow_end_ts" in flow_h5
    height, width = flow_h5["flow"].attrs["event_input_height"], flow_h5["flow"].attrs["event_input_width"]
    N, C, H, W = flow_h5["flow"].shape
    assert C == 2
    assert height == H
    assert width == W
    assert flow_h5["flow_start_ts"].size == N
    assert flow_h5["flow_end_ts"].size == N


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate flow KPIs")
    parser.add_argument("--estimated-flow", dest="estimated_flow", required=True, help="Estimated flow (in hdf5)")
    parser.add_argument("--events-sequence", dest="events_sequence", required=True,
                        help="Events sequence (either .raw, .dat, or .npy")
    parser.add_argument("--gt-flow", dest="gt_flow", required=False, help="Groundtruth flow (in hdf5)")
    parser.add_argument("--normal-flow", dest="normal_flow", action="store_true",
                        help="Use this when the estimation method computes normal flow")

    group_outputs = parser.add_argument_group("Outputs")
    group_outputs.add_argument("--output-directory", dest="output_directory",
                               help="Output directory to save evaluation results.")
    group_outputs.add_argument("--save-graphs", dest="save_graphs", action="store_true",
                               help="Save graph plots into outut_directory")
    group_outputs.add_argument("--save-npz", dest="save_npz", action="store_true", help="Save results into a npz file")
    group_outputs.add_argument("--save-video", dest="save_video",
                               action="store_true", help="Save video of the results")

    args = parser.parse_args()
    assert os.path.isfile(args.estimated_flow)
    assert os.path.isfile(args.events_sequence)
    if args.gt_flow is not None:
        assert os.path.isfile(args.gt_flow)

    if args.save_graphs or args.save_npz or args.save_video:
        if args.output_directory is None or args.output_directory == "":
            raise ValueError(
                "Error: output_directory is not defined. Please specify an output directory using --output_directory")
        if os.path.exists(args.output_directory):
            raise ValueError("Error: output directory already exists.")

    return args


def main():
    args = parse_args()
    flow_est_h5 = h5py.File(args.estimated_flow, "r")
    check_h5py_flow_consistency(flow_est_h5)

    ev_sequence_ext = os.path.splitext(args.events_sequence)[1]
    if ev_sequence_ext == ".dat":
        ev_reader = EventDatReader(event_file=args.events_sequence)
    elif ev_sequence_ext == ".raw":
        ev_reader = RawReader(record_base=args.events_sequence)
    else:
        raise NotImplemented("Only .dat and .raw are supported")

    if args.output_directory:
        os.makedirs(args.output_directory)

    if args.gt_flow is None:
        res = compute_flow_kpi_without_gt(flow_est_h5, ev_reader,
                                          output_directory=args.output_directory, save_graphs=args.save_graphs,
                                          save_npz=args.save_npz, save_video=args.save_video)
        print(res)
        if args.output_directory:
            with open(os.path.join(args.output_directory, "kpis_without_gt.json"), "w") as outfile:
                json.dump(res, outfile)
            return
    flow_gt_h5 = h5py.File(args.gt_flow, "r")
    check_h5py_flow_consistency(flow_gt_h5)

    res = compute_flow_kpi_with_gt(flow_est_h5, flow_gt_h5, ev_reader, normal_flow=args.normal_flow,
                                   output_directory=args.output_directory, save_graphs=args.save_graphs,
                                   save_npz=args.save_npz, save_video=args.save_video)
    print(res)
    if args.output_directory:
        with open(os.path.join(args.output_directory, "kpis.json"), "w") as outfile:
            json.dump(res, outfile)


if __name__ == "__main__":
    main()
