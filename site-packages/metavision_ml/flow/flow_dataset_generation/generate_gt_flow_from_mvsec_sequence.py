# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import os
import h5py
import numpy as np
from metavision_sdk_base import EventCD
from metavision_core.event_io.dat_tools import DatWriter


def generate_gt_flow_from_mvsec(filename_data_input, filename_gt_input,
                                filename_dat_output, filename_gt_output,
                                first_frame=0, nb_frames=250,
                                mask_hood=True):
    """
    Converts the groundtruth optical flow from MVSEC format to Prophesee format

    Args:
        filename_data_input (str): input HDF5 data, contains events and frames in MVSEC format
        filename_gt_input (str): input HDF5 data, contains groundtruth flows in MVSEC format
        filename_dat_output (str): output events sequences. Contains data in Prophesee format
        filename_gt_output (str): output of optical flows. Contains data in Prophesee format
        first_frame (int): index of first frame to consider
        nb_frames (int): number of frames to consider
        mask_hood (boolean): in MVSEC, reflections happen on the hood of the car. When mask_hood is set to True,
                             we set optical flow to np.inf on all pixels below the 190th row.
    """
    assert os.path.isfile(filename_data_input)
    assert os.path.isfile(filename_gt_input)
    assert not os.path.exists(filename_dat_output)
    assert not os.path.exists(filename_gt_output)

    data = h5py.File("outdoor_day1_data.hdf5", "r")
    gt = h5py.File("outdoor_day1_gt.hdf5", "r")

    N, C, H, W = gt["davis"]["left"]["flow_dist"].shape
    assert C == 2
    assert H == 260
    assert W == 346

    assert first_frame + nb_frames <= N

    first_ts = gt["davis"]["left"]["flow_dist_ts"][first_frame]
    end_ts = gt["davis"]["left"]["flow_dist_ts"][first_frame + nb_frames]

    start_ev_idx = np.searchsorted(data["davis"]["left"]["events"][:, 2], first_ts, side='right')
    end_ev_idx = np.searchsorted(data["davis"]["left"]["events"][:, 2], end_ts, side='left')

    events = data["davis"]["left"]["events"][start_ev_idx:end_ev_idx]
    events_x = events[:, 0].astype(np.int)
    assert (events_x >= 0).all()
    assert (events_x < W).all()
    events_y = events[:, 1].astype(np.int)
    assert (events_y >= 0).all()
    assert (events_y < H).all()
    events_t = ((events[:, 2] - first_ts) * 1e6).astype(np.int)
    assert (events_t >= 0).all()
    events_p = ((events[:, 3] + 1) / 2).astype(np.int)
    assert (events_p == 0).sum() + (events_p == 1).sum() == events_p.size
    events_cd = np.empty(events_x.size, dtype=EventCD)
    events_cd["x"] = events_x
    events_cd["y"] = events_y
    events_cd["t"] = events_t
    events_cd["p"] = events_p

    flows = gt["davis"]["left"]["flow_dist"][first_frame:first_frame+nb_frames]
    flows_start_ts = gt["davis"]["left"]["flow_dist_ts"][first_frame:first_frame+nb_frames]
    flows_start_ts = ((flows_start_ts - first_ts) * 1e6).astype(np.int)
    flows_end_ts = gt["davis"]["left"]["flow_dist_ts"][first_frame+1:first_frame+nb_frames+1]
    flows_end_ts = ((flows_end_ts - first_ts) * 1e6).astype(np.int)

    if mask_hood:
        # lower part of images have invalid flow due to reflections on the hood
        flows[:, :, 190:, :] = np.inf

    writer = DatWriter(filename=filename_dat_output, height=H, width=W)
    writer.write(events_cd)
    writer.close()

    hf = h5py.File(filename_gt_output, "w")
    hf.create_dataset("flow_start_ts", data=flows_start_ts, compression="gzip")
    hf.create_dataset("flow_end_ts", data=flows_end_ts, compression="gzip")
    hf.create_dataset("flow", data=flows.astype(np.float32), compression="gzip")
    hf["flow"].attrs["event_input_height"] = H
    hf["flow"].attrs["event_input_width"] = W
    hf.close()


if __name__ == "__main__":
    import fire
    fire.Fire(generate_gt_flow_from_mvsec)
