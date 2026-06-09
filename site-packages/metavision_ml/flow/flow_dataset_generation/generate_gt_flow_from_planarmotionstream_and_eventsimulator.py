# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import argparse
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import h5py

from metavision_core_ml.data.image_planar_motion_stream import PlanarMotionStream
from metavision_sdk_base import EventCD
from metavision_ml.video_to_event.simulator import EventSimulator
from metavision_core.event_io.dat_tools import DatWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a sequence with GT flow from an image")
    parser.add_argument("--input_image_filename", required=True, help="Input image")
    parser.add_argument("--tbins", default=1000, help="Number of time bins to generate with PlanarMotionStream")
    parser.add_argument("--group_bins", default=10, help="Compute GT flow for this number of bins")
    parser.add_argument("--C", default=0.2, help="Simulator contrast threshold")
    parser.add_argument("--refractory_period", default=1)
    parser.add_argument("--cutoff_hz", default=0)
    parser.add_argument("--no_display", dest="display", action="store_false")
    parser.add_argument("--output_h5_filename", required=True, help="Output HDF5 filename for ground truth flow")
    parser.add_argument("--output_events_filename", required=True, help="Output dat filename for events")

    args = parser.parse_args()
    assert os.path.isfile(args.input_image_filename)
    assert not os.path.exists(args.output_h5_filename)
    assert not os.path.exists(args.output_events_filename)
    assert args.tbins % args.group_bins == 0
    return args


def main():
    args = parse_args()
    path = args.input_image_filename

    C = args.C
    refractory_period = args.refractory_period
    cutoff_hz = args.cutoff_hz
    num_tbins = args.tbins

    img = cv2.imread(path)
    height, width, _ = img.shape

    img_gray = img.copy()

    image_stream = PlanarMotionStream(path, height, width)
    simu = EventSimulator(height, width, C, C, refractory_period)

    all_events = []
    times = []
    indices = []
    images = []

    for i, (img, ts) in enumerate(image_stream):
        total = simu.image_callback(img, ts)
        events = simu.get_events()
        if len(events) == 0:
            print('skip')
            continue
        simu.flush_events()
        all_events.append(events.copy())
        dt = events['t'][-1] - events['t'][0]
        events['t'] -= events['t'][0]
        times.append(image_stream.cam.time-1)
        indices.append(i)
        images.append(img.copy())
        if args.display:
            cv2.imshow("im", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(len(all_events))
        if len(all_events) == num_tbins:
            break

    if args.display:
        cv2.destroyAllWindows()

    larger_chunk_size = args.group_bins

    all_events_lc = []
    times_lc = []
    indices_lc = []
    images_lc = []
    flow_start_ts_list = []
    flow_end_ts_list = []
    for i in range(num_tbins//larger_chunk_size):
        events_current_chunk = np.concatenate(all_events[i*larger_chunk_size: (i+1)*larger_chunk_size])
        all_events_lc.append(events_current_chunk)
        times_lc.append(times[i*larger_chunk_size])
        indices_lc.append(indices[i*larger_chunk_size])
        images_lc.append(images[i*larger_chunk_size].copy())
        start_ts_i = events_current_chunk["t"][0]
        end_ts_i = events_current_chunk["t"][-1]
        flow_start_ts_list.append(start_ts_i)
        flow_end_ts_list.append(end_ts_i)
        print(
            "chunk {}: {} -> {}  duration: {}".format(
                i, events_current_chunk["t"][0],
                events_current_chunk["t"][-1],
                events_current_chunk["t"][-1] - events_current_chunk["t"][0] + 1))

    print("variable delta_t: ", [(all_events_lc[i]["t"][-1] - all_events_lc[i]["t"][0])
                                 for i in range(len(all_events_lc))])

    print(len(images_lc))

    fflows = []
    bflows = []
    for i in range(len(indices_lc) - 1):
        rvec1, tvec1 = image_stream.cam.rvecs[indices_lc[i]], image_stream.cam.tvecs[indices_lc[i]]
        rvec2, tvec2 = image_stream.cam.rvecs[indices_lc[i+1]], image_stream.cam.tvecs[indices_lc[i+1]]

        fflow_i = image_stream.cam.get_flow(rvec1, tvec1, rvec2, tvec2, height,
                                            width, infinite=True).astype(np.float32)
        fflows.append(fflow_i[None])

        bflow_i = image_stream.cam.get_flow(rvec2, tvec2, rvec1, tvec1, height,
                                            width, infinite=True).astype(np.float32)
        bflows.append(bflow_i[None])

    fflows = np.concatenate(fflows)
    fflows_th = torch.from_numpy(fflows).permute(0, 3, 1, 2)

    bflows = np.concatenate(bflows)
    bflows_th = torch.from_numpy(bflows).permute(0, 3, 1, 2)

    assert (len(indices_lc)-1, 2, height, width) == fflows_th.shape
    assert fflows_th.shape == bflows_th.shape

    writer = DatWriter(filename=args.output_events_filename, height=height, width=width)
    for ev in all_events_lc[:-1]:
        writer.write(ev)
    writer.close()
    del all_events_lc
    del all_events

    flow_start_ts = np.array(flow_start_ts_list[:-1])
    flow_end_ts = np.array(flow_end_ts_list[:-1])

    flow_start_ts[0] = 0

    hf = h5py.File(args.output_h5_filename, "w")
    hf.create_dataset("flow_start_ts", data=flow_start_ts, compression="gzip")
    hf.create_dataset("flow_end_ts", data=flow_end_ts, compression="gzip")
    hf.create_dataset("flow", data=fflows_th.numpy().astype(np.float32), compression="gzip")
    hf["flow"].attrs["event_input_height"] = height
    hf["flow"].attrs["event_input_width"] = width
    hf.close()


if __name__ == "__main__":
    main()
