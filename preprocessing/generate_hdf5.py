# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Script to convert one or multiple RAW or DAT files to HDF5 tensor files
"""
import argparse
import numpy as np
import glob
import os

from metavision_ml.preprocessing import get_preprocess_function_names, get_preprocess_dict
from metavision_ml.preprocessing.hdf5 import generate_hdf5


def parse_args(argv=None, only_default_values=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Convert one or multiple RAW or DAT files to HDF5 tensor files.')

    parser.add_argument('path', nargs="+",
                        help='RAW or DAT filenames. You can use shell wildcards to select more than one file.')
    #程序内部会收到一个包含所有匹配文件的列表作为 path 参数的值

    parser.add_argument('-o', '--output-folder', required=True, help='where the HDF5 is going to be written')
    parser.add_argument('--delta-t', type=int, default=50000,
                        help='duration of timeslice (in us) in which events are accumulated'
                        ' to compute features.') #默认以20Hz的频率提供ev_repr
    parser.add_argument('--start-ts', type=int, default=0, nargs="+",
                        help='timestamp (in us) from which the computation begins. '
                        'Either a single int for all files or exactly one int per input file.')
    parser.add_argument(
        '--max-duration-ms', type=int, default=None,
        help='maximum duration of the HDF5 file in ms. if the input file exceeds this duration multiple'
        'files will be produced.')
    parser.add_argument('-n', '--num-workers', type=int, default=2,
                        help='Number of processes used for precomputation')
    parser.add_argument('--preprocess', default='histo', help='name of the preprocessing function used',
                        choices=get_preprocess_function_names())
    parser.add_argument('--height_width', nargs=2, default=None, type=int,
                        help="if set, downscale the feature tensor to the requested resolution using interpolation"
                        " Possible values are only power of two of the original resolution.")
    parser.add_argument(
        '--box-labels', nargs="*", default=[], type=str, help="Optional box label files for the ground truth "
        "that goes along the input files. if `start_ts` or `max_duration` are specified, these files will"
        " be cut accordingly. You can use shell wildcards to select more than one file.")
    parser.add_argument('--store_as_uint8', action="store_true", help="Use quantization to store the underlying data "
                        "as 8bit integer and therefore save space. This will reduce precision of the features.")

    parser.add_argument('--max_val', default=None, type=float, help="maximum number of increments per pixel")
    #这个值可以设置为None(等价于event_cube处理中令k=1,符合常识,权重和为1。或者把值设置的很大 以迎合爆炸场景,然后不设置上下限)，并且event_cube的split_polarity可以把默认值改为False(C:\Program Files\Prophesee\lib\python3\site-packages\metavision_ml\preprocessing\__init__.py)
    
    parser.add_argument(
        '--neg_bit_len_quantized', default=4, type=int,
        help="Use only for Diff3D and Histo3D: the negative bits length set by the camera")
    parser.add_argument(
        '--total_bit_len_quantized', default=8, type=int,
        help="Use only for Histo3D: total bits length used by the camera")
    parser.add_argument(
        '--normalization_quantized', action="store_true",
        help="Use only for Diff3D and Histo3D: If not used, the preprocess dtype should be set as int8 for Diff3D mode "
        "and uint8 for Histo3D mode. If used, the output will be normalized and the preprocess dtype "
        "should be set as one of the floating types, such as float32.")
    parser.add_argument('--n_events', type=int, default=0,
                        help='Number of events in the timeslice, if "mode" is "n_events".')
    parser.add_argument('--mode', type=str, default="delta_t",
                        help='Load by timeslice or number of events. Either "delta_t" or "n_events".',
                        choices=("delta_t", "n_events"))

    parser.add_argument(
        '--simu_sensor_output_quantized', action="store_true",
        help="Simulate the Diff3D and Histo3D event frames output directly from the sensor. "
        "In this mode, event frames are saved in integer values and in original resolution. Thus, "
        "parameter --height_width should not be set")

    return parser.parse_args(argv) if argv is not None else parser.parse_args()


if __name__ == '__main__':

    ARGS = parse_args()

    expanded_files = []
    for pattern in ARGS.path:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"Warning: No files matched pattern: {pattern}")
            raise RuntimeError
        expanded_files.extend(expanded)

    [height, width] = ARGS.height_width if ARGS.height_width is not None else [None, None]

    if ARGS.simu_sensor_output_quantized: #默认不执行
        assert height is None and width is None, "Downsampling is not allowed if we simulate sensor ouput"
        assert not ARGS.normalization_quantized, "Normalization is not allowed if we simulate sensor output"

    preprocess_kwargs = get_preprocess_dict(ARGS.preprocess)['kwargs']
    '''
    {'cin': 2, 'events_to_tensor': histo,
              'kwargs': {'preprocess_dtype': np.dtype('float32'), 'max_incr_per_pixel': 5},
              "viz": viz_histo_binarized}

    {'cin': 10, 'events_to_tensor': event_cube,
                         'kwargs': {'preprocess_dtype': np.dtype('float32'), 'split_polarity': True,
                                    'max_incr_per_pixel': 255./4},
                         "viz": viz_event_cube_rgb}
    'event_cube_paper': {'cin': 10, 'events_to_tensor': event_cube,
                         'kwargs': {'preprocess_dtype': np.dtype('float32'), 'split_polarity': False, #这里原本为True
                                    'max_incr_per_pixel': None}, #原本为255./4
                         "viz": viz_event_cube_rgb},
    '''
    if "preprocess_dtype" in preprocess_kwargs:
        preprocess_kwargs.pop('preprocess_dtype')
    if ARGS.preprocess == "diff_quantized":
        preprocess_kwargs.update({"negative_bit_length": ARGS.neg_bit_len_quantized,
                                  "normalization": ARGS.normalization_quantized,
                                  "preprocess_dtype": np.int8 if ARGS.simu_sensor_output_quantized else np.float32})
    elif ARGS.preprocess == "histo_quantized":
        preprocess_kwargs.update({"negative_bit_length": ARGS.neg_bit_len_quantized,
                                  "total_bit_length": ARGS.total_bit_len_quantized,
                                  "normalization": ARGS.normalization_quantized,
                                  "preprocess_dtype": np.uint8 if ARGS.simu_sensor_output_quantized else np.float32})
    else:
        preprocess_kwargs.update({"max_incr_per_pixel": ARGS.max_val})#实质上就这一个元素会变，原本默认为5，需要主动设置，现在默认为None

    generate_hdf5(expanded_files, ARGS.output_folder, ARGS.preprocess, ARGS.delta_t,
                  height=height, width=width, start_ts=ARGS.start_ts,
                  max_duration=ARGS.max_duration_ms * 1000 if ARGS.max_duration_ms else None,
                  box_labels=ARGS.box_labels, n_processes=ARGS.num_workers,
                  store_as_uint8=ARGS.store_as_uint8, mode=ARGS.mode, n_events=ARGS.n_events,
                  preprocess_kwargs=preprocess_kwargs)
