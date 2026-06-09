# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_expes(names, paths, use_gt=False, output_directory=None):
    assert len(names) == len(paths)
    assert len(names) > 0
    res_np_list = [np.load(path) for path in paths]
    for res_np in res_np_list:
        assert "FWL" in res_np.files
        assert "flow_start_ts" in res_np.files
        assert "flow_end_ts" in res_np.files
        if use_gt:
            assert "FWL_gt" in res_np.files
            assert "AEE" in res_np.files
            assert "AEErel" in res_np.files
            assert "FE" in res_np.files
            assert "APEE" in res_np.files
            assert "AAE" in res_np.files

    plt.figure(figsize=[12, 12])
    for i in range(len(names)):
        res_np = res_np_list[i]
        plt.plot(res_np["flow_start_ts"], res_np["FWL"], label="FWL est {}".format(names[i]))
    if use_gt:
        plt.plot(res_np_list[0]["flow_start_ts"], res_np_list[0]["FWL_gt"], label="FWL groundtruth")
    plt.axhline(1, c="r", ls="--")
    plt.title("Evolution of Flow Warp Loss (FWL) over time")
    plt.xlabel("time")
    plt.ylabel("FWL")
    plt.legend()
    if not output_directory:
        plt.show()
    else:
        fig_filename = os.path.join(output_directory, "compare_FWL.png")
        assert not os.path.isfile(fig_filename)
        plt.savefig(fig_filename)

    if use_gt:
        fig, axes_array = plt.subplots(nrows=4, ncols=1, figsize=[16, 16])
        for i in range(len(names)):
            res_np = res_np_list[i]
            axes_array[0].plot(res_np["flow_start_ts"], res_np["AEE"], label="AEE {}".format(names[i]))
            axes_array[1].plot(res_np["flow_start_ts"], res_np["AEErel"], label="AEErel {}".format(names[i]))
            axes_array[2].plot(res_np["flow_start_ts"], res_np["FE"], label="FE {}".format(names[i]))
            axes_array[3].plot(res_np["flow_start_ts"], res_np["AAE"], label="AEE {}".format(names[i]))
        axes_array[0].set_title("Average Endpoint Error (AEE)")
        axes_array[1].set_title("Average Endpoint Error Relative to GT (AEErel)")
        axes_array[2].set_title("Flow Error (FE)")
        axes_array[3].set_title("Average Angular Error (AAE)")
        plt.legend()
        if not output_directory:
            plt.show()
        else:
            fig_filename = os.path.join(output_directory, "compare_AEE_AAE.png")
            assert not os.path.isfile(fig_filename)
            plt.savefig(fig_filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot flow KPI curves to compare several models")
    parser.add_argument("--expes", metavar="NAME:PATH_TO_JSON", nargs='+',
                        help="Set of key-value pairs.")
    parser.add_argument("--use-gt", dest="use_gt", action="store_true",
                        help="Use GT (otherwise only use unsupervised FWL)")
    parser.add_argument("--output-directory", dest="output_directory", help="Output directory to save figures")
    args = parser.parse_args()

    if args.output_directory and not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    expe_names = []
    expe_paths = []
    for expe in args.expes:
        assert expe.count(":") == 1, "Invalid key/value pair. It should contain one and only one ':'  {}".format(expe)
        name, path = expe.split(":")
        assert os.path.isfile(path)
        assert os.path.splitext(path)[1].lower() == ".npz", "Invalid expe path (should be .npz): {}".format(path)
        expe_names.append(name)
        expe_paths.append(path)
    args.expe_names = expe_names
    args.expe_paths = expe_paths
    return args


def main():
    args = parse_args()
    plot_expes(names=args.expe_names, paths=args.expe_paths,
               use_gt=args.use_gt, output_directory=args.output_directory)


if __name__ == "__main__":
    main()
