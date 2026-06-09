# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
A collection of visualization utilities for flow.

"""
from __future__ import print_function
from __future__ import division

import numpy as np
import cv2


def convert_to_gray(img):
    """"Converts the input RGB uint8 image to gray scale.

    Args:
        img (np.ndarray): 3-channels (RGB) or 1-channel (grayscale) uint8 image
    """
    # img should be converted to graylevel with three channels
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img.squeeze()
    img = np.dstack([img_gray, img_gray, img_gray])
    return img


def draw_dense_flow(img, flow, mask):
    """Draws Mixed of image converted to gray and
    the flow in HSV color space.
    Args:
        img (np.ndarray): img (RGB or Gray)
        flow (np.ndarray): u,v flow map
        mask (np.ndarray): mask to display flow or not.
    """
    if len(img.shape) == 3:
        img = convert_to_gray(img)
    else:
        img = img[..., None].repeat(3, 2)

    flow_y = flow[1]
    flow_x = flow[0]

    if mask is not None:
        flow_y[~mask] = 0
        flow_x[~mask] = 0

    hsv = np.zeros(img.shape, dtype=np.uint8)
    hsv[..., 1] = 255

    # We convert degrees from [0, 2*pi] in [0, 179] because
    # Hue's Range is [0, 179] in Opencv
    mag, ang = cv2.cartToPolar(flow_x, flow_y)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    out = cv2.addWeighted(img, 0.3, bgr, 0.7, 0.0)
    return out


def draw_arrows(img, flow, step=16, threshold_px=1, convert_img_to_gray=True, mask=None, thickness=1):
    """
    Visualizes Flow, by drawing hsv-colored arrows on top of the input image.

    Args:
        img (np.ndarray): RGB uint8 image, to draw the arrows on.
        flow (np.ndarray): of shape (2, height width), where the first index is the x component of the flow and the second is the y-component. The flow is in pixel units.
        step (int): Draws every `step` arrow. use to increase clarity, especially with fast motions.
        threshold_px (float): doesn't display arrows shorter the *threshold_px* pixels.
        mask (np.ndarray): boolean tensor of shape (H, W) indicating where flow arrows should be drawn or not.
    """
    assert step > 0, f"step must be a strictly positive integer, got {step}"
    step = int(step)
    c, height, width = flow.shape
    height_viz = int(np.ceil(height / step))
    width_viz = int(np.ceil(width / step))

    flow_y = flow[1, ::step, ::step]
    flow_x = flow[0, ::step, ::step]

    if mask is not None:
        mask = mask[::step, ::step]
        flow_y[~mask] = 0
        flow_x[~mask] = 0

    if convert_img_to_gray or len(img.shape) < 3 or img.shape[2] == 1:
        img = convert_to_gray(img)

    # We convert degrees from [0, 2*pi] in [0, 179] because
    # Hue's Range is [0, 179] in Opencv
    mag, ang = cv2.cartToPolar(flow_x, flow_y)
    ang = ang * 180 / np.pi / 2

    hsvImg = np.ones((height_viz, width_viz, 3), dtype=np.uint8) * 255
    hsvImg[..., 0] = ang
    rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB).astype('int')

    x = np.arange(0, width, step)
    y = np.arange(0, height, step)
    x_array, y_array = np.meshgrid(x, y)

    # arrow displacement
    threshold = (flow_x**2 + flow_y**2) > threshold_px

    # computes arrows ending point
    p2x = (x_array + flow_x).astype("int")
    p2y = (y_array + flow_y).astype("int")

    # displaying arrows
    for i in range(0, height // step):
        for j in range(0, width // step):
            color_list = rgbImg[i, j, :].tolist()
            if threshold[i, j]:
                img = cv2.arrowedLine(img, (x_array[i, j], y_array[i, j]),
                                      (p2x[i, j], p2y[i, j]), color_list, thickness=thickness)

    return img


def get_dense_flow(flow, base_img=None, mask=None):
    """
    Creates a flow visualization using dense image
    Optionally a RGB uint8 image can be passed as canvas to draw on.

    Args:
        flow (torch.tensor): tensor of shape (2, H, W) where the first index is the x component of the flow and the second is the y-component.
        base_img (np.ndarray): if not None, the flow arrows will be drawn on it. A visualization of the events or some features is usually a good idea. Prefer gray level for clarity.
        step (int): Draws every `step` arrow. Use to increase clarity, especially with fast motions.
        mask (np.ndarray): boolean tensor of shape (H, W) indicating where flow arrows should be drawn or not.
    """
    height, width = flow.size(1), flow.size(2)
    if base_img is None:
        base_img = np.zeros((height, width, 3)).astype(np.uint8)
    flow_viz = flow.cpu().numpy().astype(np.float64)
    img = draw_dense_flow(base_img, flow_viz, mask=mask)
    return img


def get_arrows(flow, base_img=None, step=8, mask=None):
    """
    Creates a flow visualization using colored arrows.

    Optionally a RGB uint8 image can be passed as canvas to draw on.

    Args:
        flow (torch.tensor): tensor of shape (2, H, W) where the first index is the x component of the flow and the second is the y-component.
        base_img (np.ndarray): if not None, the flow arrows will be drawn on it. A visualization of the events or some features is usually a good idea. Prefer gray level for clarity.
        step (int): Draws every `step` arrow. Use to increase clarity, especially with fast motions.
        mask (np.ndarray): boolean tensor of shape (H, W) indicating where flow arrows should be drawn or not.
    """
    # flow is an output of a network (2,H,W), on gpu
    height, width = flow.size(1), flow.size(2)
    if base_img is None:
        base_img = np.zeros((height, width, 3)).astype(np.uint8)
    flow_viz = flow.cpu().numpy().astype(np.float64)
    img = draw_arrows(base_img, flow_viz, step=step, mask=mask)
    return img


def draw_flow_on_grid(input_tensor, flows, grid, make_img_fun, scale=-1, step=8, mask_by_input=True, draw_dense=False):
    """Applies Flow drawing function to a batch of inputs.

    Inputs are going to be visualized as a sequence of 2d grids, on top of which flow
    arrows will be drawn. The feature visualization is on grayscale, whereas the arrows are hsv-colored
    depending on their orientation.

    Args:
        input_tensor (numpy ndarray): tensor of shape (num_time_bins, batchsize, channel, height, width).
        flows (torch.tensor list): list of flow tensors. The position in the list indicates the resolution
            in increasing order, each tensor is of shape (num_time_bins, batchsize, 2, height, width)
            i.e. it corresponds with the features.
        grid (np.ndarray): array of shape (num_time_bins, m * height,n * width,3) where m*n is superior to batchsize.
            This array is used to draw a sequence of batches as a RGB video.
        make_img_fun (function): visualization function corresponding to the feature type.
        scale (int): index of the flow scale to use (when regressed by an hour glass network like unet
            several resolutions of flows might be available).
        step (int): Draws every `step` arrow. Use to increase clarity, especially with fast motions.
        mask_by_input (boolean): if True only display flow arrows on pixel with non null input.
        draw_dense (boolean): if True display dense flow map
    """
    height, width = input_tensor.shape[-2:]
    ncols = grid.shape[2] // width
    mask = None
    if mask_by_input:
        masks = np.sum(np.abs(input_tensor), axis=2) > 0
    for t in range(len(input_tensor)):
        for i in range(len(input_tensor[t])):
            y = i // ncols
            x = i % ncols
            img = make_img_fun(input_tensor[t, i])
            mask = masks[t, i] if mask_by_input else None
            if draw_dense:
                grid[t, y * height:(y + 1) * height, x * width:(x + 1) * width] = get_dense_flow(
                    flows[scale][t, i], base_img=img, mask=mask)
            else:
                grid[t, y * height:(y + 1) * height, x * width:(x + 1) * width] = get_arrows(
                    flows[scale][t, i], base_img=img, step=step, mask=mask)
