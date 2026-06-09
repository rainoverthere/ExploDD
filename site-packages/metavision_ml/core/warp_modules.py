# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Layers of Neural network involving a displacement field, or interpolation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from metavision_core_ml.core.temporal_modules import seq_wise


def normalize_flow(flow):
    """
    Normalizes the flow in pixel unit
    to a flow in [-2,2] for its use grid_sample.

    Args:
        flow: tensor of shape (B,2,H,W)
    """
    height, width = flow.shape[-2:]
    height -= 1
    width -= 1
    factor = torch.tensor([2.0 / width, 2.0 / height]).to(flow)
    flow = flow * factor[None, :, None, None]
    flow = flow.clamp_(-2.0, 2.0)
    return flow


def make_grid2d(height, width, device='cpu:0'):
    """
    Generates a 2d Grid

    Args:
        height: image height
        width: image width
    """
    lin_y = torch.linspace(-1., 1., height, device=device)
    lin_x = torch.linspace(-1., 1., width, device=device)
    grid_h, grid_w = torch.meshgrid([lin_y, lin_x])
    grid = torch.cat((grid_w[None, :, :, None], grid_h[None, :, :, None]), 3)
    return grid


def warp_backward_using_forward_flow(
        img,
        flow,
        grid=None,
        align_corners=True,
        mode="bilinear",
        flow_is_normalized=False):
    """
    Generates previous images given a set of next images and forward flow
    in pixels or in [-2,2] if flow_is_normalized is set to True.

    Args:
        img (torch.Tensor): size (batch_size, channels, height, width)
        flow (torch.Tensor): size (batch_size, 2, height, width)
        grid (torch.Tensor): normalized meshgrid coordinates.
        align_corners (bool): See Pytorch Documentation for the grid_sample function
        mode (string): mode of interpolation used
        flow_is_normalized (bool): flow is in [-2,2], no call to normalize_flow
    """
    height, width = flow.shape[-2:]
    if grid is None:
        grid = make_grid2d(height, width).to(img)
    if not flow_is_normalized:
        flow = normalize_flow(flow)
    coords = grid + flow.permute(0, 2, 3, 1).contiguous()
    warps = F.grid_sample(img, coords, align_corners=align_corners, mode=mode)
    return warps


def warp_forward_using_backward_flow(
        img,
        flow,
        grid=None,
        align_corners=True,
        mode="bilinear",
        flow_is_normalized=False):
    """
    Generates next images given a set of previous images and backward flow
    in pixels or in [-2,2] if flow_is_normalized is set to True.

    Note: forward warping using the backward flow is the same computation as backward warping using the forward flow

    Args:
        img (torch.Tensor): size (batch_size, channels, height, width)
        flow (torch.Tensor): size (batch_size, 2, height, width)
        grid (torch.Tensor): normalized meshgrid coordinates.
        align_corners (bool): See Pytorch Documentation for the grid_sample function
        mode (string): mode of interpolation used
        flow_is_normalized (bool): flow is in [-2,2], no call to normalize_flow
    """
    return warp_backward_using_forward_flow(img=img, flow=flow, grid=grid, align_corners=align_corners,
                                            mode=mode, flow_is_normalized=flow_is_normalized)


def warp_forward_using_forward_flow(
        img,
        flow,
        grid=None,
        align_corners=True,
        mode="bilinear",
        flow_is_normalized=False):
    """
    Generates next images given a set of images and forward flow
    in pixels or in [-2,2] if flow_is_normalized is set to True.

    Args:
        img (torch.Tensor): size (batch_size, channels, height, width)
        flow (torch.Tensor): size (batch_size, 2, height, width)
        grid (torch.Tensor): normalized meshgrid coordinates.
        align_corners (bool): See Pytorch Documentation for the grid_sample function
        mode (string): mode of interpolation used
        flow_is_normalized (bool): flow is in [-2,2], no call to normalize_flow
    """
    height, width = flow.shape[-2:]
    if grid is None:
        grid = make_grid2d(height, width).to(img)
    if not flow_is_normalized:
        flow = normalize_flow(flow)
    coords = grid - flow.permute(0, 2, 3, 1).contiguous()
    warps = F.grid_sample(img, coords, align_corners=align_corners, mode=mode)
    return warps


def warp_to_tbin(tensor, flow, tbin, grid=None, align_corners=True, mode='bilinear', flow_is_normalized=False):
    """
    Applies flow to an input tensor to warp all time bins onto one.
    This has the effect of reducing apparent motion blur (providing the flow is correct !).
    Here we suppose a Constant Flow during num_tbins steps per sample.

    Args:
        inputs (torch.Tensor): size (num_tbins, batch_size, channels, height, width).
        flow (torch.Tensor): size (batch_size, 2, height, width) normalized in [-2,2]
        tbin (int): index of the time bin to warp to (from zero to num_tbins -1).
        grid (torch.Tensor): normalized meshgrid coordinates.
        align_corners (bool): See Pytorch Documentation for the grid_sample function
        mode (string): mode of interpolation used.
        flow_is_normalized (bool): flow is in [-2,2], no call to normalize_flow
    """
    num_tbins, batch_size, channels, height, width = tensor.shape
    times_to_target = (tbin - torch.arange(num_tbins, dtype=tensor.dtype, device=flow.device))
    flows = flow * times_to_target[:, None, None, None, None]  # T,B,C,H,W
    flows = flows.view(batch_size * num_tbins, 2, height, width)
    inputs = tensor.view(batch_size * num_tbins, channels, height, width)
    warps = warp_forward_using_forward_flow(inputs, flows, grid, align_corners, mode, flow_is_normalized)
    return warps.view(num_tbins, batch_size, channels, height, width)


def warp_to_micro_tbin(
        inputs,
        flow,
        micro_tbin,
        grid=None,
        align_corners=True,
        mode='bilinear',
        flow_is_normalized=False):
    """
    similar to "warp_to_tbin" but here we consider a sequence of
    spatio-temporal volumes with the shape [T,B,C,D,H,W]
    Here we suppose a Constant Flow per time bin (that includes multiple micro time bins).

    Args:
        inputs (torch.Tensor): size (num_time_bins, batch_size, channels, depth, height, width)
        flow (torch.Tensor): size (num_tbins, batch_size, 2, height, width) normalized in [-2,2]
        tbin (int): index of the time bin to warp to (from zero to num_tbins -1)
        grid (torch.Tensor): normalized meshgrid coordinates.
        align_corners (bool): See Pytorch Documentation for the grid_sample function
        mode (string): mode of interpolation used.
        flow_is_normalized (bool): flow is in [-2,2], no call to normalize_flow
    """
    num_tbins, batch_size, num_micro_tbins, channels, height, width = inputs.shape
    x = inputs.reshape(num_tbins * batch_size, num_micro_tbins, channels, height, width)
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    warps = warp_to_tbin(x, flow / num_micro_tbins, micro_tbin, grid, align_corners, mode, flow_is_normalized)
    warps = warps.permute(1, 0, 2, 3, 4).contiguous()
    warps = warps.reshape(num_tbins, batch_size, num_micro_tbins, channels, height, width)
    return warps


class Warping(object):
    """
    Differentiable warping module using bilinear sampling.
    This calls the internal functions above while storing a grid.
    When the resolution changes, the grid is reallocated.
    This modules handles sequences (T,B,C,H,W tensors).

    Attributes:
        grid (torch.FloatTensor): grid used for interpolation, saved to avoid reallocations.

    Args:
        mode (string): interpolation mode (can be bilinear or nearest).
    """

    def __init__(self, mode='bilinear'):
        super(Warping, self).__init__()
        self.mode = mode
        self.grid = None
        assert mode in ("bilinear", 'nearest'), "mode can be bilinear or nearest"

    def warp_sequence_by_one_time_bin(self, img, flow, align_corners=True, flow_is_normalized=False):
        """
        Uses the displacement to warp the image.

        If the displacement doesn't match the existing grid attributes, the grid is regenerated.

        Args :
            img (torch.Tensor): (num_tbins, batch_size, channel_count, height, width) tensor
            flow (torch.tensor): (num_tbins x batch_size, 2, height, width) flow in pixels per time bin
            align_corners (bool): See Pytorch Documentation for the grid_sample function
            flow_is_normalized (bool): flow is in [-2,2], no call to normalize_flow
        """
        height, width = img.shape[-2:]
        if self.grid is None:
            self.grid = make_grid2d(height, width).to(img)
        warp = seq_wise(warp_forward_using_forward_flow)(
            img, flow, self.grid, align_corners, self.mode, flow_is_normalized)
        return warp

    def sharpen_micro_tbin(
            self,
            img,
            flow,
            micro_tbin,
            is_on_off_volume=True,
            align_corners=True,
            flow_is_normalized=False):
        """
        Applies flow to an input tensor with on/off channels to warp all micro bins onto one.

        This has the effect of reducing apparent motion blur (providing the flow is correct !).
        Contrary to the function `sharpen` this is only applicable to input features that have micro time bins : id est
        Channels that are computed by slice of time within a delta_t (for instance event_cube).

        Args:
            img (torch.Tensor): size (num_time_bins, batch_size, channel_count, height, width)
            flow: size (num_tbins x batch_size, 2, height, width) in pixels/bin
            micro_tbin (int): index of the time bin to warp to (from zero to num_tbins -1)
            is_on_off_volume (bool): if input channels are organized into 2 groups of \
            "off" and "one" channels.
            This happens when you use set split_polarity option to True in a preprocessing (see for instance event_cube)
            align_corners (bool): See Pytorch Documentation for the grid_sample function
            flow_is_normalized (bool): flow is in [-2,2], no call to normalize_flow
        """
        num_tbins, batch_size, channels, height, width = img.shape
        if is_on_off_volume:
            assert channels % 2 == 0
            num_micro_tbins = channels // 2
        else:
            num_micro_tbins = channels
        tensor = img.reshape(num_tbins, batch_size, num_micro_tbins, channels // num_micro_tbins, height, width)
        warp = warp_to_micro_tbin(
            tensor,
            flow,
            micro_tbin,
            self.grid,
            align_corners,
            self.mode,
            flow_is_normalized)
        warp = warp.reshape(num_tbins, batch_size, channels, height, width)
        return warp
