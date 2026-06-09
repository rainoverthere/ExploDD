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
from metavision_sdk_base import EventCD
import numpy as np


def compute_iwe(events_np, flow, t, duration_flow=None, assert_target_time_strict=True):
    """
    Computes an image of warped events (IWE) given a numpy array of events

    Args:
        events_np (np.array): numpy array of EventCD
        flow (torch.Tensor): size (2, H, W) expressed in pixels
        t (int): timestamp to warp to
        duration_flow (int): duration of the chunk of events over which the flow is computed
        assert_target_time_strict (boolean): if True, the target timestamp t must be in the range covered by events_np
    """
    assert type(flow) == torch.Tensor
    assert events_np.dtype == EventCD
    C, H, W = flow.shape
    assert C == 2
    if assert_target_time_strict:
        assert t >= events_np[0]["t"]
        assert t <= events_np[-1]["t"]
    assert (events_np["x"] < W).all()
    assert (events_np["y"] < H).all()

    if duration_flow is None:
        duration_flow = events_np["t"][-1] - events_np["t"][0]
    assert duration_flow > 0
    xt = torch.from_numpy(events_np["x"].astype(np.float32)).to(flow.device)
    yt = torch.from_numpy(events_np["y"].astype(np.float32)).to(flow.device)
    tt = torch.from_numpy(events_np["t"].astype(np.float32)).to(flow.device)
    pt = torch.from_numpy(events_np["p"].astype(np.float32)).to(flow.device) * 2 - 1
    return compute_iwe_torch(xt, yt, tt, pt, flow, t, duration_flow)


def compute_iwe_torch(xt, yt, tt, pt, flow, t, duration_flow):
    """
    Compute an image of warped events (IWE) given torch tensors of events

    Args:
        xt (torch.Tensor): x (n,)
        yt (torch.Tensor): y (n,)
        tt (torch.Tensor): t (n,)
        pt (torch.Tensor): p (n,)
        flow (torch.Tensor): size (2, H, W) expressed in pixels
        t (int): timestamp to warp to
        duration_flow (int): duration of the chunk of events over which the flow is computed
    """
    flow_pix_per_us = flow / duration_flow
    C, H, W = flow.shape
    assert C == 2
    x_warp, y_warp = warp_events(xt, yt, tt, flow_pix_per_us, t)
    iwe = event_image(x_warp, y_warp, pt, height=H, width=W, interpolation='bilinear')
    return iwe


def warp_events(x, y, t, flow, t0):
    """Moves Events Directly

    Args:
        x (torch.Tensor): x (n,)
        y (torch.Tensor): y (n,)
        t (torch.Tensor): time (n,)
        flow (torch.Tensor): (2,height,width)
    """
    C, height, width = flow.shape
    assert C == 2
    xf = x.float() / (width - 1) * 2.0 - 1.0
    yf = y.float() / (height - 1) * 2.0 - 1.0
    coords = torch.stack((xf, yf), dim=1)[None, None]

    flow_at_event = F.grid_sample(flow[None], coords, align_corners=True, mode='bilinear').reshape(2, len(xf))
    dt = t0 - t
    xw = x + flow_at_event[0, :] * dt
    yw = y + flow_at_event[1, :] * dt
    return xw, yw


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation

    Args:
        pxs (torch.Tensor): pixel x (n,)
        pys (torch.Tensor): pixel y (n,)
        dxs (torch.Tensor): decimal part in x (n,)
        dys (torch.Tensor): decimal part in y (n,)
        weights (torch.Tensor): values to interpolate (n,)
        img (torch.Tensor): output image is updated (height, width)
    """
    img.index_put_((pys, pxs), weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
    img.index_put_((pys, pxs + 1), weights * dxs * (1.0 - dys), accumulate=True)
    img.index_put_((pys + 1, pxs), weights * (1.0 - dxs) * dys, accumulate=True)
    img.index_put_((pys + 1, pxs + 1), weights * dxs * dys, accumulate=True)
    return img


def event_image(xs, ys, ps, height, width, interpolation='bilinear'):
    """
    Differentiable Image creation from events

    Args:
        xs (torch.Tensor): x values (n,)
        ys (torch.Tensor): y values (n,)
        ps (torch.Tensor): polarities (or wheights) to apply
        height (int): height of the image
        width (int): width of the image
        interpolation (string): either 'bilinear' or 'nearest'
    """
    assert interpolation in ['bilinear', 'nearest']
    device = xs.device
    img_size = (height + 2, width + 2)

    mask = torch.ones(xs.size(), device=device)
    zero_v = torch.tensor([0.], device=device)
    ones_v = torch.tensor([1.], device=device)
    mask *= torch.where(xs >= width, zero_v, ones_v) * torch.where(xs <= -1, zero_v,
                                                                   ones_v) * torch.where(ys >= height, zero_v, ones_v) * torch.where(ys <= -1, zero_v, ones_v)
    mask = mask.to(device)

    masked_ps = ps * mask

    img = torch.zeros(img_size).to(device)
    if interpolation == 'bilinear':
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs - pxs).float()
        dys = (ys - pys).float()
        pxs = (pxs * mask).long()
        pys = (pys * mask).long()
        interpolate_to_image(pxs + 1, pys + 1, dxs, dys, masked_ps, img)
    else:
        assert interpolation == 'nearest'
        pxs = (xs.round() * mask).long()
        pys = (ys.round() * mask).long()
        img.index_put_((pys + 1, pxs + 1), masked_ps, accumulate=True)

    return img[1:height+1, 1:width+1]
