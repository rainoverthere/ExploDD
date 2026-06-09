# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Self supervised losses functions used to train a network for optical flow
"""
import math
from collections import defaultdict
from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

from metavision_core_ml.core.temporal_modules import time_to_batch, batch_to_time
from ..core.warp_modules import normalize_flow


LOSS_NAMES = ("data", "smoothness", "smoothness2", "l1", 'time_consistency', "bw_deblur")


class FlowLoss(nn.Module):
    """
    Class computing the loss for flow regression.

    Contains the following loss functions:
        **Task-specific loss functions**
        These loss functions are different formulations of the task that the flow is supposed to fulfill:
        predicting the motion of objects and \"deblurring\" moving edges.
         * `data` this loss function ensures that applying the flow to warp a tensor at time $t$ will match the tensor at time $t+1$
         * `time consistency` this loss function checks that the flow computed at timestamp $t_i$ is also correct at time $t_{i+1}$
            as most motions are consistent over time. This assumption doesn't hold for fast moving objects.
         * `bw deblur` this loss function is applied backwards to avoid the degenerate solution of a flow warping all tensors
            into one single point or away from the frame (such a flow would have a really high loss when applied
            backward). We call this \"deblurring loss\" as it allows us to warp several time channels to a single
            point and obtain an image that is sharper than the original (lower variance).

        **Regularization loss functions**
         * `smoothness` this loss function is a first-order derivative kernel applied to the flow to minimise extreme
            variations of flow.
         * `smoothness2` this loss function is a second-order derivative kernel encouraging flow values to be locally co-linear.
         * `l1` this term penalizes extreme values of flow.

    Args:
        flow_weights (dict): dictionary of weights per loss type, keys should be a subset of LOSS_NAMES, values
            should be floating point coefficients. Those coefficients are to be applied to each loss component.
        warping_head: module able to warp a feature tensor using a flow tensor. Only useful if you use
            time_consistency loss.
        smoothness_mask (string_list): list of loss functions that should be applied only for pixels with
            non negative values.
    """

    def __init__(self, flow_weights, warping_head=None, smoothness_mask=["time_consistency"]):
        super(FlowLoss, self).__init__()
        self.flow_weights = flow_weights
        self.smoothness_mask = smoothness_mask
        self.warping_head = warping_head
        assert set(flow_weights).issubset(set(LOSS_NAMES)), "flow_weights keys should be a subset of :" + ' '.join(
            LOSS_NAMES)

        if self._is_loss_active("smoothness"):
            self.smoothness = SmoothnessLoss()
        if self._is_loss_active("smoothness2"):
            self.smoothness2 = SecondOrderLoss()

    def _is_loss_active(self, key):
        """Checks if the loss has a positive weight."""
        return key in self.flow_weights and self.flow_weights[key] > 0

    def _get_mask_dict(self, inp):
        """
        Creates a dictionary containing None if mask should not be applied to a loss or a mask computed from the
        input otherwise
        """
        mask_dict = {k: None for k in LOSS_NAMES}
        if self.smoothness_mask:
            # masked used to only apply loss where there is input data
            mask = (torch.max(inp, 2)[0] > 0)[:, :, None].float().detach()
            # the param can be either an iterable of the different loss names to which it applies
            if isinstance(self.smoothness_mask, Iterable):
                assert set(self.smoothness_mask).issubset(set(LOSS_NAMES)), " should be a subset of :" + ' '.join(
                    LOSS_NAMES)
                mask_dict.update({k: mask for k in self.smoothness_mask})
            else:
                # or it applies to all of them
                mask_dict = {k: mask for k in LOSS_NAMES}
        return mask_dict

    def forward(self, flows, interpolated_inputs, batch_size):
        """
        Computes the losses and put them in a dictionary
        Here we normalize the flow only once and pass flow_is_normalized=True to
        warping modules.
        """

        scales = [2**(-i + 1) for i in range(len(flows), 0, -1)]

        loss_init = torch.tensor(0, dtype=flows[0].dtype, device=flows[0].device)
        losses = defaultdict(lambda: loss_init.clone())

        for i, (interpolated_input, flow) in enumerate(zip(interpolated_inputs, flows)):
            coef = scales[i]
            flow = normalize_flow(flow)
            warp = self.warping_head[i].warp_sequence_by_one_time_bin(
                interpolated_input, flow, align_corners=True, flow_is_normalized=True)
            mask_dict = self._get_mask_dict(interpolated_input)
            batch_size = interpolated_input.shape[1]

            if self._is_loss_active("data"):
                losses['data'] += charbonnier_loss(warp[:-1] - interpolated_input[1:], mask=mask_dict['data']) * coef
            if self._is_loss_active("smoothness"):
                losses['smoothness'] += self.smoothness(flow, ev_mask=mask_dict['smoothness']) * coef
            if self._is_loss_active("smoothness2"):
                losses['smoothness2'] += self.smoothness2(flow, ev_mask=mask_dict['smoothness2']) * coef
            if self._is_loss_active("l1"):
                losses['l1'] += extreme_displacement_loss(flow) * coef
            if self._is_loss_active('bw_deblur'):
                sharp_input = self.warping_head[i].sharpen_micro_tbin(
                    interpolated_input, flow, 0, align_corners=True, flow_is_normalized=True)
                t, b, c, h, w = sharp_input.shape
                sharp_input = sharp_input.reshape(t, b, c // 2, 2, h, w)
                sharp_input_perm = sharp_input.permute(2, 0, 1, 3, 4, 5).contiguous()
                loss1 = charbonnier_loss(sharp_input_perm[0] - sharp_input_perm[1:].mean(0))
                losses['bw_deblur'] += loss1 * coef

            if self._is_loss_active('time_consistency'):
                warped = warp
                for t in range(1, min(4, warp.shape[0] - 1)):
                    mask = mask_dict['time_consistency'][1 + t:] if mask_dict['time_consistency'] is not None else None
                    warped = warped[:-1]
                    flow_t = time_to_batch(batch_to_time(flow, batch_size)[t:])[0]
                    warped = self.warping_head[i].warp_sequence_by_one_time_bin(warped, flow_t)
                    losses["time_consistency"] += (1 - t * 0.25) * coef * charbonnier_loss(
                        warped[:-1] - interpolated_input[1 + t:], mask=mask)

        for key in losses:
            losses[key] *= self.flow_weights[key]
        return losses


def flow_length_square(flow):
    """
    Returns the length of the flow in a pixel-wise manner
    ! the flow is in the format N, C, H, W !
    """
    assert flow.shape[1] == 2
    return torch.sum(flow ** 2, 1)[..., None]


def charbonnier_loss(x, mask=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Charbonnier loss with optional masking"""
    error = torch.pow((x * beta)**2 + epsilon**2, alpha)
    if mask is not None:
        error = error * mask.to(x.device).float()

    return error.mean()


def create_mask(tensor_shape, paddings, device=torch.device('cpu')):
    """
    Creates a mask where the lateral padding values cancel out the values.
    """
    inner_width = tensor_shape[-1] - (paddings[0][0] + paddings[0][1])
    inner_height = tensor_shape[-2] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones((tensor_shape[0], 1, inner_height, inner_width), device=device)
    mask = F.pad(inner, (paddings[0][0], paddings[0][1], paddings[1][0], paddings[1][1]))
    return mask.detach()


def create_border_mask(tensor_shape, border_ratio=0.1):
    """
    Creates a mask so that the padding is a specified fraction of the image size.
    """
    size = int(math.ceil(border_ratio * min(tensor_shape[-1], tensor_shape[-2])))
    return create_mask(tensor_shape, [[size, size], [size, size]])


def extreme_displacement_loss(flow, threshold=0):
    """Penalizes extreme magnitude.

    Args:
        (torch.Tensor) Tensor on which we compute extreme values.
        threshold (float): Threshold above which penalization start to occur."""
    mag = flow_length_square(flow)
    return torch.sum(mag[mag > threshold]) / mag.numel()


class MaskedLoss(nn.Module):
    """Abstract class where a loss has a mask depending on input shape that is registered as a buffer."""

    def __init__(self):
        super(MaskedLoss, self).__init__()
        self._mask_shape = []
        self._create_mask([5, 1, 4, 4])

    def _create_mask(self, flow_shape, device=torch.device('cpu')):
        """
        Creates a mask where the lateral padding values cancel out the values.
        """
        raise NotImplementedError

    def _get_mask(self, flow, ev_mask=None):
        """
        Uses the registered border mask if shape is compatible or recreate one.

        Optionally uses an event mask.
        """
        if flow.shape != self._mask_shape:
            self._create_mask(flow.shape, device=flow.device)

        if ev_mask is not None:
            mask = self.border_mask * ev_mask
        else:
            mask = self.border_mask
        if ev_mask is not None:
            mask *= ev_mask

        return mask


class SmoothnessLoss(MaskedLoss):
    """
    First order derivative smoothness constraint to ensure colinarity of neighbouring flows.
    """

    def __init__(self):
        super(SmoothnessLoss, self).__init__()
        self.conv_filter = nn.Conv2d(1, 2, 3, bias=False, padding=1)
        self.conv_filter.weight.data = torch.Tensor([
            [[[0., 0., 0.], [0., 1., -1.], [0., 0., 0.]]], [[[0., 0., 0.], [0., 1., 0.], [0., -1., 0.]]]])

        self.conv_filter.weight.requires_grad = False

    def _create_mask(self, flow_shape, device=torch.device('cpu')):
        """
        Creates a mask where the lateral padding values cancel out the values.
        """
        self._mask_shape = flow_shape

        mask_x = create_mask(flow_shape, [[0, 0], [0, 1]])
        mask_y = create_mask(flow_shape, [[0, 1], [0, 0]])
        mask = torch.cat((mask_x, mask_y), 1)
        self.register_buffer("border_mask", mask, persistent=False)

    def forward(self, flow, ev_mask=None):
        # applying the same convolution for x and y components of the flow
        deltas = self.conv_filter(flow.view(flow.shape[0] * 2, 1, flow.shape[-2], flow.shape[-1]))
        deltas = deltas.view(flow.shape[0], 2, 2, flow.shape[-2], flow.shape[-1])
        delta_u = deltas[:, 0]
        delta_v = deltas[:, 1]

        mask = self._get_mask(flow, ev_mask=ev_mask)

        return charbonnier_loss(delta_u, mask) + charbonnier_loss(delta_v, mask)


class SecondOrderLoss(MaskedLoss):
    """
    Second order derivative smoothness constraint to ensure colinarity of neighbouring flows.
    """

    def __init__(self):
        super(SecondOrderLoss, self).__init__()
        self.conv_filter = nn.Conv2d(1, 4, 3, padding=1, bias=False)
        # the following is a sobel kernel
        self.conv_filter.weight.data = torch.Tensor([[
            [[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]]], [[[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]]],
            [[[1., 0., 0.], [0., -2., 0.], [0., 0., 1.]]], [[[0., 0., 1.], [0., -2., 0.], [0., 0., 1.]]]])

        self.conv_filter.weight.requires_grad = False

    def _create_mask(self, flow_shape, device=torch.device('cpu')):
        """
        Creates a mask where the lateral padding values cancel out the values.
        """
        self._mask_shape = flow_shape
        mask_x = create_mask(flow_shape, [[0, 0], [1, 1]])
        mask_y = create_mask(flow_shape, [[1, 1], [0, 0]])
        mask_diag = create_mask(flow_shape, [[1, 1], [1, 1]])
        mask = torch.cat((mask_x, mask_y, mask_diag, mask_diag), 1)
        self.register_buffer("border_mask", mask, persistent=False)

    def forward(self, flow, ev_mask=None):
        # convolve both component of the flow
        deltas = self.conv_filter(flow.view(flow.shape[0] * 2, 1, flow.shape[-2], flow.shape[-1]))
        deltas = deltas.view(flow.shape[0], 2, 4, flow.shape[-2], flow.shape[-1])
        delta_u = deltas[:, 0]
        delta_v = deltas[:, 1]

        mask = self._get_mask(flow, ev_mask=ev_mask)

        return charbonnier_loss(delta_u, mask) + charbonnier_loss(delta_v, mask)
