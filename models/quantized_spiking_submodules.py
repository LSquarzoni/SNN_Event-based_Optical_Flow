import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.spiking_util as spiking
from .quantization_util import QuantizationAwareModule, QuantizationConfig


class QuantizedConvLIF(QuantizationAwareModule):
    """
    Quantized convolutional spiking LIF cell.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak=(-4.0, 0.1),
        thresh=(0.8, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
        quant_config=None,
    ):
        super().__init__(quant_config)

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        if learn_leak:
            self.leak = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        else:
            self.register_buffer("leak", torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        if learn_thresh:
            self.thresh = nn.Parameter(torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])
        else:
            self.register_buffer("thresh", torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])

        # weight init
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = nn.utils.weight_norm(self.ff)
            self.norm = None
        elif norm == "group":
            groups = min(1, input_size // 4)  # at least instance norm
            self.norm = nn.GroupNorm(groups, input_size)
        else:
            self.norm = None

    def forward(self, input_, prev_state, residual=0):
        # Quantize input
        input_ = self.quantize_activation(input_)
        
        # input current
        if self.norm is not None:
            input_ = self.norm(input_)
        
        # Quantize weights before convolution
        weight = self.quantize_weight(self.ff.weight)
        ff = F.conv2d(input_, weight, self.ff.bias, 
                     self.ff.stride, self.ff.padding, self.ff.dilation, self.ff.groups)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(2, *ff.shape, dtype=ff.dtype, device=ff.device)
        
        # Quantize previous states
        v, z = prev_state  # unbind op, removes dimension
        v = self.quantize_state(v)
        z = self.quantize_state(z)

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leak
        leak = torch.sigmoid(self.leak)

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak * (1 - z) + (1 - leak) * ff
        else:
            v_out = v * leak + (1 - leak) * ff - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        # Quantize outputs
        z_out_quantized = self.quantize_activation(z_out + residual)
        v_out_quantized = self.quantize_state(v_out)
        z_out_state_quantized = self.quantize_state(z_out)

        return z_out_quantized, torch.stack([v_out_quantized, z_out_state_quantized])


class QuantizedConvLIFRecurrent(QuantizationAwareModule):
    """
    Quantized convolutional recurrent spiking LIF cell.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        activation="arctanspike",
        act_width=10.0,
        leak=(-4.0, 0.1),
        thresh=(0.8, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
        quant_config=None,
    ):
        super().__init__(quant_config)

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding, bias=False)
        self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
        if learn_leak:
            self.leak = nn.Parameter(torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        else:
            self.register_buffer("leak", torch.randn(hidden_size, 1, 1) * leak[1] + leak[0])
        if learn_thresh:
            self.thresh = nn.Parameter(torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])
        else:
            self.register_buffer("thresh", torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0])

        # weight init
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = nn.utils.weight_norm(self.ff)
            self.rec = nn.utils.weight_norm(self.rec)
            self.norm_ff = None
            self.norm_rec = None
        elif norm == "group":
            groups_ff = min(1, input_size // 4)  # at least instance norm
            groups_rec = min(1, hidden_size // 4)  # at least instance norm
            self.norm_ff = nn.GroupNorm(groups_ff, input_size)
            self.norm_rec = nn.GroupNorm(groups_rec, hidden_size)
        else:
            self.norm_ff = None
            self.norm_rec = None

    def forward(self, input_, prev_state):
        # Quantize input
        input_ = self.quantize_activation(input_)
        
        # input current
        if self.norm_ff is not None:
            input_ = self.norm_ff(input_)
        
        # Quantize weights before convolution
        ff_weight = self.quantize_weight(self.ff.weight)
        ff = F.conv2d(input_, ff_weight, self.ff.bias, 
                     self.ff.stride, self.ff.padding, self.ff.dilation, self.ff.groups)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(2, *ff.shape, dtype=ff.dtype, device=ff.device)
        
        # Quantize previous states
        v, z = prev_state  # unbind op, removes dimension
        v = self.quantize_state(v)
        z = self.quantize_state(z)

        # recurrent current
        if self.norm_rec is not None:
            z = self.norm_rec(z)
        
        # Quantize recurrent weights
        rec_weight = self.quantize_weight(self.rec.weight)
        rec = F.conv2d(z, rec_weight, self.rec.bias, 
                      self.rec.stride, self.rec.padding, self.rec.dilation, self.rec.groups)

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leak
        leak = torch.sigmoid(self.leak)

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak * (1 - z) + (1 - leak) * (ff + rec)
        else:
            v_out = v * leak + (1 - leak) * (ff + rec) - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        # Quantize outputs
        z_out_quantized = self.quantize_activation(z_out)
        v_out_quantized = self.quantize_state(v_out)
        z_out_state_quantized = self.quantize_state(z_out)

        return z_out_quantized, torch.stack([v_out_quantized, z_out_state_quantized])