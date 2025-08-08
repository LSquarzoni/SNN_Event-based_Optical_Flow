import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch.functional import quant
import brevitas
from brevitas.nn import QuantConv2d
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

import models.spiking_util as spiking

class SNNtorch_ConvLIF(nn.Module):
    """
    Convolutional spiking LIF cell using SNNTorch Leaky neuron.

    Design choices:
    - Uses snn.Leaky for LIF dynamics
    - Maintains compatibility with existing model interface
    - Supports quantization and normalization options
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
        quantization_config=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Convert leak parameters to beta for snn.Leaky (broadcastable shape for conv)
        initial_beta = torch.sigmoid(torch.randn(1, hidden_size, 1, 1) * leak[1] + leak[0])
        
        # Convert threshold parameters (broadcastable shape for conv)
        initial_thresh = torch.clamp(torch.randn(1, hidden_size, 1, 1) * thresh[1] + thresh[0], min=0.01)

        # Create snn.Leaky layer with broadcastable parameters
        reset_mechanism = "zero" if hard_reset else "subtract"

        # Quantization checking
        if quantization_config is not None and quantization_config["enabled"]:
            self.ff = QuantConv2d(
                input_size,
                hidden_size,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                return_quant_tensor=True,
                scaling_per_output_channel=True,
                per_channel_broadcastable_shape=(1, hidden_size, 1, 1),
                scaling_stats_permute_dims=(1, 0, 2, 3),
            )
            q_lif = quant.state_quant(num_bits=8, uniform=False, thr_centered=True)
            self.lif = snn.Leaky(
                beta=initial_beta,  # Shape: (1, hidden_size, 1, 1)
                threshold=initial_thresh,  # Shape: (1, hidden_size, 1, 1)
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
                state_quant=q_lif,
            )
        else:
            self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
            self.lif = snn.Leaky(
                beta=initial_beta,  # Shape: (1, hidden_size, 1, 1)
                threshold=initial_thresh,  # Shape: (1, hidden_size, 1, 1)
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
            )

        # weight init
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # Store detach option for compatibility
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = nn.utils.weight_norm(self.ff)
            self.norm = None
        elif norm == "group":
            groups = min(1, input_size // 4)
            self.norm = nn.GroupNorm(groups, input_size)
        else:
            self.norm = None

    def forward(self, input_, prev_state, residual=0):
        # input current
        if self.norm is not None:
            input_ = self.norm(input_)
        ff = self.ff(input_)

        # Extract membrane potential from prev_state for compatibility
        if prev_state is None:
            mem = None
        else:
            mem = prev_state[0]  # First element is membrane potential

        # Apply snn.Leaky neuron
        spk, mem_out = self.lif(ff, mem)

        # Detach the output membrane potential to ensure clean state transitions
        if self.detach:
            self.lif.detach_hidden()

        # Create new state compatible with original interface
        new_state = torch.stack([mem_out, spk])

        return spk + residual, new_state
    
class SNNtorch_ConvLIFRecurrent(nn.Module):
    """
    Convolutional recurrent spiking LIF cell using SNNTorch Leaky neuron.

    Design choices:
    - Uses snn.Leaky for LIF dynamics
    - Maintains compatibility with existing model interface
    - Supports quantization and normalization options
    - Includes recurrent connections
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
        quantization_config=None,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Convert leak parameters to beta for snn.Leaky (broadcastable shape for conv)
        initial_beta = torch.sigmoid(torch.randn(1, hidden_size, 1, 1) * leak[1] + leak[0])
        
        # Convert threshold parameters (broadcastable shape for conv)
        initial_thresh = torch.clamp(torch.randn(1, hidden_size, 1, 1) * thresh[1] + thresh[0], min=0.01)

        # Create snn.Leaky layer with broadcastable parameters
        reset_mechanism = "zero" if hard_reset else "subtract"
        
        # Quantization checking
        if quantization_config is not None and quantization_config["enabled"]:
            self.ff = QuantConv2d(
                input_size,
                hidden_size,
                kernel_size,
                padding=padding,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                return_quant_tensor=True,
                scaling_per_output_channel=True,
                per_channel_broadcastable_shape=(1, hidden_size, 1, 1),
                scaling_stats_permute_dims=(1, 0, 2, 3),
            )
            self.rec = QuantConv2d(
                hidden_size,
                hidden_size,
                kernel_size,
                padding=padding,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                return_quant_tensor=True,
                scaling_per_output_channel=True,
                per_channel_broadcastable_shape=(1, hidden_size, 1, 1),
                scaling_stats_permute_dims=(1, 0, 2, 3),
            )
            q_lif = quant.state_quant(num_bits=8, uniform=False, thr_centered=True)
            self.lif = snn.Leaky(
                beta=initial_beta,  # Shape: (1, hidden_size, 1, 1)
                threshold=initial_thresh,  # Shape: (1, hidden_size, 1, 1)
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
                state_quant=q_lif,
            )
        else:
            self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding, bias=False)
            self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
            self.lif = snn.Leaky(
                beta=initial_beta,  # Shape: (1, hidden_size, 1, 1)
                threshold=initial_thresh,  # Shape: (1, hidden_size, 1, 1)
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
            )

        # weight init
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

        # Store detach option for compatibility
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = nn.utils.weight_norm(self.ff)
            self.rec = nn.utils.weight_norm(self.rec)
            self.norm_ff = None
            self.norm_rec = None
        elif norm == "group":
            groups_ff = min(1, input_size // 4)
            groups_rec = min(1, hidden_size // 4)
            self.norm_ff = nn.GroupNorm(groups_ff, input_size)
            self.norm_rec = nn.GroupNorm(groups_rec, hidden_size)
        else:
            self.norm_ff = None
            self.norm_rec = None

    def forward(self, input_, prev_state):
        # input current
        if self.norm_ff is not None:
            input_ = self.norm_ff(input_)
        ff = self.ff(input_)

        # Extract membrane potential and previous spikes from prev_state
        if prev_state is None:
            mem = None
            prev_spk = torch.zeros_like(ff)
        else:
            mem = prev_state[0]  # membrane potential
            prev_spk = prev_state[1]  # previous spikes

        # recurrent current
        if self.norm_rec is not None:
            prev_spk = self.norm_rec(prev_spk)
        rec = self.rec(prev_spk)

        # Combine feedforward and recurrent currents
        total_current = ff + rec

        # Apply snn.Leaky neuron
        spk_out, mem_out = self.lif(total_current, mem)

        # Detach the output membrane potential to ensure clean state transitions
        if self.detach:
            self.lif.detach_hidden()

        # Create new state compatible with original interface
        new_state = torch.stack([mem_out, spk_out])

        return spk_out, new_state