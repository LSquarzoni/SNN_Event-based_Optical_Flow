import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch.functional import quant
import brevitas
from brevitas.nn import QuantConv2d, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
from brevitas.nn.quant_layer import QuantLayerMixin
from brevitas.core.quant import QuantType

import models.spiking_util as spiking

class SNNtorch_ConvLIF(nn.Module):
    """
    Convolutional spiking LIF cell using SNNTorch Leaky neuron.

    Design choices:
    - Uses snn.Leaky for LIF dynamics
    - Maintains compatibility with existing model interface
    - Per-channel learnable parameters like original implementation
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
        leak=(0.0, 1.0),
        thresh=(0.0, 0.8),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
        quantization_config=None,
        exporting=False,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exporting = exporting  # Store export mode flag
        
        # Per-channel parameters for initialization only
        beta_init = torch.empty(hidden_size, 1, 1).uniform_(leak[0], leak[1])
        threshold_init = torch.empty(hidden_size, 1, 1).uniform_(thresh[0], thresh[1])

        # Create snn.Leaky layer - parameters set at construction only
        reset_mechanism = "zero" if hard_reset else "subtract"
        
        self.quantization_config = quantization_config["enabled"] if quantization_config is not None else False
        self.quant_conv_only = quantization_config.get("Conv_only", False) if quantization_config is not None else False

        # Quantization checking
        if quantization_config is not None and self.quantization_config and not self.quant_conv_only:
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
            )
            # State quantization: range = [-threshold*(1+lower_limit), threshold*(1+upper_limit)]
            # Setting threshold=1.0, lower_limit=x, upper_limit=0.0 → range=[-x, 1]
            self.q_lif = quant.state_quant(
                num_bits=8,
                uniform=True,
                thr_centered=False,
                threshold=1.0,
                lower_limit=249.0,
                upper_limit=0.0
            )
            self.lif = snn.Leaky(
                beta=beta_init,
                threshold=threshold_init,
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
                state_quant=self.q_lif,
            )
        elif quantization_config is not None and self.quantization_config and self.quant_conv_only:
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
                return_quant_tensor=False,
            )
            self.lif = snn.Leaky(
                beta=beta_init,
                threshold=threshold_init,
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
            )
        else:
            self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
            self.lif = snn.Leaky(
                beta=beta_init,
                threshold=threshold_init,
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
            )

        # weight init
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        """ # Register fixed initializers so ONNX export contains clear constants
        # init_mem: values in [0.0, 0.8], shape [1, C, 1, 1]
        init_mem = torch.rand(1, hidden_size, 1, 1) * 0.8
        self.register_buffer("init_mem", init_mem) """

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
        self.lif.threshold.data.clamp_(min=0.01)
        
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
        new_state = torch.stack([mem_out, spk], dim=0)

        return spk, new_state
    
class SNNtorch_ConvLIFRecurrent(nn.Module):
    """
    Convolutional recurrent spiking LIF cell using SNNTorch Leaky neuron.

    Design choices:
    - Uses snn.Leaky for LIF dynamics
    - Maintains compatibility with existing model interface
    - Per-channel learnable parameters like original implementation
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
        leak=(0.0, 1.0),
        thresh=(0.0, 0.8),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
        quantization_config=None,
        exporting=False,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exporting = exporting  # Store export mode flag
        
        # Per-channel parameters for initialization only
        beta_init = torch.empty(hidden_size, 1, 1).uniform_(leak[0], leak[1])
        threshold_init = torch.empty(hidden_size, 1, 1).uniform_(thresh[0], thresh[1])

        # Create snn.Leaky layer - parameters set at construction only
        reset_mechanism = "zero" if hard_reset else "subtract"
        
        self.quantization_config = quantization_config["enabled"] if quantization_config is not None else False
        self.quant_conv_only = quantization_config.get("Conv_only", False) if quantization_config is not None else False

        # Quantization checking
        if quantization_config is not None and self.quantization_config and not self.quant_conv_only:
            self.quant_identity = QuantIdentity(return_quant_tensor=True)
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
            )
            # State quantization: range = [-threshold*(1+lower_limit), threshold*(1+upper_limit)]
            # Setting threshold=1.0, lower_limit=x, upper_limit=0.0 → range=[-x, 1]
            self.q_lif = quant.state_quant(
                num_bits=8,
                uniform=True,
                thr_centered=False,
                threshold=1.0,
                lower_limit=249.0,
                upper_limit=0.0
            )
            self.lif = snn.Leaky(
                beta=beta_init,
                threshold=threshold_init,
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
                state_quant=self.q_lif,
            )
        elif quantization_config is not None and self.quantization_config and self.quant_conv_only:
            self.ff = QuantConv2d(
                input_size,
                hidden_size,
                kernel_size,
                padding=padding,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
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
                return_quant_tensor=False,
            )
            self.lif = snn.Leaky(
                beta=beta_init,
                threshold=threshold_init,
                learn_beta=learn_leak,
                learn_threshold=learn_thresh,
                reset_mechanism=reset_mechanism,
                reset_delay=False,
            )
        else:
            self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding, bias=False)
            self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
            self.lif = snn.Leaky(
                beta=beta_init,
                threshold=threshold_init,
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
        """ # Register fixed initializers so ONNX export contains clear constants
        # init_mem: values in [0.0, 0.8], shape [1, C, 1, 1]
        init_mem = torch.rand(1, hidden_size, 1, 1) * 0.8
        self.register_buffer("init_mem", init_mem)
        # init_prev_spk: random binary {0,1} values with shape [1, C, 1, 1]
        init_prev_spk = (torch.rand(1, hidden_size, 1, 1) > 0.5).to(torch.float32)
        self.register_buffer("init_prev_spk", init_prev_spk) """

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

    def forward(self, input_, prev_state, residual=0):
        self.lif.threshold.data.clamp_(min=0.01)
        
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
        if self.quantization_config and not self.quant_conv_only:
            total_current = self.quant_identity(ff) + self.quant_identity(rec)
        else:
            total_current = ff + rec

        # Apply snn.Leaky neuron
        spk_out, mem_out = self.lif(total_current, mem)

        # Detach the output membrane potential to ensure clean state transitions
        if self.detach:
            self.lif.detach_hidden()

        # Create new state compatible with original interface
        new_state = torch.stack([mem_out, spk_out], dim=0)

        return spk_out, new_state
    


class custom_ConvLIF(nn.Module):
    """
    Convolutional spiking LIF cell using ONNX custom LIF operator.

    Design choices:
    - Uses ONNX custom LIF operator for LIF dynamics
    - Maintains compatibility with existing model interface
    - Per-channel parameters like original implementation
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak=(0.0, 1.0),
        thresh=(0.0, 0.8),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
        quantization_config=None,
        exporting=False,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exporting = exporting  # Store export mode flag
        
        # Per-channel parameters matching original implementation
        self.beta = nn.Parameter(torch.empty(hidden_size, 1, 1).uniform_(leak[0], leak[1]))
        self.threshold = nn.Parameter(torch.empty(hidden_size, 1, 1).uniform_(thresh[0], thresh[1]))

        self.quantization_config = quantization_config["enabled"] if quantization_config is not None else False

        # Quantization checking
        if quantization_config is not None and self.quantization_config:
            self.ff = QuantConv2d(
                input_size,
                hidden_size,
                kernel_size,
                padding=padding,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            )
            self.quant_lif_input = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
            self.quant_mem_input = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
            self.quant_spk_output = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        else:
            self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)

        # weight init
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # init_mem: values in [0.0, 0.8], shape [1, C, H, W]
        self.init_mem = torch.rand(1, hidden_size, 256, 256) * 0.8

    def forward(self, input_, prev_state, residual=0):
        ff = self.ff(input_)

        # Extract membrane potential from prev_state for compatibility
        if prev_state is None:
            mem = self.init_mem
        else:
            mem = prev_state[0]  # First element is membrane potential
            
        if self.quantization_config and self.exporting:
            ff_q = self.quant_lif_input(ff)
            mem_q = self.quant_mem_input(mem)
            out = torch.ops.SNN_implementation.LIF(ff_q, mem_q, self.beta, self.threshold)
            # Quantize the raw outputs from LIF
            spk_raw = out[0]  # shape [N, C, H, W]
            mem_raw = out[1]  # shape [N, C, H, W]
            # Apply output quantization for the return value (spike output)
            spk = self.quant_spk_output(spk_raw)
            # For state: use raw (unquantized) values to avoid graph issues
            # The quantization will be applied when these are used in the next timestep
            mem_out = mem_raw
        else:
            # Apply LIF activation
            out = torch.ops.SNN_implementation.LIF(ff, mem, self.beta, self.threshold)
            spk = out[0]  # shape [N, C, H, W]
            mem_out = out[1]  # shape [N, C, H, W]

        if self.quantization_config and self.exporting:
            new_state = torch.stack([mem_out, spk_raw], dim=0)
        else:
            # Create new state compatible with original interface
            new_state = torch.stack([mem_out, spk], dim=0)

        return spk, new_state
    

class custom_ConvLIFRecurrent(nn.Module):
    """
    Convolutional recurrent spiking LIF cell using ONNX custom LIF operator.

    Design choices:
    - Uses ONNX custom LIF operator for LIF dynamics
    - Maintains compatibility with existing model interface
    - Per-channel parameters like original implementation
    - Includes recurrent connections
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        activation="arctanspike",
        act_width=10.0,
        leak=(0.0, 1.0),
        thresh=(0.0, 0.8),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
        quantization_config=None,
        exporting=False,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exporting = exporting  # Store export mode flag
        
        # Per-channel parameters matching original implementation
        self.beta = nn.Parameter(torch.empty(hidden_size, 1, 1).uniform_(leak[0], leak[1]))
        self.threshold = nn.Parameter(torch.empty(hidden_size, 1, 1).uniform_(thresh[0], thresh[1]))

        self.quantization_config = quantization_config["enabled"] if quantization_config is not None else False

        # Quantization checking
        if quantization_config is not None and self.quantization_config:
            self.ff = QuantConv2d(
                input_size,
                hidden_size,
                kernel_size,
                padding=padding,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            )
            self.rec = QuantConv2d(
                input_size,
                hidden_size,
                kernel_size,
                padding=padding,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            )
            self.quant_identity_add = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
            # Quantization layers for LIF operator inputs/outputs
            self.quant_lif_input = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
            self.quant_mem_input = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
            self.quant_spk_output = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        else:
            self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding, bias=False)
            self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)

        # weight init
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

        # init_mem: values in [0.0, 0.8], shape [1, C, H, W]
        self.init_mem = torch.rand(1, hidden_size, 256, 256) * 0.8
        # init_prev_spk: random binary {0,1} values with shape [1, C, H, W]
        self.init_prev_spk = (torch.rand(1, hidden_size, 256, 256) > 0.5).to(torch.float32)

    def forward(self, input_, prev_state, residual=0):
        ff = self.ff(input_)

        # Extract membrane potential and previous spikes from prev_state
        if prev_state is None:
            mem = self.init_mem
            prev_spk = self.init_prev_spk
        else:
            mem = prev_state[0]  # membrane potential
            prev_spk = prev_state[1]  # previous spikes

        # recurrent current
        rec = self.rec(prev_spk)
        
        if self.quantization_config and self.exporting:
            rec_q = self.quant_identity_add(rec)
            ff_q = self.quant_identity_add(ff)
            total_current = ff_q + rec_q
            # Apply quantization to LIF inputs
            total_current_q = self.quant_lif_input(total_current)
            mem_q = self.quant_mem_input(mem)
            out = torch.ops.SNN_implementation.LIF(total_current_q, mem_q, self.beta, self.threshold)
            # Quantize the raw outputs from LIF
            spk_raw = out[0]  # shape [N, C, H, W]
            mem_raw = out[1]  # shape [N, C, H, W]
            # Apply output quantization for the return value (spike output)
            spk_out = self.quant_spk_output(spk_raw)
            # For state: use raw (unquantized) values to avoid graph issues
            # The quantization will be applied when these are used in the next timestep
            mem_out = mem_raw
        else:
            # Combine feedforward and recurrent currents
            total_current = ff + rec
            # Apply LIF activation
            out = torch.ops.SNN_implementation.LIF(total_current, mem, self.beta, self.threshold)
            spk_out = out[0]  # shape [N, C, H, W]
            mem_out = out[1]  # shape [N, C, H, W]

        if self.quantization_config and self.exporting:
            new_state = torch.stack([mem_out, spk_raw], dim=0)
        else:
            # Create new state compatible with original interface
            new_state = torch.stack([mem_out, spk_out])

        return spk_out, new_state