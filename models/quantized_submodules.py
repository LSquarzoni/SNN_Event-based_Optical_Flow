import torch
import torch.nn as nn
import torch.nn.functional as F
import models.spiking_util as spiking
from .quantization_util import QuantizationAwareModule, QuantizationConfig


class QuantizedConvLayer_(QuantizationAwareModule):
    """
    Quantized clone of ConvLayer that acts like it has state, and allows residual.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
        w_scale=None,
        quant_config=None,
    ):
        super().__init__(quant_config)

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if w_scale is not None:
            nn.init.uniform_(self.conv2d.weight, -w_scale, w_scale)
            if bias:
                nn.init.zeros_(self.conv2d.bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            else:
                self.activation = getattr(spiking, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x, prev_state, residual=0):
        # Quantize input
        x = self.quantize_activation(x)
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.tensor(0)  # not used
        
        # Quantize previous state
        prev_state = self.quantize_state(prev_state)

        # Quantize weights before convolution
        weight = self.quantize_weight(self.conv2d.weight)
        out = F.conv2d(x, weight, self.conv2d.bias, 
                      self.conv2d.stride, self.conv2d.padding, 
                      self.conv2d.dilation, self.conv2d.groups)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        out += residual
        if self.activation is not None:
            out = self.activation(out)

        # Quantize output
        out = self.quantize_activation(out)

        return out, prev_state


class QuantizedConvGRU(QuantizationAwareModule):
    """
    Quantized convolutional GRU cell.
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None, quant_config=None):
        super().__init__(quant_config)
        
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state):
        # Quantize input
        input_ = self.quantize_activation(input_)

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)
        
        # Quantize previous state
        prev_state = self.quantize_state(prev_state)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        
        # Quantize weights and perform convolutions
        reset_weight = self.quantize_weight(self.reset_gate.weight)
        update_weight = self.quantize_weight(self.update_gate.weight)
        out_weight = self.quantize_weight(self.out_gate.weight)
        
        update = torch.sigmoid(F.conv2d(stacked_inputs, update_weight, self.update_gate.bias,
                                       self.update_gate.stride, self.update_gate.padding))
        reset = torch.sigmoid(F.conv2d(stacked_inputs, reset_weight, self.reset_gate.bias,
                                      self.reset_gate.stride, self.reset_gate.padding))
        
        out_inputs = torch.tanh(F.conv2d(torch.cat([input_, prev_state * reset], dim=1), 
                                        out_weight, self.out_gate.bias,
                                        self.out_gate.stride, self.out_gate.padding))
        
        new_state = prev_state * (1 - update) + out_inputs * update

        # Quantize output state
        new_state = self.quantize_state(new_state)

        return new_state, new_state