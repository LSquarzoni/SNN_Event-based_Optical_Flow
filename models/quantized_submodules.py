import torch
import torch.nn as nn
import torch.nn.functional as F
import models.spiking_util as spiking
from .quantization_util import QuantizationAwareModule, QuantizationConfig


class QuantizedConvLayer(QuantizationAwareModule):
    """
    Quantized ConvLayer.
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

    def forward(self, x):
        # Quantize input
        x = self.quantize_activation(x)

        # Quantize weights before convolution
        weight = self.quantize_weight(self.conv2d.weight)
        out = F.conv2d(x, weight, self.conv2d.bias, 
                      self.conv2d.stride, self.conv2d.padding, 
                      self.conv2d.dilation, self.conv2d.groups)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        # Quantize output
        out = self.quantize_activation(out)

        return out