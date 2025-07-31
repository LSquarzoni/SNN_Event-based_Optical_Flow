import torch
import torch.nn as nn
import numpy as np


class QuantizationConfig:
    """Configuration for quantization parameters."""
    
    def __init__(self, data_type="fp32", activation_bits=8, weight_bits=8, state_bits=8):
        self.data_type = data_type  # "fp32" or "int8"
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        self.state_bits = state_bits
        self.use_quantization = data_type == "int8"


class QuantizationParameters:
    """Stores quantization parameters for a tensor."""
    
    def __init__(self):
        self.scale = 1.0
        self.zero_point = 0
        self.min_val = None
        self.max_val = None
    
    def update(self, tensor, bits=8):
        """Update quantization parameters based on tensor statistics."""
        if tensor.numel() == 0:
            return
            
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Symmetric quantization around zero for better numerical stability
        abs_max = max(abs(min_val), abs(max_val))
        
        # Calculate scale and zero point
        qmin = -(2**(bits-1))
        qmax = 2**(bits-1) - 1
        
        if abs_max == 0:
            self.scale = 1.0
            self.zero_point = 0
        else:
            self.scale = abs_max / (2**(bits-1) - 1)
            self.zero_point = 0  # Symmetric quantization
        
        self.min_val = min_val
        self.max_val = max_val


def quantize_tensor(tensor, scale, zero_point, bits=8):
    """Quantize a tensor to int representation."""
    qmin = -(2**(bits-1))
    qmax = 2**(bits-1) - 1
    
    # Quantize
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    
    return quantized.to(torch.int8)


def dequantize_tensor(quantized_tensor, scale, zero_point):
    """Dequantize tensor back to float."""
    return (quantized_tensor.float() - zero_point) * scale


class QuantizedLinearFunction(torch.autograd.Function):
    """Custom autograd function for quantized operations."""
    
    @staticmethod
    def forward(ctx, input_tensor, weight, bias, input_scale, input_zp, 
                weight_scale, weight_zp, output_scale, output_zp):
        # For inference, perform quantized computation
        if not input_tensor.requires_grad:
            # Quantized matrix multiplication
            input_q = quantize_tensor(input_tensor, input_scale, input_zp)
            weight_q = quantize_tensor(weight, weight_scale, weight_zp)
            
            # Perform computation in int32 to avoid overflow
            output_int32 = torch.nn.functional.conv2d(
                dequantize_tensor(input_q, input_scale, input_zp),
                dequantize_tensor(weight_q, weight_scale, weight_zp),
                bias
            )
            
            # Quantize output
            output_q = quantize_tensor(output_int32, output_scale, output_zp)
            return dequantize_tensor(output_q, output_scale, output_zp)
        else:
            # For training, use FP32
            return torch.nn.functional.conv2d(input_tensor, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradients always in FP32
        return grad_output, None, None, None, None, None, None, None, None


class QuantizationAwareModule(nn.Module):
    """Base class for quantization-aware modules."""
    
    def __init__(self, quant_config=None):
        super().__init__()
        self.quant_config = quant_config or QuantizationConfig()
        
        # Quantization parameters for different tensor types
        self.activation_quant_params = QuantizationParameters()
        self.weight_quant_params = QuantizationParameters()
        self.state_quant_params = QuantizationParameters()
        
        # Calibration mode for collecting statistics
        self.calibration_mode = False
    
    def enable_calibration(self):
        """Enable calibration mode to collect quantization statistics."""
        self.calibration_mode = True
    
    def disable_calibration(self):
        """Disable calibration mode."""
        self.calibration_mode = False
    
    def update_quantization_params(self, tensor, param_type="activation"):
        """Update quantization parameters during calibration."""
        if not self.calibration_mode or not self.quant_config.use_quantization:
            return
        
        if param_type == "activation":
            self.activation_quant_params.update(tensor, self.quant_config.activation_bits)
        elif param_type == "weight":
            self.weight_quant_params.update(tensor, self.quant_config.weight_bits)
        elif param_type == "state":
            self.state_quant_params.update(tensor, self.quant_config.state_bits)
    
    def quantize_activation(self, tensor):
        """Quantize activation tensor."""
        if not self.quant_config.use_quantization:
            return tensor
        
        if self.calibration_mode:
            self.update_quantization_params(tensor, "activation")
            return tensor
        
        # During inference, quantize and dequantize
        q_tensor = quantize_tensor(
            tensor, 
            self.activation_quant_params.scale, 
            self.activation_quant_params.zero_point,
            self.quant_config.activation_bits
        )
        return dequantize_tensor(
            q_tensor, 
            self.activation_quant_params.scale, 
            self.activation_quant_params.zero_point
        )
    
    def quantize_state(self, tensor):
        """Quantize state tensor."""
        if not self.quant_config.use_quantization or tensor is None:
            return tensor
        
        if self.calibration_mode:
            self.update_quantization_params(tensor, "state")
            return tensor
        
        # During inference, quantize and dequantize
        q_tensor = quantize_tensor(
            tensor, 
            self.state_quant_params.scale, 
            self.state_quant_params.zero_point,
            self.quant_config.state_bits
        )
        return dequantize_tensor(
            q_tensor, 
            self.state_quant_params.scale, 
            self.state_quant_params.zero_point
        )
    
    def quantize_weight(self, tensor):
        """Quantize weight tensor."""
        if not self.quant_config.use_quantization:
            return tensor
        
        if self.calibration_mode:
            self.update_quantization_params(tensor, "weight")
            return tensor
        
        # During inference, quantize and dequantize
        q_tensor = quantize_tensor(
            tensor, 
            self.weight_quant_params.scale, 
            self.weight_quant_params.zero_point,
            self.quant_config.weight_bits
        )
        return dequantize_tensor(
            q_tensor, 
            self.weight_quant_params.scale, 
            self.weight_quant_params.zero_point
        )