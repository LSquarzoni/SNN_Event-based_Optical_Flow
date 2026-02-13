"""
Spiking Neural Network models for event-based optical flow estimation.
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch
import snntorch as snn
from snntorch.functional import quant
import math

from .base import BaseModel
from .model_util import copy_states, CropParameters
from .spiking_submodules import (
    ConvLIF,
    ConvLIFRecurrent,
)
from .SNNtorch_spiking_submodules import (
    SNNtorch_ConvLIF,
    SNNtorch_ConvLIFRecurrent,
    custom_ConvLIF,
    custom_ConvLIFRecurrent,
)
from .submodules import ConvLayer
from .unet import (
    SpikingMultiResUNetRecurrent,
)


class LIFFireNet(BaseModel):
    """
    Spiking FireNet architecture of LIF neurons for dense optical flow estimation from events.
    
    Note: Per-layer quantization ranges can be automatically tuned during evaluation
    using the --auto_tune_lif flag in eval_flow_quant.py
    """
    
    head_neuron = SNNtorch_ConvLIF
    ff_neuron = SNNtorch_ConvLIF
    rec_neuron = SNNtorch_ConvLIFRecurrent
    residual = False
    num_recurrent_units = 7
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = 0.01

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        self.exporting = unet_kwargs.get("exporting", False)  # Extract export mode flag
        
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        # Create layers
        if hasattr(self, '_create_layers'):
            self._create_layers(unet_kwargs)
        else:
            self._create_default_layers(unet_kwargs)

        self.reset_states()
    
    def _create_default_layers(self, unet_kwargs):
        """Create default layers."""
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        ff_act, rec_act = unet_kwargs["activations"]
        
        quantization_config = unet_kwargs.get("quantization", {})
        exporting = self.exporting  # Use the stored export mode flag

        self.head = self.head_neuron(self.num_bins, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting)

        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R1b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R2b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )

        self.pred = ConvLayer(
            base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred, quantization_config=quantization_config
        )

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def init_cropping(self, width, height):
        pass

    def forward(self, event_voxel=None, event_cnt=None, log=False, return_dict=True):
        """
        :param event_voxel: N x num_bins x H x W (only used when exporting=False)
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity (disabled during export)
        :param return_dict: if True, return dict; if False, return flow tensor directly (for ONNX export)
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor,
                 or just the flow tensor if return_dict=False.
        """

        # input encoding
        if self.exporting:
            # Export mode: only support cnt encoding
            if self.encoding == "cnt" and self.num_bins == 2:
                x = event_cnt
            else:
                print("Model error: Incorrect input encoding for export mode.")
                raise AttributeError
        else:
            # Normal mode: support both voxel and cnt encoding
            if self.encoding == "voxel":
                x = event_voxel
            elif self.encoding == "cnt" and self.num_bins == 2:
                x = event_cnt
            else:
                print("Model error: Incorrect input encoding.")
                raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # forward pass
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        x4, self._states[3] = self.R1b(x3, self._states[3], residual=x2 if self.residual else 0)

        x5, self._states[4] = self.G2(x4, self._states[4])
        x6, self._states[5] = self.R2a(x5, self._states[5])
        x7, self._states[6] = self.R2b(x6, self._states[6], residual=x5 if self.residual else 0)

        flow = self.pred(x7)  # [B, 2, H, W]
        
        # For FX tracing / ONNX export, return flow tensor directly
        if not return_dict:
            return flow

        # log activity
        # Skip logging during FX tracing to avoid control flow issues
        activity = None
        if isinstance(log, bool) and log and not self.exporting:
            activity = {}
            name = [
                "0:input",
                "1:head",
                "2:G1",
                "3:R1a",
                "4:R1b",
                "5:G2",
                "6:R2a",
                "7:R2b",
                "8:pred",
            ]
            for n, l in zip(name, [x, x1, x2, x3, x4, x5, x6, x7, flow]):
                activity[n] = l.detach().ne(0).float().mean().item()

        return {"flow": [flow], "activity": activity}


class LIFFireNet_short(BaseModel):
    """
    Shortened spiking FireNet architecture of LIF neurons with R1b and R2b layers removed.
    
    Note: Per-layer quantization ranges can be automatically tuned during evaluation
    using the --auto_tune_lif flag in eval_flow_quant.py
    """
    
    head_neuron = SNNtorch_ConvLIF
    ff_neuron = SNNtorch_ConvLIF
    rec_neuron = SNNtorch_ConvLIFRecurrent
    residual = False
    num_recurrent_units = 5  # Reduced from 7 to 5
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = 0.01

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        self.exporting = unet_kwargs.get("exporting", False)  # Extract export mode flag
        
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        # Create layers
        if hasattr(self, '_create_layers'):
            self._create_layers(unet_kwargs)
        else:
            self._create_default_layers(unet_kwargs)

        self.reset_states()
    
    def _create_default_layers(self, unet_kwargs):
        """Create default layers."""
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        ff_act, rec_act = unet_kwargs["activations"]
        
        quantization_config = unet_kwargs.get("quantization", {})
        exporting = self.exporting  # Use the stored export mode flag

        self.head = self.head_neuron(
            self.num_bins, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        
        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        # R1b removed

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        # R2b removed

        self.pred = ConvLayer(
            base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred, quantization_config=quantization_config
        )

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def init_cropping(self, width, height):
        pass

    def forward(self, event_voxel=None, event_cnt=None, log=False, return_dict=True):
        """
        :param event_voxel: N x num_bins x H x W (only used when exporting=False)
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity (disabled during export)
        :param return_dict: if True, return dict; if False, return flow tensor directly (for ONNX export)
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor,
                 or just the flow tensor if return_dict=False.
        """

        # input encoding
        if self.exporting:
            # Export mode: only support cnt encoding
            if self.encoding == "cnt" and self.num_bins == 2:
                x = event_cnt
            else:
                print("Model error: Incorrect input encoding for export mode.")
                raise AttributeError
        else:
            # Normal mode: support both voxel and cnt encoding
            if self.encoding == "voxel":
                x = event_voxel
            elif self.encoding == "cnt" and self.num_bins == 2:
                x = event_cnt
            else:
                print("Model error: Incorrect input encoding.")
                raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # forward pass (R1b and R2b removed)
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        # Skip R1b

        x4, self._states[3] = self.G2(x3, self._states[3])  # G2 now takes x3 instead of x4
        x5, self._states[4] = self.R2a(x4, self._states[4])
        # Skip R2b

        flow = self.pred(x5)  # [B, 2, H, W]

        # For FX tracing / ONNX export, return flow tensor directly
        if not return_dict:
            return flow

        # log activity
        # Skip logging during FX tracing to avoid control flow issues
        activity = None
        if isinstance(log, bool) and log and not self.exporting:
            activity = {}
            name = [
                "0:input",
                "1:head",
                "2:G1",
                "3:R1a",
                "4:G2",
                "5:R2a",
                "6:pred",
            ]
            for n, l in zip(name, [x, x1, x2, x3, x4, x5, flow]):
                activity[n] = l.detach().ne(0).float().mean().item()

        return {"flow": [flow], "activity": activity}


class LIFFireFlowNet(BaseModel):
    """
    Spiking FireFlowNet architecture to investigate the power of implicit recurrency in SNNs.
    Uses feed-forward LIF neurons instead of recurrent neurons.
    """

    head_neuron = SNNtorch_ConvLIF
    ff_neuron = SNNtorch_ConvLIF
    rec_neuron = SNNtorch_ConvLIF  # Feed-forward instead of recurrent
    residual = False
    num_recurrent_units = 7
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = 0.01

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        self.exporting = unet_kwargs.get("exporting", False)  # Extract export mode flag
        
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        # Create layers
        if hasattr(self, '_create_layers'):
            self._create_layers(unet_kwargs)
        else:
            self._create_default_layers(unet_kwargs)

        self.reset_states()
    
    def _create_default_layers(self, unet_kwargs):
        """Create default layers."""
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        ff_act, rec_act = unet_kwargs["activations"]
        
        quantization_config = unet_kwargs.get("quantization", {})
        exporting = self.exporting  # Use the stored export mode flag

        self.head = self.head_neuron(self.num_bins, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting)

        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R1b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R2b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )

        self.pred = ConvLayer(
            base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred, quantization_config=quantization_config
        )

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def init_cropping(self, width, height):
        pass

    def forward(self, event_voxel=None, event_cnt=None, log=False, return_dict=True):
        """
        :param event_voxel: N x num_bins x H x W (only used when exporting=False)
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity (disabled during export)
        :param return_dict: if True, return dict; if False, return flow tensor directly (for ONNX export)
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor,
                 or just the flow tensor if return_dict=False.
        """

        # input encoding
        if self.exporting:
            # Export mode: only support cnt encoding
            if self.encoding == "cnt" and self.num_bins == 2:
                x = event_cnt
            else:
                print("Model error: Incorrect input encoding for export mode.")
                raise AttributeError
        else:
            # Normal mode: support both voxel and cnt encoding
            if self.encoding == "voxel":
                x = event_voxel
            elif self.encoding == "cnt" and self.num_bins == 2:
                x = event_cnt
            else:
                print("Model error: Incorrect input encoding.")
                raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # forward pass
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        x4, self._states[3] = self.R1b(x3, self._states[3], residual=x2 if self.residual else 0)

        x5, self._states[4] = self.G2(x4, self._states[4])
        x6, self._states[5] = self.R2a(x5, self._states[5])
        x7, self._states[6] = self.R2b(x6, self._states[6], residual=x5 if self.residual else 0)

        flow = self.pred(x7)  # [B, 2, H, W]

        # For FX tracing / ONNX export, return flow tensor directly
        if not return_dict:
            return flow

        # log activity
        # Skip logging during FX tracing to avoid control flow issues
        activity = None
        if isinstance(log, bool) and log and not self.exporting:
            activity = {}
            name = [
                "0:input",
                "1:head",
                "2:G1",
                "3:R1a",
                "4:R1b",
                "5:G2",
                "6:R2a",
                "7:R2b",
                "8:pred",
            ]
            for n, l in zip(name, [x, x1, x2, x3, x4, x5, x6, x7, flow]):
                activity[n] = l.detach().ne(0).float().mean().item()

        return {"flow": [flow], "activity": activity}


class LIFFireFlowNet_short(BaseModel):
    """
    Shortened spiking FireFlowNet architecture to investigate the power of implicit recurrency in SNNs.
    Uses feed-forward LIF neurons. R1b and R2b layers removed.
    """

    head_neuron = SNNtorch_ConvLIF
    ff_neuron = SNNtorch_ConvLIF
    rec_neuron = SNNtorch_ConvLIF  # Feed-forward instead of recurrent
    residual = False
    num_recurrent_units = 5  # Reduced from 7 to 5
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = 0.01

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        self.exporting = unet_kwargs.get("exporting", False)  # Extract export mode flag
        
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        # Create layers
        if hasattr(self, '_create_layers'):
            self._create_layers(unet_kwargs)
        else:
            self._create_default_layers(unet_kwargs)

        self.reset_states()
    
    def _create_default_layers(self, unet_kwargs):
        """Create default layers."""
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        ff_act, rec_act = unet_kwargs["activations"]
        
        quantization_config = unet_kwargs.get("quantization", {})
        exporting = self.exporting  # Use the stored export mode flag

        self.head = self.head_neuron(
            self.num_bins, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        
        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        # R1b removed

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config, exporting=exporting
        )
        # R2b removed

        self.pred = ConvLayer(
            base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred, quantization_config=quantization_config
        )

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def init_cropping(self, width, height):
        pass

    def forward(self, event_voxel=None, event_cnt=None, log=False, return_dict=True):
        """
        :param event_voxel: N x num_bins x H x W (only used when exporting=False)
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity (disabled during export)
        :param return_dict: if True, return dict; if False, return flow tensor directly (for ONNX export)
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor,
                 or just the flow tensor if return_dict=False.
        """

        # input encoding
        if self.exporting:
            # Export mode: only support cnt encoding
            if self.encoding == "cnt" and self.num_bins == 2:
                x = event_cnt
            else:
                print("Model error: Incorrect input encoding for export mode.")
                raise AttributeError
        else:
            # Normal mode: support both voxel and cnt encoding
            if self.encoding == "voxel":
                x = event_voxel
            elif self.encoding == "cnt" and self.num_bins == 2:
                x = event_cnt
            else:
                print("Model error: Incorrect input encoding.")
                raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # forward pass (R1b and R2b removed)
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        # Skip R1b

        x4, self._states[3] = self.G2(x3, self._states[3])  # G2 now takes x3
        x5, self._states[4] = self.R2a(x4, self._states[4])
        # Skip R2b

        flow = self.pred(x5)  # [B, 2, H, W]

        # For FX tracing / ONNX export, return flow tensor directly
        if not return_dict:
            return flow

        # log activity
        # Skip logging during FX tracing to avoid control flow issues
        activity = None
        if isinstance(log, bool) and log and not self.exporting:
            activity = {}
            name = [
                "0:input",
                "1:head",
                "2:G1",
                "3:R1a",
                "4:G2",
                "5:R2a",
                "6:pred",
            ]
            for n, l in zip(name, [x, x1, x2, x3, x4, x5, flow]):
                activity[n] = l.detach().ne(0).float().mean().item()

        return {"flow": [flow], "activity": activity}


class SpikingRecEVFlowNet(BaseModel):
    """
    Spiking recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = SpikingMultiResUNetRecurrent
    recurrent_block_type = "lif"
    spiking_feedforward_block_type = "lif"

    def __init__(self, unet_kwargs):
        super().__init__()

        norm = None
        use_upsample_conv = True
        if "norm" in unet_kwargs.keys():
            norm = unet_kwargs["norm"]
        if "use_upsample_conv" in unet_kwargs.keys():
            use_upsample_conv = unet_kwargs["use_upsample_conv"]

        RecEVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": norm,
            "use_upsample_conv": use_upsample_conv,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "recurrent_block_type": self.recurrent_block_type,
            "final_activation": "tanh",
            "spiking_feedforward_block_type": self.spiking_feedforward_block_type,
            "spiking_neuron": unet_kwargs["spiking_neuron"],
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = RecEVFlowNet_kwargs["num_encoders"]

        unet_kwargs.update(RecEVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("mask_output", None)

        self.multires_unetrec = self.unet_type(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.multires_unetrec.states)

    @states.setter
    def states(self, states):
        self.multires_unetrec.states = states

    def detach_states(self):
        detached_states = []
        for state in self.multires_unetrec.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.multires_unetrec.states = detached_states

    def reset_states(self):
        self.multires_unetrec.states = [None] * self.multires_unetrec.num_states

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H x W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt" and self.num_bins == 2:
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = self.multires_unetrec.forward(x)

        # log activity
        if log:
            raise NotImplementedError("Activity logging not implemented")
        else:
            activity = None

        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[2] / flow.shape[2],
                        multires_flow[-1].shape[3] / flow.shape[3],
                    ),
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        return {"flow": flow_list, "activity": activity}


# For ONNX export, swap LIF neurons for dummy ReLU modules
from .SNNtorch_spiking_submodules import SNNtorch_ConvReLU, SNNtorch_ConvReLURecurrent


# Utility classes for custom LIF layers and ONNX export

class LIF(torch.nn.Module):
    """
    Minimal single-layer LIF model using snn.Leaky.
    Input: N x channels x H x W
    Output: N x channels x H x W (spikes)
    """
    def __init__(self, channels=4, leak=(0.0, 1.0), thresh=(0.0, 0.8), use_custom_op=False):
        super().__init__()
        # Per-channel learnable parameters
        self.beta = torch.nn.Parameter(torch.empty(channels, 1, 1).uniform_(leak[0], leak[1]))
        self.threshold = torch.nn.Parameter(torch.empty(channels, 1, 1).uniform_(thresh[0], thresh[1]))
        # Internal membrane potential (non-learnable state). Register as buffer so it's saved with state_dict.
        # Initialized as None; will be created on first forward with the same shape as the input.
        self.lif = snn.Leaky(beta=self.beta, threshold=self.threshold, reset_mechanism="zero", surrogate_disable=True)
        self.use_custom_op = use_custom_op

    def forward(self, input, mem):
        """
        Forward uses an internal membrane potential buffer `self.mem`.
        - Input: input (tensor)
        - Output: spk (tensor)

        The internal `self.mem` is created/reset to zeros with the same shape as `input` if it is None
        or if its shape does not match `input` (e.g., different batch size).
        After computing the new membrane potential, it is stored in `self.mem` and only the
        spikes tensor `spk` is returned.
        """        

        if self.use_custom_op:
            # Use custom operator for ONNX export
            out = torch.ops.SNN_implementation.LIF(input, mem, self.beta, self.threshold)
            spk = out[0]  # shape [N, C, H, W]
            mem_out = out[1]  # shape [N, C, H, W]

            return spk, mem_out
        else:
            # Use snn.Leaky for training/inference
            self.lif.threshold.data.clamp_(min=0.01)
            spk, mem_out = self.lif(input, mem)

            return spk, mem_out


class ConvLIF(torch.nn.Module):
    """
    Single convolutional LIF layer model for ONNX export.
    Input: N x input_channels x H x W
    Output: N x hidden_channels x H x W (spikes)
    
    Wraps custom_ConvLIF to provide a simple model interface for export.
    """
    def __init__(self, input_channels=2, hidden_channels=256, kernel_size=3, 
                 leak=(0.0, 1.0), thresh=(0.0, 0.8), use_custom_op=True):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.use_custom_op = use_custom_op

        # Create quantization config (disabled by default)
        quantization_config = {"enabled": False}
        
        # Create three ConvLIF layers stacked sequentially.
        self.conv_lif1 = custom_ConvLIF(
            input_size=input_channels,
            hidden_size=hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            leak=leak,
            thresh=thresh,
            learn_leak=True,
            learn_thresh=True,
            hard_reset=True,
            detach=True,
            norm=None,
            quantization_config=quantization_config,
        )

        self.conv_lif2 = custom_ConvLIF(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            leak=leak,
            thresh=thresh,
            learn_leak=True,
            learn_thresh=True,
            hard_reset=True,
            detach=True,
            norm=None,
            quantization_config=quantization_config,
        ) 

        self.conv_lif3 = custom_ConvLIF(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            leak=leak,
            thresh=thresh,
            learn_leak=True,
            learn_thresh=True,
            hard_reset=True,
            detach=True,
            norm=None,
            quantization_config=quantization_config,
        ) 

        self.conv = ConvLayer(
            hidden_channels, out_channels=2, kernel_size=1, activation="tanh", quantization_config=quantization_config
        )

        # State for the stacked layers
        self.mem = [None, None, None]
    
    def reset_states(self):
        """Reset the internal state to None."""
        self.mem = [None, None, None]

    def forward(self, input, mem=None):
        # Layer 1: conv1 -> LIF1
        ff1 = self.conv_lif1.ff(input)
        
        # Use init_mem from the layer itself for initialization
        mem1 = self.conv_lif1.init_mem
        
        # Apply LIF1
        out1 = torch.ops.SNN_implementation.LIF(ff1, mem1, self.conv_lif1.beta, self.conv_lif1.threshold)
        spk1 = out1[0]  # spikes from layer 1
        mem_out1 = out1[1]  # membrane state from layer 1

        pred = self.conv(spk1)

        return pred
