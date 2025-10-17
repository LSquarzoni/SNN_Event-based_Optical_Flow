"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch
import snntorch as snn
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
from .submodules import ConvGRU, ConvLayer, ConvLayer_, ConvRecurrent
from .unet import (
    UNetRecurrent,
    MultiResUNet,
    MultiResUNetRecurrent,
    SpikingMultiResUNetRecurrent,
)


class FireNet(BaseModel):
    """
    FireNet architecture (adapted for optical flow estimation), as described in the paper "Fast Image
    Reconstruction with an Event Camera", Scheerlinck et al., WACV 2020.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvGRU
    residual = False
    num_recurrent_units = 7
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = None

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        # Create layers (can be overridden by subclasses)
        if hasattr(self, '_create_layers'):
            self._create_layers(unet_kwargs)
        else:
            self._create_default_layers(unet_kwargs)

        self.reset_states()
    
    def _create_default_layers(self, unet_kwargs):
        """Create default layers (original implementation)."""
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        ff_act, rec_act = unet_kwargs["activations"]
        
        quantization_config = unet_kwargs.get("quantization", {})

        self.head = self.head_neuron(self.num_bins, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[0], quantization_config=quantization_config)

        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=rec_act, **self.kwargs[1], quantization_config=quantization_config
        )
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[2], quantization_config=quantization_config
        )
        self.R1b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[3], quantization_config=quantization_config
        )

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=rec_act, **self.kwargs[4], quantization_config=quantization_config
        )
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[5], quantization_config=quantization_config
        )
        self.R2b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[6], quantization_config=quantization_config
        )

        self.pred = ConvLayer(
            base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred, quantization_config=quantization_config
            #base_num_channels, out_channels=2, kernel_size=1, activation=None, w_scale=self.w_scale_pred, quantization_config=quantization_config
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

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
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

        # forward pass
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        x4, self._states[3] = self.R1b(x3, self._states[3], residual=x2 if self.residual else 0)

        x5, self._states[4] = self.G2(x4, self._states[4])
        x6, self._states[5] = self.R2a(x5, self._states[5])
        x7, self._states[6] = self.R2b(x6, self._states[6], residual=x5 if self.residual else 0)

        # Predict dense flow then pool to a single (x, y) pair per sample
        flow = self.pred(x7)  # [B, 2, H, W]
        #flow = torch.nn.functional.adaptive_avg_pool2d(flow, (1, 1))  # [B, 2, 1, 1]

        # log activity
        if log:
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
        else:
            activity = None

        return {"flow": [flow], "activity": activity}
    
    
class FireNet_short(BaseModel):
    """
    Shortened FireNet architecture with R1b and R2b layers removed.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvGRU
    residual = False
    num_recurrent_units = 5  # Reduced from 7 to 5
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = None

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        # Create layers (can be overridden by subclasses)
        if hasattr(self, '_create_layers'):
            self._create_layers(unet_kwargs)
        else:
            self._create_default_layers(unet_kwargs)

        self.reset_states()
    
    def _create_default_layers(self, unet_kwargs):
        """Create default layers (original implementation)."""
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        ff_act, rec_act = unet_kwargs["activations"]
        
        quantization_config = unet_kwargs.get("quantization", {})

        self.head = self.head_neuron(
            self.num_bins, base_num_channels, kernel_size, quantization_config=quantization_config
        )
        
        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config
        )
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config
        )
        # R1b removed

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config
        )
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, quantization_config=quantization_config
        )
        # R2b removed

        self.pred = ConvLayer(
            base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred, quantization_config=quantization_config
            #base_num_channels, out_channels=2, kernel_size=1, activation=None, w_scale=self.w_scale_pred, quantization_config=quantization_config
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

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
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

        # forward pass (R1b and R2b removed)
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        # Skip R1b

        x4, self._states[3] = self.G2(x3, self._states[3])  # G2 now takes x3 instead of x4
        x5, self._states[4] = self.R2a(x4, self._states[4])
        # Skip R2b

        # Predict dense flow then pool to a single (x, y) pair per sample
        flow = self.pred(x5)  # [B, 2, H, W]
        #flow = torch.nn.functional.adaptive_avg_pool2d(flow, (1, 1))  # [B, 2, 1, 1]

        # log activity
        if log:
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
        else:
            activity = None

        return {"flow": [flow], "activity": activity}


class EVFlowNet(BaseModel):
    """
    EV-FlowNet architecture, as described in the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    def __init__(self, unet_kwargs):
        super().__init__()

        EVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": None,
            "use_upsample_conv": True,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "final_activation": "tanh",
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = EVFlowNet_kwargs["num_encoders"]

        unet_kwargs.update(EVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("eval", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("spiking_neuron", None)

        self.multires_unet = MultiResUNet(unet_kwargs)

    def detach_states(self):
        pass

    def reset_states(self):
        pass

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
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
        multires_flow = self.multires_unet.forward(x)

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


class FireFlowNet(FireNet):
    """
    EV-FireFlowNet architecture, as described in the paper "Back to Event Basics: Self
    Supervised Learning of Image Reconstruction from Event Data via Photometric Constancy",
    Paredes-Valles et al., CVPR 2021.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvLayer_
    residual = False
    w_scale_pred = 0.01
    
    
class FireFlowNet_short(FireNet_short):
    """
    EV-FireFlowNet architecture, as described in the paper "Back to Event Basics: Self
    Supervised Learning of Image Reconstruction from Event Data via Photometric Constancy",
    Paredes-Valles et al., CVPR 2021.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvLayer_
    residual = False
    w_scale_pred = 0.01


class RecEVFlowNet(BaseModel):
    """
    Recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = MultiResUNetRecurrent
    recurrent_block_type = "convgru"
    spiking_feedforward_block_type = None

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


class SpikingRecEVFlowNet(RecEVFlowNet):
    """
    Spiking recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    unet_type = SpikingMultiResUNetRecurrent
    recurrent_block_type = "lif"
    spiking_feedforward_block_type = "lif"



# For ONNX export, swap LIF neurons for dummy ReLU modules
from .SNNtorch_spiking_submodules import SNNtorch_ConvReLU, SNNtorch_ConvReLURecurrent

class LIFFireNet(FireNet):
    """
    Spiking FireNet architecture of LIF neurons for dense optical flow estimation from events.
    """
    head_neuron = custom_ConvLIF
    ff_neuron = custom_ConvLIF
    rec_neuron = custom_ConvLIFRecurrent
    residual = False
    w_scale_pred = 0.01


class LIFFireNet_short(FireNet_short):
    """
    Shortened spiking FireNet architecture of LIF neurons with R1b and R2b layers removed.
    """
    head_neuron = custom_ConvLIF
    ff_neuron = custom_ConvLIF
    rec_neuron = custom_ConvLIFRecurrent
    residual = False
    w_scale_pred = 0.01


class LIFFireFlowNet(FireNet):
    """
    Spiking FireFlowNet architecture to investigate the power of implicit recurrency in SNNs.
    """

    head_neuron = custom_ConvLIF
    ff_neuron = custom_ConvLIF
    rec_neuron = custom_ConvLIF
    residual = False
    w_scale_pred = 0.01
    
    
class LIFFireFlowNet_short(FireNet_short):
    """
    Spiking FireFlowNet architecture to investigate the power of implicit recurrency in SNNs.
    """

    head_neuron = custom_ConvLIF
    ff_neuron = custom_ConvLIF
    rec_neuron = custom_ConvLIF
    residual = False
    w_scale_pred = 0.01
    
    
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
    

class LIF_stateful(torch.nn.Module):
    """
    Stateful single-layer LIF using custom op with internal persistent membrane.
    Input: N x channels x H x W
    Output: N x channels x H x W (spikes)
    The internal membrane is keyed by state_id and persists across forward calls.
    If mem_in is provided (tensor), it overrides the internal state for that call.
    """
    def __init__(self, channels=4, leak=(0.0, 1.0), thresh=(0.0, 0.8), state_id=None, use_custom_op=True):
        super().__init__()
        self.channels = channels
        # Auto-generate a unique state id per module if not provided
        self.state_id = int(state_id) if state_id is not None else (id(self) & 0x7FFF_FFFF_FFFF_FFFF)
        self.use_custom_op = use_custom_op
        # beta and threshold per-channel
        self.register_buffer("beta", torch.linspace(leak[0], leak[1], steps=channels))
        self.register_buffer("threshold", torch.linspace(thresh[0], thresh[1], steps=channels))

    def reset_state(self, mem_init=None, shape_hint=None, device=None, dtype=None):
        """
        Optionally prime the internal state by calling the op once with mem_in.
        shape_hint: tuple (N,C,H,W) to create zeros if mem_init is None.
        """
        if not self.use_custom_op:
            return
        if mem_init is None:
            if shape_hint is None:
                return
            N, C, H, W = shape_hint
            device = device or (self.beta.device)
            dtype = dtype or torch.float32
            mem_init = torch.zeros((N, C, H, W), device=device, dtype=dtype)
        dummy_inp = torch.zeros_like(mem_init)
        # Calling once with mem_in seeds internal state.
        _ = torch.ops.SNN_implementation.LIF_stateful(dummy_inp, mem_init, self.beta, self.threshold, self.state_id)

    def forward(self, input, mem_in=None):
        if self.use_custom_op:
            # mem_in can be None to reuse internal state
            return torch.ops.SNN_implementation.LIF_stateful(input, mem_in, self.beta, self.threshold, self.state_id)
        else:
            # Fallback minimal python behavior (stateless): if mem_in is None, use zeros
            if mem_in is None:
                mem_in = torch.zeros_like(input)
            beta = self.beta.view(1, -1, 1, 1)
            thr = self.threshold.view(1, -1, 1, 1)
            mem = beta * mem_in + input
            spk = (mem >= thr).to(input.dtype)
            # do not persist in fallback path; return spikes only to match API
            return spk
        
        
class SNNtorch_FCLIF(torch.nn.Module):
    """
    Fully connected spiking LIF small model using SNNTorch Leaky neuron.
    """
    def __init__(self, leak=(0.0, 1.0), thresh=(0.0, 0.8), detach=True):
        super().__init__()
        self.detach = detach
        # Fixed input resolution: C=2, H=8, W=8
        self.in_features = 2 * 8 * 8  # 128
        self.out_features = 2 # only one x and y flow output
        self.leak = leak
        self.thresh = thresh

        # Layers defined up-front to register parameters before optimizer creation
        self.fc1 = torch.nn.Linear(self.in_features, self.in_features, bias=False)
        self.lif1 = snn.Leaky(
            beta=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.leak[0], self.leak[1])),
            threshold=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.thresh[0], self.thresh[1])),
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="zero",
            reset_delay=False,
            spike_grad=snn.surrogate.atan(alpha=2),
        )
        self.fc2 = torch.nn.Linear(self.in_features, self.in_features, bias=False)
        self.lif2 = snn.Leaky(
            beta=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.leak[0], self.leak[1])),
            threshold=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.thresh[0], self.thresh[1])),
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="zero",
            reset_delay=False,
            spike_grad=snn.surrogate.atan(alpha=2),
        )
        #self.fc3 = torch.nn.Linear(self.in_features, self.in_features, bias=False)
        self.fc3 = torch.nn.Linear(self.in_features, self.out_features, bias=False)

        # LIF states
        self.states = [None, None]

    def reset_states(self):
        self.states = [None, None]

    def forward(self, input_, prev_state=None, log=False):
        # input_: [B, C, H, W]
        B, C, H, W = input_.shape
        flattened_size = C * H * W
        # Sanity check for expected input size
        if flattened_size != self.in_features:
            raise ValueError(f"SNNtorch_FCLIF expects C*H*W={self.in_features}, got {flattened_size} (C={C}, H={H}, W={W}).")

        # Reshape input to [B, C*H*W]
        x = input_.reshape(B, self.in_features)

        # Use prev_state if provided, otherwise use self.states
        if prev_state is not None:
            mem1, mem2 = prev_state
        else:
            mem1, mem2 = self.states

        # Block 1: FC -> LIF
        x1 = self.fc1(x)
        spk1, mem1 = self.lif1(x1, mem1)
        if self.detach:
            self.lif1.detach_hidden()

        # Block 2: FC -> LIF
        x2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(x2, mem2)
        if self.detach:
            self.lif2.detach_hidden()

        # Block 3: FC
        out = self.fc3(spk2)

        # Reshape output
        #out = out.reshape(B, C, H, W)
        out = out.reshape(B, C, 1, 1)

        # Always update self.states with new membrane values
        self.states = [mem1, mem2]

        new_state = [mem1, mem2]

        return {"flow": [out], "state": new_state}


class SNNtorch_ConvFCLIF(torch.nn.Module):
    """
    Fully connected spiking LIF model with an initial 2D convolutional layer.
    """
    def __init__(self, leak=(0.0, 1.0), thresh=(0.0, 0.8), detach=True):
        super().__init__()
        self.detach = detach
        # Fixed input resolution: C=2, H=8, W=8
        self.in_features = 2 * 8 * 8  # 128 after flatten
        self.out_features = 2 # only one x and y flow output
        self.leak = leak
        self.thresh = thresh

        # Define layers up-front
        self.conv = torch.nn.Conv2d(2, 2, kernel_size=3, padding=1)
        # LIF after flattening conv output
        self.lif_conv = snn.Leaky(
            beta=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.leak[0], self.leak[1])),
            threshold=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.thresh[0], self.thresh[1])),
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="zero",
            reset_delay=False,
            spike_grad=snn.surrogate.atan(alpha=2),
        )
        self.fc1 = torch.nn.Linear(self.in_features, self.in_features, bias=False)
        self.lif1 = snn.Leaky(
            beta=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.leak[0], self.leak[1])),
            threshold=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.thresh[0], self.thresh[1])),
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="zero",
            reset_delay=False,
            spike_grad=snn.surrogate.atan(alpha=2),
        )
        self.fc2 = torch.nn.Linear(self.in_features, self.in_features, bias=False)
        self.lif2 = snn.Leaky(
            beta=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.leak[0], self.leak[1])),
            threshold=torch.nn.Parameter(torch.empty(self.in_features).uniform_(self.thresh[0], self.thresh[1])),
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="zero",
            reset_delay=False,
            spike_grad=snn.surrogate.atan(alpha=2),
        )
        #self.fc3 = torch.nn.Linear(self.in_features, self.in_features, bias=False)
        self.fc3 = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        # LIF states
        self.states = [None, None, None]

    def reset_states(self):
        self.states = [None, None, None]

    def forward(self, input_, prev_state=None, log=False):
        # input_: [B, C, H, W]
        B, C, H, W = input_.shape
        flattened_size = C * H * W
        # Sanity check for expected input size
        if flattened_size != self.in_features or C != 2:
            raise ValueError(f"SNNtorch_ConvFCLIF expects C=2 and C*H*W={self.in_features}, got C={C}, H={H}, W={W}.")

        # Use prev_state if provided, otherwise use self.states
        if prev_state is not None:
            mem_conv, mem1, mem2 = prev_state
        else:
            mem_conv, mem1, mem2 = self.states

        # Block 1: CONV2D -> LIF
        x_conv = self.conv(input_)
        # Flatten the output of the convolution
        x_flat = x_conv.reshape(B, self.in_features)  # [B, 128]
        # LIF on flattened features
        spk_conv, mem_conv = self.lif_conv(x_flat, mem_conv)
        if self.detach:
            self.lif_conv.detach_hidden()

        # Block 2: FC -> LIF
        x1 = self.fc1(spk_conv)
        spk1, mem1 = self.lif1(x1, mem1)
        if self.detach:
            self.lif1.detach_hidden()

        # Block 3: FC -> LIF
        x2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(x2, mem2)
        if self.detach:
            self.lif2.detach_hidden()

        # Block 4: FC
        out = self.fc3(spk2)

        # Reshape output
        #out = out.reshape(B, C, H, W)
        out = out.reshape(B, C, 1, 1)

        # Always update self.states with new membrane values
        self.states = [mem_conv, mem1, mem2]

        new_state = [mem_conv, mem1, mem2]

        return {"flow": [out], "state": new_state}