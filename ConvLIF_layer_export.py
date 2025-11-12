import torch
import numpy as np
import os
from models.model import ConvLIF
import onnx
from onnxsim import simplify
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper as sym_help
import sys

def lif_leaky_symbolic(g, input, mem, beta, threshold):
    return g.op("SNN_implementation::LIF", input, mem, beta, threshold, outputs=2)

def set_valueinfo_shape(value_proto, dims):
    tt = value_proto.type.tensor_type
    # clear existing dims
    tt.shape.dim.clear()
    for d in dims:
        di = tt.shape.dim.add()
        di.dim_value = int(d)

# Settings for dummy input and model
batch_size = 1
input_channels = 2  # Input channels (e.g., event voxel bins or cnt encoding)
hidden_channels = 256  # Hidden channels for ConvLIF layer
height = 8
width = 8
kernel_size = 3

pred_shape = [batch_size, input_channels, height, width]
in_shape = [batch_size, input_channels, height, width]

# Load the custom LIF operators
register_custom_op_symbolic('SNN_implementation::LIF', lif_leaky_symbolic, 11)
# Ensure Torch can find its own libs when loading the custom op
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"
# Load the compiled custom op shared library
custom_op_path = os.path.join('ONNX_LIF_operator', 'build', 'lib.linux-x86_64-cpython-39', 'lif_op.cpython-39-x86_64-linux-gnu.so')
torch.ops.load_library(custom_op_path)

os.makedirs("exported_models", exist_ok=True)

# Create dummy input (N x input_channels x H x W) with strictly positive values
eps = 1e-6
dummy_input = torch.rand(*in_shape) + eps

# Save example input for reference
numpy_inputs_path = 'exported_models/inputs.npz'
np.savez(numpy_inputs_path,
         input=dummy_input.cpu().numpy())

# Initialize ConvLIF model
model = ConvLIF(
    input_channels=input_channels,
    hidden_channels=hidden_channels,
    kernel_size=kernel_size,
    leak=(0.0, 1.0),
    thresh=(0.0, 0.8),
    use_custom_op=True
)
model.eval()

# Run model to get output
# Register hooks on Conv2d modules to capture their output shapes
conv_pred_shapes = []
hooks = []
def make_hook():
    def hook(module, inp, pred):
        o = pred
        try:
            if isinstance(o, torch.Tensor):
                conv_pred_shapes.append(tuple(o.shape))
            elif isinstance(o, (list, tuple)) and len(o) > 0 and isinstance(o[0], torch.Tensor):
                conv_pred_shapes.append(tuple(o[0].shape))
        except Exception:
            pass
    return hook

for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        hooks.append(m.register_forward_hook(make_hook()))

with torch.no_grad():
    pred = model(dummy_input)

# Remove hooks
for h in hooks:
    try:
        h.remove()
    except Exception:
        pass

# Save example output for reference
numpy_outputs_path = 'exported_models/outputs.npz'
np.savez(numpy_outputs_path,
         pred=pred.cpu().numpy())

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {pred.shape}")

# Get actual output shape from runtime execution
actual_pred_shape = list(pred.shape)

# Export to ONNX
onnx_path = "exported_models/network.onnx"
onnx_simpler_path = "exported_models/network_simpler.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['pred'],
    custom_opsets={"SNN_implementation": 11}
)

# Verify the exported model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print(f"ONNX model exported successfully and verified!")

# try to update graph.output entries; if not found, add to value_info
# Use actual runtime shapes instead of predefined shapes
names_to_fix = ["pred"]
shapes_to_fix = [actual_pred_shape]
for name, shape in zip(names_to_fix, shapes_to_fix):
    found = False
    for pred in onnx_model.graph.output:
        if pred.name == name:
            set_valueinfo_shape(pred, shape)
            found = True
            break
    if not found:
        # add a new graph.output ValueInfo (if exporter didn't create it as expected)
        vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        onnx_model.graph.output.append(vi)

# Also ensure Conv node outputs have explicit value_info shapes
try:
    # Build conv shapes from runtime and map them onto Conv nodes in the ONNX graph
    conv_shapes = [list(s) for s in conv_pred_shapes]
    conv_node_outputs = []
    for n in onnx_model.graph.node:
        if n.op_type == 'Conv':
            if len(n.output) > 0:
                conv_node_outputs.append(n.output[0])

    # Assign shapes from conv_shapes list to conv outputs in graph order
    for i, out_name in enumerate(conv_node_outputs):
        target_shape = None
        if i < len(conv_shapes):
            target_shape = conv_shapes[i]
        elif len(conv_shapes) > 0:
            target_shape = conv_shapes[-1]
        if target_shape is not None:
            # update or append value_info
            found = False
            for vi in onnx_model.graph.value_info:
                if vi.name == out_name:
                    set_valueinfo_shape(vi, target_shape)
                    found = True
                    break
            if not found:
                vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, target_shape)
                onnx_model.graph.value_info.append(vi)

    # Build mapping conv_output_name -> shape for quick lookup
    conv_output_shape_by_name = {}
    for i, out_name in enumerate(conv_node_outputs):
        if i < len(conv_shapes):
            conv_output_shape_by_name[out_name] = conv_shapes[i]
        elif len(conv_shapes) > 0:
            conv_output_shape_by_name[out_name] = conv_shapes[-1]

    # 1) Try shape inference now that conv outputs have shapes. This should populate Add outputs, etc.
    try:
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        # Replace model with inferred one for downstream edits
        onnx_model = inferred_model
    except Exception:
        # If inference fails, continue with manual heuristics
        pass

    # 2) Build a lookup of all known value_info shapes (post-inference if succeeded)
    value_info_shape_by_name = {}
    def collect_shapes_from_value_info(model):
        for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
            if not hasattr(vi, 'type'):
                continue
            tt = vi.type.tensor_type
            if not hasattr(tt, 'shape'):
                continue
            dims = []
            for d in tt.shape.dim:
                if getattr(d, 'dim_value', 0):
                    dims.append(int(d.dim_value))
                else:
                    dims = []
                    break
            if dims:
                value_info_shape_by_name[vi.name] = dims

    collect_shapes_from_value_info(onnx_model)

    # 3) For each LIF node, set outputs to match the shape of its first input.
    # Prefer shapes from value_info (post-inference). Fall back to conv mapping and name heuristics.
    def find_shape_for_input_name(name):
        # direct value_info mapping first
        if name in value_info_shape_by_name:
            return value_info_shape_by_name[name]
        # conv direct mapping
        if name in conv_output_shape_by_name:
            return conv_output_shape_by_name[name]
        # try suffix and last-segment matches against conv outputs
        for k, v in conv_output_shape_by_name.items():
            if k.endswith(name) or name.endswith(k):
                return v
        base = name.split('/')[-1]
        for k, v in conv_output_shape_by_name.items():
            if k.split('/')[-1] == base:
                return v
        return None

    for n in onnx_model.graph.node:
        if (n.domain == 'SNN_implementation') or (n.op_type == 'LIF') or ('LIF' in n.op_type):
            if len(n.input) > 0:
                inp0 = n.input[0]
                target_shape = find_shape_for_input_name(inp0)
                if target_shape is not None:
                    for out_name in n.output:
                        updated = False
                        for vi in onnx_model.graph.value_info:
                            if vi.name == out_name:
                                set_valueinfo_shape(vi, target_shape)
                                updated = True
                                break
                        if not updated:
                            vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, target_shape)
                            onnx_model.graph.value_info.append(vi)
except Exception as e:
    print(f"Warning: failed to set Conv/LIF shapes: {e}")

# Simplify the model
try:
    simpler_model, check = simplify(onnx_model)
    if check:
        onnx.save(simpler_model, onnx_simpler_path)
        print("Simplified ONNX model saved successfully!")
    else:
        print("Simplification check failed, saving original model")
        onnx.save(onnx_model, onnx_simpler_path)
except Exception as e:
    print(f"Simplification failed: {e}")
    print("Saving original model instead")
    onnx.save(onnx_model, onnx_simpler_path)

# Print model info
print(f"Model size: {len(onnx_model.SerializeToString()) / 1024 / 1024:.2f} MB")
print(f"\nModel exported successfully!")
print(f"  - Original: {onnx_path}")
print(f"  - Simplified: {onnx_simpler_path}")
print(f"  - Inputs saved: {numpy_inputs_path}")
print(f"  - Outputs saved: {numpy_outputs_path}")
