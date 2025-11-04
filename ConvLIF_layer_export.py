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
hidden_channels = 4  # Hidden channels for ConvLIF layer
height = 32
width = 32
kernel_size = 3

# Output shape will be [batch_size, hidden_channels, height, width]
out_shape = [batch_size, hidden_channels, height, width]
# Input shape
in_shape = [batch_size, input_channels, height, width]
# Membrane state shape is [2, batch_size, hidden_channels, height, width]
# The first dim=2 represents [membrane_potential, spikes]
mem_shape = [2, batch_size, hidden_channels, height, width]

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
# Create dummy membrane state [2, N, hidden_channels, H, W]
dummy_mem = torch.rand(*mem_shape) + eps

# Save example input for reference
np.savez('exported_models/convlif_inputs.npz',
         input=dummy_input.cpu().numpy(),
         mem=dummy_mem.cpu().numpy())

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
with torch.no_grad():
    spk, mem_out = model(dummy_input, dummy_mem)

# Save example output for reference
np.savez('exported_models/convlif_outputs.npz',
         spk=spk.cpu().numpy(),
         mem_out=mem_out.cpu().numpy())

print(f"Input shape: {dummy_input.shape}")
print(f"Membrane state shape: {dummy_mem.shape}")
print(f"Output spike shape: {spk.shape}")
print(f"Output membrane state shape: {mem_out.shape}")

# Export to ONNX
onnx_path = "exported_models/convlif_network.onnx"
onnx_simpler_path = "exported_models/convlif_network_simpler.onnx"
torch.onnx.export(
    model,
    (dummy_input, dummy_mem),
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input', 'mem'],
    output_names=['spk', 'mem_out'],
    custom_opsets={"SNN_implementation": 11}
)

# Verify the exported model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print(f"ONNX model exported successfully and verified!")

# try to update graph.output entries; if not found, add to value_info
names_to_fix = ["spk", "mem_out"]
shapes_to_fix = [out_shape, mem_shape]
for name, shape in zip(names_to_fix, shapes_to_fix):
    found = False
    for out in onnx_model.graph.output:
        if out.name == name:
            set_valueinfo_shape(out, shape)
            found = True
            break
    if not found:
        # add a new graph.output ValueInfo (if exporter didn't create it as expected)
        vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        onnx_model.graph.output.append(vi)

# also set any intermediate value_info entries created by custom op with symbolic dim names
for vi in onnx_model.graph.value_info:
    if vi.name.startswith("LIF") or vi.name in names_to_fix:
        try:
            # Try to infer the correct shape based on the name
            if "spk" in vi.name or vi.name == "spk":
                set_valueinfo_shape(vi, out_shape)
            elif "mem" in vi.name or vi.name == "mem_out":
                set_valueinfo_shape(vi, mem_shape)
        except Exception:
            pass

# optional: run shape inference to propagate shapes further
try:
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    print("Ran ONNX shape inference")
except Exception as e:
    print("shape inference failed:", e)

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
print(f"  - Inputs saved: exported_models/convlif_inputs.npz")
print(f"  - Outputs saved: exported_models/convlif_outputs.npz")
