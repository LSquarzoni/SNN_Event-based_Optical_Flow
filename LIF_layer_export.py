import torch
import numpy as np
import os
from models.model import LIF
import onnx
from onnxsim import simplify
from torch.onnx import register_custom_op_symbolic
import sys

def lif_leaky_symbolic(g, input, mem, beta, threshold):
    return g.op("SNN_implementation::LIF", input, mem, beta, threshold, outputs=2)

# Settings for dummy input and model
batch_size = 1
channels = 4  # hidden_size for LIF
height = 32
width = 32

# Load the custom LIF operator
register_custom_op_symbolic('SNN_implementation::LIF', lif_leaky_symbolic, 11)
# Ensure Torch can find its own libs when loading the custom op
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"
# Load the compiled custom op shared library
custom_op_path = os.path.join('ONNX_LIF_operator', 'build', 'lib.linux-x86_64-cpython-39', 'lif_op.cpython-39-x86_64-linux-gnu.so')
torch.ops.load_library(custom_op_path)

os.makedirs("exported_models", exist_ok=True)

# Create dummy input (N x channels x H x W) with strictly positive values
eps = 1e-6
dummy_input = torch.rand(batch_size, channels, height, width) + eps

# Save example input for reference
np.savez('exported_models/inputs.npz',
         input=dummy_input.cpu().numpy())

# Initialize LIF model
model = LIF(channels=channels, use_custom_op=True)
model.eval()

# Run model to get output
with torch.no_grad():
    spk = model(dummy_input)

# Save example output for reference
np.savez('exported_models/outputs.npz',
         spk=spk.cpu().numpy())

# Export to ONNX
onnx_path = "exported_models/network.onnx"
onnx_simpler_path = "exported_models/network_simpler.onnx"
torch.onnx.export(
    model,
    (dummy_input),
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['x'],
    output_names=['spk'],
    custom_opsets={"SNN_implementation": 11}
)

# Verify the exported model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print(f"ONNX model exported successfully and verified!")

simpler_model, check = simplify(onnx_model)
if check:
    onnx.save(simpler_model, onnx_simpler_path)
    print("Simplified ONNX model saved successfully!")

# Print model info
print(f"Model size: {len(onnx_model.SerializeToString()) / 1024 / 1024:.2f} MB")