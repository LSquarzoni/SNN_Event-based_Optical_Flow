import torch
import numpy as np
import os
from models.model import LIF
import onnx
from onnxsim import simplify

# Settings for dummy input and model
batch_size = 1
channels = 4  # hidden_size for LIF
height = 32
width = 32

os.makedirs("exported_models", exist_ok=True)

# Create dummy input (N x channels x H x W)
dummy_input = torch.randn(batch_size, channels, height, width)
mem = torch.zeros(batch_size, channels, height, width)

# Save example input for reference
np.savez('exported_models/inputs.npz',
         input=dummy_input.cpu().numpy(),
         mem=mem.cpu().numpy())

# Initialize LIF model
model = LIF(channels=channels)
model.eval()

# Run model to get output
with torch.no_grad():
    spk, mem = model(dummy_input, mem)

# Save example output for reference
np.savez('exported_models/outputs.npz',
         spk=spk.cpu().numpy(),
         mem=mem.cpu().numpy())

# Export to ONNX
onnx_path = "exported_models/network.onnx"
onnx_simpler_path = "exported_models/network_simpler.onnx"
torch.onnx.export(
    model,
    (dummy_input, None),  # LIF expects (x, prev_mem)
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['x', 'prev_mem'],
    output_names=['spk', 'mem'],
    # dynamic_axes={
    #     'x': {0: 'batch_size'},
    #     'spk': {0: 'batch_size'},
    #     'mem': {0: 'batch_size'}
    # }
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