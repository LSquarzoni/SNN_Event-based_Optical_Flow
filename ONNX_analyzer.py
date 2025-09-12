import onnx

# Load the model
model = onnx.load("/home/msc25h1/event_flow/exported_models/LIFFireFlowNet_SNNtorch_fp32_simpler_TEST.onnx")

# Model metadata
print("Producer:", model.producer_name)
print("Version:", model.ir_version)
print("Opset:", model.opset_import)

# Inputs
for input_tensor in model.graph.input:
    print("Input:", input_tensor.name)
    print("  Type:", input_tensor.type)
    # Shape extraction
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    print("  Shape:", shape)

# Outputs
for output_tensor in model.graph.output:
    print("Output:", output_tensor.name)
    print("  Type:", output_tensor.type)
    shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
    print("  Shape:", shape)

# Layers/Nodes
print("Number of layers/operators:", len(model.graph.node))
print("First few layers:")
for node in model.graph.node[:5]:
    print("  OpType:", node.op_type, "| Name:", node.name)