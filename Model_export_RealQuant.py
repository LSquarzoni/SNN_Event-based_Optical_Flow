import argparse
import torch
import onnx
import mlflow
import os
from brevitas import config as cf
from brevitas.export import export_onnx_qcdq, export_qonnx
from brevitas.graph.calibrate import calibration_mode
from onnxsim import simplify
import numpy as np
from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from models.model import LIFFireNet, LIFFireFlowNet, LIFFireNet_short, LIFFireFlowNet_short
from utils.utils import load_model
from torch.onnx import symbolic_helper as sym_help
from DeepQuant.ExportBrevitas import exportBrevitas


def set_valueinfo_shape(value_proto, dims):
    tt = value_proto.type.tensor_type
    # clear existing dims
    tt.shape.dim.clear()
    for d in dims:
        di = tt.shape.dim.add()
        di.dim_value = int(d)

# Ensure Torch can find its own libs when loading the custom op
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"
# Load the compiled custom op shared library FIRST
custom_op_path = os.path.join('ONNX_LIF_operator', 'build', 'lib.linux-x86_64-cpython-311', 'lif_op.cpython-311-x86_64-linux-gnu.so')
torch.ops.load_library(custom_op_path)

# Register ONNX symbolic function for TorchScript-based exporter
from torch.onnx import register_custom_op_symbolic

def lif_leaky_symbolic(g, input, mem, beta, threshold):
    """Symbolic function for TorchScript-based exporter"""
    output = g.op("SNN_implementation::LIF", input, mem, beta, threshold, outputs=2)
    if hasattr(input, 'type'):
        input_type = input.type()
        if isinstance(output, (list, tuple)) and len(output) == 2:
            for out in output:
                if hasattr(out, 'setType'):
                    out.setType(input_type)
    return output

register_custom_op_symbolic('SNN_implementation::LIF', lif_leaky_symbolic, 18)
print("✓ Registered LIF operator for legacy TorchScript ONNX exporter")

cf.IGNORE_MISSING_KEYS = True

def calibrate_model(calibration_loader, quant_model, device, num_batches=1000):
    """Calibrate the quantized model"""
    quant_model = quant_model.to(device)
    quant_model.eval()
    
    with torch.no_grad():
        with calibration_mode(quant_model):
            for i, inputs in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                    
                event_voxel = inputs["event_voxel"].to(device)
                event_cnt = inputs["event_cnt"].to(device)
                
                print(f'Calibration iteration {i+1}/{num_batches}')
                quant_model.reset_states()  # Reset states for each calibration sample
                quant_model(event_voxel, event_cnt)
                
    return quant_model

def export(args, config_parser):    
    mlflow.set_tracking_uri(args.path_mlflow)

    # If runid is 'dummy', skip MLflow and use config as-is
    if args.runid == "dummy":
        config = config_parser.config
    else:
        run = mlflow.get_run(args.runid)
        config = config_parser.merge_configs(run.data.params)

    # Initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # Model initialization
    # Enable quantization for quantized export
    config["model"]["quantization"]["enabled"] = True
    config["model"]["quantization"]["PTQ"] = True
    model_suffix = "_quantized"

    model = eval(config["model"]["name"])(config["model"]).to(device)

    # Only load weights if not dummy
    if args.runid != "dummy":
        model_path_dir = "mlruns/0/models/LIFFN/38/model.pth" # runid: e1965c33f8214d139624d7e08c7ec9c1
        model = load_model(args.runid, model, device, model_path_dir)
        pass

    model.eval()
    
    # Data loader
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=1,
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )
    
    # Calibrate if quantized export
    print("Calibrating quantized model...")
    model = calibrate_model(dataloader, model, device, num_batches=50)
        
    # Reset dataloader for export
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=1,
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )
    
    # Get sample input for export
    with torch.no_grad():
        for inputs in dataloader:
            event_voxel = inputs["event_voxel"].to(device)
            event_cnt = inputs["event_cnt"].to(device)

            # Require at least 10% non-zero values in event_cnt for export
            nonzero_cnt = torch.count_nonzero(event_cnt)
            total_cnt = event_cnt.numel()
            nonzero_ratio = nonzero_cnt.float() / total_cnt
            if nonzero_ratio < 0.1:
                print(f"Skipping sparse input batch for ONNX export (nonzero ratio: {nonzero_ratio:.3f})...")
                continue

            # Reset model states before export (again, for ONNX)
            model.reset_states()

            # Call exportBrevitas with inputs as a dictionary to specify concrete args
            # This tells the tracer that 'log' should be treated as a constant False
            # Use a wrapper function to handle the input correctly
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, event_cnt):
                    # Unpack the tuple input and call model with return_dict=False
                    # to get flow tensor directly without dictionary operations
                    flow = self.model(event_cnt, log=False, return_dict=False)
                    return flow
            
            wrapped_model = ModelWrapper(model)
            # Pass as a single tuple so exportBrevitas can call model(exampleInput)
            exportBrevitas(wrapped_model, event_cnt, debug=True)

            break
    
    # Simplify the exported ONNX model
    print("\nSimplifying ONNX model...")
    onnx_path = "exported_models/4_model_dequant_moved.onnx"

    model_onnx = onnx.load(onnx_path)

    model_simplified, check = simplify(model_onnx)
        
    if check:
        onnx.save(model_simplified, onnx_path)
        print(f"✓ Simplified model saved to {onnx_path}")
    else:
        print(f"⚠ Simplification failed for {onnx_path}")
    
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_MVSEC.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    args = parser.parse_args()

    # Create export directory
    import os
    os.makedirs("exported_models", exist_ok=True)

    export(args, YAMLParser(args.config))