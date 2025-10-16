import argparse
import torch
import onnx
import mlflow
import os
from brevitas import config as cf
from brevitas.export import export_onnx_qcdq
from brevitas.graph.calibrate import calibration_mode
from onnxsim import simplify
import numpy as np
from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from models.model import LIFFireNet, LIFFireFlowNet, LIFFireNet_short, LIFFireFlowNet_short
from utils.utils import load_model
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper as sym_help


def set_valueinfo_shape(value_proto, dims):
    tt = value_proto.type.tensor_type
    # clear existing dims
    tt.shape.dim.clear()
    for d in dims:
        di = tt.shape.dim.add()
        di.dim_value = int(d)

def lif_leaky_symbolic(g, input, mem, beta, threshold):
    return g.op("SNN_implementation::LIF", input, mem, beta, threshold, outputs=2)

# Load the custom LIF operators
register_custom_op_symbolic('SNN_implementation::LIF', lif_leaky_symbolic, 11)
# Ensure Torch can find its own libs when loading the custom op
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"
# Load the compiled custom op shared library
custom_op_path = os.path.join('ONNX_LIF_operator', 'build', 'lib.linux-x86_64-cpython-39', 'lif_op.cpython-39-x86_64-linux-gnu.so')
torch.ops.load_library(custom_op_path)

cf.IGNORE_MISSING_KEYS = True

def calibrate_model(calibration_loader, quant_model, device, num_batches=50):
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


def export_to_onnx(args, config_parser, export_quantized=False):    
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
    if export_quantized:
        # Enable quantization for quantized export
        config["model"]["quantization"]["enabled"] = True
        config["model"]["quantization"]["PTQ"] = True
        model_suffix = "_quantized"
    else:
        # Disable quantization for FP32 export
        config["model"]["quantization"]["enabled"] = False
        model_suffix = "_fp32"

    model = eval(config["model"]["name"])(config["model"]).to(device)

    # Only load weights if not dummy
    if args.runid != "dummy":
        model_path_dir = "mlruns/0/models/LIFFFN/24/model.pth" # runid: cc75ff82496a4dc6896f2464898f774f
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
    if export_quantized:
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

            # Save example input for Deeploy
            np.savez('exported_models/inputs.npz',
                     event_cnt=event_cnt.cpu().numpy().astype(np.float32))

            # Run model to get output
            # Register hooks on Conv2d modules to capture their output shapes
            conv_out_shapes = []
            hooks = []
            def make_hook():
                def hook(module, inp, out):
                    o = out
                    try:
                        if isinstance(o, torch.Tensor):
                            conv_out_shapes.append(tuple(o.shape))
                        elif isinstance(o, (list, tuple)) and len(o) > 0 and isinstance(o[0], torch.Tensor):
                            conv_out_shapes.append(tuple(o[0].shape))
                    except Exception:
                        pass
                return hook

            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    hooks.append(m.register_forward_hook(make_hook()))

            model.reset_states()
            x = model(event_voxel, event_cnt)

            # Remove hooks
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            flow = x["flow"][0].cpu().numpy().astype(np.float32)
            np.savez('exported_models/outputs.npz', flow=flow)

            # Export paths
            onnx_file_path = f"exported_models/{config['model']['name']}_SNNtorch{model_suffix}.onnx"
            onnx_simpler_file_path = f"exported_models/{config['model']['name']}_SNNtorch{model_suffix}_simpler.onnx"

            print(f"Exporting model to: {onnx_file_path}")
            print(f"Input shape - event_cnt: {event_cnt.shape}")

            # Reset model states before export (again, for ONNX)
            model.reset_states()

            if export_quantized:
                # Export quantized model using Brevitas
                print("Exporting quantized model...")
                export_onnx_qcdq(
                    model,
                    input_t=(event_voxel, event_cnt),
                    export_path=onnx_file_path,
                    input_names=['event_voxel', 'event_cnt'],
                    output_names=['flow'],
                    opset_version=13,  # Use higher opset for quantization
                )
            else:
                # Standard FP32 export
                print("Exporting FP32 model...")
                torch.onnx.export(
                    model,
                    (event_voxel, event_cnt),
                    onnx_file_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['event_voxel', 'event_cnt'],
                    output_names=['flow'],
                    custom_opsets={"SNN_implementation": 11},
                    # dynamic_axes={
                    #     'event_voxel': {0: 'batch_size'},
                    #     'event_cnt': {0: 'batch_size'},
                    #     'flow': {0: 'batch_size'}
                    # }
                )

            # Verify the exported model
            onnx_model = onnx.load(onnx_file_path)
            # Fix output/value_info shapes for clearer graphs (LIF nodes)
            try:
                # set flow output shape from example saved flow (add batch dim)
                flow_shape = list(flow.shape)
                flow_out_shape = [1] + flow_shape
                names_to_fix = ["flow"]
                for name in names_to_fix:
                    found = False
                    for out in onnx_model.graph.output:
                        if out.name == name:
                            set_valueinfo_shape(out, flow_out_shape)
                            found = True
                            break
                    if not found:
                        vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, flow_out_shape)
                        onnx_model.graph.output.append(vi)

                # For internal LIF nodes, set their shape to match the output of
                # the previous convolution. We use the ordered list collected
                # by the Conv2d forward hooks during the example run.
                try:
                    conv_shapes = [list(s) for s in conv_out_shapes]
                    lif_vi_list = [vi for vi in onnx_model.graph.value_info if vi.name.startswith("LIF") or "LIF" in vi.name]
                    for i, vi in enumerate(lif_vi_list):
                        if i < len(conv_shapes):
                            set_valueinfo_shape(vi, conv_shapes[i])
                        elif len(conv_shapes) > 0:
                            # fallback to last conv shape
                            set_valueinfo_shape(vi, conv_shapes[-1])
                except Exception:
                    pass
                # Also ensure Conv node outputs have explicit value_info shapes
                try:
                    # collect Conv node output names in graph order
                    conv_node_outputs = []
                    for n in onnx_model.graph.node:
                        if n.op_type == 'Conv':
                            # take first output name if present
                            if len(n.output) > 0:
                                conv_node_outputs.append(n.output[0])

                    # assign shapes from conv_shapes to these outputs
                    for i, out_name in enumerate(conv_node_outputs):
                        target_shape = None
                        if i < len(conv_shapes):
                            target_shape = conv_shapes[i]
                        elif len(conv_shapes) > 0:
                            target_shape = conv_shapes[-1]
                        if target_shape is not None:
                            # find existing value_info or append new one
                            found = False
                            for vi in onnx_model.graph.value_info:
                                if vi.name == out_name:
                                    set_valueinfo_shape(vi, target_shape)
                                    found = True
                                    break
                            if not found:
                                vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, target_shape)
                                onnx_model.graph.value_info.append(vi)
                except Exception:
                    pass

            except Exception as e:
                print("Warning: failed to fix value_info shapes:", e)

            onnx.checker.check_model(onnx_model)
            print(f"ONNX model exported successfully and verified!")

            simpler_model, check = simplify(onnx_model)
            if check:
                onnx.save(simpler_model, onnx_simpler_file_path)
                print("Simplified ONNX model saved successfully!")

            # Print model info
            print(f"Model size: {len(onnx_model.SerializeToString()) / 1024 / 1024:.2f} MB")

            break
    
    mlflow.end_run()


def export_both_models(args, config_parser):
    """Export both FP32 and quantized versions"""
    print("=== Exporting FP32 Model ===")
    export_to_onnx(args, config_parser, export_quantized=False)
    
    print("\n=== Exporting Quantized Model ===")
    export_to_onnx(args, config_parser, export_quantized=True)


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
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Export quantized model only"
    )
    parser.add_argument(
        "--fp32",
        action="store_true", 
        help="Export FP32 model only"
    )
    args = parser.parse_args()

    # Create export directory
    import os
    os.makedirs("exported_models", exist_ok=True)

    # Launch export based on arguments
    if args.quantized:
        export_to_onnx(args, YAMLParser(args.config), export_quantized=True)
    elif args.fp32:
        export_to_onnx(args, YAMLParser(args.config), export_quantized=False)
    else:
        # Export both by default
        export_both_models(args, YAMLParser(args.config))