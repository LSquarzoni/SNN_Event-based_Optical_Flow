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


            # Require at least 10% non-zero values in event_cnt for export
            nonzero_cnt = torch.count_nonzero(event_cnt)
            total_cnt = event_cnt.numel()
            nonzero_ratio = nonzero_cnt.float() / total_cnt
            if nonzero_ratio < 0.1:
                print(f"Skipping sparse input batch for ONNX export (nonzero ratio: {nonzero_ratio:.3f})...")
                continue

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
            # Extract flow array robustly (handle tensor or list/tuple outputs)
            flow_out_obj = x.get("flow") if isinstance(x, dict) else None
            if flow_out_obj is None:
                flow_arr = np.array([])
            else:
                if isinstance(flow_out_obj, torch.Tensor):
                    flow_arr = flow_out_obj.cpu().numpy()
                elif isinstance(flow_out_obj, (list, tuple)) and len(flow_out_obj) > 0 and isinstance(flow_out_obj[0], torch.Tensor):
                    flow_arr = flow_out_obj[0].cpu().numpy()
                else:
                    # fallback
                    flow_arr = np.asarray(flow_out_obj)

            flow = flow_arr.astype(np.float32)
            np.savez('exported_models/outputs.npz', flow=flow)

            # Export paths
            #onnx_file_path = f"exported_models/{config['model']['name']}_SNNtorch{model_suffix}.onnx"
            onnx_file_path = f"exported_models/network.onnx"
            onnx_simpler_file_path = f"exported_models/{config['model']['name']}_SNNtorch{model_suffix}_simpler.onnx"

            print(f"Exporting model to: {onnx_file_path}")
            print(f"Input shape - event_cnt: {event_cnt.shape}")

            # Reset model states before export (again, for ONNX)
            model.reset_states()

            if export_quantized:
                # Export quantized model using Brevitas
                print("Exporting quantized model...")
                export_qonnx(
                    model,
                    input_t=(event_voxel, event_cnt),
                    export_path=onnx_file_path,
                    export_params=True,
                    opset_version=11,
                    input_names=['event_voxel', 'event_cnt'],
                    output_names=['flow'],
                    custom_opsets={"SNN_implementation": 11},
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
                )

            # Verify the exported model
            onnx_model = onnx.load(onnx_file_path)
            # Fix output/value_info shapes for clearer graphs (LIF nodes)
            try:
                # set flow output shape from example saved flow
                # Use the last 4 dimensions as (N, C, H, W) to avoid double-prepending
                flow_shape = list(flow.shape)
                if flow.ndim >= 4:
                    flow_out_shape = list(flow.shape[-4:])
                else:
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

                # Also ensure Conv node outputs have explicit value_info shapes
                try:
                    # Build conv shapes from runtime and map them onto Conv nodes in the ONNX graph
                    conv_shapes = [list(s) for s in conv_out_shapes]
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
                except Exception:
                    pass

            except Exception as e:
                print("Warning: failed to fix value_info shapes:", e)

            onnx.checker.check_model(onnx_model)
            print(f"ONNX model exported successfully and verified!")
            onnx.save(onnx_model, onnx_file_path)

            """ simpler_model, check = simplify(onnx_model)
            if check:
                onnx.save(simpler_model, onnx_simpler_file_path)
                print("Simplified ONNX model saved successfully!") """

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