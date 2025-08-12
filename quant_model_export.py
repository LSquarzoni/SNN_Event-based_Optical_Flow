import argparse
import torch
import onnx
import mlflow
from brevitas import config as cf

cf.IGNORE_MISSING_KEYS = True

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from models.model import LIFFireNet
from utils.utils import load_model


def export_to_onnx(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)
    
    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)
    
    # Initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    
    # Model initialization - disable quantization for ONNX export
    config["model"]["quantization"]["enabled"] = False
    model = eval(config["model"]["name"])(config["model"]).to(device)
    
    # Load model weights (update path as needed)
    model_path_dir = "mlruns/0/models/LIFFireNet_SNNtorch/26/model.pth" # runid: 3ab96c99fced453e91ed83b5e48ac3ca
    model = load_model(args.runid, model, device, model_path_dir)
    model.eval()
    
    # Data loader to get sample input
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=1,  # Use batch size 1 for ONNX export
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )
    
    # Get sample input
    with torch.no_grad():
        for inputs in dataloader:
            event_voxel = inputs["event_voxel"].to(device)
            event_cnt = inputs["event_cnt"].to(device)
            
            # Export to ONNX
            onnx_file_path = f"exported_models/{config['model']['name']}_SNNtorch.onnx"
            
            print(f"Exporting model to: {onnx_file_path}")
            print(f"Input shapes - event_voxel: {event_voxel.shape}, event_cnt: {event_cnt.shape}")
            
            # Reset model states before export
            model.reset_states()
            
            torch.onnx.export(
                model,
                (event_voxel, event_cnt),
                onnx_file_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['event_voxel', 'event_cnt'],
                output_names=['flow'],
                dynamic_axes={
                    'event_voxel': {0: 'batch_size'},
                    'event_cnt': {0: 'batch_size'},
                    'flow': {0: 'batch_size'}
                }
            )
            
            # Verify the exported model
            onnx_model = onnx.load(onnx_file_path)
            onnx.checker.check_model(onnx_model)
            print(f"ONNX model exported successfully and verified!")
            
            break
    
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
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

    # Launch export
    export_to_onnx(args, YAMLParser(args.config))