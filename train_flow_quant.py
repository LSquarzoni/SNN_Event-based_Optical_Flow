import argparse
import os
import shutil
import gc

import mlflow
import torch
from torch.optim import *
from brevitas import config as cf
from brevitas.export import export_onnx_qcdq

# CRITICAL: Enable expandable segments to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping
from models.model import (
    LIFFireNet,
    LIFFireNet_short,
    LIFFireFlowNet,
    LIFFireFlowNet_short,
)
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_diff
from utils.visualization import Visualization

cf.IGNORE_MISSING_KEYS = True


def save_quantized_model(model, path, epoch=None, optimizer=None, loss=None, additional_info=None):
    """
    Save quantized model with all quantization metadata.
    
    Args:
        model: The quantized model to save
        path: Path where to save the model
        epoch: Current epoch number (optional)
        optimizer: Optimizer state (optional)
        loss: Current loss value (optional)
        additional_info: Dictionary with additional metadata (optional)
    """
    # Prepare checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'quantization_enabled': True,
    }
    
    # Add optional information
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if loss is not None:
        checkpoint['loss'] = loss
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Extract quantization parameters from the model
    quant_params = {}
    for name, module in model.named_modules():
        # Store Brevitas quantization parameters for convolutions
        if hasattr(module, 'weight_quant') and hasattr(module.weight_quant, 'scale'):
            quant_params[f'{name}.weight_quant'] = {
                'bit_width': module.weight_quant.bit_width() if hasattr(module.weight_quant, 'bit_width') else None,
                'scale': module.weight_quant.scale().detach().cpu() if hasattr(module.weight_quant, 'scale') else None,
                'zero_point': module.weight_quant.zero_point().detach().cpu() if hasattr(module.weight_quant, 'zero_point') else None,
            }
        if hasattr(module, 'input_quant') and hasattr(module.input_quant, 'scale'):
            quant_params[f'{name}.input_quant'] = {
                'bit_width': module.input_quant.bit_width() if hasattr(module.input_quant, 'bit_width') else None,
                'scale': module.input_quant.scale().detach().cpu() if hasattr(module.input_quant, 'scale') else None,
                'zero_point': module.input_quant.zero_point().detach().cpu() if hasattr(module.input_quant, 'zero_point') else None,
            }
        if hasattr(module, 'output_quant') and hasattr(module.output_quant, 'scale'):
            quant_params[f'{name}.output_quant'] = {
                'bit_width': module.output_quant.bit_width() if hasattr(module.output_quant, 'bit_width') else None,
                'scale': module.output_quant.scale().detach().cpu() if hasattr(module.output_quant, 'scale') else None,
                'zero_point': module.output_quant.zero_point().detach().cpu() if hasattr(module.output_quant, 'zero_point') else None,
            }
        
        # Store LIF quantization parameters (for Full QAT mode)
        if hasattr(module, 'q_lif') and not isinstance(module.q_lif, torch.nn.Identity):
            # This is a quantized LIF state quantizer (Full QAT)
            if hasattr(module.q_lif, 'scale'):
                quant_params[f'{name}.q_lif'] = {
                    'bit_width': module.q_lif.bit_width if hasattr(module.q_lif, 'bit_width') else 8,
                    'scale': module.q_lif.scale.detach().cpu() if hasattr(module.q_lif, 'scale') else None,
                    'zero_point': module.q_lif.zero_point.detach().cpu() if hasattr(module.q_lif, 'zero_point') else None,
                }
        
        # Store learnable LIF parameters (beta, threshold) - always present
        if hasattr(module, 'beta'):
            quant_params[f'{name}.beta'] = module.beta.data.detach().cpu().clone()
        if hasattr(module, 'threshold'):
            quant_params[f'{name}.threshold'] = module.threshold.data.detach().cpu().clone()
    
    checkpoint['quantization_params'] = quant_params
    
    # Save checkpoint
    torch.save(checkpoint, path)
    print(f"✓ Quantized model saved to {path}")
    
    # Also save a separate file with just quantization metadata for easy inspection
    metadata_path = path.replace('.pth', '_quant_metadata.pth')
    torch.save({'quantization_params': quant_params}, metadata_path)
    print(f"✓ Quantization metadata saved to {metadata_path}")

def train_qat(args, config_parser):
    """
    Quantization Aware Training (QAT) for SNN models.
    
    Supports two QAT modes:
    1. Full QAT: Quantize both convolutions and LIF layers during training
       - Config: quantization.Conv_only = False
       - Saves model with all quantization metadata (weights, activations, LIF parameters)
       
    2. Conv-only QAT: Quantize only convolutions, keep LIF at FP32 during training
       - Config: quantization.Conv_only = True
       - LIF quantization can be added later at evaluation time via calibration
       - Saves model with convolution quantization metadata only
    """
    mlflow.set_tracking_uri(args.path_mlflow)

    # configs
    config = config_parser.config
    if config["data"]["mode"] == "frames":
        print("Config error: Training pipeline not compatible with frames mode.")
        raise AttributeError

    # Ensure quantization is enabled
    if not config["model"].get("quantization", {}).get("enabled", False):
        print("Warning: Quantization not enabled in config. Enabling it for QAT training.")
        config["model"]["quantization"]["enabled"] = True

    # log config
    mlflow.set_experiment(config["experiment"])
    mlflow.start_run()
    mlflow.log_params(config)
    mlflow.log_param("prev_runid", args.prev_runid)
    mlflow.log_param("training_type", "QAT")
    
    # Determine QAT mode
    conv_only = config["model"].get("quantization", {}).get("Conv_only", False)
    mlflow.log_param("qat_mode", "Conv_only" if conv_only else "Full")
    
    config = config_parser.combine_entries(config)
    mlflow.pytorch.autolog()
    print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])

    # log git diff
    save_diff("train_diff.txt")

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)

    # data loader
    data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # loss function
    loss_function = EventWarping(config, device)

    # model initialization with quantization enabled
    print(f"Initializing quantized model: {config['model']['name']}")
    model = eval(config["model"]["name"])(config["model"].copy()).to(device)
    
    print("\n" + "="*80)
    if conv_only:
        print("QAT MODE: Convolution-Only Quantization")
        print("-" * 80)
        print("Training with:")
        print("  ✓ Convolutions: INT8 quantized (weights + activations)")
        print("  ✓ LIF parameters (beta, threshold): FP32 trainable")
        print("  ✓ LIF states (membrane): FP32")
        print("-" * 80)
        print("At evaluation time, you can:")
        print("  • Keep LIF at FP32 for mixed precision (best accuracy)")
        print("  • Add LIF quantization via calibration (smaller model)")
    else:
        print("QAT MODE: Full Quantization (Convolutions + LIF)")
        print("-" * 80)
        print("Training with:")
        print("  ✓ Convolutions: INT8 quantized (weights + activations)")
        print("  ✓ LIF parameters (beta, threshold): INT8 quantized")
        print("  ✓ LIF states (membrane): INT8 quantized")
        print("-" * 80)
        print("At evaluation: Model is fully quantized, no calibration needed")
    print("="*80)
    
    # Load pre-trained FP32 model if provided (for transfer learning)
    if args.prev_runid:
        print(f"\nLoading FP32 pre-trained model from run: {args.prev_runid}")
        model = load_model(args.prev_runid, model, device)
        print("✓ FP32 weights loaded - will train with quantization from these weights")
    
    model.train()

    # optimizer
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # Create model save directory based on QAT mode
    run_id = mlflow.active_run().info.run_id
    if conv_only:
        model_save_dir = f"mlruns/0/models/LIFFN_ConvOnly_QAT"
    else:
        model_save_dir = f"mlruns/0/models/LIFFN_Full_QAT"
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\nModel checkpoints will be saved to: {model_save_dir}")

    # simulation variables
    patience = 50
    epochs_without_improvement = 0
    train_loss = 0
    best_loss = 1.0e6
    end_train = False
    grads_w = []

    # training loop
    data.shuffle()
    while True:
        for inputs in dataloader:

            if data.new_seq:
                data.new_seq = False

                loss_function.reset()
                model.reset_states()
                optimizer.zero_grad()
                
                # Empty cache at sequence boundaries
                torch.cuda.empty_cache()

            if data.seq_num >= len(data.files):
                avg_train_loss = train_loss / (data.samples + 1)
                mlflow.log_metric("loss", avg_train_loss, step=data.epoch)

                # Print epoch summary
                print(f"Epoch {data.epoch}/{config['loader']['n_epochs']} - Train Loss: {avg_train_loss:.6f}")

                # Save model checkpoints (memory-efficient: only best and latest)
                with torch.no_grad():
                    # Check if this is the best model so far
                    is_best = avg_train_loss < best_loss
                    if is_best:
                        best_loss = avg_train_loss
                        epochs_without_improvement = 0
                        print(f"New best loss: {best_loss:.6f}")
                        
                        # Save best model (overwrites previous best)
                        best_model_path = os.path.join(model_save_dir, "model_quant_best.pth")
                        save_quantized_model(
                            model, 
                            best_model_path,
                            epoch=data.epoch,
                            optimizer=optimizer,
                            loss=avg_train_loss,
                            additional_info={
                                'best_train_loss': best_loss,
                            }
                        )
                        mlflow.log_artifact(best_model_path)
                        
                        # Also log quantization metadata separately
                        metadata_path = best_model_path.replace('.pth', '_quant_metadata.pth')
                        if os.path.exists(metadata_path):
                            mlflow.log_artifact(metadata_path)
                    else:
                        epochs_without_improvement += 1
                    
                    # Save latest model (overwrites previous latest each epoch)
                    latest_model_path = os.path.join(model_save_dir, "model_quant_latest.pth")
                    save_quantized_model(
                        model,
                        latest_model_path,
                        epoch=data.epoch,
                        optimizer=optimizer,
                        loss=avg_train_loss,
                    )
                    mlflow.log_artifact(latest_model_path)

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)

                # save grads to file
                if config["vis"]["store_grads"]:
                    save_csv([{name: values for name, values in zip(grads_w[0].keys(), values)} for values in zip(*grads_w[0].values())], "grads.csv")
                    grads_w = []

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"] or epochs_without_improvement >= patience:
                    print(f"\nTraining finished at epoch {data.epoch}")
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping: No improvement for {patience} epochs")
                    
                    # Final model is already saved as either best or latest, no need for a separate final copy
                    print(f"Best model saved at: {os.path.join(model_save_dir, 'model_quant_best.pth')}")
                    print(f"Latest model saved at: {os.path.join(model_save_dir, 'model_quant_latest.pth')}")
                    
                    end_train = True
                    break

            # forward pass
            x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))

            # event flow association
            loss_function.event_flow_association(
                x["flow"],
                inputs["event_list"].to(device),
                inputs["event_list_pol_mask"].to(device),
                inputs["event_mask"].to(device),
            )

            # backward pass
            if loss_function.num_events >= config["data"]["window_loss"]:

                # overwrite intermediate flow estimates with the final ones
                if config["loss"]["overwrite_intermediate"]:
                    loss_function.overwrite_intermediate_flow(x["flow"])

                # loss
                loss = loss_function()
                train_loss += loss.item()

                # update number of loss samples seen by the network
                data.samples += config["loader"]["batch_size"]
                
                loss.backward()

                # clip and save grads
                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
                if config["vis"]["store_grads"]:
                    grads_w.append(get_grads(model.named_parameters()))

                optimizer.step()

                model.detach_states()
                
                loss_function.reset()
                
                # Periodic garbage collection to free fragmented memory
                if data.seq_num % 50 == 0:
                    torch.cuda.empty_cache()

            # print training info
            if config["vis"]["verbose"]:
                print(
                    "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                        data.epoch,
                        data.seq_num,
                        len(data.files),
                        int(100 * data.seq_num / len(data.files)),
                        train_loss / (data.samples + 1),
                    ),
                    end="\r",
                )

            # store gradients
            if config["vis"]["store_grads"]:
                grads_w.append(get_grads(model))

            """ # Print memory usage and sequence info every 10 sequences
            if data.seq_num % 10 == 0:
                print(f"\nMemory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
                print(f"Sequence: {data.seq_num}, Events in loss: {loss_function.num_events}") """

        if end_train:
            break

    mlflow.end_run()
    print(f"\nQuantized models saved in: {model_save_dir}")
    print("Saved files:")
    print("  - model_quant_best.pth: Best model based on training loss")
    print("  - model_quant_latest.pth: Latest model (from last epoch)")
    print("  - *_quant_metadata.pth: Quantization metadata files")
    print("\nNote: Only best and latest models are saved to conserve memory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization Aware Training (QAT) for SNN models")
    parser.add_argument(
        "--config",
        default="configs/train_SNN.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="Optional: FP32 pre-trained model run ID to use as starting point for transfer learning",
    )
    args = parser.parse_args()

    # Verify config has quantization enabled
    config_parser = YAMLParser(args.config)
    if not config_parser.config["model"].get("quantization", {}).get("enabled", False):
        print("\n" + "="*60)
        print("WARNING: Quantization not enabled in config file!")
        print("QAT training requires quantization to be enabled.")
        print("Please ensure your config has:")
        print("  quantization:")
        print("    enabled: True")
        print("    type: int8")
        print("="*60 + "\n")

    # launch QAT training
    print("="*60)
    print("Starting Quantization Aware Training (QAT)")
    print("="*60)
    train_qat(args, config_parser)
