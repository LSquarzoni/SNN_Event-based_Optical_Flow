import argparse
import os
import shutil

import mlflow
import torch
from torch.optim import *
from brevitas import config as cf
from brevitas.graph.calibrate import calibration_mode
from brevitas.export import export_onnx_qcdq

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
        # Store Brevitas quantization parameters
        if hasattr(module, 'quant_weight'):
            quant_params[f'{name}.weight_quant'] = {
                'bit_width': module.quant_weight.bit_width() if hasattr(module.quant_weight, 'bit_width') else None,
                'scale': module.quant_weight.scale() if hasattr(module.quant_weight, 'scale') else None,
                'zero_point': module.quant_weight.zero_point() if hasattr(module.quant_weight, 'zero_point') else None,
            }
        if hasattr(module, 'quant_act'):
            quant_params[f'{name}.act_quant'] = {
                'bit_width': module.quant_act.bit_width() if hasattr(module.quant_act, 'bit_width') else None,
                'scale': module.quant_act.scale() if hasattr(module.quant_act, 'scale') else None,
                'zero_point': module.quant_act.zero_point() if hasattr(module.quant_act, 'zero_point') else None,
            }
        # Store learnable LIF parameters (leak, threshold)
        if hasattr(module, 'beta'):
            quant_params[f'{name}.beta'] = module.beta.data.clone()
        if hasattr(module, 'threshold'):
            quant_params[f'{name}.threshold'] = module.threshold.data.clone()
    
    checkpoint['quantization_params'] = quant_params
    
    # Save checkpoint
    torch.save(checkpoint, path)
    print(f"Quantized model saved to {path}")
    
    # Also save a separate file with just quantization metadata for easy inspection
    metadata_path = path.replace('.pth', '_quant_metadata.pth')
    torch.save({'quantization_params': quant_params}, metadata_path)
    print(f"Quantization metadata saved to {metadata_path}")


def load_quantized_model(model, path, device, load_optimizer=False, optimizer=None):
    """
    Load quantized model with all quantization metadata.
    
    Args:
        model: The model instance to load weights into
        path: Path to the saved checkpoint
        device: Device to load the model on
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer instance (required if load_optimizer=True)
    
    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dictionary with metadata
    """
    if not os.path.isfile(path):
        print(f"No quantized model found at {path}")
        return model, None
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Quantized model loaded from {path}")
    else:
        print("Warning: 'model_state_dict' not found in checkpoint")
    
    # Load optimizer state if requested
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # Print quantization info
    if 'quantization_params' in checkpoint:
        print(f"Loaded quantization metadata for {len(checkpoint['quantization_params'])} parameters")
    
    return model, checkpoint


def calibrate_model(calibration_loader, model, device, num_batches=50):
    """
    Calibrate the quantized model to initialize quantization parameters.
    This should be called before QAT training begins.
    """
    print("Starting model calibration...")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        with calibration_mode(model):
            batch_count = 0
            for inputs in calibration_loader:
                if batch_count >= num_batches:
                    break
                    
                event_voxel = inputs["event_voxel"].to(device)
                event_cnt = inputs["event_cnt"].to(device)
                
                model.reset_states()
                model(event_voxel, event_cnt)
                
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f'Calibration: {batch_count}/{num_batches} batches')
    
    print("Calibration completed!")
    return model


def train_qat(args, config_parser):
    """
    Quantization Aware Training (QAT) for SNN models.
    This function trains a model with quantization enabled and saves all quantization metadata.
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
    mlflow.log_param("calibration_batches", args.calibration_batches)
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
    
    # CRITICAL: Disable state quantization during training to save VRAM and improve stability
    # State quantization will be calibrated during evaluation
    print("\nDisabling state quantization during training (will be calibrated at evaluation)...")
    state_quantizers_disabled = 0
    for name, module in model.named_modules():
        if hasattr(module, 'q_lif'):
            # Replace state quantizer with identity function
            module.q_lif = torch.nn.Identity()
            state_quantizers_disabled += 1
    if state_quantizers_disabled > 0:
        print(f"Disabled {state_quantizers_disabled} state quantizers to save VRAM")
        print("Note: Weights, activations, beta, threshold are still quantized (QAT)")
        print("      States will be calibrated during evaluation with --calibration_batches")
    
    # Load pre-trained model if provided
    if args.prev_runid:
        if args.load_quantized:
            # Load from a previous QAT checkpoint
            model_dir = f"mlruns/0/models/{args.prev_runid}/model_quant.pth"
            model, checkpoint = load_quantized_model(model, model_dir, device)
        else:
            # Load from a standard checkpoint (FP32)
            model = load_model(args.prev_runid, model, device)
            print("Loaded FP32 model. Will perform calibration before QAT training.")
    
    # Calibrate the model if starting from FP32 or if explicitly requested
    if args.calibrate or (args.prev_runid and not args.load_quantized):
        model = calibrate_model(dataloader, model, device, num_batches=args.calibration_batches)
    
    model.train()

    # optimizer
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    
    # Load optimizer state if resuming from QAT checkpoint
    if args.prev_runid and args.load_quantized:
        model_dir = f"mlruns/0/models/{args.prev_runid}/model_quant.pth"
        if os.path.exists(model_dir):
            checkpoint = torch.load(model_dir, map_location=device, weights_only=False)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
    
    optimizer.zero_grad()

    # Create model save directory
    run_id = mlflow.active_run().info.run_id
    model_save_dir = f"mlruns/0/models/LIFFN_int8"
    os.makedirs(model_save_dir, exist_ok=True)

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
                optimizer.zero_grad()

                model.detach_states()
                loss_function.reset()

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
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "--load_quantized",
        action="store_true",
        help="load from a quantized checkpoint (otherwise loads FP32)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="perform calibration before training (automatically done when loading FP32)",
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=50,
        help="number of batches to use for calibration",
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
