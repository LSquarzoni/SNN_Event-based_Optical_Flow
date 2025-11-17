import os

import mlflow
import pandas as pd
import torch
import numpy as np


def load_model(prev_runid, model, device, model_path_dir="", strict=True):
    import sys
    print(f"\n{'='*80}", flush=True)
    print(f"*** load_model called with strict={strict} ***", flush=True)
    print(f"{'='*80}\n", flush=True)
    sys.stdout.flush()
    try:
        run = mlflow.get_run(prev_runid)
    except:
        return model
    
    if model_path_dir == "":
        model_dir = run.info.artifact_uri + "/model/data/model.pth"
    else:
        model_dir = model_path_dir
        
    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    if os.path.isfile(model_dir):
        checkpoint = torch.load(model_dir, map_location=device, weights_only=False)
        
        # Check if the loaded object is a dictionary (your new format) or a model object (old format)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Fallback: assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            # Old format: the checkpoint is the model object itself
            state_dict = checkpoint.state_dict()
        
        # Special handling for PTQ: If loading with strict=False, copy trained beta/threshold
        # from snn.Leaky (*.lif.beta, *.lif.threshold) to the model's direct parameters
        if not strict:
            print(f"\n{'#'*80}", flush=True)
            print("ENTERING PTQ MODE BLOCK!", flush=True)
            print(f"{'#'*80}\n", flush=True)
            print("PTQ mode: Checking for trained LIF parameters in checkpoint...", flush=True)
            lif_param_mapping = {}
            for key in state_dict.keys():
                if '.lif.beta' in key:
                    # Map head.lif.beta -> head.beta
                    new_key = key.replace('.lif.beta', '.beta')
                    lif_param_mapping[new_key] = state_dict[key].clone()
                    print(f"  DEBUG: Found {key}, will copy to {new_key}")
                elif '.lif.threshold' in key:
                    # Map head.lif.threshold -> head.threshold
                    new_key = key.replace('.lif.threshold', '.threshold')
                    lif_param_mapping[new_key] = state_dict[key].clone()
                    print(f"  DEBUG: Found {key}, will copy to {new_key}")
            
            # Update state_dict with the trained LIF parameters
            if lif_param_mapping:
                print(f"  Found {len(lif_param_mapping)} trained LIF parameters to copy")
                state_dict.update(lif_param_mapping)
                print("  Successfully updated state_dict with trained LIF parameters")
            else:
                print("  WARNING: No .lif.beta or .lif.threshold keys found in checkpoint!")
                print("  This means the model was trained with the new architecture.")
                print("  Beta/threshold should load correctly from the checkpoint.")
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if strict:
                print(f"Warning: Failed to load with strict=True: {e}")
                print("Retrying with strict=False (will ignore mismatched keys)...")
                model.load_state_dict(state_dict, strict=False)
            else:
                raise
            
        print("Model restored from " + prev_runid + "\n")
    else:
        print("No model found at " + prev_runid + "\n")

    return model


def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model):
    
    mlflow.pytorch.log_model(model, name="model")


def save_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
    if not os.path.isfile(path):
        mlflow.log_text("", fname)
        pd.DataFrame(data).to_csv(path)
    # else append
    else:
        pd.DataFrame(data).to_csv(path, mode="a", header=False)


def save_diff(fname="git_diff.txt"):
    # .txt to allow showing in mlflow
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":
        path = path[7:]
    mlflow.log_text("", fname)
    os.system(f"git diff > {path}")


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


def inspect_quantized_model(path):
    """
    Inspect and print quantization metadata from a saved quantized model.
    
    Args:
        path: Path to the quantized model checkpoint
    """
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    print("="*80)
    print(f"Quantized Model Inspection: {os.path.basename(path)}")
    print("="*80)
    
    # Print basic info
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Loss: {checkpoint['loss']:.6f}")
    if 'best_val_aee' in checkpoint:
        print(f"Best Validation AEE: {checkpoint['best_val_aee']:.6f}")
    
    print("\nQuantization Status:")
    print(f"  Quantization Enabled: {checkpoint.get('quantization_enabled', False)}")
    
    # Print quantization parameters
    if 'quantization_params' in checkpoint:
        quant_params = checkpoint['quantization_params']
        print(f"\nQuantization Parameters ({len(quant_params)} entries):")
        print("-"*80)
        
        for name, params in quant_params.items():
            if isinstance(params, dict):
                print(f"\n  {name}:")
                for key, value in params.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            print(f"    {key}: shape={value.shape}, mean={value.mean():.6f}, std={value.std():.6f}")
                        else:
                            print(f"    {key}: {value}")
            elif isinstance(params, torch.Tensor):
                print(f"\n  {name}: shape={params.shape}, mean={params.mean():.6f}, std={params.std():.6f}")
    else:
        print("\nNo quantization parameters found in checkpoint")
    
    # Print additional info
    if 'training_completed' in checkpoint:
        print(f"\nTraining Completed: {checkpoint['training_completed']}")
    
    print("="*80)
