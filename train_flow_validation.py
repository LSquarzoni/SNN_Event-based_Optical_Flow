import argparse
import os
import shutil

import mlflow
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping, AAE, AEE
from models.model import (
    FireNet,
    FireNet_short,
    FireFlowNet,
    FireFlowNet_short,
)
from models.model import (
    LIFFireNet,
    LIFFireNet_short,
    LIFFireFlowNet,
    LIFFireFlowNet_short,
)
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_diff, save_model
from utils.visualization import Visualization

def get_next_model_folder(base_path="mlruns/0/models/"):
    index = 0
    while os.path.exists(os.path.join(base_path, str(index))):
        index += 1
    return os.path.join(base_path, str(index))

def validate_on_mvsec(model, val_config, val_config_parser, device, verbose=True):
    """
    Run validation on the entire MVSEC validation dataset.
    Returns the average AAE and AEE across all sequences.
    
    This function saves and restores the model's internal states to avoid
    disrupting the temporal continuity during training.
    """
    # Save the current training states before validation
    saved_states = None
    if hasattr(model, '_states') and model._states is not None:
        saved_states = [state.clone() if state is not None else None for state in model._states]
    
    model.eval()
    
    # Create validation dataloader
    val_data = H5Loader(val_config, val_config["model"]["num_bins"], val_config["model"]["round_encoding"])
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        drop_last=True,
        batch_size=val_config["loader"]["batch_size"],
        collate_fn=val_data.custom_collate,
        worker_init_fn=val_config_parser.worker_init_fn,
        **val_config_parser.loader_kwargs,
    )
    
    # Create AAE metric
    aae_metric = AAE(val_config, device, flow_scaling=val_config["metrics"]["flow_scaling"])
    
    aae_sum = 0.0
    aae_count = 0
    num_sequences = len(val_data.files)
    
    # Create AEE metric
    aee_metric = AEE(val_config, device, flow_scaling=val_config["metrics"]["flow_scaling"])
    
    aee_sum = 0.0
    aee_count = 0
    
    with torch.no_grad():
        val_data.seq_num = 0
        val_data.samples = 0
        
        if verbose:
            print(f"  Starting validation on all {num_sequences} MVSEC sequences...")
        
        # Reset states for validation (different resolution)
        model.reset_states()
        
        for inputs in val_dataloader:
            # Handle new sequence
            if val_data.new_seq:
                val_data.new_seq = False
                model.reset_states()
                aae_metric.reset()
                aee_metric.reset()
                if verbose:
                    print(f"  Processing sequence {val_data.seq_num + 1}/{num_sequences}...")
            
            # Stop after processing all sequences
            if val_data.seq_num >= num_sequences:
                break
            
            # Forward pass
            x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))
            
            # Compute AAE and AEE metric
            aae_metric.event_flow_association(x["flow"], inputs)
            aee_metric.event_flow_association(x["flow"], inputs)
            
            if aae_metric.num_events > 0:
                result_aae = aae_metric()
                result_aee = aee_metric()
                # Handle tuple returns (error, percentage_outliers)
                if isinstance(result_aae, tuple):
                    aae_error = result_aae[0].item()
                    aee_error = result_aee[0].item()
                else:
                    aae_error = result_aae.item()
                    aee_error = result_aee.item()
                
                aae_sum += aae_error
                aee_sum += aee_error
                aae_count += 1
                aee_count += 1
    
    # Compute average AAE and AEE
    avg_aae = aae_sum / aae_count if aae_count > 0 else float('inf')
    avg_aee = aee_sum / aee_count if aee_count > 0 else float('inf')
    
    if verbose:
        print(f"  Validation complete: AAE = {avg_aae:.4f} (from {aae_count} samples across {num_sequences} sequences)")
        print(f"  Validation complete: AEE = {avg_aee:.4f} (from {aee_count} samples across {num_sequences} sequences)")
    
    # Restore the saved training states
    if saved_states is not None and hasattr(model, '_states'):
        model._states = saved_states
    else:
        # If no states were saved, reset to ensure clean state
        model.reset_states()
    
    model.train()  # Switch back to training mode
    return avg_aae, avg_aee

def train(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    # configs
    config = config_parser.config
    if config["data"]["mode"] == "frames":
        print("Config error: Training pipeline not compatible with frames mode.")
        raise AttributeError

    # log config
    mlflow.set_experiment(config["experiment"])
    mlflow.start_run()
    mlflow.log_params(config)
    mlflow.log_param("prev_runid", args.prev_runid)
    mlflow.log_param("val_config_path", args.val_config)
    mlflow.log_param("val_every_n_epochs", args.val_every_n_epochs)
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

    # data loader (training)
    data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # Load validation config
    print(f"\nLoading validation config from: {args.val_config}")
    val_config_parser = YAMLParser(args.val_config)
    val_config = val_config_parser.config
    
    # Copy essential model settings from training config to validation config
    val_config["model"]["num_bins"] = config["model"]["num_bins"]
    val_config["model"]["round_encoding"] = config["model"].get("round_encoding", False)
    val_config["model"]["name"] = config["model"]["name"]
    val_config["model"]["encoding"] = config["model"]["encoding"]
    val_config["model"]["norm_input"] = config["model"]["norm_input"]
    val_config["model"]["base_num_channels"] = config["model"]["base_num_channels"]
    val_config["model"]["kernel_size"] = config["model"]["kernel_size"]
    val_config["model"]["activations"] = config["model"]["activations"]
    val_config["model"]["mask_output"] = config["model"]["mask_output"]
    
    # Copy spiking neuron settings if present
    if "spiking_neuron" in config:
        val_config["spiking_neuron"] = config["spiking_neuron"]
    
    # Copy loss settings for overwrite_intermediate if needed
    if "loss" not in val_config:
        val_config["loss"] = {}
    val_config["loss"]["overwrite_intermediate"] = config["loss"].get("overwrite_intermediate", False)
    
    print(f"Training resolution: {config['loader']['resolution']}")
    print(f"Validation resolution: {val_config['loader']['resolution']}")
    print(f"Validation will run every {args.val_every_n_epochs} epochs on all MVSEC sequences\n")

    # loss function
    loss_function = EventWarping(config, device)

    # model initialization and settings
    model = eval(config["model"]["name"])(config["model"].copy()).to(device)
    model = load_model(args.prev_runid, model, device)
    model.train()

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # simulation variables
    patience = 50
    epochs_without_improvement = 0
    train_loss = 0
    best_loss = 1.0e6
    best_val_aae = 1.0e6
    best_val_aee = 1.0e6
    end_train = False
    grads_w = []
    
    # Separate checkpoint tracking for loss and validation
    checkpoint_counter_loss = 0
    last_checkpoint_path_loss = None
    checkpoint_counter_val = 0
    last_checkpoint_path_val = None

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

                # Run validation every N epochs
                val_aae = None
                val_aee = None
                if data.epoch > 0 and data.epoch % args.val_every_n_epochs == 0:
                    print(f"\n{'='*60}")
                    print(f"Running validation at epoch {data.epoch}...")
                    print(f"{'='*60}")
                    val_aae, val_aee = validate_on_mvsec(model, val_config, val_config_parser, device, verbose=True)
                    mlflow.log_metric("val_AAE", val_aae, step=data.epoch)
                    mlflow.log_metric("val_AEE", val_aee, step=data.epoch)
                    print(f"{'='*60}\n")

                # Print epoch summary
                if val_aae is not None:
                    print(f"Epoch {data.epoch:04d} - Training Loss: {avg_train_loss:.6f}, Validation AAE: {val_aae:.4f}, Validation AEE: {val_aee:.4f}")
                else:
                    print(f"Epoch {data.epoch:04d} - Training Loss: {avg_train_loss:.6f}")

                with torch.no_grad():
                    # Save model based on training loss (always)
                    if avg_train_loss < best_loss - 1e-6:
                        # Delete previous loss-based checkpoint folder if it exists
                        if last_checkpoint_path_loss is not None and os.path.exists(last_checkpoint_path_loss):
                            try:
                                shutil.rmtree(last_checkpoint_path_loss)
                                print(f"[Loss] Deleted previous checkpoint: {last_checkpoint_path_loss}")
                            except Exception as e:
                                print(f"[Loss] Warning: Could not delete previous checkpoint: {e}")
                        
                        # Create new checkpoint folder for loss
                        base_model_path_loss = "mlruns/0/models/LIFFN_validation_loss/"
                        model_save_path_loss = os.path.join(base_model_path_loss, str(checkpoint_counter_loss))
                        
                        try:
                            os.makedirs(model_save_path_loss, exist_ok=True)
                            print(f"[Loss] Created checkpoint directory: {model_save_path_loss}")
                        except Exception as e:
                            print(f"[Loss] Error creating directory {model_save_path_loss}: {e}")
                            os.makedirs(base_model_path_loss, exist_ok=True)
                            os.makedirs(model_save_path_loss, exist_ok=True)

                        save_data_loss = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': data.epoch,
                            'loss': avg_train_loss,
                            'config': config
                        }
                        if val_aae is not None:
                            save_data_loss['val_AAE'] = val_aae
                        if val_aee is not None:
                            save_data_loss['val_AEE'] = val_aee

                        model_pth_path_loss = os.path.join(model_save_path_loss, 'model.pth')
                        try:
                            temp_path = model_pth_path_loss + '.tmp'
                            torch.save(save_data_loss, temp_path)
                            os.replace(temp_path, model_pth_path_loss)
                            print(f"[Loss] Model checkpoint saved to: {model_pth_path_loss} (Loss: {avg_train_loss:.6f})")
                            
                            try:
                                mlflow.log_artifact(model_pth_path_loss)
                            except Exception as e:
                                print(f"[Loss] Warning: Could not log artifact to mlflow: {e}")
                            
                            last_checkpoint_path_loss = model_save_path_loss
                            checkpoint_counter_loss += 1
                        except Exception as e:
                            print(f"[Loss] Error saving checkpoint to {model_pth_path_loss}: {e}")
                            print(f"Available space in {base_model_path_loss}:")
                            os.system(f"df -h {base_model_path_loss}")
                            raise

                        best_loss = avg_train_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    # Save model based on validation AAE (only when validation runs)
                    if val_aae is not None and val_aae < best_val_aae - 1e-6:
                        # Delete previous validation-based checkpoint folder if it exists
                        if last_checkpoint_path_val is not None and os.path.exists(last_checkpoint_path_val):
                            try:
                                shutil.rmtree(last_checkpoint_path_val)
                                print(f"[Val AAE] Deleted previous checkpoint: {last_checkpoint_path_val}")
                            except Exception as e:
                                print(f"[Val AAE] Warning: Could not delete previous checkpoint: {e}")
                        
                        # Create new checkpoint folder for validation
                        base_model_path_val = "mlruns/0/models/LIFFN_validation_AAE/"
                        model_save_path_val = os.path.join(base_model_path_val, str(checkpoint_counter_val))
                        
                        try:
                            os.makedirs(model_save_path_val, exist_ok=True)
                            print(f"[Val AAE] Created checkpoint directory: {model_save_path_val}")
                        except Exception as e:
                            print(f"[Val AAE] Error creating directory {model_save_path_val}: {e}")
                            os.makedirs(base_model_path_val, exist_ok=True)
                            os.makedirs(model_save_path_val, exist_ok=True)

                        save_data_val = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': data.epoch,
                            'loss': avg_train_loss,
                            'val_AAE': val_aae,
                            'val_AEE': val_aee,
                            'config': config
                        }

                        model_pth_path_val = os.path.join(model_save_path_val, 'model.pth')
                        try:
                            temp_path = model_pth_path_val + '.tmp'
                            torch.save(save_data_val, temp_path)
                            os.replace(temp_path, model_pth_path_val)
                            print(f"[Val AAE] Model checkpoint saved to: {model_pth_path_val} (AAE: {val_aae:.4f})")
                            
                            try:
                                mlflow.log_artifact(model_pth_path_val)
                            except Exception as e:
                                print(f"[Val AAE] Warning: Could not log artifact to mlflow: {e}")
                            
                            last_checkpoint_path_val = model_save_path_val
                            checkpoint_counter_val += 1
                        except Exception as e:
                            print(f"[Val AAE] Error saving checkpoint to {model_pth_path_val}: {e}")
                            print(f"Available space in {base_model_path_val}:")
                            os.system(f"df -h {base_model_path_val}")
                            raise

                        best_val_aae = val_aae

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)

                # save grads to file
                if config["vis"]["store_grads"]:
                    save_csv(grads_w, "grads_w.csv")
                    grads_w = []

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"] or epochs_without_improvement >= patience:
                    print(f"Stopping at epoch {data.epoch}.")
                    end_train = True

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
                
                # Log gradients for all layers with respect to parameters
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        mlflow.log_metric(f"grad_mean_{name}", param.grad.mean().item(), step=data.epoch)

                # clip and save grads
                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
                if config["vis"]["store_grads"]:
                    grads_w.append(get_grads(model.named_parameters()))

                optimizer.step()
                optimizer.zero_grad()

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask and config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    flow_vis *= loss_function.event_mask

                model.detach_states()
                loss_function.reset()

                # visualize
                with torch.no_grad():
                    if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                        vis.update(inputs, flow_vis, None)

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

        if end_train:
            break

        # --- LOGGING OF MODEL PARAMETERS AND GRADIENTS ---
        # Log only self.lif.beta and self.lif.threshold for all LIF layers and weights for Conv2d layers
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                mlflow.log_metric(f"conv_weight_mean_{module_name}", module.weight.data.mean().item(), step=data.epoch)
            if hasattr(module, 'lif'):
                # lif.beta and lif.threshold should be nn.Parameter with shape [channels, 1, 1]
                beta = getattr(module.lif, 'beta', None)
                threshold = getattr(module.lif, 'threshold', None)
                if beta is not None:
                    mlflow.log_metric(f"lif_beta_mean_{module_name}", beta.data.mean().item(), step=data.epoch)
                if threshold is not None:
                    mlflow.log_metric(f"lif_threshold_mean_{module_name}", threshold.data.mean().item(), step=data.epoch)

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_SNN.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--val_config",
        default="configs/eval_MVSEC.yml",
        help="validation configuration (MVSEC dataset)",
    )
    parser.add_argument(
        "--val_every_n_epochs",
        type=int,
        default=1,
        help="run validation every N epochs (default: 1)",
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
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))
