import argparse
import os
import shutil

import mlflow
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping, AEE, AE
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
    FC_1l_100x100
)
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_diff, save_model
from utils.visualization import Visualization

def get_next_model_folder(base_path="mlruns/0/models/"):
    index = 0
    while os.path.exists(os.path.join(base_path, str(index))):
        index += 1
    return os.path.join(base_path, str(index))

def create_validation_loader(train_config, validation_config_path):
    """Create validation dataloader for MVSEC dataset"""
    if not validation_config_path or not os.path.exists(validation_config_path):
        print(f"Validation config not found: {validation_config_path}")
        return None, None, None, None
        
    try:
        val_config_parser = YAMLParser(validation_config_path)
        val_config = val_config_parser.config
        
        # Keep validation config completely separate
        # Only copy model settings needed for dataloader compatibility
        val_config["model"] = {
            "num_bins": train_config["model"]["num_bins"],
            "round_encoding": train_config["model"].get("round_encoding", False)
        }
        
        # Create validation dataloader with pure validation config
        val_data = H5Loader(val_config, val_config["model"]["num_bins"], val_config["model"]["round_encoding"])
        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            drop_last=True,
            batch_size=val_config["loader"]["batch_size"],
            collate_fn=val_data.custom_collate,
            shuffle=False,
            **val_config_parser.loader_kwargs,
        )
        
        # Create validation metrics
        val_metrics = {
            "AEE": AEE(val_config, val_config_parser.device, flow_scaling=val_config["metrics"]["flow_scaling"]),
            "AE": AE(val_config, val_config_parser.device, flow_scaling=val_config["metrics"]["flow_scaling"])
        }
        
        return val_dataloader, val_data, val_metrics, val_config
    except Exception as e:
        print(f"Warning: Could not create validation loader: {e}")
        return None, None, None, None

def validate_model(model, val_dataloader, val_data, val_metrics, val_config, device):
    """Run validation on MVSEC dataset"""
    model.eval()
    val_results = {"AEE": 0.0, "AE": 0.0}
    val_counts = {"AEE": 0, "AE": 0}
    
    with torch.no_grad():
        val_data.seq_num = 0
        val_data.samples = 0
        
        for inputs in val_dataloader:
            if val_data.new_seq:
                val_data.new_seq = False
                model.reset_states()
                # Reset metrics
                for metric in val_metrics.values():
                    metric.reset()
            
            if val_data.seq_num >= len(val_data.files):
                break
                
            # Forward pass
            x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))
            
            # Compute validation metrics
            for metric_name, metric in val_metrics.items():
                metric.event_flow_association(x["flow"], inputs)
                
                if metric.num_events > 0:
                    result = metric()
                    # Handle tuple returns (error, percentage_outliers)
                    if isinstance(result, tuple):
                        error = result[0]  # Take the first element (main error)
                    else:
                        error = result
                    val_results[metric_name] += error.item()
                    val_counts[metric_name] += 1
    
    # Compute average validation errors
    avg_val_results = {}
    for metric_name in val_results:
        if val_counts[metric_name] > 0:
            avg_val_results[metric_name] = val_results[metric_name] / val_counts[metric_name]
        else:
            avg_val_results[metric_name] = float('inf')
    
    model.train()  # Switch back to training mode
    return avg_val_results

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

    # Create validation dataloader
    # val_dataloader, val_data, val_metrics, val_config = create_validation_loader(config, args.val_config)
    val_dataloader = None
    validation_enabled = val_dataloader is not None

    # loss function
    loss_function = EventWarping(config, device)
    
    #model_path_dir = "mlruns/0/models/LIFFN/38/model.pth" # runid: e1965c33f8214d139624d7e08c7ec9c1

    # model initialization and settings
    model = eval(config["model"]["name"])(config["model"].copy()).to(device)
    model = load_model(args.prev_runid, model, device)
    #model = load_model(args.runid, model, device, model_path_dir)
    
    model.train()

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # simulation variables
    patience = 50
    epochs_without_improvement = 0
    train_loss = 0
    best_loss = 1.0e6
    best_val_aee = 1.0e6  # Track best validation AEE
    end_train = False
    grads_w = []
    checkpoint_counter = 0  # Counter for checkpoint folders
    last_checkpoint_path = None  # Path to the last saved checkpoint

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

                # Run validation if enabled
                val_results = {}
                if validation_enabled:
                    print(f"\nRunning validation at epoch {data.epoch}...")
                    val_results = validate_model(model, val_dataloader, val_data, val_metrics, val_config, device)
                    
                    # Log validation metrics
                    for metric_name, value in val_results.items():
                        mlflow.log_metric(f"val_{metric_name}", value, step=data.epoch)
                
                # Print epoch summary with both training loss and validation results
                if validation_enabled and val_results:
                    print(f"Epoch {data.epoch:04d} - Training Loss: {avg_train_loss:.6f}, Validation AEE: {val_results.get('AEE', 'N/A'):.4f}, Validation AE: {val_results.get('AE', 'N/A'):.4f}")
                else:
                    print(f"Epoch {data.epoch:04d} - Training Loss: {avg_train_loss:.6f}")

                with torch.no_grad():
                    # Use combined metric for model selection if validation is enabled
                    if validation_enabled and val_results:
                        # Weighted sum (equal weights here, adjust as needed)
                        current_metric = 0.5 * val_results.get('AEE', avg_train_loss) + 0.5 * val_results.get('AE', avg_train_loss)
                        best_metric = 0.5 * best_val_aee + 0.5 * best_val_ae if ('best_val_ae' in locals()) else float('inf')
                    else:
                        current_metric = avg_train_loss
                        best_metric = best_loss

                    # Save model if combined metric improves
                    if current_metric < best_metric - 1e-6:
                        # Delete previous checkpoint folder if it exists
                        if last_checkpoint_path is not None and os.path.exists(last_checkpoint_path):
                            try:
                                shutil.rmtree(last_checkpoint_path)
                                print(f"Deleted previous checkpoint: {last_checkpoint_path}")
                            except Exception as e:
                                print(f"Warning: Could not delete previous checkpoint: {e}")
                        
                        # Create new checkpoint folder with incremented counter
                        base_model_path = "mlruns/0/models/LIFFN_best2/"
                        model_save_path = os.path.join(base_model_path, str(checkpoint_counter))
                        
                        try:
                            os.makedirs(model_save_path, exist_ok=True)
                            print(f"Created checkpoint directory: {model_save_path}")
                        except Exception as e:
                            print(f"Error creating directory {model_save_path}: {e}")
                            print(f"Attempting to create parent directory {base_model_path}")
                            os.makedirs(base_model_path, exist_ok=True)
                            os.makedirs(model_save_path, exist_ok=True)

                        save_data = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': data.epoch,
                            'loss': avg_train_loss,
                            'config': config
                        }
                        if validation_enabled:
                            save_data['validation_results'] = val_results

                        model_pth_path = os.path.join(model_save_path, 'model.pth')
                        try:
                            # Save to temporary file first, then rename (atomic operation on most filesystems)
                            temp_path = model_pth_path + '.tmp'
                            torch.save(save_data, temp_path)
                            # Rename temp file to final name (more atomic than direct write)
                            os.replace(temp_path, model_pth_path)
                            print(f"Model checkpoint saved to: {model_pth_path}")
                            
                            try:
                                mlflow.log_artifact(model_pth_path)
                            except Exception as e:
                                print(f"Warning: Could not log artifact to mlflow: {e}")
                            
                            # Update checkpoint tracking
                            last_checkpoint_path = model_save_path
                            checkpoint_counter += 1
                        except Exception as e:
                            print(f"Error saving checkpoint to {model_pth_path}: {e}")
                            print(f"Available space in {base_model_path}:")
                            os.system(f"df -h {base_model_path}")
                            raise

                        if validation_enabled and val_results:
                            best_val_aee = val_results.get('AEE', best_val_aee)
                            best_val_ae = val_results.get('AE', float('inf'))
                        else:
                            best_loss = current_metric
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

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

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("runid", default="", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/train_SNN.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--val_config",
        default="configs/validation_config.yml",
        help="validation configuration (optional)",
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
