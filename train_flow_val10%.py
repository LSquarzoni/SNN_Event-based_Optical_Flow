import argparse
import os
import random
import copy

import h5py
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
)
from models.model import (
    LIFFireNet,
    LIFFireNet_short,
    LIFFireFlowNet,
)
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_diff, save_model
from utils.visualization import Visualization

def get_next_model_folder(base_path="mlruns/0/models/"):
    index = 0
    while os.path.exists(os.path.join(base_path, str(index))):
        index += 1
    return os.path.join(base_path, str(index))

def create_train_val_split(files, val_ratio=0.1, random_seed=42):
    """Split files into training and validation sets without altering global RNG state"""
    rng = random.Random(random_seed)
    files_copy = files.copy()
    rng.shuffle(files_copy)

    val_size = max(1, int(len(files_copy) * val_ratio)) if len(files_copy) > 1 else 0
    val_files = files_copy[:val_size]
    train_files = files_copy[val_size:]

    print(f"Total files: {len(files_copy)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    return train_files, val_files

class CustomH5Loader(H5Loader):
    """Custom H5Loader that can be initialized with a specific file list"""
    def __init__(self, config, num_bins, round_encoding, file_list=None):
        super().__init__(config, num_bins, round_encoding)
        if file_list is not None:
            # Close any existing files
            for f in self.open_files:
                f.close()
            
            # Override with custom file list
            self.files = file_list
            
            # Re-initialize with new file list
            self.open_files = []
            self.batch_last_ts = []
            
            # Open files (limited by batch size from config)
            batch_size = config["loader"]["batch_size"]
            num_files_to_open = min(batch_size, len(self.files))
            for batch in range(num_files_to_open):
                self.open_files.append(h5py.File(self.files[batch], "r"))
                self.batch_last_ts.append(self.open_files[-1]["events/ts"][-1] - self.open_files[-1].attrs["t0"])
            
            # Reset batch indices
            self.batch_idx = [i for i in range(num_files_to_open)]
            self.batch_row = [0 for i in range(num_files_to_open)]

def validate_model(model, val_dataloader, val_data, val_config, device):
    """Run validation and return average validation loss"""
    model.eval()
    val_loss = 0.0
    val_samples = 0  # count number of loss windows
    
    # Create validation loss function
    val_loss_function = EventWarping(val_config, device)
    
    with torch.no_grad():
        val_data.seq_num = 0
        val_data.samples = 0
        
        for inputs in val_dataloader:
            if val_data.new_seq:
                val_data.new_seq = False
                model.reset_states()
                val_loss_function.reset()
            
            if val_data.seq_num >= len(val_data.files):
                break
                
            # Forward pass
            x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))
            
            # Event flow association
            val_loss_function.event_flow_association(
                x["flow"],
                inputs["event_list"].to(device),
                inputs["event_list_pol_mask"].to(device),
                inputs["event_mask"].to(device),
            )
            
            # Compute validation loss
            if val_loss_function.num_events >= val_config["data"]["window_loss"]:
                # Overwrite intermediate flow estimates with the final ones if configured
                if val_config["loss"]["overwrite_intermediate"]:
                    val_loss_function.overwrite_intermediate_flow(x["flow"])
                
                loss = val_loss_function()
                val_loss += loss.item()
                val_samples += 1
                val_loss_function.reset()
    
    model.train()  # Switch back to training mode
    avg_val_loss = val_loss / max(val_samples, 1) if val_samples > 0 else float('inf')
    return avg_val_loss

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

    # Create train/validation split
    temp_data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    all_files = temp_data.files
    train_files, val_files = create_train_val_split(all_files, val_ratio=0.1)
    
    # Clean up temporary loader
    for f in temp_data.open_files:
        f.close()

    # Create training dataloader with custom file list
    data = CustomH5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"], train_files)
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # Create validation dataloader
    val_dataloader = None
    val_data = None
    validation_enabled = False
    
    if val_files:
        # Deep copy to avoid mutating training config (critical!)
        val_config = copy.deepcopy(config)
        val_config["loader"]["batch_size"] = 1  # Use batch size 1 for validation
        val_config["loader"]["augment"] = []  # No augmentation for validation
        val_config["loader"]["augment_prob"] = []
        
        val_data = CustomH5Loader(val_config, val_config["model"]["num_bins"], val_config["model"]["round_encoding"], val_files)
        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            drop_last=True,
            batch_size=val_config["loader"]["batch_size"],
            collate_fn=val_data.custom_collate,
            shuffle=False,
        )
        validation_enabled = True

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
    best_val_loss = 1.0e6
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
                avg_train_loss_raw = train_loss / max(data.samples, 1)
                avg_train_loss_legacy = avg_train_loss_raw / config["loader"]["batch_size"]
                # Log both: keep 'train_loss' as legacy-style for backward comparability
                mlflow.log_metric("train_loss", avg_train_loss_legacy, step=data.epoch)
                #mlflow.log_metric("train_loss_raw", avg_train_loss_raw, step=data.epoch)

                # Run validation if enabled
                avg_val_loss = None
                if validation_enabled:
                    print(f"\nRunning validation at epoch {data.epoch}...")
                    avg_val_loss = validate_model(model, val_dataloader, val_data, val_config, device)
                    mlflow.log_metric("val_loss", avg_val_loss, step=data.epoch)
                    # Ensure recurrent states don't leak validation sequence context
                    model.reset_states()
                
                # Print epoch summary
                if validation_enabled and avg_val_loss is not None:
                    print(f"Epoch {data.epoch:04d} - Training Loss: {avg_train_loss_legacy:.6f}, Validation Loss: {avg_val_loss:.6f}")
                else:
                    print(f"Epoch {data.epoch:04d} - Training Loss: {avg_train_loss_legacy:.6f}")

                with torch.no_grad():
                    # Save model based on validation loss if available, otherwise training loss
                    current_metric = avg_val_loss if validation_enabled and avg_val_loss is not None else avg_train_loss_raw
                    
                    if current_metric < best_val_loss - 1e-6:  # small delta to prevent stopping on tiny changes
                        model_save_path = get_next_model_folder("mlruns/0/models/LIFFireNet_SNNtorch_val10%/")
                        os.makedirs(model_save_path, exist_ok=True)
                        
                        # Save just the state dict instead of the full model
                        save_data = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': data.epoch,
                            #'train_loss_raw': avg_train_loss_raw,
                            'train_loss': avg_train_loss_legacy,
                            'config': config
                        }
                        
                        # Add validation loss to saved data
                        if validation_enabled and avg_val_loss is not None:
                            save_data['val_loss'] = avg_val_loss
                            
                        torch.save(save_data, os.path.join(model_save_path, 'model.pth'))
                        
                        # Also log with MLflow but without autolog to avoid duplication
                        mlflow.log_artifact(os.path.join(model_save_path, 'model.pth'))
                        
                        best_val_loss = current_metric
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                # Shuffle training file order each epoch for better generalization
                data.shuffle()
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
                # Count number of loss windows instead of samples*batch_size
                data.samples += 1

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
                _raw = train_loss / max(data.samples, 1)
                _legacy = _raw / config["loader"]["batch_size"]
                print(
                    "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                        data.epoch,
                        data.seq_num,
                        len(data.files),
                        int(100 * data.seq_num / len(data.files)),
                        _legacy,
                    ),
                    end="\r",
                )

        if end_train:
            break

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))
