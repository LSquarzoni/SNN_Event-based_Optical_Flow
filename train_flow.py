import argparse
import os
import shutil

import mlflow
import numpy as np
import torch
from torch.optim import *

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
from utils.utils import load_model, save_csv, save_diff, save_model
from utils.visualization import Visualization

def get_next_model_folder(base_path="mlruns/0/models/"):
    index = 0
    while os.path.exists(os.path.join(base_path, str(index))):
        index += 1
    return os.path.join(base_path, str(index))

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
    end_train = False
    grads_w = []
    checkpoint_counter = 0  # Counter for checkpoint folders
    last_checkpoint_path = None  # Path to the last saved checkpoint
    
    # Multi-checkpoint tracking: save 3 diverse checkpoints for later evaluation
    recent_losses = []  # Track losses for variance calculation
    loss_history_window = 50  # Number of recent batches to track
    checkpoints = {
        'lowest_loss': {'loss': float('inf'), 'path': None, 'epoch': None},
        'smoothest_loss': {'variance': float('inf'), 'path': None, 'epoch': None},
        'most_recent': {'path': None, 'epoch': None}
    }
    base_model_path = "mlruns/0/models/LIFFN_short_BN_4ch/"  # Base path for all checkpoints
    
    """ # Anti-overfitting tracking
    consecutive_small_loss_decrease = 0
    min_loss_decrease_threshold = 1e-4  # Threshold to detect plateau """

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
                loss_variance = np.var(recent_losses) if len(recent_losses) > 1 else float('inf')
                mlflow.log_metric("loss", avg_train_loss, step=data.epoch)
                mlflow.log_metric("loss_variance", loss_variance, step=data.epoch)

                # Print epoch summary
                print(f"Epoch {data.epoch:04d} - Training Loss: {avg_train_loss:.6f} | Loss Variance: {loss_variance:.6f}")

                with torch.no_grad():
                    # Prepare save data
                    save_data = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': data.epoch,
                        'loss': avg_train_loss,
                        'loss_variance': loss_variance,
                        'config': config
                    }
                    
                    # Helper function to save checkpoint
                    def save_checkpoint(checkpoint_type, metric_value):
                        checkpoint_subdir = os.path.join(base_model_path, checkpoint_type, str(data.epoch))
                        try:
                            os.makedirs(checkpoint_subdir, exist_ok=True)
                            model_pth_path = os.path.join(checkpoint_subdir, 'model.pth')
                            
                            # Save to temporary file first, then rename (atomic)
                            temp_path = model_pth_path + '.tmp'
                            torch.save(save_data, temp_path)
                            os.replace(temp_path, model_pth_path)
                            
                            print(f"  Saved {checkpoint_type} checkpoint (epoch {data.epoch}): {metric_value:.6f}")
                            
                            try:
                                mlflow.log_artifact(model_pth_path)
                            except Exception as e:
                                print(f"  Warning: Could not log {checkpoint_type} checkpoint to mlflow: {e}")
                            
                            return checkpoint_subdir
                        except Exception as e:
                            print(f"  Error saving {checkpoint_type} checkpoint: {e}")
                            raise
                    
                    # Check and save 'lowest_loss' checkpoint
                    if avg_train_loss < checkpoints['lowest_loss']['loss']:
                        # Delete old checkpoint if exists
                        if checkpoints['lowest_loss']['path'] is not None and os.path.exists(checkpoints['lowest_loss']['path']):
                            try:
                                shutil.rmtree(checkpoints['lowest_loss']['path'])
                            except Exception as e:
                                print(f"  Warning: Could not delete old lowest_loss checkpoint: {e}")
                        
                        # Save new checkpoint
                        new_path = save_checkpoint('lowest_loss', avg_train_loss)
                        checkpoints['lowest_loss'] = {
                            'loss': avg_train_loss,
                            'path': new_path,
                            'epoch': data.epoch
                        }
                        best_loss = avg_train_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    # Check and save 'smoothest_loss' checkpoint (lowest variance = most stable)
                    if loss_variance < checkpoints['smoothest_loss']['variance'] and len(recent_losses) > 10:
                        # Delete old checkpoint if exists
                        if checkpoints['smoothest_loss']['path'] is not None and os.path.exists(checkpoints['smoothest_loss']['path']):
                            try:
                                shutil.rmtree(checkpoints['smoothest_loss']['path'])
                            except Exception as e:
                                print(f"  Warning: Could not delete old smoothest_loss checkpoint: {e}")
                        
                        # Save new checkpoint
                        new_path = save_checkpoint('smoothest_loss', loss_variance)
                        checkpoints['smoothest_loss'] = {
                            'variance': loss_variance,
                            'path': new_path,
                            'epoch': data.epoch
                        }
                    
                    # Always save 'most_recent' checkpoint (overwrite previous)
                    if checkpoints['most_recent']['path'] is not None and os.path.exists(checkpoints['most_recent']['path']):
                        try:
                            shutil.rmtree(checkpoints['most_recent']['path'])
                        except Exception as e:
                            print(f"  Warning: Could not delete old most_recent checkpoint: {e}")
                    
                    new_path = save_checkpoint('most_recent', avg_train_loss)
                    checkpoints['most_recent'] = {
                        'path': new_path,
                        'epoch': data.epoch
                    }

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
                
                # Track loss for variance calculation
                batch_loss = loss.item()
                recent_losses.append(batch_loss)
                if len(recent_losses) > loss_history_window:
                    recent_losses.pop(0)

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
