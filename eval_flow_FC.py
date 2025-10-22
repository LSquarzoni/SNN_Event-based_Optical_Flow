import argparse

import mlflow
import numpy as np
import torch
from torch.optim import *
from torchinfo import summary

from brevitas.graph.calibrate import calibration_mode
from brevitas import config as cf
import torchvision.transforms as transforms

cf.IGNORE_MISSING_KEYS = True

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import AEE, NEE, AE
from models.model import (
    SNNtorch_FCLIF,
    SNNtorch_ConvFCLIF
)
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization, vis_activity

def test(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)

    # configs
    if config["loader"]["batch_size"] > 1:
        config["vis"]["enabled"] = False
        config["vis"]["store"] = False
        config["vis"]["bars"] = False  # progress bars not yet compatible batch_size > 1

    # asserts
    if "AEE" in config["metrics"]["name"]:
        assert (
            config["data"]["mode"] == "gtflow_dt1" or config["data"]["mode"] == "gtflow_dt4"
        ), "AEE computation not possible without ground truth mode"

    if "AEE" in config["metrics"]["name"]:
        assert config["data"]["window"] <= 1, "AEE computation not compatible with window > 1"
        assert np.isclose(
            (1.0 / config["data"]["window"]) % 1.0, 0.0
        ), "AEE computation not compatible with windows whose inverse is not a round number"

    if config["data"]["mode"] == "frames":
        if config["data"]["window"] <= 1.0:
            assert np.isclose(
                (1.0 / config["data"]["window"]) % 1.0, 0.0
            ), "Frames mode not compatible with < 1 windows whose inverse is not a round number"
        else:
            assert np.isclose(
                config["data"]["window"] % 1.0, 0.0
            ), "Frames mode not compatible with > 1 fractional windows"

    if not args.debug:
        # create directory for inference results
        path_results = create_model_dir(args.path_results, args.runid)

        # store validation settings
        eval_id = log_config(path_results, args.runid, config)
    else:
        path_results = None
        eval_id = -1

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool: prefer slim FCVisualization for FC models
    vis = None
    if config["vis"]["enabled"] or config["vis"]["store"]:
        # Use specialized FC visualization (8x8, vectors + GT gradient, video only)
        vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # model initialization and settings
    #model_path_dir = "mlruns/0/models/LIF_FC_3l/28/model.pth" # runid: 2a74ca0b11eb4ba69f4a550c20331119
    #model_path_dir = "mlruns/0/models/LIF_Conv_FC/29/model.pth" # runid: a9d7723f16d544bfa155282fc231ac1b
    #model_path_dir = "mlruns/0/models/LIF_FC_newOUT/18/model.pth" # runid: 15a795fdc3e343019118f363e947be7d
    #model_path_dir = "mlruns/0/models/LIF_Conv_FC_newOUT/11/model.pth" # runid: 9311acda2e714108bc3779b439b1639f
    #model_path_dir = "mlruns/0/models/LIF_FC_newIN/32/model.pth" # runid: 043cad279b7b4194b4cde29330ca03a8
    #model_path_dir = "mlruns/0/models/LIF_Conv_FC_newIN//model.pth" # runid:
    #model_path_dir = "mlruns/0/models/LIF_FC_newIN_OUT/21/model.pth" # runid: 4a4532f90d1e4863be45632a806b1d39
    #model_path_dir = "mlruns/0/models/LIF_Conv_FC_newIN_OUT//model.pth" # runid:
    model_path_dir = "/scratch2/msc25h1/models/LIF_FC_attempt2/0/model.pth" # runid: 5a5c4048f888433ea507930fce247283
    
    model = eval(config["model"]["name"])().to(device)
    
    #model = load_model(args.runid, model, device) #                                         MODEL PATH AUTOMATIC (from runid) --------------------
    model = load_model(args.runid, model, device, model_path_dir) #                         MODEL PATH FROM MY TRAINING ---------------------------
    model.eval()

    # validation metric
    criteria = []
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))

    # data loader
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # Initialize center crop transform if needed
    center_crop = None
    if config["loader"]["output_crop"]:
        center_crop = transforms.CenterCrop((config["loader"]["resolution"][0], config["loader"]["resolution"][1]))

    # inference loop
    idx_AEE = 0
    val_results = {}
    end_test = False
    activity_log = None
    with torch.no_grad():
        while True:
            for inputs in dataloader:

                if data.new_seq:
                    data.new_seq = False
                    activity_log = None
                    model.reset_states()

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # forward pass
                x = model(
                    inputs["event_cnt"].to(device), log=config["vis"]["activity"]
                )
                    
                if config["loader"]["output_crop"]:
                    resolution = config["loader"]["std_resolution"]
                else:
                    resolution = config["loader"]["resolution"]

                # Ensure model flow outputs are at the requested evaluation resolution
                # (models may output a single global flow [B,2,1,1] or small maps)
                target_h, target_w = int(resolution[0]), int(resolution[1])
                resized_flows = []
                for f in x["flow"]:
                    h_in, w_in = f.shape[2], f.shape[3]
                    if h_in == 1 and w_in == 1:
                        # exact tiling for a single global flow value
                        try:
                            resized_flows.append(f.expand(-1, -1, target_h, target_w))
                        except Exception:
                            # expand may return a non-contiguous view; fall back to repeat
                            resized_flows.append(f.repeat(1, 1, target_h, target_w))
                    else:
                        resized_flows.append(f)
                x["flow"] = resized_flows

                # Update visualization flows from resized outputs
                flow_vis_unmasked = x["flow"][-1].clone()
                flow_vis = x["flow"][-1].clone()

                # image of warped events
                iwe = compute_pol_iwe(
                    x["flow"][-1],
                    inputs["event_list"].to(device),
                    resolution,
                    inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                    inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                    flow_scaling=config["metrics"]["flow_scaling"],
                    round_idx=True,
                )

                iwe_window_vis = None
                events_window_vis = None
                masked_window_flow_vis = None
                if "metrics" in config.keys():
                    if config["loader"]["output_crop"]:
                        # Cropping the output of the model
                        x["flow"] = [center_crop(flow).contiguous() for flow in x["flow"]]
                        
                        # Create filtered inputs for metric computation
                        inputs_filtered = {}
                        for key, value in inputs.items():
                            inputs_filtered[key] = value
                        
                        # Apply center crop to ground truth flow if it exists
                        if "gtflow" in inputs and inputs["gtflow"].numel() > 0:
                            inputs_filtered["gtflow"] = center_crop(inputs["gtflow"]).contiguous()
                        
                        # Apply center crop to other resolution-dependent tensors
                        for tensor_key in ["event_voxel", "event_cnt", "event_mask"]:
                            if tensor_key in inputs and inputs[tensor_key].numel() > 0:
                                inputs_filtered[tensor_key] = center_crop(inputs[tensor_key]).contiguous()
                        
                        # Filter event_list based on crop boundaries if output_crop is enabled
                        if inputs["event_list"].numel() > 0:
                            crop_y_start = (config["loader"]["std_resolution"][0] - config["loader"]["resolution"][0]) // 2
                            crop_y_end = crop_y_start + config["loader"]["resolution"][0]
                            crop_x_start = (config["loader"]["std_resolution"][1] - config["loader"]["resolution"][1]) // 2
                            crop_x_end = crop_x_start + config["loader"]["resolution"][1]
                            
                            # Handle event_list shape: [batch, N_events, 4] where last dim is [ts, y, x, p]
                            event_list = inputs["event_list"]
                            batch_size = event_list.shape[0]
                            
                            filtered_events = []
                            for b in range(batch_size):
                                batch_events = event_list[b, :, :]  # [N_events, 4]
                                
                                if batch_events.shape[0] > 0:
                                    # Create mask for events within crop region (y, x are at indices 1, 2)
                                    event_mask = ((batch_events[:, 1] >= crop_y_start) & (batch_events[:, 1] < crop_y_end) & 
                                                (batch_events[:, 2] >= crop_x_start) & (batch_events[:, 2] < crop_x_end))
                                    
                                    # Filter events and adjust coordinates
                                    filtered_batch = batch_events[event_mask].clone()
                                    if filtered_batch.shape[0] > 0:
                                        filtered_batch[:, 1] -= crop_y_start  # Adjust y coordinates
                                        filtered_batch[:, 2] -= crop_x_start  # Adjust x coordinates
                                    
                                    filtered_events.append(filtered_batch)
                                else:
                                    # Empty batch
                                    filtered_events.append(torch.zeros(0, 4, device=batch_events.device, dtype=batch_events.dtype))
                            
                            # Reconstruct the tensor - pad to same length and stack
                            if len(filtered_events) > 0:
                                max_events = max(fe.shape[0] for fe in filtered_events) if any(fe.shape[0] > 0 for fe in filtered_events) else 0
                                
                                if max_events > 0:
                                    padded_events = []
                                    for fe in filtered_events:
                                        if fe.shape[0] < max_events:
                                            padding = torch.zeros(max_events - fe.shape[0], 4, device=fe.device, dtype=fe.dtype)
                                            fe = torch.cat([fe, padding], dim=0)
                                        padded_events.append(fe.unsqueeze(0))  # Add batch dimension back
                                    inputs_filtered["event_list"] = torch.cat(padded_events, dim=0)
                                else:
                                    # No events after filtering
                                    inputs_filtered["event_list"] = torch.zeros(batch_size, 0, 4, device=event_list.device, dtype=event_list.dtype)
                            else:
                                inputs_filtered["event_list"] = torch.zeros_like(event_list)
                            
                            # Also filter polarity mask if it exists
                            if inputs["event_list_pol_mask"].numel() > 0:
                                pol_mask = inputs["event_list_pol_mask"]
                                # Apply the same filtering to polarity mask - needs to handle same shape as event_list
                                filtered_pol_masks = []
                                for b in range(batch_size):
                                    batch_events = event_list[b, :, :]  # [N_events, 4]
                                    batch_pol_mask = pol_mask[b, :, :] if pol_mask.shape[0] > 1 else pol_mask[0, :, :]  # [N_events, 2]
                                    
                                    if batch_events.shape[0] > 0:
                                        event_mask = ((batch_events[:, 1] >= crop_y_start) & (batch_events[:, 1] < crop_y_end) & 
                                                    (batch_events[:, 2] >= crop_x_start) & (batch_events[:, 2] < crop_x_end))
                                        filtered_pol_batch = batch_pol_mask[event_mask].clone()
                                        filtered_pol_masks.append(filtered_pol_batch)
                                    else:
                                        filtered_pol_masks.append(torch.zeros(0, batch_pol_mask.shape[1], device=batch_pol_mask.device, dtype=batch_pol_mask.dtype))
                                
                                # Reconstruct polarity mask with same padding as event_list
                                if max_events > 0:
                                    padded_pol_masks = []
                                    for fpm in filtered_pol_masks:
                                        if fpm.shape[0] < max_events:
                                            padding = torch.zeros(max_events - fpm.shape[0], fpm.shape[1], device=fpm.device, dtype=fpm.dtype)
                                            fpm = torch.cat([fpm, padding], dim=0)
                                        padded_pol_masks.append(fpm.unsqueeze(0))
                                    inputs_filtered["event_list_pol_mask"] = torch.cat(padded_pol_masks, dim=0)
                                else:
                                    inputs_filtered["event_list_pol_mask"] = torch.zeros(batch_size, 0, pol_mask.shape[2], device=pol_mask.device, dtype=pol_mask.dtype)
                        
                    # event flow association
                    for metric in criteria:
                        if config["loader"]["output_crop"]:
                            metric.event_flow_association(x["flow"], inputs_filtered)
                        else:
                            metric.event_flow_association(x["flow"], inputs)

                    # validation
                    for i, metric in enumerate(config["metrics"]["name"]):
                        if criteria[i].num_events >= config["data"]["window_eval"]:

                            # overwrite intermedia flow estimates with the final ones
                            if config["loss"]["overwrite_intermediate"]:
                                criteria[i].overwrite_intermediate_flow(x["flow"])
                            if metric == "AEE" and inputs["dt_gt"] <= 0.0:
                                continue
                            if metric == "AEE":
                                idx_AEE += 1
                                if idx_AEE != np.round(1.0 / config["data"]["window"]):
                                    continue

                            # compute metric
                            val_metric = criteria[i]()
                            if metric == "AEE":
                                idx_AEE = 0

                            # accumulate results
                            for batch in range(config["loader"]["batch_size"]):
                                filename = data.files[data.batch_idx[batch] % len(data.files)].split("/")[-1]
                                if filename not in val_results.keys():
                                    val_results[filename] = {}
                                    for metric in config["metrics"]["name"]:
                                        val_results[filename][metric] = {}
                                        val_results[filename][metric]["metric"] = 0
                                        val_results[filename][metric]["it"] = 0
                                        if metric in ["AEE", "NEE", "AE"]:
                                            val_results[filename][metric]["percent"] = 0

                                val_results[filename][metric]["it"] += 1
                                if metric in ["AEE", "NEE", "AE"]:
                                    val_results[filename][metric]["metric"] += val_metric[0][batch].cpu().numpy()
                                    val_results[filename][metric]["percent"] += val_metric[1][batch].cpu().numpy()
                                else:
                                    val_results[filename][metric]["metric"] += val_metric[batch].cpu().numpy()

                            # visualize
                            if (
                                i == 0
                                and (config["vis"]["enabled"] or config["vis"]["store"]) 
                                and config["data"]["window"] < config["data"]["window_eval"]
                            ):
                                events_window_vis = criteria[i].compute_window_events()
                                iwe_window_vis = criteria[i].compute_window_iwe()
                                masked_window_flow_vis = criteria[i].compute_masked_window_flow()

                            # reset criteria
                            criteria[i].reset()

                # visualize
                if config["vis"]["bars"]:
                    for bar in data.open_files_bar:
                        bar.next()
                if config["vis"]["enabled"]: # flow_vis_unmasked -> show the unmasked flow, flow_vis -> show the masked flow
                    vis.update(inputs, flow_vis_unmasked, iwe, events_window_vis, flow_vis, iwe_window_vis)
                if config["vis"]["store"]:
                    sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]
                    vis.store(
                        inputs,
                        flow_vis,
                        iwe,
                        sequence,
                        events_window_vis,
                        masked_window_flow_vis,
                        iwe_window_vis,
                        ts=data.last_proc_timestamp,
                    )

                # visualize activity
                if config["vis"]["activity"]:
                    activity_log = vis_activity(x["activity"], activity_log)

            if end_test:
                break

    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
            bar.finish()

    # store validation config and results
    results = {}
    if not args.debug and "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            results[metric] = {}
            if metric in ["AEE", "NEE", "AE"]:
                results[metric + "_percent"] = {}
            for key in val_results.keys():
                results[metric][key] = str(val_results[key][metric]["metric"] / val_results[key][metric]["it"])
                if metric in ["AEE", "NEE", "AE"]:
                    results[metric + "_percent"][key] = str(
                        val_results[key][metric]["percent"] / val_results[key][metric]["it"]
                    )
            log_results(args.runid, results, path_results, eval_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_MVSEC_FC.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))