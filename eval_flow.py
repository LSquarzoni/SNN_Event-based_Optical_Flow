import argparse
import os

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
from dataloader.h5 import H5Loader, H5Loader_original
from loss.flow import AEE, NEE, AAE, NAAE, AE_ofMeans, AAE_Weighted, AAE_Filtered
from models.model import (
    LIFFireNet,
    LIFFireNet_short,
    LIFFireFlowNet,
    LIFFireFlowNet_short,
    SpikingRecEVFlowNet,
)
from utils.iwe import compute_pol_iwe, upsample_flow
from utils.utils import load_model, create_model_dir
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization, vis_activity

def calibrate_model(calibration_loader, quant_model, device, args):
    quant_model = quant_model.to(device)
    quant_model.eval()
    with torch.no_grad():
        # Put the model in calibration mode to collect statistics
        # Quantization is automatically disabled during the calibration, and re-enabled at the end
        with calibration_mode(quant_model):
            for i, inputs in enumerate(calibration_loader):
                event_voxel = inputs["event_voxel"].to(device)
                event_cnt = inputs["event_cnt"].to(device)
                    
                print(f'Calibration iteration {i}')
                quant_model(event_voxel, event_cnt)
                
                if i >= 50:  # Calibrate on first 50 batches
                    break
    return quant_model

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

    # visualization tool
    vis_type = config["vis"].get("type", "gradients")  # Default to "gradients" if not specified
    vis = None
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id, path_results=path_results, vis_type=vis_type)

    # model initialization and settings
    
    # FINAL MODELS: simplification of the LIF code
    #model_path_dir = "mlruns/0/models/LIFFN/38/model.pth" # runid: e1965c33f8214d139624d7e08c7ec9c1
    #model_path_dir = "mlruns/0/models/LIFFN_16ch/38/model.pth" # runid: b6764e1aa848462c89dc70ea9d99246e
    #model_path_dir = "mlruns/0/models/LIFFN_8ch/12/model.pth" # runid: b41ac25a81064a72ac818dce9b25d4d6
    #model_path_dir = "mlruns/0/models/LIFFN_4ch/12/model.pth" # runid: d27de9a1834748f8857b891ab6eba05e
    #model_path_dir = "mlruns/0/models/LIFFN_short/39/model.pth" # runid: bb4ece23356043fca1204176cb270c7d
    #model_path_dir = "mlruns/0/models/LIFFN_16ch_short/33/model.pth" # runid: 7b3c8e69807d44c79abc682e96ff57e1
    #model_path_dir = "mlruns/0/models/LIFFN_8ch_short/23/model.pth" # runid: b61534e5119a4704a66638c1ba78f308
    #model_path_dir = "mlruns/0/models/LIFFN_4ch_short/5/model.pth" # runid: 4ea97793680843e99fd7aaffc2a717ef
    
    #model_path_dir = "mlruns/0/models/LIFFFN/24/model.pth" # runid: cc75ff82496a4dc6896f2464898f774f
    #model_path_dir = "mlruns/0/models/LIFFFN_16ch/23/model.pth" # runid: 5263c20879994d469904e950f8835953
    #model_path_dir = "mlruns/0/models/LIFFFN_8ch/15/model.pth" # runid: 4899a1984ba74c91a44925426aa7c397
    #model_path_dir = "mlruns/0/models/LIFFFN_4ch/5/model.pth" # runid: 24a9b249620b48318d4b12dc4ce7b7a2
    #model_path_dir = "mlruns/0/models/LIFFFN_short/31/model.pth" # runid: 5d9d714fafa144a888f770199de6ac46
    #model_path_dir = "mlruns/0/models/LIFFFN_16ch_short/24/model.pth" # runid: 9b45bea0c1cd481a94a046e71793ef32
    #model_path_dir = "mlruns/0/models/LIFFFN_8ch_short/11/model.pth" # runid: f056dc2aa6e04f20b7760408eb563f1c
    #model_path_dir = "mlruns/0/models/LIFFFN_4ch_short/9/model.pth" # runid: 4ba018c376724267aee4bc66cd18d35c
    
    # 256x256 DATASET:
    #model_path_dir = "mlruns/0/models/LIFFN_256x256//model.pth" # runid: 97538a1b16bb4eed982a4da6db8bad16
    
    # POOLED MODELS:
    #model_path_dir = "mlruns/0/models/LIFFN_128x128/5/model.pth" # runid: 84cfb35b11e749d891d8d17b56fa75e0
    
    # NORMALIZATION MODELS:
    #model_path_dir = "mlruns/0/models/LIFFN_TEBN/30/model.pth" # runid: 4dbd002b28d448f59d620c2d94626907
    #model_path_dir = "mlruns/0/models/LIFFN_MPBN/27/model.pth" # runid: 4a07c11db1f04d50ae354b0919471be4
    #model_path_dir = "mlruns/0/models/LIFFN_GN/48/model.pth" # runid: 5be494fe70934ca4b5585df4d1384e53
    #model_path_dir = "mlruns/0/models/LIFFN_GN2/25/model.pth" # runid: d48a0fecb3744a878094155524f3cd91
    
    model_path_dir = "mlruns/0/models/LIFFN_BN/33/model.pth" # runid: ebbe836259ba488fa1714c7c36a5b3da
    #model_path_dir = "mlruns/0/models/LIFFN_BN_16ch/30/model.pth" # runid: 66c2e7eb3dd747b3a3ae529768871836
    #model_path_dir = "mlruns/0/models/LIFFN_BN_8ch/23/model.pth" # runid: 5beef02b4ac2463dae9c68dd90c03fb9
    #model_path_dir = "mlruns/0/models/LIFFN_BN_4ch/10/model.pth" # runid: 3c0f8ac17d254580a4f0246320bbb095
    #model_path_dir = "mlruns/0/models/LIFFN_short_BN/28/model.pth" # runid: 1b39ec58c0094ee7b8e27c46d46f7935
    #model_path_dir = "mlruns/0/models/LIFFN_short_BN_16ch/30/model.pth" # runid: 9de5f4a3c2e442fab8ce686d5afa99ec
    #model_path_dir = "mlruns/0/models/LIFFN_short_BN_8ch/18/model.pth" # runid: 23b1cb23745747f590a297d9b0027460
    #model_path_dir = "mlruns/0/models/LIFFN_short_BN_4ch/9/model.pth" # runid: 09911d92cbaf435ba7179e8466156d5c
    
    # NEW TRAINING TESTS:
    #model_path_dir = "mlruns/0/models/LIFFN_short_BN_16ch_2/smoothest_loss//model.pth" # runid: 71c14720756d4f2aa2ff957ad96c71fc
    #model_path_dir = "mlruns/0/models/LIFFN_short_BN_8ch_2/smoothest_loss//model.pth" # runid: e7895ea06daa4b7e9934e37f42730906
    
    model = eval(config["model"]["name"])(config["model"]).to(device)
    
    #model = load_model(args.runid, model, device) #                                         MODEL PATH AUTOMATIC (from runid) --------------------
    model = load_model(args.runid, model, device, model_path_dir) #                         MODEL PATH FROM MY TRAINING ---------------------------
    model.eval()

    # validation metric
    criteria = []
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))
    
    # Update metrics resolution and flow_scaling if keeping GT at full resolution
    keep_gt_full_res = config["loader"].get("keep_gt_full_res", False)
    if keep_gt_full_res and criteria:
        std_resolution = config["loader"].get("std_resolution", config["loader"]["resolution"])
        model_resolution = config["loader"]["resolution"]
        base_flow_scaling = config["metrics"]["flow_scaling"]
        
        # Adjust flow_scaling proportionally to inference resolution
        training_resolution = 128  # Models trained at 128×128
        adjusted_flow_scaling = base_flow_scaling * (model_resolution[0] / training_resolution) / 2
        
        for criterion in criteria:
            criterion.res = std_resolution
            criterion.flow_scaling = adjusted_flow_scaling

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
    
    # Quantization calibration for Post Training Uqantization
    if config['model']['quantization']['PTQ']:
        model = calibrate_model(dataloader, model, device, args)
        
        # Reset the dataloader for actual inference
        data = H5Loader(config, config["model"]["num_bins"])
        dataloader = torch.utils.data.DataLoader(
            data,
            drop_last=True,
            batch_size=config["loader"]["batch_size"],
            collate_fn=data.custom_collate,
            worker_init_fn=config_parser.worker_init_fn,
            **kwargs,
        )

    # inference loop
    idx_AEE = 0
    val_results = {}
    end_test = False
    activity_log = None
    try:
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
                        inputs["event_voxel"].to(device), inputs["event_cnt"].to(device), log=config["vis"]["activity"]
                    )
                    
                    # mask flow for visualization (before upsampling, at model resolution)
                    flow_vis_unmasked = x["flow"][-1].clone()
                    flow_vis = x["flow"][-1].clone()
                    
                    # image of warped events (computed at model resolution before upsampling)
                    iwe = compute_pol_iwe(
                        x["flow"][-1],
                        inputs["event_list"].to(device),
                        config["loader"]["resolution"],
                        inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                        inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                        flow_scaling=config["metrics"]["flow_scaling"],
                        round_idx=True,
                    )
                    
                    # Upsample predictions if GT is at full resolution and model at lower resolution
                    keep_gt_full_res = config["loader"].get("keep_gt_full_res", False)
                    if keep_gt_full_res and "gtflow" in inputs:
                        gt_flow_h, gt_flow_w = inputs["gtflow"].shape[2], inputs["gtflow"].shape[3]
                        pred_flow_h, pred_flow_w = x["flow"][-1].shape[2], x["flow"][-1].shape[3]
                        
                        # If GT is at higher resolution, upsample predictions to match
                        if gt_flow_h > pred_flow_h or gt_flow_w > pred_flow_w:
                            x["flow"][-1] = upsample_flow(x["flow"][-1], gt_flow_h, gt_flow_w)
                            # Also scale flow values: if spatial dimensions increased by 2x, flow should also increase by 2x
                            scale_factor_h = gt_flow_h / pred_flow_h
                            scale_factor_w = gt_flow_w / pred_flow_w
                            x["flow"][-1][:, 0, :, :] *= scale_factor_h  # Scale y component
                            x["flow"][-1][:, 1, :, :] *= scale_factor_w  # Scale x component
                    
                    # Apply mask after upsampling to ensure dimensions match
                    flow_vis = x["flow"][-1].clone()
                    if model.mask:
                        flow_vis *= inputs["event_mask"].to(device)
                    
                    iwe_window_vis = None
                    events_window_vis = None
                    masked_window_flow_vis = None
                    error_map_vis = None  # Initialize error map
                    if "metrics" in config.keys():
                        # event flow association
                        for metric in criteria:
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
                                
                                # Extract error map immediately after metric computation (for visualization)
                                if metric == "AAE":
                                    error_map_vis = criteria[i].get_error_map()
                                
                                # accumulate results
                                for batch in range(config["loader"]["batch_size"]):
                                    filename = data.files[data.batch_idx[batch] % len(data.files)].split("/")[-1]
                                    if filename not in val_results.keys():
                                        val_results[filename] = {}
                                        for metric in config["metrics"]["name"]:
                                            val_results[filename][metric] = {}
                                            val_results[filename][metric]["metric"] = 0
                                            val_results[filename][metric]["it"] = 0
                                            if metric in ["AEE", "NEE", "AAE"]:
                                                val_results[filename][metric]["percent"] = 0
                                    val_results[filename][metric]["it"] += 1
                                    if metric in ["AEE", "NEE", "AAE"]:
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
                                    # Prepare visualization data (works for both events and gtflow modes)
                                    if config["data"]["mode"] == "events":
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
                            flow_vis_unmasked,  # full flow (unmasked)
                            iwe,
                            sequence,
                            events_window_vis,
                            flow_vis,           # masked flow for masked visualizations
                            iwe_window_vis,
                            ts=data.last_proc_timestamp,
                            error_map=error_map_vis,
                        )
                    if config["vis"]["activity"]:
                        activity_log = vis_activity(x["activity"], activity_log)
                if end_test:
                    break
        if config["vis"]["bars"]:
            for bar in data.open_files_bar:
                bar.finish()
    except KeyboardInterrupt:
        print("Evaluation interrupted. Closing video files...")
    finally:
        if vis is not None:
            vis.close_videos()

    # store validation config and results
    results = {}
    if not args.debug and "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            results[metric] = {}
            if metric in ["AEE", "NEE", "AAE"]:
                results[metric + "_percent"] = {}
            for key in val_results.keys():
                results[metric][key] = str(val_results[key][metric]["metric"] / val_results[key][metric]["it"])
                if metric in ["AEE", "NEE", "AAE"]:
                    results[metric + "_percent"][key] = str(
                        val_results[key][metric]["percent"] / val_results[key][metric]["it"]
                    )
            log_results(args.runid, results, path_results, eval_id)

    # Save aggregated error heatmaps if enabled
    if not args.debug and "metrics" in config.keys() and config["metrics"].get("heat_map", False):
        print("\nSaving aggregated error heatmaps...")
        
        # Get magnitude threshold from config (default: 0.5)
        mag_threshold = config["metrics"].get("mag_threshold", 0.5)
        show_overlay = config["metrics"].get("show_magnitude_overlay", True)
        save_magnitude = config["metrics"].get("save_magnitude_map", True)
        save_combined = config["metrics"].get("save_combined", True)
        
        for i, metric in enumerate(config["metrics"]["name"]):
            if metric in ["AEE", "AAE", "NAAE"]:  # Only these metrics support heatmap visualization
                heatmap_dir = os.path.join(path_results, "heatmaps")
                os.makedirs(heatmap_dir, exist_ok=True)
                
                heatmap_path = os.path.join(heatmap_dir, f"{metric}_heatmap.png")
                success = criteria[i].save_error_heatmap(
                    save_path=heatmap_path,
                    title=f"Aggregated {metric} Error Distribution (Full Evaluation)",
                    mag_threshold=mag_threshold,
                    show_magnitude_overlay=show_overlay,
                    save_magnitude_map=save_magnitude,
                    save_combined=save_combined
                )
                if success:
                    print(f"  ✓ Saved {metric} heatmap to {heatmap_path}")
                else:
                    print(f"  ✗ Failed to save {metric} heatmap")

    # Close video writers if needed
    if vis is not None and config["vis"]["store_type"] == "video":
        vis.close_videos()


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
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
