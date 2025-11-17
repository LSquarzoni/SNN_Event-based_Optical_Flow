from torch.optim import *
from torchinfo import summary
import torch
import numpy as np

from brevitas.graph.calibrate import calibration_mode
from brevitas import config as cf
import argparse
import mlflow

cf.IGNORE_MISSING_KEYS = True

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import AEE, NEE, AAE, NAAE, AE_ofMeans
from models.model import (
    LIFFireNet,
    LIFFireNet_short,
    LIFFireFlowNet,
    LIFFireFlowNet_short,
)
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, load_quantized_model, create_model_dir
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization


def calibrate_model_ptq(calibration_loader, model, device, num_batches=50):
    """
    Calibrate the model for Post-Training Quantization (PTQ).
    This collects statistics to initialize quantization parameters without training.
    """
    print("="*60)
    print("Starting PTQ Calibration...")
    print(f"Using {num_batches} batches for calibration")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    with calibration_mode(model):
        with torch.no_grad():
            for batch_idx, item in enumerate(calibration_loader):
                if batch_idx >= num_batches:
                    break
                
                x = item["event_voxel"].to(device)
                cnt = item["event_cnt"].to(device)
                
                # Forward pass for calibration
                model(x, cnt)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Calibration progress: {batch_idx + 1}/{num_batches} batches")
    
    print("PTQ Calibration completed!")
    print("="*60)
    return model


def test_quantized(args, config_parser):
    """
    Evaluate quantized models with two modes:
    1. QAT: Load models trained with Quantization-Aware Training
    2. PTQ: Apply Post-Training Quantization to FP32 models
    """
    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)

    # configs
    if config["loader"]["batch_size"] > 1:
        raise NotImplementedError("Batch size > 1 not supported for evaluation")

    # asserts
    if "AEE" in config["metrics"]["name"]:
        if not config["data"]["mode"] in ["gtflow_dt1", "gtflow_dt4"]:
            raise Exception("Ground truth mode not enabled. AEE cannot be computed.")

    if "AEE" in config["metrics"]["name"]:
        if not config["loss"]["overwrite_intermediate"]:
            config["loss"]["overwrite_intermediate"] = True
            print("Warning: 'overwrite_intermediate' set to True for AEE computation")

    if config["data"]["mode"] == "frames":
        raise NotImplementedError("Frame mode not supported")

    # Determine quantization mode
    use_ptq = config["model"].get("quantization", {}).get("PTQ", False)
    quantization_enabled = config["model"].get("quantization", {}).get("enabled", False)
    
    if not quantization_enabled:
        raise ValueError("Quantization not enabled in config! Set model.quantization.enabled: True")
    
    print("="*80)
    if use_ptq:
        print("Evaluation Mode: POST-TRAINING QUANTIZATION (PTQ)")
        print("Will apply quantization to FP32 model during evaluation")
    else:
        print("Evaluation Mode: QUANTIZATION-AWARE TRAINING (QAT)")
        print("Will load model trained with QAT")
    print("="*80)

    if not args.debug:
        # create directory for inference results
        path_results = create_model_dir(args.path_results, args.runid)
        
        # store validation settings (this will start/end an mlflow run)
        eval_id = log_config(path_results, args.runid, config)
        
        # Now start a new run for logging metrics
        mlflow.set_experiment(config["experiment"])
        mlflow.start_run()
        mlflow.log_params(config)
    else:
        path_results = None
        eval_id = -1

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    vis = None
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # data loader
    data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=False,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # model initialization with quantization enabled
    print(f"Initializing quantized model: {config['model']['name']}")
    model = eval(config["model"]["name"])(config["model"].copy()).to(device)
    
    
    
    if use_ptq:
        # PTQ Mode: Load FP32 model, then calibrate
        print("\nLoading FP32 pre-trained model for PTQ...")
        
        # FP32 MODELS:
        model_path_dir = "mlruns/0/models/LIFFN/38/model.pth" # runid: e1965c33f8214d139624d7e08c7ec9c1
        #model_path_dir = "mlruns/0/models/LIFFN_16ch/38/model.pth" # runid: b6764e1aa848462c89dc70ea9d99246e
        #model_path_dir = "mlruns/0/models/LIFFN_8ch/12/model.pth" # runid: b41ac25a81064a72ac818dce9b25d4d6
        #model_path_dir = "mlruns/0/models/LIFFN_4ch/12/model.pth" # runid: d27de9a1834748f8857b891ab6eba05e
        #model_path_dir = "mlruns/0/models/LIFFN_short/39/model.pth" # runid: bb4ece23356043fca1204176cb270c7d
        #model_path_dir = "mlruns/0/models/LIFFN_16ch_short/33/model.pth" # runid: 7b3c8e69807d44c79abc682e96ff57e1
        #model_path_dir = "mlruns/0/models/LIFFN_8ch_short/23/model.pth" # runid: b61534e5119a4704a66638c1ba78f308
        #model_path_dir = "mlruns/0/models/LIFFN_4ch_short/5/model.pth" # runid: 4ea97793680843e99fd7aaffc2a717ef
        
        # Use strict=False for PTQ to handle architecture mismatches between old/new LIF implementations
        print("Note: Loading with strict=False to handle potential architecture differences")
        model = load_model(args.runid, model, device, model_path_dir=model_path_dir, strict=False)
        
        # Calibration for PTQ
        print("\nCalibrating model for PTQ...")
        calibration_batches = args.calibration_batches if hasattr(args, 'calibration_batches') else 50
        model = calibrate_model_ptq(dataloader, model, device, num_batches=calibration_batches)
        
        print("\nPTQ model ready for evaluation")
        print("Note: This uses fake quantization (FP32 compute with quantization simulation)")
    else:
        # QAT Mode: Load quantized checkpoint
        print("\nLoading QAT quantized model checkpoint...")
        
        # QAT MODELS:
        model_path_dir = "mlruns/0/models/LIFFN_int8/model_quant_best.pth" # runid: d5ac111464894b2e8379baaa538eeca7
        #model_path_dir = "mlruns/0/models/LIFFN_16ch_int8/model_quant_best.pth" # runid: 
        #model_path_dir = "mlruns/0/models/LIFFN_8ch_int8/model_quant_best.pth" # runid: 
        #model_path_dir = "mlruns/0/models/LIFFN_short_int8/model_quant_best.pth" # runid: 
        #model_path_dir = "mlruns/0/models/LIFFN_short_16ch_int8/model_quant_best.pth" # runid: 
        
        if model_path_dir:
            model, checkpoint = load_quantized_model(model, model_path_dir, device)
            if checkpoint:
                print(f"Loaded from: {model_path_dir}")
                if 'epoch' in checkpoint:
                    print(f"  Trained for {checkpoint['epoch']} epochs")
                if 'loss' in checkpoint:
                    print(f"  Final loss: {checkpoint['loss']:.6f}")
            
            # Fix device mismatch: Move all quantization parameters to the correct device
            print(f"\nMoving quantization metadata to {device}...")
            model = model.to(device)
            
            # Calibrate state quantization if QAT was trained without it
            # (QAT models typically don't have state quantization due to VRAM constraints)
            if hasattr(args, 'calibration_batches') and args.calibration_batches > 0:
                print("\nCalibrating state quantization for QAT model...")
                print("(QAT trained weights/activations, now calibrating states only)")
                calibration_batches = args.calibration_batches
                model = calibrate_model_ptq(dataloader, model, device, num_batches=calibration_batches)
        else:
            print("Error: model_path_dir not set for QAT evaluation")
            return
    
    model.eval()

    # validation metric
    criteria = []
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))

    # inference loop
    idx_AEE = 0
    val_results = {}
    end_test = False
    
    print("\n" + "="*80)
    print("Starting Evaluation")
    print("="*80)
    
    try:
        with torch.no_grad():
            while True:
                for inputs in dataloader:
                    if data.new_seq:
                        data.new_seq = False
                        model.reset_states()
                    # finish inference loop
                    if data.seq_num >= len(data.files):
                        end_test = True
                        break
                    # forward pass
                    x = model(
                        inputs["event_voxel"].to(device), inputs["event_cnt"].to(device)
                    )
                    
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
                                # reset criteria
                                criteria[i].reset()
                    # visualize
                    if config["vis"]["bars"]:
                        for bar in data.open_files_bar:
                            bar.next()
                    if config["vis"]["enabled"]:
                        flow_vis = x["flow"][-1].clone()
                        if model.mask:
                            flow_vis *= inputs["event_mask"].to(device)
                        iwe = compute_pol_iwe(
                            x["flow"][-1],
                            inputs["event_list"].to(device),
                            config["loader"]["resolution"],
                            inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                            inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                            flow_scaling=config["metrics"]["flow_scaling"],
                            round_idx=True,
                        )
                        vis.update(inputs, x["flow"][-1].clone(), iwe, None, flow_vis, None)
                    if config["vis"]["store"]:
                        sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]
                        flow_vis = x["flow"][-1].clone()
                        if model.mask:
                            flow_vis *= inputs["event_mask"].to(device)
                        iwe = compute_pol_iwe(
                            x["flow"][-1],
                            inputs["event_list"].to(device),
                            config["loader"]["resolution"],
                            inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                            inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                            flow_scaling=config["metrics"]["flow_scaling"],
                            round_idx=True,
                        )
                        vis.store(
                            inputs,
                            x["flow"][-1].clone(),
                            iwe,
                            sequence,
                            None,
                            flow_vis,
                            None,
                            ts=data.last_proc_timestamp,
                        )
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
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            results[metric] = {}
            if metric in ["AEE", "NEE", "AE"]:
                results[metric + "_percent"] = {}
            for filename in val_results.keys():
                if metric in ["AEE", "NEE", "AE"]:
                    results[metric][filename] = str(val_results[filename][metric]["metric"] / val_results[filename][metric]["it"])
                    results[metric + "_percent"][filename] = str(val_results[filename][metric]["percent"] / val_results[filename][metric]["it"])
                else:
                    results[metric][filename] = str(val_results[filename][metric]["metric"] / val_results[filename][metric]["it"])
        
        # Save results to file
        if not args.debug:
            log_results(args.runid, results, path_results, eval_id)
    
    # Print final results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Model: {config['model']['name']}")
    print(f"Quantization Mode: {'PTQ' if use_ptq else 'QAT'}")
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            if metric in results:
                # Calculate average from stored results
                metric_values = [float(v) for v in results[metric].values()]
                avg_metric = np.mean(metric_values)
                print(f"Average {metric}: {avg_metric:.6f}")
                if metric in ["AEE", "NEE", "AE"] and metric + "_percent" in results:
                    percent_values = [float(v) for v in results[metric + "_percent"].values()]
                    avg_percent = np.mean(percent_values)
                    print(f"  Percent: {avg_percent:.2f}%")
    print("="*80)

    # log to mlflow
    if not args.debug and "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            if metric in results:
                metric_values = [float(v) for v in results[metric].values()]
                mlflow.log_metric(f"test_{metric.lower()}", np.mean(metric_values))
                if metric in ["AEE", "NEE", "AE"] and metric + "_percent" in results:
                    percent_values = [float(v) for v in results[metric + "_percent"].values()]
                    mlflow.log_metric(f"test_{metric.lower()}_percent", np.mean(percent_values))
        mlflow.log_param("quantization_mode", "PTQ" if use_ptq else "QAT")
        mlflow.end_run()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Quantized SNN Models (QAT or PTQ)")
    parser.add_argument(
        "--config",
        default="configs/eval_MVSEC.yml",
        help="evaluation configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--runid",
        required=True,
        help="MLflow run ID of the model to evaluate",
    )
    parser.add_argument(
        "--model_path",
        default="",
        help="Optional: Direct path to model checkpoint (for QAT models: path to .pth file)",
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=50,
        help="Number of batches for PTQ calibration (only used if PTQ=True)",
    )
    parser.add_argument(
        "--path_results",
        default="results_inference/",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: disable MLflow logging",
    )
    
    args = parser.parse_args()

    # Load and validate config
    config_parser = YAMLParser(args.config)
    
    if not config_parser.config["model"].get("quantization", {}).get("enabled", False):
        print("Error: Quantization not enabled in config!")
        print("Please set in your config:")
        print("  model:")
        print("    quantization:")
        print("      enabled: True")
        print("      type: int8")
        print("      PTQ: True/False  # True for PTQ, False for QAT")
        exit(1)
    
    # Print evaluation info
    use_ptq = config_parser.config["model"].get("quantization", {}).get("PTQ", False)
    print("\n" + "="*80)
    print("Quantized Model Evaluation")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Run ID: {args.runid}")
    print(f"Mode: {'PTQ (Post-Training Quantization)' if use_ptq else 'QAT (Quantization-Aware Training)'}")
    if args.model_path:
        print(f"Model Path: {args.model_path}")
    if use_ptq:
        print(f"PTQ Calibration Batches: {args.calibration_batches}")
    print("="*80 + "\n")
    
    # Run evaluation
    test_quantized(args, config_parser)
