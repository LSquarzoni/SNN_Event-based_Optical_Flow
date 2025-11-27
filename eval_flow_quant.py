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
from loss.flow import AEE, NEE, AAE, NAAE, AE_ofMeans, AAE_Weighted, AAE_Filtered
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


def calibrate_model_ptq(calibration_loader, model, device, num_batches=50, calibrate_conv_only=False, calibrate_states_only=False):
    """
    Calibrate the model for Post-Training Quantization (PTQ).
    This collects statistics to initialize quantization parameters without training.
    
    Args:
        calibrate_conv_only: If True, only calibrate Conv layers (not LIF). Used for PTQ Conv-only mode.
        calibrate_states_only: If True, only calibrate LIF state quantizers (for Conv-only QAT + PTQ LIF).
                               Preserves trained Conv quantizers.
        
    Three calibration modes:
    1. Full PTQ (both False): Calibrate Conv + LIF using calibration_mode
    2. Conv-only PTQ (calibrate_conv_only=True): Calibrate only Conv, LIF stays FP32
    3. QAT Conv + PTQ LIF (calibrate_states_only=True): Preserve Conv, calibrate LIF only
    """
    print("="*60)
    if calibrate_states_only:
        print("Starting LIF-Only Calibration (Conv-only QAT + PTQ LIF)...")
        print("Preserving trained Conv weight/activation quantizers")
        print("Calibrating LIF layers with PTQ")
    elif calibrate_conv_only:
        print("Starting Conv-Only PTQ Calibration...")
        print("LIF layers will remain at FP32 (mixed precision)")
    else:
        print("Starting Full PTQ Calibration (Conv + LIF)...")
    print(f"Using up to {num_batches} batches for calibration (across all sequences)")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    if calibrate_states_only:
        # CASE 2: Hybrid mode - Conv-only QAT + PTQ LIF
        # Model has full quantization structure, Conv quantizers are trained, LIF quantizers need calibration
        
        # Find all q_lif state quantizers
        state_quantizers = []
        lif_modules = []
        for name, module in model.named_modules():
            if hasattr(module, 'q_lif') and not isinstance(module.q_lif, torch.nn.Identity):
                state_quantizers.append((name, module.q_lif))
                lif_modules.append((name, module))
        
        if len(state_quantizers) > 0:
            print(f"Found {len(state_quantizers)} LIF state quantizers to calibrate")
            print("Preserving trained Conv quantizers...")
            
            # Put ONLY the q_lif quantizers into calibration mode
            # We need to manually enable calibration for state quantizers since they're not regular layers
            # The forward pass through snn.Leaky with state_quant will trigger calibration
            
            # Use calibration_mode but it will only affect the q_lif quantizers during forward pass
            with calibration_mode(model):
                with torch.no_grad():
                    batch_count = 0
                    end_calibration = False
                    
                    while not end_calibration:
                        for item in calibration_loader:
                            # Handle new sequence - reset model states
                            if calibration_loader.dataset.new_seq:
                                calibration_loader.dataset.new_seq = False
                                model.reset_states()
                            
                            # Check if we've processed all sequences
                            if calibration_loader.dataset.seq_num >= len(calibration_loader.dataset.files):
                                end_calibration = True
                                break
                            
                            # Check if we've reached the batch limit
                            if batch_count >= num_batches:
                                end_calibration = True
                                break
                            
                            x = item["event_voxel"].to(device)
                            cnt = item["event_cnt"].to(device)
                            
                            # Forward pass - state quantizers will collect statistics
                            model(x, cnt)
                            
                            batch_count += 1
                            if batch_count % 10 == 0:
                                print(f"Calibration progress: {batch_count}/{num_batches} batches (sequence {calibration_loader.dataset.seq_num + 1}/{len(calibration_loader.dataset.files)})")
                        
                        if end_calibration:
                            break
                    
                    print(f"\nCalibration completed: {batch_count} batches across {calibration_loader.dataset.seq_num + 1} sequence(s)")
            
            print(f"\n✓ Successfully calibrated {len(state_quantizers)} LIF state quantizers")
        else:
            print("\nERROR: No state quantizers (q_lif) found!")
            print("This model doesn't have LIF quantization structure.")
            print("Please ensure the model is initialized with full quantization (Conv_only=False).")
            raise ValueError("Cannot apply PTQ to LIF: q_lif quantizers not found in model")
    
    elif calibrate_conv_only:
        # CASE 1b: PTQ from FP32, Conv-only mode
        # Calibrate ONLY Conv layers (weights, input_quant, output_quant)
        # Do NOT calibrate LIF state quantizers (q_lif)
        
        # We need to selectively enable calibration mode ONLY for Conv quantizers
        # Unfortunately, Brevitas calibration_mode() affects all quantizers
        # Workaround: Temporarily disable LIF quantizers
        
        lif_quantizers = []
        for name, module in model.named_modules():
            if hasattr(module, 'q_lif') and not isinstance(module.q_lif, torch.nn.Identity):
                lif_quantizers.append((name, module, module.q_lif))
                # Temporarily replace with Identity to prevent calibration
                module.q_lif = torch.nn.Identity()
        
        print(f"Temporarily disabled {len(lif_quantizers)} LIF quantizers")
        
        # Now calibrate with calibration_mode (will only affect Conv layers)
        with calibration_mode(model):
            with torch.no_grad():
                batch_count = 0
                end_calibration = False
                
                while not end_calibration:
                    for item in calibration_loader:
                        # Handle new sequence - reset model states
                        if calibration_loader.dataset.new_seq:
                            calibration_loader.dataset.new_seq = False
                            model.reset_states()
                        
                        # Check if we've processed all sequences
                        if calibration_loader.dataset.seq_num >= len(calibration_loader.dataset.files):
                            end_calibration = True
                            break
                        
                        # Check if we've reached the batch limit
                        if batch_count >= num_batches:
                            end_calibration = True
                            break
                        
                        x = item["event_voxel"].to(device)
                        cnt = item["event_cnt"].to(device)
                        
                        # Forward pass for calibration
                        model(x, cnt)
                        
                        batch_count += 1
                        if batch_count % 10 == 0:
                            print(f"Calibration progress: {batch_count}/{num_batches} batches (sequence {calibration_loader.dataset.seq_num + 1}/{len(calibration_loader.dataset.files)})")
                    
                    if end_calibration:
                        break
                
                print(f"\nCalibration completed: {batch_count} batches across {calibration_loader.dataset.seq_num + 1} sequence(s)")
        
        # Restore LIF quantizers (but they remain uncalibrated/disabled for FP32 operation)
        for name, module, original_q_lif in lif_quantizers:
            module.q_lif = original_q_lif
        
        print(f"LIF quantizers kept at FP32 (mixed precision mode)")
    
    else:
        # CASE 1a: Full PTQ from FP32 - calibrate everything
        with calibration_mode(model):
            with torch.no_grad():
                batch_count = 0
                end_calibration = False
                
                while not end_calibration:
                    for item in calibration_loader:
                        # Handle new sequence - reset model states
                        if calibration_loader.dataset.new_seq:
                            calibration_loader.dataset.new_seq = False
                            model.reset_states()
                        
                        # Check if we've processed all sequences
                        if calibration_loader.dataset.seq_num >= len(calibration_loader.dataset.files):
                            end_calibration = True
                            break
                        
                        # Check if we've reached the batch limit
                        if batch_count >= num_batches:
                            end_calibration = True
                            break
                        
                        x = item["event_voxel"].to(device)
                        cnt = item["event_cnt"].to(device)
                        
                        # Forward pass for calibration
                        model(x, cnt)
                        
                        batch_count += 1
                        if batch_count % 10 == 0:
                            print(f"Calibration progress: {batch_count}/{num_batches} batches (sequence {calibration_loader.dataset.seq_num + 1}/{len(calibration_loader.dataset.files)})")
                    
                    if end_calibration:
                        break
                
                print(f"\nCalibration completed: {batch_count} batches across {calibration_loader.dataset.seq_num + 1} sequence(s)")
    
    print("Calibration completed!")
    print("="*60)
    return model


def test_quantized(args, config_parser):
    """
    Evaluate quantized models with multiple modes:
    
    1. PTQ from FP32:
       - Load FP32 trained model
       - Apply calibration (Conv-only or Full)
       - Evaluate with quantized inference
    
    2. QAT Conv-only + PTQ LIF:
       - Load Conv-only QAT model (convs already quantized)
       - Optionally calibrate LIF layers (PTQ)
       - Evaluate with mixed or full quantization
    
    3. QAT Full (no calibration needed):
       - Load fully quantized QAT model
       - All layers already quantized during training
       - Direct evaluation
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
            config["loss"]["overwrite_intermediate"] = False
            #print("Warning: 'overwrite_intermediate' set to True for AEE computation")

    if config["data"]["mode"] == "frames":
        raise NotImplementedError("Frame mode not supported")

    # Determine quantization mode
    use_ptq = config["model"].get("quantization", {}).get("PTQ", False)
    quantization_enabled = config["model"].get("quantization", {}).get("enabled", False)
    conv_only_config = config["model"].get("quantization", {}).get("Conv_only", False)
    
    if not quantization_enabled:
        raise ValueError("Quantization not enabled in config! Set model.quantization.enabled: True")
    
    print("="*80)
    if use_ptq:
        print("EVALUATION MODE: Post-Training Quantization (PTQ)")
        print("-" * 80)
        print("Starting from: FP32 trained model")
        if conv_only_config:
            print("Calibration: Conv-only (LIF stays FP32)")
        else:
            print("Calibration: Full (Conv + LIF)")
    else:
        print("EVALUATION MODE: Quantization-Aware Training (QAT)")
        print("-" * 80)
        if conv_only_config:
            print("Model type: Conv-only QAT")
            print("  - Convolutions: Already quantized (trained)")
            print("  - LIF: Can add PTQ calibration or keep FP32")
        else:
            print("Model type: Full QAT")
            print("  - All layers: Already quantized (trained)")
            print("  - No calibration needed")
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
    
    
    # ========================================================================
    # CASE 1: PTQ from FP32 model
    # ========================================================================
    if use_ptq:
        print("\n" + "="*80)
        print("CASE 1: PTQ - Post-Training Quantization from FP32 Model")
        print("="*80)
        print("\nLoading FP32 pre-trained model...")
        
        # FP32 MODELS:
        model_path_dir = "mlruns/0/models/LIFFN/38/model.pth" # runid: e1965c33f8214d139624d7e08c7ec9c1
        #model_path_dir = "mlruns/0/models/LIFFN_16ch/38/model.pth" # runid: b6764e1aa848462c89dc70ea9d99246e
        #model_path_dir = "mlruns/0/models/LIFFN_8ch/12/model.pth" # runid: b41ac25a81064a72ac818dce9b25d4d6
        #model_path_dir = "mlruns/0/models/LIFFN_4ch/12/model.pth" # runid: d27de9a1834748f8857b891ab6eba05e
        #model_path_dir = "mlruns/0/models/LIFFN_short/39/model.pth" # runid: bb4ece23356043fca1204176cb270c7d
        #model_path_dir = "mlruns/0/models/LIFFN_16ch_short/33/model.pth" # runid: 7b3c8e69807d44c79abc682e96ff57e1
        #model_path_dir = "mlruns/0/models/LIFFN_8ch_short/23/model.pth" # runid: b61534e5119a4704a66638c1ba78f308
        #model_path_dir = "mlruns/0/models/LIFFN_4ch_short/5/model.pth" # runid: 4ea97793680843e99fd7aaffc2a717ef
        
        print(f"Loading weights from: {model_path_dir}")
        print("Note: Using strict=False to handle architecture differences")
        model = load_model(args.runid, model, device, model_path_dir=model_path_dir, strict=False)
        
        # Calibration for PTQ
        calibration_batches = args.calibration_batches if hasattr(args, 'calibration_batches') else 50
        
        if conv_only_config:
            print(f"\nApplying PTQ calibration: Conv-only mode ({calibration_batches} batches)")
            print("  ✓ Convolutions: Will be calibrated and quantized")
            print("  ✓ LIF layers: Stay at FP32 (mixed precision)")
            model = calibrate_model_ptq(dataloader, model, device, num_batches=calibration_batches, 
                                       calibrate_conv_only=True, calibrate_states_only=False)
        else:
            print(f"\nApplying PTQ calibration: Full mode ({calibration_batches} batches)")
            print("  ✓ Convolutions: Will be calibrated and quantized")
            print("  ✓ LIF layers: Will be calibrated and quantized")
            model = calibrate_model_ptq(dataloader, model, device, num_batches=calibration_batches,
                                       calibrate_conv_only=False, calibrate_states_only=False)
        
        # CRITICAL: Reset dataloader after calibration
        print("\nResetting dataloader for full dataset evaluation...")
        data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
        dataloader = torch.utils.data.DataLoader(
            data,
            drop_last=False,
            batch_size=config["loader"]["batch_size"],
            collate_fn=data.custom_collate,
            worker_init_fn=config_parser.worker_init_fn,
            **kwargs,
        )
        
        print("\n✓ PTQ model ready for evaluation")
        print("  Mode: Fake quantization (FP32 compute with quantization simulation)")
        
    # ========================================================================
    # CASE 2 & 3: QAT models (Conv-only or Full)
    # ========================================================================
    else:
        print("\n" + "="*80)
        if conv_only_config:
            print("CASE 2: Conv-only QAT + Optional PTQ for LIF")
        else:
            print("CASE 3: Full QAT (No Calibration Needed)")
        print("="*80)
        
        # QAT MODELS:
        #model_path_dir = "mlruns/0/models/LIFFN_Full_QAT/model_quant_best.pth" # runid: 
        model_path_dir = "mlruns/0/models/LIFFN_ConvOnly_QAT/model_quant_best.pth" # runid: bc8a6af04f5146bd84e61df2a3b0ad0b
        
        calibration_batches = args.calibration_batches if hasattr(args, 'calibration_batches') else 0
        
        # ----------------------------------------------------------------
        # CASE 2: Conv-only QAT - decide on LIF quantization
        # ----------------------------------------------------------------
        if conv_only_config and calibration_batches > 0:
            print("\n" + "-"*80)
            print("HYBRID MODE: Conv-only QAT + PTQ LIF")
            print("-"*80)
            print("\nStrategy:")
            print("  1. Initialize model with FULL quantization structure (Conv + LIF quantizers)")
            print("  2. Load Conv-only QAT checkpoint:")
            print("     - Conv quantizer metadata (scales, zero-points) → preserves QAT training")
            print("     - LIF parameters (beta, threshold) → preserves learned dynamics")
            print("     - Conv weights → trained values")
            print("     - LIF q_lif quantizers → exist but uninitialized (will be calibrated)")
            print("  3. Apply PTQ calibration ONLY to LIF state quantizers (q_lif)")
            print("     - Conv quantizers remain unchanged (preserve QAT training)")
            print("     - LIF beta/threshold remain unchanged (preserve learned dynamics)")
            print("  4. Result: True INT8 for both Conv and LIF with optimal parameters")
            print("-"*80)
            
            # Step 1: Model already initialized with full quant structure (from config)
            print("\n✓ Step 1: Model initialized with full quantization structure")
            print(f"  - Config has Conv_only={conv_only_config} but model creation uses full structure")
            
            # Verify model has q_lif
            has_q_lif = False
            for name, module in model.named_modules():
                if hasattr(module, 'q_lif') and not isinstance(module.q_lif, torch.nn.Identity):
                    has_q_lif = True
                    break
            
            if not has_q_lif:
                print("\nERROR: Model doesn't have q_lif quantizers!")
                print("The model must be initialized with Conv_only=False for this hybrid mode.")
                print("Please update the config to use full quantization structure during model init.")
                return
            
            # Step 2: Load Conv-only QAT checkpoint
            print("\n✓ Step 2: Loading Conv-only QAT checkpoint...")
            print(f"  From: {model_path_dir}")
            print("  This will restore:")
            print("    - Conv quantizer metadata (scales, zero-points)")
            print("    - LIF parameters (beta, threshold) from training")
            print("    - Conv weights")
            
            model, checkpoint = load_quantized_model(model, model_path_dir, device)
            if checkpoint:
                print(f"  ✓ Loaded successfully!")
                if 'epoch' in checkpoint:
                    print(f"    Trained epochs: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"    Final loss: {checkpoint['loss']:.6f}")
                
                # Verify LIF parameters were loaded
                lif_params_count = 0
                for name, param in model.named_parameters():
                    if 'beta' in name or 'threshold' in name:
                        lif_params_count += 1
                if lif_params_count > 0:
                    print(f"    ✓ Restored {lif_params_count} LIF parameters (beta, threshold)")
            else:
                print("  ERROR: Failed to load checkpoint!")
                return
            
            # Move model to device
            print(f"\n  Moving model to {device}...")
            model = model.to(device)
            
            # Step 3: Apply PTQ calibration to LIF only
            print(f"\n✓ Step 3: Applying PTQ calibration to LIF layers ({calibration_batches} batches)...")
            print("  ✓ Conv quantizers: Keep trained QAT metadata (no recalibration)")
            print("  ✓ LIF beta/threshold: Keep trained values (no modification)")
            print("  ✓ LIF q_lif quantizers: Calibrate with PTQ (initialize scales/zero-points)")
            
            # Verify beta and threshold are reasonable before calibration
            beta_values = []
            thresh_values = []
            for name, module in model.named_modules():
                if hasattr(module, 'beta'):
                    beta_values.append(module.beta.mean().item())
                if hasattr(module, 'threshold'):
                    thresh_values.append(module.threshold.mean().item())
            
            if beta_values and thresh_values:
                print(f"\n  LIF parameters from training:")
                print(f"    Beta (leak): min={min(beta_values):.4f}, max={max(beta_values):.4f}, mean={sum(beta_values)/len(beta_values):.4f}")
                print(f"    Threshold: min={min(thresh_values):.4f}, max={max(thresh_values):.4f}, mean={sum(thresh_values)/len(thresh_values):.4f}")
            
            model = calibrate_model_ptq(dataloader, model, device, num_batches=calibration_batches,
                                       calibrate_conv_only=False, calibrate_states_only=True)
            
            # Reset dataloader
            print("\nResetting dataloader for full dataset evaluation...")
            data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
            dataloader = torch.utils.data.DataLoader(
                data,
                drop_last=False,
                batch_size=config["loader"]["batch_size"],
                collate_fn=data.custom_collate,
                worker_init_fn=config_parser.worker_init_fn,
                **kwargs,
            )
            
            print("\n✓ Step 4: Hybrid model ready!")
            print("  ✓ Conv: INT8 (QAT trained)")
            print("  ✓ LIF: INT8 (PTQ calibrated)")
            
        # ----------------------------------------------------------------
        # CASE 2 alternative: Conv-only QAT without LIF quantization
        # ----------------------------------------------------------------
        elif conv_only_config:
            print("\n" + "-"*80)
            print("MIXED PRECISION MODE: Conv-only QAT")
            print("-"*80)
            print("\nLoading Conv-only QAT checkpoint...")
            print(f"From: {model_path_dir}")
            
            model, checkpoint = load_quantized_model(model, model_path_dir, device)
            if checkpoint:
                print(f"✓ Loaded from: {model_path_dir}")
                if 'epoch' in checkpoint:
                    print(f"  Trained epochs: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"  Final loss: {checkpoint['loss']:.6f}")
            
            # Move model to correct device
            print(f"\nMoving model to {device}...")
            model = model.to(device)
            
            print("\nMixed Precision Configuration:")
            print("  ✓ Convolutions: INT8 (QAT trained)")
            print("  ✓ LIF layers: FP32 (no quantization)")
            print("\n✓ Mixed precision model ready for evaluation")
        
        # ----------------------------------------------------------------
        # CASE 3: Full QAT - already fully quantized
        # ----------------------------------------------------------------
        else:
            print("\n" + "-"*80)
            print("FULL QAT MODE")
            print("-"*80)
            print("\nLoading Full QAT checkpoint...")
            print(f"From: {model_path_dir}")
            
            model, checkpoint = load_quantized_model(model, model_path_dir, device)
            if checkpoint:
                print(f"✓ Loaded from: {model_path_dir}")
                if 'epoch' in checkpoint:
                    print(f"  Trained epochs: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"  Final loss: {checkpoint['loss']:.6f}")
            
            # Move model to correct device
            print(f"\nMoving model to {device}...")
            model = model.to(device)
            
            if calibration_batches > 0:
                print("\nWARNING: Calibration requested but not needed for Full QAT!")
                print("Skipping calibration to preserve trained quantizers...")
            
            print("\nFully Quantized Configuration:")
            print("  ✓ Convolutions: INT8 (QAT trained)")
            print("  ✓ LIF layers: INT8 (QAT trained)")
            print("\n✓ Fully quantized model ready for evaluation")
    
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
    
    # Detailed mode reporting
    if use_ptq:
        print(f"Mode: PTQ from FP32")
        if conv_only_config:
            print("  Quantization: Conv-only (LIF at FP32)")
        else:
            print("  Quantization: Full (Conv + LIF)")
    else:
        if conv_only_config:
            if calibration_batches > 0:
                print(f"Mode: Hybrid (QAT Conv + PTQ LIF)")
            else:
                print(f"Mode: Mixed Precision (QAT Conv, FP32 LIF)")
        else:
            print(f"Mode: Full QAT")
    
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
        
        # Log detailed mode information
        if use_ptq:
            mlflow.log_param("eval_mode", "PTQ")
            mlflow.log_param("ptq_type", "Conv_only" if conv_only_config else "Full")
        else:
            if conv_only_config:
                if calibration_batches > 0:
                    mlflow.log_param("eval_mode", "Hybrid_QAT_Conv_PTQ_LIF")
                else:
                    mlflow.log_param("eval_mode", "Mixed_Precision_QAT_Conv")
            else:
                mlflow.log_param("eval_mode", "Full_QAT")
        
        mlflow.end_run()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Quantized SNN Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluation Modes:
-----------------
1. PTQ from FP32:
   Set in config: quantization.PTQ = True
   - Conv-only: quantization.Conv_only = True (LIF stays FP32)
   - Full: quantization.Conv_only = False (Conv + LIF quantized)
   
2. QAT Conv-only + Optional PTQ LIF:
   Set in config: quantization.PTQ = False, quantization.Conv_only = True
   - Mixed precision: --calibration_batches 0 (LIF stays FP32)
   - Hybrid: --calibration_batches > 0 (add PTQ for LIF)
   
3. Full QAT (no calibration):
   Set in config: quantization.PTQ = False, quantization.Conv_only = False
   - All layers already quantized during training
        """
    )
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
        help="Optional: Direct path to model checkpoint",
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=50,
        help="Number of batches for calibration. PTQ: calibrates based on Conv_only flag. QAT Conv-only: if >0 calibrates LIF (PTQ), if 0 keeps LIF at FP32. Full QAT: ignored.",
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
    conv_only = config_parser.config["model"].get("quantization", {}).get("Conv_only", False)
    
    print("\n" + "="*80)
    print("Quantized Model Evaluation")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Run ID: {args.runid}")
    
    if use_ptq:
        print(f"Base: PTQ from FP32 model")
        print(f"Calibration: {'Conv-only' if conv_only else 'Full (Conv + LIF)'}")
        print(f"Calibration Batches: {args.calibration_batches}")
    else:
        print(f"Base: QAT model")
        if conv_only:
            if args.calibration_batches > 0:
                print(f"Type: Hybrid (QAT Conv + PTQ LIF)")
                print(f"LIF Calibration Batches: {args.calibration_batches}")
            else:
                print(f"Type: Mixed Precision (QAT Conv, FP32 LIF)")
        else:
            print(f"Type: Full QAT (no calibration needed)")
    
    if args.model_path:
        print(f"Model Path: {args.model_path}")
    print("="*80 + "\n")
    
    # Run evaluation
    test_quantized(args, config_parser)
