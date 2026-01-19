from torch.optim import *
from torchinfo import summary
import torch
import numpy as np
import json
import os

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


def print_quantization_info(model, config):
    """
    Print detailed information about what is being quantized in the model.
    Shows Conv layers, LIF layers, their parameters, and quantization status.
    """
    print("\n" + "="*100)
    print("QUANTIZATION ANALYSIS")
    print("="*100)
    
    # Configuration summary
    quant_config = config["model"].get("quantization", {})
    print(f"\nConfiguration:")
    print(f"  - Quantization enabled: {quant_config.get('enabled', False)}")
    print(f"  - PTQ mode: {quant_config.get('PTQ', False)}")
    print(f"  - Conv-only mode: {quant_config.get('Conv_only', False)}")
    print(f"  - Quantization type: {quant_config.get('type', 'N/A')}")
    
    # Count different component types
    total_conv_layers = 0
    total_lif_layers = 0
    quant_conv_layers = 0
    quant_lif_layers = 0
    
    print("\n" + "-"*100)
    print(f"{'Layer Name':<40} {'Type':<25} {'Input→Output':<20} {'Quantized':<15}")
    print("-"*100)
    
    for name, module in model.named_modules():
        # Skip the top-level model and intermediate containers
        if name == '' or 'pred' in name.lower():
            continue
            
        # Check for Convolutional layers
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            from brevitas.nn import QuantConv2d
            is_quant = isinstance(module, QuantConv2d)
            
            total_conv_layers += 1
            if is_quant:
                quant_conv_layers += 1
            
            in_ch = module.in_channels
            out_ch = module.out_channels
            k_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            
            layer_type = "QuantConv2d" if is_quant else "Conv2d"
            quant_status = "✓ QUANTIZED" if is_quant else "✗ FP32"
            
            print(f"{name:<40} {layer_type:<25} {in_ch}→{out_ch} (k={k_size}){'':<5} {quant_status:<15}")
            
            # Print quantization details for QuantConv2d
            if is_quant:
                if hasattr(module, 'weight_quant'):
                    print(f"{'  └─ Weights':<40} {'Quantizer':<25} {'':<20} {'Int8 (per-tensor)':<15}")
                if hasattr(module, 'input_quant'):
                    print(f"{'  └─ Input':<40} {'Quantizer':<25} {'':<20} {'Int8 activations':<15}")
                if hasattr(module, 'output_quant'):
                    print(f"{'  └─ Output':<40} {'Quantizer':<25} {'':<20} {'Int8 activations':<15}")
        
        # Check for LIF neuron layers
        elif hasattr(module, 'lif') and hasattr(module, 'ff'):
            import snntorch as snn
            from brevitas.nn import QuantConv2d
            
            total_lif_layers += 1
            
            # Check if Conv part is quantized
            conv_is_quant = isinstance(module.ff, QuantConv2d)
            
            # Check if LIF state is quantized
            lif_state_quant = hasattr(module, 'q_lif') and module.q_lif is not None
            
            if conv_is_quant or lif_state_quant:
                quant_lif_layers += 1
            
            # Get dimensions
            in_ch = module.input_size
            out_ch = module.hidden_size
            
            # Determine layer type
            is_recurrent = hasattr(module, 'rec')
            layer_type = "ConvLIF+Rec" if is_recurrent else "ConvLIF"
            
            quant_parts = []
            if conv_is_quant:
                quant_parts.append("Conv")
            if lif_state_quant:
                quant_parts.append("LIF-state")
            
            if quant_parts:
                quant_status = f"✓ {'+'.join(quant_parts)}"
            else:
                quant_status = "✗ FP32"
            
            print(f"{name:<40} {layer_type:<25} {in_ch}→{out_ch}{'':<12} {quant_status:<15}")
            
            # Print LIF parameter details
            if hasattr(module, 'lif'):
                lif = module.lif
                beta_shape = lif.beta.shape if hasattr(lif, 'beta') else "N/A"
                thresh_shape = lif.threshold.shape if hasattr(lif, 'threshold') else "N/A"
                
                # Check if parameters are per-channel
                per_channel = False
                if hasattr(lif, 'beta'):
                    beta_numel = lif.beta.numel()
                    if beta_numel == out_ch:
                        per_channel = True
                
                param_type = "per-channel" if per_channel else f"shape={beta_shape}"
                print(f"{'  └─ LIF params (β,θ)':<40} {param_type:<25} {'':<20} {'Learnable':<15}")
                
                if lif_state_quant:
                    print(f"{'  └─ Membrane state':<40} {'State Quantizer':<25} {'':<20} {'Int8 (8-bit)':<15}")
                else:
                    print(f"{'  └─ Membrane state':<40} {'FP32':<25} {'':<20} {'Not quantized':<15}")
            
            # For recurrent layers, show recurrent conv
            if is_recurrent and hasattr(module, 'rec'):
                rec_is_quant = isinstance(module.rec, QuantConv2d)
                rec_status = "✓ Quantized" if rec_is_quant else "✗ FP32"
                print(f"{'  └─ Recurrent Conv':<40} {'QuantConv2d' if rec_is_quant else 'Conv2d':<25} {out_ch}→{out_ch}{'':<12} {rec_status:<15}")
    
    print("-"*100)
    print(f"\nSummary:")
    print(f"  Total Conv layers: {total_conv_layers}")
    print(f"  Quantized Conv layers: {quant_conv_layers}")
    print(f"  Total LIF layers: {total_lif_layers}")
    print(f"  LIF layers with quantization: {quant_lif_layers}")
    print(f"  Coverage: Conv {quant_conv_layers}/{total_conv_layers} ({100*quant_conv_layers/max(total_conv_layers,1):.1f}%), " +
          f"LIF {quant_lif_layers}/{total_lif_layers} ({100*quant_lif_layers/max(total_lif_layers,1):.1f}%)")
    
    # Additional warnings
    print(f"\nNotes:")
    if quant_config.get('Conv_only', False):
        print(f"  ⚠ Conv-only mode: LIF neuron states remain in FP32 (mixed precision)")
    if quant_config.get('PTQ', False):
        print(f"  ℹ PTQ mode: Model will be calibrated using sample data")
    else:
        print(f"  ℹ QAT mode: Model was quantized during training")
    
    # Check per-channel parameters
    for name, module in model.named_modules():
        if hasattr(module, 'lif'):
            lif = module.lif
            if hasattr(lif, 'beta'):
                beta_shape = lif.beta.shape
                if len(beta_shape) == 3 and beta_shape[0] > 1:
                    print(f"  ✓ Per-channel LIF parameters detected: beta shape = {beta_shape}")
                    print(f"    (SNNtorch quantization typically doesn't support per-channel LIF params,")
                    print(f"     but they are preserved as learnable FP32 parameters)")
                    break
    
    print("="*100 + "\n")


def profile_membrane_ranges(model, calibration_loader, device, num_batches=10):
    """
    Profile membrane potential ranges for each LIF layer.
    Returns dict mapping layer_name -> (min_value, max_value)
    
    This temporarily disables quantization to measure the true FP32
    operating range of each layer's membrane potentials.
    This gives us the actual membrane dynamics before quantization.
    
    Uses robust statistical analysis to identify and filter outliers.
    """
    print("\n" + "="*80)
    print("PROFILING MEMBRANE POTENTIAL RANGES")
    print("="*80)
    print(f"Running {num_batches} batches to measure membrane operating ranges...")
    print("(Quantization DISABLED - profiling FP32 behavior)")
    
    model.eval()
    
    # Find all LIF layers
    lif_layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lif') and hasattr(module, 'ff'):
            # Extract layer name (head, G1, R1a, etc.)
            layer_name = name.split('.')[-1] if '.' in name else name
            if layer_name:  # Skip empty names
                lif_layers[layer_name] = module
    
    if not lif_layers:
        print("No LIF layers found for profiling!")
        return {}
    
    print(f"Found {len(lif_layers)} LIF layers: {list(lif_layers.keys())}")
    
    # DISABLE quantization temporarily to get true FP32 ranges
    saved_quantizers = {}
    for layer_name, module in lif_layers.items():
        if hasattr(module, 'lif') and hasattr(module.lif, 'state_quant'):
            saved_quantizers[layer_name] = module.lif.state_quant
            module.lif.state_quant = None  # Disable quantization
    
    print(f"Temporarily disabled {len(saved_quantizers)} LIF state quantizers")
    
    # Statistics collectors - now collecting full distributions
    layer_stats = {name: {
        'values': [],  # Store all values for distribution analysis
        'min': float('inf'), 
        'max': float('-inf')
    } for name in lif_layers.keys()}
    
    # Profile membrane potentials
    batch_count = 0
    print(f"Profiling configuration: mode={calibration_loader.dataset.seq_dir if hasattr(calibration_loader.dataset, 'seq_dir') else 'unknown'}")
    with torch.no_grad():
        for i_batch, item in enumerate(calibration_loader):
            if batch_count >= num_batches:
                break
            
            # Handle new sequence - reset model states
            if calibration_loader.dataset.new_seq:
                calibration_loader.dataset.new_seq = False
                model.reset_states()
            
            # Check if we've processed all sequences
            if calibration_loader.dataset.seq_num >= len(calibration_loader.dataset.files):
                break
            
            # Get inputs (direct keys, not nested)
            event_voxel = item["event_voxel"].to(device)
            event_cnt = item["event_cnt"].to(device)
            
            # Forward pass
            try:
                _ = model(event_voxel, event_cnt, log=False)
                
                # Collect statistics directly from module.lif.mem
                for layer_name, module in lif_layers.items():
                    if hasattr(module.lif, 'mem') and module.lif.mem is not None:
                        mem = module.lif.mem
                        
                        # Store min/max
                        min_val = mem.min().item()
                        max_val = mem.max().item()
                        layer_stats[layer_name]['min'] = min(layer_stats[layer_name]['min'], min_val)
                        layer_stats[layer_name]['max'] = max(layer_stats[layer_name]['max'], max_val)
                        
                        # Sample values for distribution (to avoid memory issues with 1000 batches)
                        # Take random sample of 10000 values per batch
                        mem_flat = mem.flatten()
                        sample_size = min(10000, mem_flat.numel())
                        if mem_flat.numel() > sample_size:
                            indices = torch.randperm(mem_flat.numel())[:sample_size]
                            sampled = mem_flat[indices].cpu().numpy()
                        else:
                            sampled = mem_flat.cpu().numpy()
                        layer_stats[layer_name]['values'].extend(sampled.tolist())
                
                batch_count += 1
                if batch_count % 100 == 0:
                    # Report progress and verify data collection
                    total_samples = sum(len(layer_stats[ln]['values']) for ln in layer_stats.keys())
                    print(f"  Progress: {batch_count}/{num_batches} batches, {total_samples:,} samples collected")
                
            except Exception as e:
                print(f"Warning: Error during profiling batch {i_batch}: {e}")
    
    # RESTORE quantizers
    for layer_name, quantizer in saved_quantizers.items():
        lif_layers[layer_name].lif.state_quant = quantizer
    print(f"Restored {len(saved_quantizers)} LIF state quantizers")
    
    # Verify data collection before analysis
    print("\n" + "="*80)
    print("DATA COLLECTION VERIFICATION")
    print("="*80)
    total_collected = 0
    for layer_name in sorted(layer_stats.keys()):
        num_samples = len(layer_stats[layer_name]['values'])
        total_collected += num_samples
        print(f"{layer_name}: {num_samples:,} samples")
    print(f"Total samples collected: {total_collected:,}")
    
    if total_collected == 0:
        print("\n⚠⚠⚠ ERROR: NO DATA COLLECTED! Profiling failed!")
        print("This will cause the statistical analysis to fail.")
        print("Check that the dataloader is providing data correctly.")
        return {}
    
    # Analyze distributions and compute robust statistics
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS OF MEMBRANE DISTRIBUTIONS")
    print("="*80)
    
    layer_robust_stats = {}
    
    for layer_name in sorted(layer_stats.keys()):
        stats = layer_stats[layer_name]
        if stats['min'] == float('inf') or len(stats['values']) == 0:
            print(f"\n{layer_name}: No data collected - SKIPPING ANALYSIS")
            continue
        
        values = np.array(stats['values'])
        
        print(f"\n{layer_name}: Analyzing {len(values):,} samples")
        
        # Basic statistics
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)
        
        # Percentiles for outlier detection
        p1 = np.percentile(values, 1)
        p2_5 = np.percentile(values, 2.5)  # Midpoint between P1 and P5
        p5 = np.percentile(values, 5)
        p6 = np.percentile(values, 6)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        p95 = np.percentile(values, 95)
        p99 = np.percentile(values, 99)
        p999 = np.percentile(values, 99.9)
        
        # IQR method for outlier detection
        iqr = p75 - p25
        lower_fence = p25 - 1.5 * iqr
        upper_fence = p75 + 1.5 * iqr
        
        # Robust range (excluding extreme outliers beyond 3 IQR)
        lower_robust = p25 - 3 * iqr
        upper_robust = p75 + 3 * iqr
        
        # Filter values within robust range
        robust_values = values[(values >= lower_robust) & (values <= upper_robust)]
        outlier_count = len(values) - len(robust_values)
        outlier_percent = 100 * outlier_count / len(values)
        
        # Store robust statistics
        layer_robust_stats[layer_name] = {
            'min': stats['min'],
            'max': stats['max'],
            'mean': mean,
            'median': median,
            'std': std,
            'p1': p1,
            'p2_5': p2_5,
            'p5': p5,
            'p6': p6,
            'p95': p95,
            'p99': p99,
            'p999': p999,
            'iqr': iqr,
            'lower_fence': lower_fence,
            'upper_fence': upper_fence,
            'robust_min': float(np.min(robust_values)) if len(robust_values) > 0 else stats['min'],
            'robust_max': float(np.max(robust_values)) if len(robust_values) > 0 else stats['max'],
            'outlier_count': outlier_count,
            'outlier_percent': outlier_percent,
            'total_samples': len(values)
        }
        
        # Print detailed analysis
        print(f"\n{layer_name}:")
        print(f"  Total samples: {len(values):,}")
        print(f"  Absolute range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mean: {mean:.4f}, Median: {median:.4f}, Std: {std:.4f}")
        print(f"  Percentiles:")
        print(f"    1%: {p1:.4f}, 2.5%: {p2_5:.4f}, 5%: {p5:.4f}, 6%: {p6:.4f}, 95%: {p95:.4f}, 99%: {p99:.4f}, 99.9%: {p999:.4f}")
        print(f"  IQR (Q1-Q3): [{p25:.4f}, {p75:.4f}], width: {iqr:.4f}")
        print(f"  Outlier fences (1.5×IQR): [{lower_fence:.4f}, {upper_fence:.4f}]")
        print(f"  Robust range (3×IQR): [{lower_robust:.4f}, {upper_robust:.4f}]")
        print(f"  Outliers beyond 3×IQR: {outlier_count:,} ({outlier_percent:.2f}%)")
        
        # Recommendation
        if outlier_percent > 1.0:
            print(f"  ⚠ High outlier rate! Consider using P99 or robust range for quantization")
        elif outlier_percent > 0.1:
            print(f"  ℹ Moderate outliers present. P99.9 or robust range recommended")
        else:
            print(f"  ✓ Low outlier rate. Absolute range or P99.9 suitable")
    
    print("\n" + "="*80)
    print("RECOMMENDED QUANTIZATION RANGES")
    print("="*80)
    print(f"{'Layer':<15} {'Strategy':<20} {'Range':<35} {'vs Absolute':<15}")
    print("-"*85)
    
    # Define layer-specific percentile strategies
    layer_strategies = {
        'head': ('P2.5-P99', 'p2_5', 'p99'),
        'G1': ('P1-P99', 'p1', 'p99'),
        'R1a': ('P1-P99', 'p1', 'p99'),
        'R1b': ('P2.5-P99', 'p2_5', 'p99'),
        'G2': ('P1-P99', 'p1', 'p99'),
        'R2a': ('P1-P99', 'p1', 'p99'),
        'R2b': ('P6-P99', 'p6', 'p99')
    }
    
    for layer_name in sorted(layer_robust_stats.keys()):
        rstats = layer_robust_stats[layer_name]
        
        # Choose strategy based on layer name
        if layer_name in layer_strategies:
            strategy, min_key, max_key = layer_strategies[layer_name]
            rec_min = rstats[min_key]
            rec_max = rstats[max_key]
        else:
            # Default to P1-P99 for unknown layers
            strategy = "P1-P99 (default)"
            rec_min = rstats['p1']
            rec_max = rstats['p99']
        
        # Calculate coverage vs absolute range
        abs_range = rstats['max'] - rstats['min']
        rec_range = rec_max - rec_min
        coverage_pct = 100 * rec_range / abs_range if abs_range > 0 else 100
        
        print(f"{layer_name:<15} {strategy:<20} [{rec_min:6.2f}, {rec_max:5.2f}]{'':<18} {coverage_pct:5.1f}%")
        
        # Store recommended values for use in apply function
        layer_robust_stats[layer_name]['recommended_min'] = rec_min
        layer_robust_stats[layer_name]['recommended_max'] = rec_max
    
    print("-"*85)
    print("✓ Statistical analysis complete\n")
    
    # Print summary comparison for dt1 vs dt4 awareness
    print("="*80)
    print("PROFILING MODE SUMMARY")
    print("="*80)
    avg_abs_range = np.mean([rstats['max'] - rstats['min'] for rstats in layer_robust_stats.values()])
    avg_recommended_range = np.mean([rstats['recommended_max'] - rstats['recommended_min'] for rstats in layer_robust_stats.values()])
    print(f"Average absolute range across layers: {avg_abs_range:.2f}")
    print(f"Average recommended range: {avg_recommended_range:.2f}")
    print(f"Number of layers profiled: {len(layer_robust_stats)}")
    print("\nNote: These ranges are specific to the data mode used during profiling.")
    print("If evaluating on different data (dt1 vs dt4), ranges may be suboptimal.")
    print("="*80 + "\n")
    
    return layer_robust_stats


def apply_per_layer_quantization_ranges(model, layer_stats, safety_margin=0.001, clip_extreme_negatives=True):
    """
    Apply per-layer quantization ranges based on profiled membrane statistics.
    
    Args:
        model: The model with LIF layers
        layer_stats: Dict from profile_membrane_ranges with statistical analysis including 'recommended_min/max'
        safety_margin: Additional margin to add (as fraction, e.g., 0.001 = 0.1%)
        clip_extreme_negatives: If True, clip very negative values (e.g., < -200)
    """
    from snntorch.functional import quant
    import math
    
    print("\n" + "="*80)
    print("APPLYING PER-LAYER QUANTIZATION RANGES")
    print("="*80)
    
    # Find all LIF layers
    lif_layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lif') and hasattr(module, 'ff'):
            layer_name = name.split('.')[-1] if '.' in name else name
            if layer_name and layer_name in layer_stats:
                lif_layers[layer_name] = module
    
    if not lif_layers:
        print("No LIF layers found!")
        return
    
    print(f"Configuring {len(lif_layers)} LIF layers with custom ranges:")
    print(f"  - Safety margin: {safety_margin*100:.0f}%")
    print(f"  - Clip extreme negatives: {clip_extreme_negatives}")
    print()
    
    print(f"{'Layer':<15} {'Observed Range':<30} {'Quantization Range':<30} {'Step Size':<15}")
    print("-"*90)
    
    for layer_name, module in lif_layers.items():
        if layer_name not in layer_stats:
            continue
            
        stats = layer_stats[layer_name]
        if 'recommended_min' not in stats:
            print(f"{layer_name:<15} No data - skipping")
            continue
        
        # Use recommended ranges from statistical analysis
        min_obs = stats['recommended_min']
        max_obs = stats['recommended_max']
        
        # Determine quantization bounds
        # Goal: Create TIGHTER ranges than observed (round toward zero)
        # This gives smaller step size and better precision
        # Safety margin is subtracted (not added) to make range tighter
        
        if clip_extreme_negatives and min_obs < -250:
            # Clip very extreme negatives to -x (prevents outliers from wasting quantization levels)
            lower_bound = -250
        else:
            # For negative values: round TOWARD zero (less negative) for tighter range
            # e.g., observed -45.7 → -45 (ceil toward zero)
            # With small safety margin: -45.7 * (1 - 0.001) = -45.65 → -46 (ceil)
            if min_obs < 0:
                lower_bound = math.floor(min_obs * (1 - safety_margin))
            else:
                # For positive mins (shouldn't happen), round up toward zero
                lower_bound = math.floor(min_obs)
        
        # Enforce minimum floor of -x for small negative ranges
        # This ensures numerical stability and avoids overly tight ranges
        if lower_bound > -15:
            lower_bound = -15
        
        # Upper bound: round TOWARD zero (less positive) for tighter range
        # e.g., observed 0.94 → 0 (floor toward zero)
        # With small safety margin: 0.94 * (1 + 0.001) = 0.94 → 0 (floor)
        upper_bound = math.ceil(max_obs * (1 + safety_margin))
        upper_bound = max(upper_bound, 1.0)  # Ensure at least 1.0
        
        # Calculate SNNtorch parameters
        # Range formula: [-threshold*(1+lower_limit), threshold*(1+upper_limit)]
        # We want: [lower_bound, upper_bound]
        
        # Set threshold to upper_bound for simplicity
        threshold = float(upper_bound)
        
        # Solve for lower_limit: -threshold*(1+lower_limit) = lower_bound
        # -> lower_limit = -lower_bound/threshold - 1
        lower_limit = abs(lower_bound) / threshold - 1.0
        
        # upper_limit = 0 since threshold = upper_bound
        upper_limit = 0.0
        
        # Recreate state_quant with new parameters
        if hasattr(module.lif, 'state_quant'):
            module.lif.state_quant = quant.state_quant(
                num_bits=8,
                uniform=True,
                thr_centered=False,
                threshold=threshold,
                lower_limit=lower_limit,
                upper_limit=upper_limit
            )
            
            # Calculate step size for reporting
            num_levels = 256  # 8-bit
            range_width = upper_bound - lower_bound
            step_size = range_width / (num_levels - 1)
            
            obs_range = f"[{min_obs:.2f}, {max_obs:.2f}]"
            quant_range = f"[{lower_bound:.0f}, {upper_bound:.0f}]"
            
            print(f"{layer_name:<15} {obs_range:<30} {quant_range:<30} {step_size:<15.4f}")
    
    print("-"*90)
    print("✓ Per-layer quantization ranges configured\n")


def calibrate_model_ptq(calibration_loader, model, device, num_batches=50, calibrate_conv_only=False, calibrate_states_only=False, auto_tune_lif_ranges=False, config=None, loader_kwargs=None, worker_init_fn=None):
    """
    Calibrate the model for Post-Training Quantization (PTQ).
    This collects statistics to initialize quantization parameters without training.
    
    Args:
        calibrate_conv_only: If True, only calibrate Conv layers (not LIF). Used for PTQ Conv-only mode.
        calibrate_states_only: If True, only calibrate LIF state quantizers (for Conv-only QAT + PTQ LIF).
                               Preserves trained Conv quantizers.
        auto_tune_lif_ranges: If True, profile membrane ranges and set per-layer quantization bounds.
        config: Configuration dict (required for dataloader reset)
        loader_kwargs: Loader kwargs (required for dataloader reset)
        worker_init_fn: Worker init function (required for dataloader reset)
        
    Three calibration modes:
    1. Full PTQ (both False): Calibrate Conv + LIF using calibration_mode
    2. Conv-only PTQ (calibrate_conv_only=True): Calibrate only Conv, LIF stays FP32
    3. QAT Conv + PTQ LIF (calibrate_states_only=True): Preserve Conv, calibrate LIF only
    
    Auto-tuning (auto_tune_lif_ranges=True):
    - Profiles membrane ranges during first pass
    - Applies per-layer quantization bounds
    - Then performs standard calibration
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
    
    if auto_tune_lif_ranges:
        print("Auto-tuning enabled: Will profile and optimize per-layer ranges")
    
    print(f"Using up to {num_batches} batches for calibration (across all sequences)")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    # Step 0: Auto-tune LIF ranges if requested
    if auto_tune_lif_ranges and not calibrate_conv_only:
        # Profile membrane ranges (use more batches for better statistics)
        # IMPORTANT: Profile batches adjusted based on data mode
        # dt1: Every RGB frame (smaller flow, more frequent)
        # dt4: Every 4th RGB frame (larger flow, less frequent)
        data_mode = config["data"]["mode"]
        print(f"\n⚠ IMPORTANT: Profiling for data mode: {data_mode}")
        
        if data_mode == "gtflow_dt1":
            profile_batches = 1000
            print("  dt1 mode: Small flow values, high temporal resolution")
        elif data_mode == "gtflow_dt4":
            profile_batches = 1000
            print("  dt4 mode: Large flow values (4x), lower temporal resolution")
            print("  ⚠ WARNING: dt4 has 4x larger flow → expect larger membrane ranges!")
        else:
            profile_batches = 1000
            print(f"  Unknown mode {data_mode}, using default profile batches")
        
        layer_stats = profile_membrane_ranges(model, calibration_loader, device, num_batches=profile_batches)
        
        # Apply per-layer quantization ranges
        if layer_stats:
            apply_per_layer_quantization_ranges(model, layer_stats, safety_margin=0.001, clip_extreme_negatives=True)
        
        # Reset dataloader for actual calibration by recreating it
        print("Resetting dataloader for calibration pass...")
        if config is not None and loader_kwargs is not None:
            data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
            calibration_loader = torch.utils.data.DataLoader(
                data,
                drop_last=False,
                batch_size=config["loader"]["batch_size"],
                collate_fn=data.custom_collate,
                worker_init_fn=worker_init_fn,
                **loader_kwargs,
            )
            print("✓ Dataloader reset complete")
        else:
            print("⚠ Warning: Cannot reset dataloader (config/kwargs not provided)")
    
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
            
            # Reset dataloader after LIF calibration
            print("\nResetting dataloader after LIF calibration...")
            if config is not None and loader_kwargs is not None:
                data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
                calibration_loader = torch.utils.data.DataLoader(
                    data,
                    drop_last=False,
                    batch_size=config["loader"]["batch_size"],
                    collate_fn=data.custom_collate,
                    worker_init_fn=worker_init_fn,
                    **loader_kwargs,
                )
                print("✓ Dataloader reset complete")
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
        
        # Reset dataloader after Conv-only calibration
        print("\nResetting dataloader after Conv calibration...")
        if config is not None and loader_kwargs is not None:
            data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
            calibration_loader = torch.utils.data.DataLoader(
                data,
                drop_last=False,
                batch_size=config["loader"]["batch_size"],
                collate_fn=data.custom_collate,
                worker_init_fn=worker_init_fn,
                **loader_kwargs,
            )
            print("✓ Dataloader reset complete")
    
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
        
        # Reset dataloader after full calibration
        print("\nResetting dataloader after full calibration...")
        if config is not None and loader_kwargs is not None:
            data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
            calibration_loader = torch.utils.data.DataLoader(
                data,
                drop_last=False,
                batch_size=config["loader"]["batch_size"],
                collate_fn=data.custom_collate,
                worker_init_fn=worker_init_fn,
                **loader_kwargs,
            )
            print("✓ Dataloader reset complete")
    
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
    
    # Determine if loading from TensorBoard or MLflow
    use_tensorboard = args.model_path and not args.runid
    use_mlflow = args.runid is not None
    
    if use_tensorboard:
        print("Loading model from TensorBoard directory...")
        # Extract experiment directory from model path
        # e.g., "runs/Default_QAT_20251202_095245_ConvOnly_finetuning_0.72/checkpoints/ConvOnly_QAT/model_quant_best.pth"
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Find config.json in the experiment directory
        experiment_dir = model_path.split('/checkpoints/')[0] if '/checkpoints/' in model_path else os.path.dirname(os.path.dirname(model_path))
        config_file = os.path.join(experiment_dir, 'config.json')
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}. Expected in TensorBoard experiment directory.")
        
        print(f"Loading config from: {config_file}")
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        # Merge with base config from YAML, but prioritize evaluation-specific settings
        # Start with evaluation config from YAML
        config = config_parser.config.copy()
        
        # Selectively update with training config - only model architecture and quantization settings
        # Keep evaluation-specific settings (loader, data paths, metrics, visualization) from YAML
        preserve_keys = ["loader", "data", "metrics", "vis"]  # Keep these from eval config
        for key, value in saved_config.items():
            if key not in preserve_keys:
                config[key] = value
            elif key == "model":
                # For model, merge carefully - take architecture from training but keep eval overrides
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
        
        print("Using evaluation config for: loader, data, metrics, visualization")
        print("Using training config for: model architecture, quantization settings")
        
    elif use_mlflow:
        print("Loading model from MLflow...")
        mlflow.set_tracking_uri(args.path_mlflow)
        run = mlflow.get_run(args.runid)
        config = config_parser.merge_configs(run.data.params)
    else:
        raise ValueError("Must provide either --model_path (TensorBoard) or --runid (MLflow)")

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
        if use_tensorboard:
            # For TensorBoard models, use model path as identifier
            model_identifier = os.path.basename(os.path.dirname(args.model_path))
            path_results = os.path.join(args.path_results, model_identifier)
            os.makedirs(path_results, exist_ok=True)
            eval_id = model_identifier
            print(f"Results will be saved to: {path_results}")
        else:
            # For MLflow models, use existing logic
            path_results = create_model_dir(args.path_results, args.runid)
            eval_id = log_config(path_results, args.runid, config)
        
        # Start MLflow run only if using MLflow
        if use_mlflow:
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
    
    # Print quantization info for the initialized model
    print("\n" + "="*80)
    print("INITIAL MODEL STRUCTURE")
    print("="*80)
    print_quantization_info(model, config)
    
    # ========================================================================
    # CASE 1: PTQ from FP32 model
    # ========================================================================
    if use_ptq:
        print("\n" + "="*80)
        print("CASE 1: PTQ - Post-Training Quantization from FP32 Model")
        print("="*80)
        print("\nLoading FP32 pre-trained model...")
        
        # Determine model path
        if use_tensorboard and args.model_path:
            model_path_dir = args.model_path
        else:
            # FP32 MODELS (default MLflow paths):
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
        runid_for_load = "" if use_tensorboard else args.runid
        model = load_model(runid_for_load, model, device, model_path_dir=model_path_dir, strict=False)
        
        # Print model state after loading FP32 checkpoint but before calibration
        print("\n" + "="*80)
        print("MODEL AFTER LOADING FP32 CHECKPOINT (Before Calibration)")
        print("="*80)
        print_quantization_info(model, config)
        
        # Calibration for PTQ
        calibration_batches = args.calibration_batches if hasattr(args, 'calibration_batches') else 50
        auto_tune = getattr(args, 'auto_tune_lif', False)
        
        if conv_only_config:
            print(f"\nApplying PTQ calibration: Conv-only mode ({calibration_batches} batches)")
            print("  ✓ Convolutions: Will be calibrated and quantized")
            print("  ✓ LIF layers: Stay at FP32 (mixed precision)")
            model = calibrate_model_ptq(dataloader, model, device, num_batches=calibration_batches, 
                                       calibrate_conv_only=True, calibrate_states_only=False, auto_tune_lif_ranges=False,
                                       config=config, loader_kwargs=kwargs, worker_init_fn=config_parser.worker_init_fn)
        else:
            print(f"\nApplying PTQ calibration: Full mode ({calibration_batches} batches)")
            print("  ✓ Convolutions: Will be calibrated and quantized")
            print("  ✓ LIF layers: Will be calibrated and quantized")
            if auto_tune:
                print("  ✓ Auto-tuning: Per-layer LIF ranges will be profiled and optimized")
            model = calibrate_model_ptq(dataloader, model, device, num_batches=calibration_batches,
                                       calibrate_conv_only=False, calibrate_states_only=False, auto_tune_lif_ranges=auto_tune,
                                       config=config, loader_kwargs=kwargs, worker_init_fn=config_parser.worker_init_fn)
        
        # Print model state after calibration
        print("\n" + "="*80)
        print("MODEL AFTER PTQ CALIBRATION")
        print("="*80)
        print_quantization_info(model, config)
        
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
        
        # Determine model path
        if use_tensorboard and args.model_path:
            model_path_dir = args.model_path
        else:
            # QAT MODELS (default MLflow paths):
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
            
            # Print model state after loading Conv-only QAT checkpoint
            print("\n" + "="*80)
            print("MODEL AFTER LOADING CONV-ONLY QAT CHECKPOINT (Before LIF Calibration)")
            print("="*80)
            print_quantization_info(model, config)
            
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
            
            auto_tune = getattr(args, 'auto_tune_lif', False)
            if auto_tune:
                print("\n  ✓ Auto-tuning enabled: Per-layer LIF ranges will be profiled")
            
            model = calibrate_model_ptq(dataloader, model, device, num_batches=calibration_batches,
                                       calibrate_conv_only=False, calibrate_states_only=True, auto_tune_lif_ranges=auto_tune,
                                       config=config, loader_kwargs=kwargs, worker_init_fn=config_parser.worker_init_fn)
            
            # Print model state after LIF calibration
            print("\n" + "="*80)
            print("MODEL AFTER LIF PTQ CALIBRATION (Hybrid Mode Complete)")
            print("="*80)
            print_quantization_info(model, config)
            
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
            
            # Print model state for Conv-only QAT (mixed precision)
            print("\n" + "="*80)
            print("MODEL LOADED: CONV-ONLY QAT (Mixed Precision)")
            print("="*80)
            print_quantization_info(model, config)
            
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
            
            # Print model state for Full QAT
            print("\n" + "="*80)
            print("MODEL LOADED: FULL QAT")
            print("="*80)
            print_quantization_info(model, config)
            
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
            if use_mlflow:
                log_results(args.runid, results, path_results, eval_id)
            else:
                # For TensorBoard, save results directly
                results_file = os.path.join(path_results, 'eval_results.json')
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to: {results_file}")
    
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

    # log to mlflow (only if using mlflow)
    if not args.debug and use_mlflow and "metrics" in config.keys():
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
        default=None,
        help="MLflow run ID of the model to evaluate (for MLflow models)",
    )
    parser.add_argument(
        "--model_path",
        default="",
        help="Path to model checkpoint. Use for TensorBoard models (e.g., runs/Default_QAT_20251202_095245/checkpoints/ConvOnly_QAT/model_quant_best.pth). For MLflow models, use --runid instead.",
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=50,
        help="Number of batches for calibration. PTQ: calibrates based on Conv_only flag. QAT Conv-only: if >0 calibrates LIF (PTQ), if 0 keeps LIF at FP32. Full QAT: ignored.",
    )
    parser.add_argument(
        "--auto_tune_lif",
        action="store_true",
        help="Auto-tune LIF quantization ranges: profiles membrane potentials during calibration and sets per-layer optimal ranges. Recommended for different model variants (e.g., 16ch vs 32ch).",
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
    
    # Validate that either runid or model_path is provided
    if not args.runid and not args.model_path:
        print("Error: Must provide either --runid (for MLflow) or --model_path (for TensorBoard)")
        exit(1)
    
    if args.runid and args.model_path:
        print("Warning: Both --runid and --model_path provided. Using --model_path (TensorBoard mode)")
        args.runid = None  # Prioritize TensorBoard

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
    if args.runid:
        print(f"Source: MLflow (Run ID: {args.runid})")
    else:
        print(f"Source: TensorBoard (Path: {args.model_path})")
    
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
