"""
Voltage dynamics analysis script for LIF neurons in optical flow models.

This script profiles:
- Membrane potential statistics (min, max, mean, std) per layer
- Spike rates per layer and per-channel
- Voltage distribution histograms
- Per-channel voltage ranges
- Neuron "deadness" (neurons with zero spike rate)

Usage:
    python analyze_voltage_dynamics.py <runid> [--config configs/eval_MVSEC.yml] [--num_batches 100]
    
    --skip_visualization to avoid producing the final plots and just compute statistics (useful for quick analysis)
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from models.model import LIFFireNet, LIFFireNet_short, LIFFireFlowNet, LIFFireFlowNet_short
from utils.utils import load_model


class VoltageProfiler:
    """Hooks into LIF layers to collect voltage statistics during inference."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.hooks = []
        self.voltage_stats = defaultdict(dict)
        self.spike_stats = defaultdict(dict)
        self.layer_names = {}
        
        # Map module references to readable layer names
        self._register_layer_names()
        
        # Attach hooks to all SNNtorch_ConvLIF and SNNtorch_ConvLIFRecurrent layers
        self._attach_hooks()
    
    def _register_layer_names(self):
        """Create mapping of layer indices to readable names."""
        layer_counter = defaultdict(int)
        for name, module in self.model.named_modules():
            module_type = module.__class__.__name__
            if 'ConvLIF' in module_type:
                key = id(module)
                self.layer_names[key] = name
    
    def _attach_hooks(self):
        """Attach forward hooks to LIF layers."""
        for name, module in self.model.named_modules():
            module_type = module.__class__.__name__
            
            # Hook into SNNtorch_ConvLIF and SNNtorch_ConvLIFRecurrent
            if module_type in ['SNNtorch_ConvLIF', 'SNNtorch_ConvLIFRecurrent']:
                hook = module.register_forward_hook(self._lif_hook(name))
                self.hooks.append(hook)
    
    def _lif_hook(self, layer_name):
        """Create a hook function for a specific LIF layer."""
        def hook(module, input, output):
            spks, state = output
            
            # state is [mem_out, spk_out] stacked
            if isinstance(state, torch.Tensor) and state.shape[0] == 2:
                mem = state[0]  # Membrane potential
                spk = state[1]  # Spikes
            else:
                return
            
            # Detach and move to CPU for storage
            mem_np = mem.detach().cpu().numpy()
            spk_np = spk.detach().cpu().numpy()
            
            # Store statistics
            if layer_name not in self.voltage_stats:
                self.voltage_stats[layer_name] = {
                    'voltages': [],
                    'spikes': [],
                }
            
            self.voltage_stats[layer_name]['voltages'].append(mem_np)
            self.voltage_stats[layer_name]['spikes'].append(spk_np)
        
        return hook
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_statistics(self):
        """Compute aggregated statistics from collected data."""
        import time
        stats = {}
        
        for layer_idx, layer_name in enumerate(sorted(self.voltage_stats.keys())):
            print(f"  Processing layer {layer_idx+1}/{len(self.voltage_stats)}: {layer_name}...", end='', flush=True)
            layer_start = time.time()
            
            data = self.voltage_stats[layer_name]
            voltages = np.concatenate(data['voltages'], axis=0)  # [B*T, C, H, W]
            spikes = np.concatenate(data['spikes'], axis=0)
            
            batch_size, channels, height, width = voltages.shape
            
            # Compute all percentiles at once on flattened 1D array (faster than separate calls)
            print(".", end='', flush=True)
            voltages_flat_1d = voltages.flatten()
            percentiles = np.percentile(voltages_flat_1d, [5, 50, 95])
            del voltages_flat_1d
            
            # Reshape spike data once for channel-wise operations
            print(".", end='', flush=True)
            spikes_reshaped = spikes.reshape(batch_size, channels, -1)  # [B*T, C, H*W]
            voltages_reshaped = voltages.reshape(batch_size, channels, -1)  # [B*T, C, H*W]
            
            # Compute per-channel stats efficiently
            print(".", end='', flush=True)
            channel_spike_rates = spikes_reshaped.mean(axis=(0, 2))  # Mean over time and spatial
            per_neuron_spike_rates = spikes_reshaped.mean(axis=0)  # Mean over time only: [C, H*W]
            
            # Compute per-channel voltage stats
            print(".", end='', flush=True)
            channel_voltage_mins = voltages_reshaped.min(axis=(0, 2))  # [C]
            channel_voltage_maxs = voltages_reshaped.max(axis=(0, 2))  # [C]
            channel_voltage_means = voltages_reshaped.mean(axis=(0, 2))  # [C]
            channel_voltage_stds = voltages_reshaped.std(axis=(0, 2))  # [C]
            
            # Voltage range per channel
            channel_voltage_ranges = channel_voltage_maxs - channel_voltage_mins
            
            print(".", end='', flush=True)
            stats[layer_name] = {
                'num_samples': batch_size,
                'num_channels': channels,
                'spatial_size': height * width,
                
                # Global statistics
                'voltage_min': float(voltages.min()),
                'voltage_max': float(voltages.max()),
                'voltage_mean': float(voltages.mean()),
                'voltage_std': float(voltages.std()),
                'voltage_median': float(percentiles[1]),
                'voltage_q5': float(percentiles[0]),
                'voltage_q95': float(percentiles[2]),
                
                # Per-channel statistics
                'channel_voltage_mins': channel_voltage_mins,
                'channel_voltage_maxs': channel_voltage_maxs,
                'channel_voltage_means': channel_voltage_means,
                'channel_voltage_stds': channel_voltage_stds,
                
                # Spike rate statistics
                'spike_rate_global': float(spikes.mean()),
                'channel_spike_rates': channel_spike_rates,
                'per_neuron_spike_rates': per_neuron_spike_rates,
                
                # Detailed spike rate analysis
                'channels_dead': int((channel_spike_rates == 0).sum()),
                'channels_very_low': int((channel_spike_rates < 0.01).sum()),
                'channels_low': int((channel_spike_rates < 0.05).sum()),
                'channels_moderate': int((channel_spike_rates < 0.20).sum()),
                
                # Per-neuron analysis
                'neurons_dead': int((per_neuron_spike_rates == 0).sum()),
                'neurons_very_low': int((per_neuron_spike_rates < 0.01).sum()),
                'neurons_low': int((per_neuron_spike_rates < 0.05).sum()),
                'neurons_moderate': int((per_neuron_spike_rates < 0.20).sum()),
                'total_neurons': int(per_neuron_spike_rates.size),
                
                # Voltage range per channel
                'channel_voltage_ranges': channel_voltage_ranges,
                
                # Raw data for histograms
                'voltages_raw': voltages.flatten(),
                'spikes_raw': spikes.flatten(),
            }
            
            # Clean up large arrays to free memory
            del voltages, spikes, spikes_reshaped, voltages_reshaped
            
            layer_time = time.time() - layer_start
            print(f" ({layer_time:.2f}s)")
        
        return stats


def analyze_model(args, config_parser):
    """Run voltage dynamics analysis on a trained model."""
    
    import time
    start_time = time.time()
    
    # Load configuration
    import mlflow
    mlflow.set_tracking_uri(args.path_mlflow)
    
    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)
    
    # Override some config settings for analysis
    config["vis"]["enabled"] = False
    config["vis"]["store"] = False
    config["vis"]["bars"] = True
    config["loader"]["batch_size"] = 1
    
    print(f"\n{'='*80}")
    print(f"VOLTAGE DYNAMICS ANALYSIS: {args.runid}")
    print(f"{'='*80}\n")
    
    # Setup
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    
    # Model
    model = eval(config["model"]["name"])(config["model"]).to(device)
    model = load_model(args.runid, model, device)
    model.eval()
    
    print(f"Model: {config['model']['name']}")
    print(f"Device: {device}")
    print(f"Quantization enabled: {config['model']['quantization']['enabled']}")
    
    # Data
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )
    
    # Profiler
    profiler = VoltageProfiler(model, device)
    
    # Inference loop
    print("\nRunning inference to collect voltage statistics...")
    batch_count = 0
    inference_start = time.time()
    
    try:
        with torch.no_grad():
            while batch_count < args.num_batches:
                # Recreate dataloader iterator each time to handle multiple sequences
                dataloader_iter = iter(dataloader)
                
                for inputs in dataloader_iter:
                    if data.new_seq:
                        data.new_seq = False
                        model.reset_states()
                    
                    # Forward pass
                    _ = model(
                        inputs["event_voxel"].to(device),
                        inputs["event_cnt"].to(device),
                        log=False
                    )
                    
                    batch_count += 1
                    if batch_count % 50 == 0:
                        elapsed = time.time() - inference_start
                        throughput = batch_count / max(elapsed, 1)
                        eta = (args.num_batches - batch_count) / max(throughput, 1)
                        print(f"  Processed {batch_count}/{args.num_batches} batches ({throughput:.2f} batches/sec, ETA: {eta:.1f}s)...")
                    
                    if batch_count >= args.num_batches:
                        raise StopIteration
    
    except StopIteration:
        pass
    
    elapsed_inference = time.time() - inference_start
    print(f"\nTotal batches processed: {batch_count} in {elapsed_inference:.2f}s\n")
    
    # Compute statistics
    print("Computing statistics...")
    start_stats = time.time()
    stats = profiler.compute_statistics()
    elapsed_stats = time.time() - start_stats
    print(f"Statistics computation took {elapsed_stats:.2f}s")
    
    # Print results
    print_statistics(stats)
    
    # Generate visualizations (optional)
    if not args.skip_visualizations:
        output_dir = f"voltage_analysis/{args.runid}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations in {output_dir}...")
        start_viz = time.time()
        generate_visualizations(stats, output_dir)
        elapsed_viz = time.time() - start_viz
        print(f"Visualization generation took {elapsed_viz:.2f}s")
        
        # Save statistics to file
        save_statistics_csv(stats, output_dir)
    else:
        print("\nVisualization generation skipped (--skip_visualizations flag set)")
    
    profiler.remove_hooks()
    total_time = time.time() - start_time
    print(f"\nAnalysis complete! (Total time: {total_time:.2f}s)\n")


def print_statistics(stats):
    """Print voltage and spike statistics in tabular format."""
    
    print(f"\n{'LAYER':<40} {'MIN':<12} {'MAX':<12} {'MEAN':<12} {'STD':<12}")
    print(f"{'-'*88}")
    
    for layer_name in sorted(stats.keys()):
        s = stats[layer_name]
        print(f"{layer_name:<40} {s['voltage_min']:<12.4f} {s['voltage_max']:<12.4f} "
              f"{s['voltage_mean']:<12.4f} {s['voltage_std']:<12.4f}")
    
    print(f"\n{'LAYER':<40} {'SPIKE RATE':<15} {'DEAD CHANNELS':<20}")
    print(f"{'-'*75}")
    
    for layer_name in sorted(stats.keys()):
        s = stats[layer_name]
        spike_rate = s['spike_rate_global'] * 100
        print(f"{layer_name:<40} {spike_rate:<14.2f}% {s['channels_dead']:<20}")
    
    print(f"\n{'LAYER':<40} {'Q5':<12} {'MEDIAN':<12} {'Q95':<12} {'RANGE':<12}")
    print(f"{'-'*88}")
    
    for layer_name in sorted(stats.keys()):
        s = stats[layer_name]
        v_range = s['voltage_max'] - s['voltage_min']
        print(f"{layer_name:<40} {s['voltage_q5']:<12.4f} {s['voltage_median']:<12.4f} "
              f"{s['voltage_q95']:<12.4f} {v_range:<12.4f}")
    
    # ENHANCED: Print detailed channel utilization analysis
    print(f"\n{'='*100}")
    print("DETAILED CHANNEL UTILIZATION ANALYSIS")
    print(f"{'='*100}")
    print(f"{'LAYER':<40} {'DEAD':<12} {'<1%':<12} {'<5%':<12} {'<20%':<12}")
    print(f"{'-'*88}")
    
    for layer_name in sorted(stats.keys()):
        s = stats[layer_name]
        print(f"{layer_name:<40} {s['channels_dead']:<12} {s['channels_very_low']:<12} "
              f"{s['channels_low']:<12} {s['channels_moderate']:<12}")
    
    # ENHANCED: Print detailed neuron utilization analysis
    print(f"\n{'='*100}")
    print("DETAILED NEURON UTILIZATION ANALYSIS (Per-Neuron across all spatial locations)")
    print(f"{'='*100}")
    print(f"{'LAYER':<40} {'DEAD':<15} {'<1%':<15} {'<5%':<15} {'<20%':<15} {'TOTAL':<12}")
    print(f"{'-'*105}")
    
    for layer_name in sorted(stats.keys()):
        s = stats[layer_name]
        print(f"{layer_name:<40} {s['neurons_dead']:<15} {s['neurons_very_low']:<15} "
              f"{s['neurons_low']:<15} {s['neurons_moderate']:<15} {s['total_neurons']:<12}")
    
    # Print summary statistics
    print(f"\n{'='*100}")
    print("LEGEND: Neurons with spike rates <20% may underutilize capacity")
    print("        Neurons with spike rates <1% are essentially inactive")
    print(f"{'='*100}\n")


def generate_visualizations(stats, output_dir):
    """Generate histogram and distribution plots."""
    
    num_layers = len(stats)
    
    # 1. Voltage distribution histograms
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for idx, layer_name in enumerate(sorted(stats.keys())):
        s = stats[layer_name]
        voltages = s['voltages_raw']
        
        axes[idx].hist(voltages, bins=100, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{layer_name} - Voltage Distribution (n={len(voltages):,})')
        axes[idx].set_xlabel('Membrane Potential (V)')
        axes[idx].set_ylabel('Count')
        axes[idx].axvline(s['voltage_mean'], color='r', linestyle='--', label=f"Mean: {s['voltage_mean']:.3f}")
        axes[idx].axvline(s['voltage_median'], color='g', linestyle='--', label=f"Median: {s['voltage_median']:.3f}")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'voltage_distributions.png'), dpi=150)
    plt.close()
    
    # 2. Per-channel voltage ranges
    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for idx, layer_name in enumerate(sorted(stats.keys())):
        s = stats[layer_name]
        channels = np.arange(s['num_channels'])
        mins = s['channel_voltage_mins']
        maxs = s['channel_voltage_maxs']
        ranges = maxs - mins
        
        axes[idx].bar(channels, ranges, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{layer_name} - Per-Channel Voltage Range')
        axes[idx].set_xlabel('Channel')
        axes[idx].set_ylabel('Voltage Range (max - min)')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_voltage_ranges.png'), dpi=150)
    plt.close()
    
    # 3. Per-channel spike rates
    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for idx, layer_name in enumerate(sorted(stats.keys())):
        s = stats[layer_name]
        channels = np.arange(s['num_channels'])
        spike_rates = s['channel_spike_rates'] * 100
        
        colors = ['red' if rate == 0 else 'blue' for rate in spike_rates]
        axes[idx].bar(channels, spike_rates, edgecolor='black', alpha=0.7, color=colors)
        axes[idx].set_title(f'{layer_name} - Per-Channel Spike Rate (red = dead channels)')
        axes[idx].set_xlabel('Channel')
        axes[idx].set_ylabel('Spike Rate (%)')
        axes[idx].set_ylim([0, 100])
        axes[idx].axhline(50, color='gray', linestyle='--', alpha=0.5, label='50%')
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_spike_rates.png'), dpi=150)
    plt.close()
    
    # 4. Voltage statistics summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    layer_names = sorted(stats.keys())
    voltage_mins = [stats[l]['voltage_min'] for l in layer_names]
    voltage_maxs = [stats[l]['voltage_max'] for l in layer_names]
    voltage_means = [stats[l]['voltage_mean'] for l in layer_names]
    spike_rates = [stats[l]['spike_rate_global'] * 100 for l in layer_names]
    
    x = np.arange(len(layer_names))
    width = 0.35
    
    # Min/Max voltages
    axes[0, 0].bar(x - width/2, voltage_mins, width, label='Min', alpha=0.7)
    axes[0, 0].bar(x + width/2, voltage_maxs, width, label='Max', alpha=0.7)
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].set_title('Min/Max Voltages per Layer')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Mean voltages
    axes[0, 1].bar(x, voltage_means, alpha=0.7, color='orange')
    axes[0, 1].set_ylabel('Mean Voltage (V)')
    axes[0, 1].set_title('Mean Voltages per Layer')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Spike rates
    axes[1, 0].bar(x, spike_rates, alpha=0.7, color='green')
    axes[1, 0].set_ylabel('Spike Rate (%)')
    axes[1, 0].set_title('Global Spike Rates per Layer')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].axhline(50, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Dead channels
    dead_channels = [stats[l]['channels_dead'] for l in layer_names]
    axes[1, 1].bar(x, dead_channels, alpha=0.7, color='red')
    axes[1, 1].set_ylabel('Number of Dead Channels')
    axes[1, 1].set_title('Dead Channels per Layer')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'voltage_summary.png'), dpi=150)
    plt.close()
    
    # 5. Spike rate distribution (NEW: shows how many neurons have each spike rate)
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for idx, layer_name in enumerate(sorted(stats.keys())):
        s = stats[layer_name]
        spike_rates = s['per_neuron_spike_rates'].flatten() * 100  # Flatten and convert to percentage
        
        # Histogram of spike rates
        axes[idx].hist(spike_rates, bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[idx].set_title(f'{layer_name} - Spike Rate Distribution (Per-Neuron)')
        axes[idx].set_xlabel('Spike Rate (%)')
        axes[idx].set_ylabel('Number of Neurons')
        axes[idx].axvline(np.mean(spike_rates), color='r', linestyle='--', 
                        label=f"Mean: {np.mean(spike_rates):.2f}%")
        axes[idx].axvline(np.median(spike_rates), color='g', linestyle='--', 
                        label=f"Median: {np.median(spike_rates):.2f}%")
        
        # Mark threshold ranges
        axes[idx].axvline(1, color='orange', linestyle=':', alpha=0.7, label="1% threshold")
        axes[idx].axvline(5, color='yellow', linestyle=':', alpha=0.7, label="5% threshold")
        axes[idx].axvline(20, color='cyan', linestyle=':', alpha=0.7, label="20% threshold")
        
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spike_rate_distribution.png'), dpi=150)
    plt.close()
    
    # 6. Neuron utilization summary (stacked bar chart showing categories)
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    
    layer_names = sorted(stats.keys())
    dead_neurons = [stats[l]['neurons_dead'] for l in layer_names]
    very_low_neurons = [stats[l]['neurons_very_low'] - stats[l]['neurons_dead'] for l in layer_names]
    low_neurons = [stats[l]['neurons_low'] - stats[l]['neurons_very_low'] for l in layer_names]
    moderate_neurons = [stats[l]['neurons_moderate'] - stats[l]['neurons_low'] for l in layer_names]
    active_neurons = [stats[l]['total_neurons'] - stats[l]['neurons_moderate'] for l in layer_names]
    
    x = np.arange(len(layer_names))
    width = 0.6
    
    ax = axes
    p1 = ax.bar(x, dead_neurons, width, label='Dead (0%)', color='red', alpha=0.8)
    p2 = ax.bar(x, very_low_neurons, width, bottom=dead_neurons, label='Very Low (<1%)', color='orange', alpha=0.8)
    
    bottom2 = np.array(dead_neurons) + np.array(very_low_neurons)
    p3 = ax.bar(x, low_neurons, width, bottom=bottom2, label='Low (<5%)', color='yellow', alpha=0.8)
    
    bottom3 = bottom2 + np.array(low_neurons)
    p4 = ax.bar(x, moderate_neurons, width, bottom=bottom3, label='Moderate (<20%)', color='lightblue', alpha=0.8)
    
    bottom4 = bottom3 + np.array(moderate_neurons)
    p5 = ax.bar(x, active_neurons, width, bottom=bottom4, label='Active (≥20%)', color='green', alpha=0.8)
    
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Neuron Utilization Categories per Layer')
    ax.set_xticks(x)
    ax.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neuron_utilization.png'), dpi=150)
    plt.close()
    
    print(f"  - voltage_distributions.png")
    print(f"  - channel_voltage_ranges.png")
    print(f"  - channel_spike_rates.png")
    print(f"  - voltage_summary.png")
    print(f"  - spike_rate_distribution.png")
    print(f"  - neuron_utilization.png")


def save_statistics_csv(stats, output_dir):
    """Save statistics to CSV file for further analysis."""
    import csv
    
    csv_path = os.path.join(output_dir, 'voltage_statistics.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header - ENHANCED with neuron utilization metrics
        writer.writerow([
            'Layer', 'Voltage Min', 'Voltage Max', 'Voltage Mean', 'Voltage Std',
            'Voltage Q5', 'Voltage Median', 'Voltage Q95', 'Voltage Range',
            'Global Spike Rate (%)', 
            'Channels Dead', 'Channels <1%', 'Channels <5%', 'Channels <20%',
            'Neurons Dead', 'Neurons <1%', 'Neurons <5%', 'Neurons <20%', 'Total Neurons',
            'Num Channels', 'Num Samples'
        ])
        
        # Data
        for layer_name in sorted(stats.keys()):
            s = stats[layer_name]
            v_range = s['voltage_max'] - s['voltage_min']
            writer.writerow([
                layer_name,
                f"{s['voltage_min']:.6f}",
                f"{s['voltage_max']:.6f}",
                f"{s['voltage_mean']:.6f}",
                f"{s['voltage_std']:.6f}",
                f"{s['voltage_q5']:.6f}",
                f"{s['voltage_median']:.6f}",
                f"{s['voltage_q95']:.6f}",
                f"{v_range:.6f}",
                f"{s['spike_rate_global']*100:.2f}",
                s['channels_dead'],
                s['channels_very_low'],
                s['channels_low'],
                s['channels_moderate'],
                s['neurons_dead'],
                s['neurons_very_low'],
                s['neurons_low'],
                s['neurons_moderate'],
                s['total_neurons'],
                s['num_channels'],
                s['num_samples'],
            ])
    
    print(f"  - voltage_statistics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze voltage dynamics in LIF neurons')
    parser.add_argument('runid', help='MLflow run ID')
    parser.add_argument(
        '--config',
        default='configs/eval_MVSEC.yml',
        help='Config file for evaluation'
    )
    parser.add_argument(
        '--path_mlflow',
        default='',
        help='Location of MLflow UI'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=100,
        help='Number of batches to analyze (default: 100)'
    )
    parser.add_argument(
        '--skip_visualizations',
        action='store_true',
        help='Skip visualization generation to speed up analysis (only compute statistics)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_model(args, YAMLParser(args.config))
