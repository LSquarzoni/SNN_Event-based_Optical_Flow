import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# Data - Number of Channels
n_channels = [32, 16, 8, 4]
aae_degrees = [24, 29, 41, 60]
memory_mb = [76.7, 38.8, 19.9, 10.4]
operations_mmacs = [7350, 1863, 478, 126]

# Data - Resolution
resolutions = ['256x256', '128x128', '64x64', '32x32']
aae_degrees_res = [24, 30, 49, 65]
memory_mb_res = [76.7, 19.3, 5, 1.4]

# Generate 4 plots with incremental data
for num_models in range(1, len(n_channels) + 1):
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Define x positions (use all positions to keep spacing consistent)
    x_pos_all = np.arange(len(n_channels)) * 0.7
    bar_width = 0.5
    
    # Subset data for this iteration
    current_x_pos = x_pos_all[:num_models]
    current_aae = aae_degrees[:num_models]
    current_mem = memory_mb[:num_models]
    current_channels = n_channels[:num_models]

    # Plot bars for AAE (Average Angular Error)
    bars = ax1.bar(current_x_pos, current_aae, bar_width, 
                   color='#2D9B3A', alpha=1, edgecolor='black', linewidth=1, zorder=3)

    # Customize primary y-axis (AAE)
    ylabel1 = ax1.set_ylabel('AAE [degrees]', fontsize=18, fontweight='bold', color='#2D9B3A', labelpad=10)
    ax1.set_xlabel('N. of Channels', fontsize=18, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.set_xticks(current_x_pos)
    ax1.set_xticklabels([f'{ch}' for ch in current_channels])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(aae_degrees) * 1.15)  # Keep consistent scale
    # Keep x-axis limits consistent and centered across all plots
    margin = 0.5
    ax1.set_xlim(-margin, x_pos_all[-1] + margin)

    # Add "baseline" text under the first bar
    ax1.text(current_x_pos[0], -0.08 * max(aae_degrees), 'baseline', 
             ha='center', va='top', fontsize=16, style='italic', fontweight='bold')

    # Create secondary y-axis for memory allocation
    ax2 = ax1.twinx()
    line = ax2.plot(current_x_pos, current_mem, '^', color='#D84315', markersize=16, 
                    label='Memory', alpha=1, zorder=4, markeredgecolor='black', markeredgewidth=1)

    # Customize secondary y-axis (Memory)
    ylabel2 = ax2.set_ylabel('Static Memory Allocation [MB]', fontsize=18, fontweight='bold', color='#B8390E', labelpad=11)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_ylim(-15, max(memory_mb) * 1.15)  # Keep consistent scale
    
    # Add horizontal reference lines (behind other elements)
    ax2.axhline(y=32, color='#D84315', linestyle='--', linewidth=2, alpha=0.8, zorder=0)
    ax2.axhline(y=1.5, color='#D84315', linestyle='--', linewidth=2, alpha=0.8, zorder=0)
    
    # Add labels on the lines
    l3_text = ax2.text(x_pos_all[-1] + 0.15, 32, 'L3 - 32MB', va='bottom', ha='left', 
             fontsize=14, color='#B8390E', fontweight='bold')
    #l3_text.set_path_effects([pe.withStroke(linewidth=1, foreground='#999999')])
    l2_text = ax2.text(x_pos_all[-1] + 0.13, 1.5, 'L2 - 1.5MB', va='bottom', ha='left', 
             fontsize=14, color='#B8390E', fontweight='bold')
    #l2_text.set_path_effects([pe.withStroke(linewidth=1, foreground='#999999')])

    # Add value labels on bars
    for i, (aae, mem, x_p) in enumerate(zip(current_aae, current_mem, current_x_pos)):
        # AAE values on bars
        aae_text = ax1.text(x_p, aae,
                 f'{aae}°',
                 ha='center', va='bottom', fontsize=16, fontweight='bold', color='black', zorder=10)
        #aae_text.set_path_effects([pe.withStroke(linewidth=1, foreground='#999999')])
        
        # Memory values on dots
        ax2.text(x_p - 0.05, mem, f'{mem} MB',
                 ha='right', va='center', fontsize=16, fontweight='bold', color='black', zorder=10,
                 #path_effects=[pe.withStroke(linewidth=1, foreground='#999999')]
        )

    # Tight layout and save
    plt.tight_layout()
    filename = f'model_comparison_{num_models}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Plot saved as '{filename}'")
    
    plt.close()

print("\nAll channel comparison plots generated successfully!")

# Generate 4 plots with incremental data for Resolution
for num_models in range(1, len(resolutions) + 1):
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Define x positions (use all positions to keep spacing consistent)
    x_pos_all = np.arange(len(resolutions)) * 0.7
    bar_width = 0.5
    
    # Subset data for this iteration
    current_x_pos = x_pos_all[:num_models]
    current_aae = aae_degrees_res[:num_models]
    current_mem = memory_mb_res[:num_models]
    current_res = resolutions[:num_models]

    # Plot bars for AAE (Average Angular Error)
    bars = ax1.bar(current_x_pos, current_aae, bar_width, 
                   color='#2D9B3A', alpha=1, edgecolor='black', linewidth=1, zorder=3)

    # Customize primary y-axis (AAE) 
    ylabel1 = ax1.set_ylabel('AAE [degrees]', fontsize=18, fontweight='bold', color='#2D9B3A', labelpad=10)
    ax1.set_xlabel('Input resolution', fontsize=18, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.set_xticks(current_x_pos)
    ax1.set_xticklabels(current_res)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(aae_degrees_res) * 1.15)  # Keep consistent scale
    # Keep x-axis limits consistent and centered across all plots
    margin = 0.5
    ax1.set_xlim(-margin, x_pos_all[-1] + margin)

    # Add "baseline" text under the first bar
    ax1.text(current_x_pos[0], -0.08 * max(aae_degrees_res), 'baseline', 
             ha='center', va='top', fontsize=16, style='italic', fontweight='bold')

    # Create secondary y-axis for memory allocation
    ax2 = ax1.twinx()
    line = ax2.plot(current_x_pos, current_mem, '^', color='#D84315', markersize=16, 
                    label='Memory', alpha=1, zorder=4, markeredgecolor='black', markeredgewidth=1)

    # Customize secondary y-axis (Memory)
    ylabel2 = ax2.set_ylabel('Static Memory Allocation [MB]', fontsize=18, fontweight='bold', color='#B8390E', labelpad=11)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_ylim(-8, max(memory_mb_res) * 1.15)  # Keep consistent scale
    
    # Add horizontal reference lines (behind other elements)
    ax2.axhline(y=32, color='#D84315', linestyle='--', linewidth=2, alpha=0.8, zorder=0)
    ax2.axhline(y=1.5, color='#D84315', linestyle='--', linewidth=2, alpha=0.8, zorder=0)
    
    # Add labels on the lines
    l3_text = ax2.text(x_pos_all[-1] + 0.15, 32, 'L3 - 32MB', va='bottom', ha='left', 
             fontsize=14, color='#B8390E', fontweight='bold')
    #l3_text.set_path_effects([pe.withStroke(linewidth=1, foreground='#999999')])
    l2_text = ax2.text(x_pos_all[-1] + 0.13, 1.5, 'L2 - 1.5MB', va='bottom', ha='left', 
             fontsize=14, color='#B8390E', fontweight='bold')
    #l2_text.set_path_effects([pe.withStroke(linewidth=1, foreground='#999999')])

    # Add value labels on bars
    for i, (aae, mem, x_p) in enumerate(zip(current_aae, current_mem, current_x_pos)):
        # AAE values on bars
        aae_text = ax1.text(x_p, aae,
                 f'{aae}°',
                 ha='center', va='bottom', fontsize=16, fontweight='bold', color='black', zorder=10)
        #aae_text.set_path_effects([pe.withStroke(linewidth=1, foreground='#999999')])
        
        # Memory values on dots
        mem_text = ax2.text(x_p - 0.05, mem, f'{mem} MB',
                 ha='right', va='center', fontsize=16, fontweight='bold', color='black', zorder=10)
        #mem_text.set_path_effects([pe.withStroke(linewidth=1, foreground='#999999')])

    # Tight layout and save
    plt.tight_layout()
    filename = f'model_comparison_resolution_{num_models}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Plot saved as '{filename}'")
    
    plt.close()

print("\nAll resolution comparison plots generated successfully!")
