import matplotlib.pyplot as plt
import numpy as np

# Data
resolutions = ['256×256', '128×128', '64×64', '32×32']
x_pos = np.arange(len(resolutions))

# Plot 1: Old approach (Average pooling on GT and input)
aae_1 = [23.6, 29.6, 49.1, 64.9]
aee_1 = [2.7, 2.4, 2.7, 2.8]

# Plot 2: New approach (Average pooling on input, upsampling on output)
aae_2 = [23.6, 27.8, 45.1, 60.4]
aee_2 = [2.7, 2.8, 3.2, 3.5]

# Create figures
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))

old_aee_color = "#57A8FF"
old_aae_color = "#86ED74"
new_aee_color = "#007EFC"
new_aae_color = "#469C32"

# Function to create plot
def create_plot(fig, ax, resolutions, x_pos, aae, aee, title, aae_color='green', aee_color='blue', use_pale_colors=False):
    # Primary y-axis (AAE)
    ax.set_xlabel('Resolution', fontsize=12, fontweight='bold')
    ax.set_ylabel('AAE (degrees)', fontsize=12, fontweight='bold', color=aae_color)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(resolutions)
    ax.set_ylim(20, 70)
    
    # Plot AAE on primary y-axis (green)
    line1 = ax.plot(x_pos, aae, 'o-', color=aae_color, linewidth=2, markersize=8, label='AAE (angular error) (lower is better)', alpha=0.7 if use_pale_colors else 1.0)
    ax.tick_params(axis='y', labelcolor=aae_color)
    
    # Add labels on AAE points
    for i, (x, y) in enumerate(zip(x_pos, aae)):
        ax.text(x, y - 3, f'{y:.1f}', ha='center', va='bottom', fontsize=12, color=aae_color, fontweight='bold')
    
    # Secondary y-axis (AEE)
    ax2_twin = ax.twinx()
    ax2_twin.set_ylim(2.35, 3.65)
    ax2_twin.set_ylabel('AEE (pixels)', fontsize=12, fontweight='bold', color=aee_color)
    
    # Plot AEE on secondary y-axis (blue)
    line2 = ax2_twin.plot(x_pos, aee, 's-', color=aee_color, linewidth=2, markersize=8, label='AEE (modulus error) (lower is better)', alpha=0.7 if use_pale_colors else 1.0)
    ax2_twin.tick_params(axis='y', labelcolor=aee_color)
    
    # Add labels on AEE points
    for i, (x, y) in enumerate(zip(x_pos, aee)):
        ax2_twin.text(x, y + 0.03, f'{y:.1f}', ha='center', va='bottom', fontsize=12, color=aee_color, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, ax, ax2_twin

# Create Plot 1 (Old approach with pale colors)
fig1, ax1, ax1_twin = create_plot(fig1, ax1, resolutions, x_pos, aae_1, aee_1, 
                                   'Baseline model at different resolutions', 
                                   aae_color=old_aae_color, aee_color=old_aee_color, use_pale_colors=True)

# Create Plot 2 (New approach with bright colors)
fig2, ax2, ax2_twin = create_plot(fig2, ax2, resolutions, x_pos, aae_2, aee_2, 
                                   'Baseline model at different resolutions',
                                   aae_color=new_aae_color, aee_color=new_aee_color)

# Save figures
fig1.savefig('/home/msc25h1/event_flow/plots/evaluation_comparison_old_approach.png', dpi=300, bbox_inches='tight')
fig2.savefig('/home/msc25h1/event_flow/plots/evaluation_comparison_new_approach.png', dpi=300, bbox_inches='tight')

# Plot 3: Comparison of AAE between old and new approach
fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.set_xlabel('Resolution', fontsize=12, fontweight='bold')
ax3.set_ylabel('AAE (degrees) (lower is better)', fontsize=12, fontweight='bold')
ax3.set_title('Baseline model at different resolutions', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(resolutions)
ax3.set_ylim(20, 70)

# Plot AAE for old approach (pale blue)
line1 = ax3.plot(x_pos, aae_1, 'o-', color=old_aae_color, linewidth=2.5, markersize=8, label='AAE (old approach)', alpha=0.85)

# Plot AAE for new approach (dark green)
line2 = ax3.plot(x_pos, aae_2, 's-', color=new_aae_color, linewidth=2.5, markersize=8, label='AAE (new approach)', alpha=0.9)

# Add labels on old approach points
for i, (x, y) in enumerate(zip(x_pos, aae_1)):
    ax3.text(x, y + 1, f'{y:.1f}', ha='right', va='bottom', fontsize=12, color=old_aae_color, fontweight='bold')

# Add labels on new approach points
for i, (x, y) in enumerate(zip(x_pos, aae_2)):
    ax3.text(x, y - 1.5, f'{y:.1f}', ha='left', va='top', fontsize=12, color=new_aae_color, fontweight='bold')

# Legend
ax3.legend(loc='upper left', fontsize=11)

# Grid
ax3.grid(True, alpha=0.3)

fig3.tight_layout()

# Save figures
fig3.savefig('/home/msc25h1/event_flow/plots/evaluation_comparison_aae_comparison.png', dpi=300, bbox_inches='tight')

print("✓ Plots saved:")
print("  - /home/msc25h1/event_flow/plots/evaluation_comparison_old_approach.png")
print("  - /home/msc25h1/event_flow/plots/evaluation_comparison_new_approach.png")
print("  - /home/msc25h1/event_flow/plots/evaluation_comparison_aae_comparison.png")

plt.show()
