import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import csv


def make_transparent(fig, ax):
    """Make figure and axis transparent: remove background, grids, ticks and spines.
    Works for 2D and 3D axes (tries to clear 3D panes when available).
    """
    # Figure background
    try:
        fig.patch.set_alpha(0)
        fig.patch.set_facecolor('none')
    except Exception:
        pass

    # Axis background (2D)
    try:
        ax.set_facecolor('none')
    except Exception:
        pass

    # For 3D axes, clear pane colors (matplotlib >= 3.x)
    try:
        # hide pane colors and gridlines
        for waxis in (getattr(ax, 'w_xaxis', None), getattr(ax, 'w_yaxis', None), getattr(ax, 'w_zaxis', None)):
            if waxis is None:
                continue
            try:
                waxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass
            try:
                # hide axis line (edge) if present
                if hasattr(waxis, 'line'):
                    waxis.line.set_color((0.0, 0.0, 0.0, 0.0))
                    waxis.line.set_visible(False)
            except Exception:
                pass
            try:
                # disable grid lines info
                if hasattr(waxis, '_axinfo') and 'grid' in waxis._axinfo:
                    waxis._axinfo['grid']['linewidth'] = 0.0
                    waxis._axinfo['grid']['color'] = (1.0, 1.0, 1.0, 0.0)
            except Exception:
                pass

    except Exception:
        # outer exception for 3D axis operations
        pass

    # Remove grid, ticks and labels
    try:
        ax.grid(False)
    except Exception:
        pass
    try:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    except Exception:
        pass
    try:
        ax.set_zticks([])
        ax.set_zticklabels([])
    except Exception:
        pass

    # Hide spines if present (2D)
    try:
        for spine in getattr(ax, 'spines', {}).values():
            spine.set_visible(False)
    except Exception:
        pass


def save_transparent(fig, ax, filename):
    """Make figure/axis transparent and save as PNG with no background."""
    make_transparent(fig, ax)
    fig.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)

# Generate sparser events
np.random.seed(42)
num_events = 500              
grid_size = 10
time_range = 1000        

xs = np.random.randint(0, grid_size, num_events)
ys = np.random.randint(0, grid_size, num_events)
ts = np.sort(np.random.rand(num_events) * time_range)

polarities = np.random.choice([-1, 1], size=num_events)
colors = ["limegreen" if p > 0 else "r" for p in polarities]

# Histograms
hist_pos = np.zeros((grid_size, grid_size), dtype=int)
hist_neg = np.zeros((grid_size, grid_size), dtype=int)

for x, y, p in zip(xs, ys, polarities):
    if p > 0:
        hist_pos[y, x] += 1
    else:
        hist_neg[y, x] += 1

# Plotting
# 3D scatter plot (events)
fig1 = plt.figure(figsize=(12, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(ts, ys, xs, c=colors, s=50)
yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# removed the central time plane to keep the events plot clean
# (previously: ts_plane = np.full_like(yy, 500)
# ax1.plot_surface(ts_plane, yy, xx, color='lightgray', alpha=0.3, zorder=0))
ax1.set_xticks(np.arange(0, time_range + 1, 100))  # Show a tick every 100 ts
ax1.set_yticklabels([])
ax1.set_zticklabels([])
'''legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='pol. +', markerfacecolor='limegreen', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='pol. -', markerfacecolor='r', markersize=8)
] 

ax1.legend(handles=legend_elements, loc='upper left') '''
ax1.set_box_aspect(aspect = (3,1,1))
plt.tight_layout()
save_transparent(fig1, ax1, 'events_transparent.png')
plt.show()

# 3D positive histogram
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
_x, _y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
xpos = _x.ravel()
ypos = _y.ravel()
zpos = np.zeros_like(xpos)
dx_pos = hist_pos.ravel()
mask_pos = dx_pos > 0
ax2.bar3d(zpos[mask_pos], ypos[mask_pos], xpos[mask_pos], dx=dx_pos[mask_pos], dy=0.8, dz=0.8, color='limegreen', alpha=0.7)
ax2.set_zlabel('')
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.set_title('')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_box_aspect(aspect = (1,1,1))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
save_transparent(fig2, ax2, 'hist_pos_transparent.png')
plt.show()

# 3D negative histogram
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
dx_neg = hist_neg.ravel()
mask_neg = dx_neg > 0
ax3.bar3d(zpos[mask_neg], ypos[mask_neg], xpos[mask_neg], dx=dx_neg[mask_neg], dy=0.8, dz=0.8, color='r', alpha=0.7)
ax3.set_zlabel('')
ax3.set_ylabel('')
ax3.set_xlabel('')
ax3.set_title('')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_zticklabels([])
ax3.set_box_aspect(aspect = (1,1,1))
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
save_transparent(fig3, ax3, 'hist_neg_transparent.png')
plt.show()

# smaller time interval accumulation
fig4, axes = plt.subplots(1, 10, figsize=(28, 5), subplot_kw={'projection': '3d'})
interval = time_range // 10

for i in range(10):
    t_start = i * interval
    t_end = (i + 1) * interval
    mask_pos = (ts >= t_start) & (ts < t_end) & (polarities > 0)
    mask_neg = (ts >= t_start) & (ts < t_end) & (polarities < 0)
    xs_pos = xs[mask_pos]
    ys_pos = ys[mask_pos]
    xs_neg = xs[mask_neg]
    ys_neg = ys[mask_neg]
    
    hist_pos_interval = np.zeros((grid_size, grid_size), dtype=int)
    hist_neg_interval = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in zip(xs_pos, ys_pos):
        hist_pos_interval[y, x] += 1
    for x, y in zip(xs_neg, ys_neg):
        hist_neg_interval[y, x] += 1

    ax = axes[i]
    _x, _y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    xpos = _x.ravel()
    ypos = _y.ravel()
    zpos = np.zeros_like(xpos)

    dx_pos = hist_pos_interval.ravel()
    mask_hist_pos = dx_pos > 0
    ax.bar3d(zpos[mask_hist_pos], ypos[mask_hist_pos], xpos[mask_hist_pos], dx=dx_pos[mask_hist_pos], dy=0.8, dz=0.8, color='limegreen', alpha=0.7)

    dx_neg = hist_neg_interval.ravel()
    mask_hist_neg = dx_neg > 0
    ax.bar3d(zpos[mask_hist_neg], ypos[mask_hist_neg], xpos[mask_hist_neg], dx=dx_neg[mask_hist_neg], dy=0.8, dz=0.8, color='r', alpha=0.7)

    # Remove all axis text
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_box_aspect((0.5, 1, 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.05)
# Save the grid of interval subplots. We pass the first axis to clear 3D panes on all axes;
# make_transparent iterates per-axis when needed but saving the whole figure keeps layout.
save_transparent(fig4, axes[0], 'intervals_transparent.png')
# Also save the whole figure explicitly (safer for multi-axis figures)
fig4.savefig('intervals_transparent_whole.png', transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()

""" # Example settings
grid_size = 10
hist_pos = np.zeros((grid_size, grid_size), dtype=int)
hist_neg = np.zeros((grid_size, grid_size), dtype=int)

# Simulate some events to activate random pixels
np.random.seed(42)
for _ in range(50):
    x = np.random.randint(0, grid_size)
    y = np.random.randint(0, grid_size)
    if np.random.rand() > 0.5:
        hist_pos[y, x] += 1
    else:
        hist_neg[y, x] += 1

# Active mask
active_mask = (hist_pos + hist_neg) > 0

# Generate (dx, dy) for active pixels, round to 1 decimal
dx_map = np.zeros((grid_size, grid_size))
dy_map = np.zeros((grid_size, grid_size))
dx_map[active_mask] = np.round(np.random.rand(np.count_nonzero(active_mask)) * 3, 1)
dy_map[active_mask] = np.round(np.random.rand(np.count_nonzero(active_mask)) * 3, 1)

# Create a 2D table with (dx, dy) strings or empty cells
table = []
for y in range(grid_size):  # Row-wise (Y axis)
    row = []
    for x in range(grid_size):  # Column-wise (X axis)
        if active_mask[y, x]:
            cell = f"({dx_map[y, x]:.1f}, {dy_map[y, x]:.1f})"
        else:
            cell = ""
        row.append(cell)
    table.append(row)

# Save as CSV
csv_filename = "displacement_table.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(table)

print(f"CSV table '{csv_filename}' saved. Each cell shows (dx, dy) or is empty.") """