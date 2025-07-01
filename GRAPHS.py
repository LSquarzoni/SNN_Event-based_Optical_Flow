import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import csv

# Generate sparser events
np.random.seed(42)
num_events = 100              
grid_size = 10
time_range = 100           

xs = np.random.randint(0, grid_size, num_events)
ys = np.random.randint(0, grid_size, num_events)
ts = np.sort(np.random.rand(num_events) * time_range)

polarities = np.random.choice([-1, 1], size=num_events)
colors = ['lightskyblue' if p > 0 else 'coral' for p in polarities]

# Histograms
hist_pos = np.zeros((grid_size, grid_size), dtype=int)
hist_neg = np.zeros((grid_size, grid_size), dtype=int)

for x, y, p in zip(xs, ys, polarities):
    if p > 0:
        hist_pos[y, x] += 1
    else:
        hist_neg[y, x] += 1

# Plotting
fig = plt.figure(figsize=(18, 5))

# 3D scatter plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(ts, ys, xs, c=colors, s=50)

ax1.set_yticks(np.arange(grid_size))
ax1.set_zticks(np.arange(grid_size))
ax1.set_xlabel('ts')
ax1.set_ylabel('y')
ax1.set_zlabel('x')
ax1.set_title('event_list')
ax1.grid(True)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='pol. +', markerfacecolor='lightskyblue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='pol. -', markerfacecolor='coral', markersize=8)
]
ax1.legend(handles=legend_elements, loc='upper left')

# 3D positive histogram
ax2 = fig.add_subplot(132, projection='3d')
_x, _y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
xpos = _x.ravel()
ypos = _y.ravel()
zpos = np.zeros_like(xpos)
dx_pos = hist_pos.ravel()
mask_pos = dx_pos > 0  # Only keep non-zero bars

ax2.bar3d(zpos[mask_pos], ypos[mask_pos], xpos[mask_pos], dx=dx_pos[mask_pos], dy=0.8, dz=0.8, color='lightskyblue', alpha=0.7)
ax2.set_zlabel('x')
ax2.set_ylabel('y')
ax2.set_xlabel('num.')
ax2.set_title('positive event_cnt')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # Positive event histogram

# 3D negative histogram
ax3 = fig.add_subplot(133, projection='3d')
dx_neg = hist_neg.ravel()
mask_neg = dx_neg > 0  # Only keep non-zero bars

ax3.bar3d(zpos[mask_neg], ypos[mask_neg], xpos[mask_neg], dx=dx_neg[mask_neg], dy=0.8, dz=0.8, color='coral', alpha=0.7)
ax3.set_zlabel('x')
ax3.set_ylabel('y')
ax3.set_xlabel('num.')
ax3.set_title('negative event_cnt')
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))  # Negative event histogram

plt.tight_layout()
plt.show()

# Example settings
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

print(f"CSV table '{csv_filename}' saved. Each cell shows (dx, dy) or is empty.")