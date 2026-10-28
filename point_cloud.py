
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Assuming rgb and depth have been loaded
# from our previous load_data.py
from load_data import rgb, depth

# TUM Freiburg 1 intrinsics
fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5

height, width = depth.shape

# Create pixel coordinates
u, v = np.meshgrid(
    np.arange(width),
    np.arange(height)
)

# Keep only valid depth pixels
valid = depth > 0

z = depth[valid]
x = (u[valid] - cx) * z / fx
y = (v[valid] - cy) * z / fy

# Combine into 3D points
points = np.column_stack((x, y, z))

# Get corresponding RGB colors
colors = rgb[valid] / 255.0

print("3D points shape:", points.shape)
print("Colors shape:", colors.shape)

# Preview points in 3D
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot a subset for faster visualization
step = 20

ax.scatter(
    points[::step, 0],
    points[::step, 2],
    -points[::step, 1],
    c=colors[::step],
    s=1
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_zlabel("-Y (m)")
ax.set_title("Single Frame 3D Point Cloud")

plt.show()
