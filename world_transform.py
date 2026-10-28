
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_data import pose
from point_cloud import points, colors

# Camera translation
t = pose[:3]

# Quaternion: qx, qy, qz, qw
quaternion = pose[3:]

# Convert quaternion to rotation matrix
R = Rotation.from_quat(quaternion).as_matrix()

# TUM uses optical camera coordinates:
# X right, Y down, Z forward.
# Its camera pose maps these coordinates into world space.
points_world = (R @ points.T).T + t

print("Rotation matrix:\n", R)
print("Translation:", t)
print("World points shape:", points_world.shape)

# Display transformed point cloud
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

step = 20

ax.scatter(
    points_world[::step, 0],
    points_world[::step, 1],
    points_world[::step, 2],
    c=colors[::step],
    s=1
)

ax.set_xlabel("World X (m)")
ax.set_ylabel("World Y (m)")
ax.set_zlabel("World Z (m)")
ax.set_title("Point Cloud in World Coordinates")

plt.show()
