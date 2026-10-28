
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "rgbd_dataset_freiburg1_room"

# TUM Freiburg 1 intrinsics
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5


def read_file(filename):
    rows = []

    with open(DATA / filename) as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue

            values = line.split()
            rows.append((float(values[0]), values[1:]))

    return rows


def closest(rows, timestamp):
    return min(rows, key=lambda row: abs(row[0] - timestamp))


def voxel_downsample(points, colors, voxel_size):
    voxel_ids = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_ids, axis=0, return_inverse=True)
    voxel_counts = np.bincount(inverse)

    downsampled_points = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis]) / voxel_counts
            for axis in range(3)
        ]
    )
    downsampled_colors = np.column_stack(
        [
            np.bincount(inverse, weights=colors[:, channel]) / voxel_counts
            for channel in range(3)
        ]
    )
    return downsampled_points, downsampled_colors


def save_colored_ply(filename, points, colors):
    rgb = np.clip(np.round(colors * 255), 0, 255).astype(np.uint8)
    with filename.open("w", encoding="ascii") as ply_file:
        ply_file.write(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        np.savetxt(
            ply_file,
            np.column_stack((points, rgb)),
            fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"],
        )


if not DATA.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {DATA}")


rgb_list = read_file("rgb.txt")
depth_list = read_file("depth.txt")
pose_list = read_file("groundtruth.txt")

if not rgb_list or not depth_list or not pose_list:
    raise ValueError("RGB, depth, or ground-truth timestamp list is empty.")

all_points = []
all_colors = []

# Select 10 frames spread across the recording
indices = np.linspace(
    0, len(rgb_list) - 1, 10, dtype=int
)

for index in indices:
    rgb_time, rgb_info = rgb_list[index]
    depth_time, depth_info = closest(depth_list, rgb_time)
    pose_time, pose_info = closest(pose_list, rgb_time)

    # Skip frames that are not synchronized
    if abs(rgb_time - depth_time) > 0.05:
        continue

    if abs(rgb_time - pose_time) > 0.05:
        continue

    rgb = np.array(
        Image.open(DATA / rgb_info[0]).convert("RGB")
    )

    depth = np.array(
        Image.open(DATA / depth_info[0]),
        dtype=np.float32
    ) / 5000.0

    height, width = depth.shape
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB/depth dimensions differ for frame {index}: "
            f"{rgb.shape[:2]} vs {depth.shape}"
        )

    u, v = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    valid = depth > 0

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    points = np.column_stack((x, y, z))
    colors = rgb[valid] / 255.0

    # Camera-to-world transformation
    pose = np.array(pose_info, dtype=float)
    translation = pose[:3]
    quaternion = pose[3:]

    R = Rotation.from_quat(quaternion).as_matrix()

    points_world = points @ R.T + translation

    all_points.append(points_world)
    all_colors.append(colors)

    print(f"Processed frame {index}")

# Combine all frames
if not all_points:
    raise ValueError(
        "No frames were processed: none met the 50 ms synchronization threshold."
    )

merged_points = np.vstack(all_points)
merged_colors = np.vstack(all_colors)

print("Total points before filtering:", len(merged_points))

# Reduce overlapping points
merged_points, merged_colors = voxel_downsample(
    merged_points, merged_colors, voxel_size=0.02
)

print("Points after downsampling:", len(merged_points))

# Save beside this script so the output location does not depend on the
# directory from which the script was launched.
output_path = BASE_DIR / "merged_room.ply"
save_colored_ply(output_path, merged_points, merged_colors)
print(f"Saved point cloud to: {output_path}")

# Display a subset to keep rendering responsive.
display_step = max(1, len(merged_points) // 50_000)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    merged_points[::display_step, 0],
    merged_points[::display_step, 1],
    merged_points[::display_step, 2],
    c=merged_colors[::display_step],
    s=1,
)
ax.set_xlabel("World X (m)")
ax.set_ylabel("World Y (m)")
ax.set_zlabel("World Z (m)")
ax.set_title("Merged RGB-D Point Cloud")
preview_path = BASE_DIR / "merged_room_preview.png"
fig.savefig(preview_path, dpi=150, bbox_inches="tight")
print(f"Saved point-cloud preview to: {preview_path}")
plt.show()
