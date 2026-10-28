
from pathlib import Path
from PIL import Image
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA= BASE_DIR / "rgbd_dataset_freiburg1_room"



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


rgb_list = read_file("rgb.txt")
depth_list = read_file("depth.txt")
pose_list = read_file("groundtruth.txt")

# Select the first RGB frame
rgb_time, rgb_info = rgb_list[0]

# Find nearest depth and camera pose
depth_time, depth_info = closest(depth_list, rgb_time)
pose_time, pose_info = closest(pose_list, rgb_time)

# Check timestamp differences
assert abs(rgb_time - depth_time) < 0.05
assert abs(rgb_time - pose_time) < 0.05

# Load images
rgb = np.array(Image.open(DATA / rgb_info[0]).convert("RGB"))
depth = np.array(Image.open(DATA / depth_info[0]))

# TUM depth: 5000 units = 1 meter
depth = depth.astype(np.float32) / 5000.0

# Camera pose: tx, ty, tz, qx, qy, qz, qw
pose = np.array(pose_info, dtype=float)

print("RGB shape:", rgb.shape)
print("Depth shape:", depth.shape)
print("Camera position:", pose[:3])
print("Camera quaternion:", pose[3:])

valid = depth[depth > 0]
print("Valid depth pixels:", valid.size)
print("Minimum depth (m):", valid.min())
print("Maximum depth (m):", valid.max())
