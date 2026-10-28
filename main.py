from PIL import Image


# RGB = Image.open( "rgbd_dataset_freiburg1_room\rgb\1305031910.765238.png")

RGB = Image.open("C:\Users\GCS.RAHUL\Downloads\geometric_reconstruction\rgbd_dataset_freiburg1_room\rgb\1305031910.797230.png")
DEPTH = Image.open("rgbd_dataset_freiburg1_room\depth\1305031910.765238.png")

print("RGB size:", RGB.size)    
print("RGB format:", RGB.mode)
print("Depth size:", DEPTH.size)
print("Depth format:", DEPTH.mode)

RGB.show()
DEPTH.show()

# from pathlib import Path
# from PIL import Image

# base = Path(__file__).resolve().parent / "rgbd_dataset_freiburg1_room"

# rgb_path = base / "rgb" / "1305031910.765238.png"
# depth_dir = base / "depth"

# # RGB image timestamp
# rgb_timestamp = float(rgb_path.stem)

# # Find nearest depth timestamp
# depth_files = list(depth_dir.glob("*.png"))

# depth_path = min(
#     depth_files,
#     key=lambda p: abs(float(p.stem) - rgb_timestamp)
# )

# print("RGB file:", rgb_path.name)
# print("Depth file:", depth_path.name)
# print("Timestamp difference:", abs(float(depth_path.stem) - rgb_timestamp), "seconds")

# RGB = Image.open(rgb_path)
# DEPTH = Image.open(depth_path)

# print("RGB size:", RGB.size)
# print("RGB format:", RGB.mode)
# print("Depth size:", DEPTH.size)
# print("Depth format:", DEPTH.mode)

# RGB.show()
# DEPTH.show()