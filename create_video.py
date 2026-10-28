
from pathlib import Path
import cv2

DATA = Path("rgbd_dataset_freiburg1_room")
OUTPUT = "room_scan_video.mp4"

FPS = 30

# Read RGB frames in timestamp order
frames = []

with open(DATA / "rgb.txt") as file:
    for line in file:
        if line.startswith("#") or not line.strip():
            continue

        timestamp, filename = line.split()[:2]
        frames.append((float(timestamp), filename))

frames.sort()

# Get first frame dimensions
first_image = cv2.imread(str(DATA / frames[0][1]))
height, width = first_image.shape[:2]

# Create MP4 video
video = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (width, height)
)

if not video.isOpened():
    raise RuntimeError("Could not create video")

for index, (timestamp, filename) in enumerate(frames):
    image = cv2.imread(str(DATA / filename))

    if image is None:
        raise FileNotFoundError(filename)

    video.write(image)

video.release()

print("Total frames:", len(frames))
print("Video resolution:", width, "x", height)
print("Video FPS:", FPS)
print("Saved video:", Path(OUTPUT).resolve())
