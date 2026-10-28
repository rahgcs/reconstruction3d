
import open3d as o3d

pcd = o3d.io.read_point_cloud("merged_room.ply")

remaining = pcd
colors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1]
]

detected_planes = []

for i in range(5):

    if len(remaining.points) < 1000:
        break

    model, inliers = remaining.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )

    if len(inliers) < 1000:
        break

    plane = remaining.select_by_index(inliers)
    plane.paint_uniform_color(colors[i])

    detected_planes.append(plane)

    print(f"\nPlane {i + 1}")
    print("Equation:", model)
    print("Points:", len(inliers))

    remaining = remaining.select_by_index(
        inliers, invert=True
    )

# Visualize all detected planes
o3d.visualization.draw_geometries(
    detected_planes + [remaining],
    window_name="Multiple Plane Detection",
    width=1200,
    height=800
)
