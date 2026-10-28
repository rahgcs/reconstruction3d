
import open3d as o3d

# Load merged point cloud
pcd = o3d.io.read_point_cloud("merged_room.ply")

print("Total points:", len(pcd.points))

# Detect largest plane using RANSAC
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000
)

a, b, c, d = plane_model

print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
print("Plane points:", len(inliers))
print("Remaining points:", len(pcd.points) - len(inliers))

# Separate plane and remaining points
plane_cloud = pcd.select_by_index(inliers)
remaining_cloud = pcd.select_by_index(inliers, invert=True)

# Highlight detected plane in red
plane_cloud.paint_uniform_color([1, 0, 0])

# Save detected plane
o3d.io.write_point_cloud("detected_plane.ply", plane_cloud)

# Visualize
o3d.visualization.draw_geometries(
    [plane_cloud, remaining_cloud],
    window_name="RANSAC Plane Detection",
    width=1200,
    height=800
)
