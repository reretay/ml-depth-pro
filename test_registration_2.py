import open3d as o3d
import numpy as np
from glob import glob
import os

# Step 1: Load all point clouds
input_dir = "output"
ply_files = sorted(glob(os.path.join(input_dir, "*.ply")))

pointclouds = []
for path in ply_files:
    pcd = o3d.io.read_point_cloud(path)
    # Normalize scale by mean distance from center
    points = np.asarray(pcd.points)
    center = pcd.get_center()
    distances = np.linalg.norm(points - center, axis=1)
    scale = 1.0 / np.mean(distances)
    pcd.scale(scale, center=center)
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pointclouds.append(pcd)

# Step 2: Use the first point cloud as the base
fused_pcd = pointclouds[0]

# Step 3: Register and merge subsequent point clouds
voxel_size = 0.02
for i in range(1, len(pointclouds)):
    source = pointclouds[i]
    target = fused_pcd

    # Downsample
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # Re-estimate normals
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute FPFH features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

    # Global alignment via RANSAC
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )

    # ICP refinement
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 0.4, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Transform and merge
    source.transform(result_icp.transformation)
    fused_pcd += source

# Step 4: Post-processing
# 1. Downsample
fused_down = fused_pcd.voxel_down_sample(voxel_size=0.01)

# 2. Remove outliers
fused_down, _ = fused_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Step 5: Save in binary format
output_path = "fused_pointcloud_optimized.ply"
o3d.io.write_point_cloud(output_path, fused_down, write_ascii=False)
print(f"✔ Saved optimized point cloud: {output_path}")
