import open3d as o3d
import numpy as np
import os
from glob import glob

input_dir = "output"
ply_files = sorted(glob(os.path.join(input_dir, "*.ply")))

voxel_size = 0.05  # 다운샘플링 크기
distance_threshold = voxel_size * 1.5

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result.transformation

def refine_registration(source, target, init_transform):
    return o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    ).transformation

# 초기 기준
pcd_combined = o3d.io.read_point_cloud(ply_files[0])
pcd_combined_down, pcd_combined_fpfh = preprocess_point_cloud(pcd_combined, voxel_size)

for ply in ply_files[1:]:
    source = o3d.io.read_point_cloud(ply)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    # 1. 자동 초기 정렬
    init_transform = execute_global_registration(
        source_down, pcd_combined_down, source_fpfh, pcd_combined_fpfh
    )

    # 2. 정밀 정합
    refined_transform = refine_registration(source_down, pcd_combined_down, init_transform)

    # 3. 변환 및 누적
    source.transform(refined_transform)
    pcd_combined += source

    # 기준 업데이트
    pcd_combined_down, pcd_combined_fpfh = preprocess_point_cloud(pcd_combined, voxel_size)

# 정리 및 저장
pcd_combined = pcd_combined.voxel_down_sample(voxel_size / 2)
pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
o3d.io.write_point_cloud("fused_pointcloud.ply", pcd_combined)
print("✔ 융합 완료 및 저장: fused_pointcloud.ply")
