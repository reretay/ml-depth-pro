import open3d as o3d
import numpy as np
from glob import glob
import os


def umeyama_alignment(source_points, target_points):
    assert source_points.shape == target_points.shape

    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    cov_matrix = np.dot(tgt_centered.T, src_centered) / source_points.shape[0]

    U, D, Vt = np.linalg.svd(cov_matrix)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    var_src = np.var(src_centered)
    scale = np.trace(np.dot(np.diag(D), np.eye(3))) / var_src

    t = mu_tgt - scale * np.dot(R, mu_src)

    transformation = np.eye(4)
    transformation[:3, :3] = scale * R
    transformation[:3, 3] = t
    return transformation


input_dir = "output"
ply_files = sorted(glob(os.path.join(input_dir, "*.ply")))

pointclouds = []
for path in ply_files:
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pointclouds.append(pcd)

fused_pcd = pointclouds[0]
for i in range(1, len(pointclouds)):
    source = pointclouds[i]
    target = fused_pcd

    # Extract keypoints
    source_down = source.voxel_down_sample(0.02)
    target_down = target.voxel_down_sample(0.02)
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute FPFH
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

    # RANSAC 매칭으로 대응점 얻기
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=0.03,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.03)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )

    # Umeyama를 이용해 스케일 포함 변환 계산
    corres = np.asarray(ransac_result.correspondence_set)
    src_pts = np.asarray(source_down.points)[corres[:, 0]]
    tgt_pts = np.asarray(target_down.points)[corres[:, 1]]

    sim3_transform = umeyama_alignment(src_pts, tgt_pts)

    # 변환 적용 및 병합
    source.transform(sim3_transform)
    fused_pcd += source

output_path = "fused_pointcloud_umeyama.ply"
o3d.io.write_point_cloud(output_path, fused_pcd)
print(f"Saved: {output_path}")
