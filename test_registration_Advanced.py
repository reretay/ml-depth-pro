import open3d as o3d
import numpy as np
from glob import glob
import os

def estimate_normals(pcd, radius=0.1, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    _, inliers = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(inliers)

def umeyama_alignment(source_points, target_points):
    assert source_points.shape == target_points.shape
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)
    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    cov = np.dot(tgt_centered.T, src_centered) / source_points.shape[0]
    U, _, Vt = np.linalg.svd(cov)
    R = np.dot(U, Vt)

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)

    var_src = np.var(source_points, axis=0).sum()
    scale = np.trace(np.dot(np.dot(U, np.diag(_)), Vt)) / var_src
    t = mu_tgt - scale * np.dot(R, mu_src)

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T

def get_correspondences(src, tgt, max_dist=0.1):
    pcd_tree = o3d.geometry.KDTreeFlann(tgt)
    src_pts = np.asarray(src.points)
    tgt_pts = []
    src_valid = []
    for i, pt in enumerate(src_pts):
        [_, idx, dist] = pcd_tree.search_knn_vector_3d(pt, 1)
        if dist[0] < max_dist ** 2:
            tgt_pts.append(tgt.points[idx[0]])
            src_valid.append(pt)
    return np.array(src_valid), np.array(tgt_pts)

def register_pointclouds(source, target, voxel_size):
    # Downsample
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    estimate_normals(source_down)
    estimate_normals(target_down)

    try:
        # Find correspondences
        src_corr, tgt_corr = get_correspondences(source_down, target_down, max_dist=voxel_size * 2.0)
        if len(src_corr) < 10:
            raise ValueError("Too few correspondences for Umeyama")

        # Apply Umeyama similarity transformation
        T_umeyama = umeyama_alignment(src_corr, tgt_corr)
        source.transform(T_umeyama)

    except Exception as e:
        print(f"⚠ Umeyama 정합 실패: {e}")

    # FPFH feature + RANSAC
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    estimate_normals(source_down)
    estimate_normals(target_down)

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )

    try:
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size * 0.5,
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        source.transform(result_icp.transformation)
        return source
    except Exception as e:
        print(f"⚠ ICP 실패: {e}")
        return None

# === Main Process ===
input_dir = "output"
ply_files = sorted(glob(os.path.join(input_dir, "*.ply")))
voxel_size = 0.05

# Load and normalize point clouds
pointclouds = []
for path in ply_files:
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    center = pcd.get_center()
    distances = np.linalg.norm(points - center, axis=1)
    scale = 1.0 / np.mean(distances)
    pcd.scale(scale, center=center)
    estimate_normals(pcd)
    pointclouds.append(pcd)

# Initialize with the first point cloud
fused = pointclouds[0]

for i in range(1, len(pointclouds)):
    print(f"▶ 정합 중: {i-1} ➕ {i}")
    registered = register_pointclouds(pointclouds[i], fused, voxel_size)
    if registered is not None:
        fused += registered
    else:
        print(f"⚠ {i}번째 포인트클라우드 정합 실패, 스킵됨")

# Remove outliers
print("📦 Outlier 제거 중...")
fused = remove_outliers(fused, nb_neighbors=30, std_ratio=2.0)

# Optional: remove duplicate points
fused = fused.voxel_down_sample(voxel_size=voxel_size / 2.0)

# Save
output_path = "fused_pointcloud_advanced.ply"
o3d.io.write_point_cloud(output_path, fused)
print(f"✅ 최종 정합된 포인트 클라우드 저장 완료: {output_path}")
