import open3d as o3d
import numpy as np
from glob import glob
import os


def umeyama_alignment(source_points, target_points):
    """
    Umeyama 방식으로 similarity transformation(Sim3) 계산
    """
    assert source_points.shape == target_points.shape
    src_center = np.mean(source_points, axis=0)
    tgt_center = np.mean(target_points, axis=0)

    src_centered = source_points - src_center
    tgt_centered = target_points - tgt_center

    cov_matrix = np.dot(tgt_centered.T, src_centered) / source_points.shape[0]
    U, _, Vt = np.linalg.svd(cov_matrix)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)

    var_src = np.var(source_points, axis=0).sum()
    scale = 1.0 / var_src * np.trace(np.dot(np.diag(np.ones(3)), np.dot(U, Vt)))
    t = tgt_center - scale * R @ src_center

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=30))
    return pcd_down


def remove_outliers(pcd):
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd_clean


# 경로 설정
input_dir = "output"
ply_files = sorted(glob(os.path.join(input_dir, "*.ply")))

# 모든 포인트클라우드 로드 및 정규화
voxel_size = 0.05
pointclouds = []
for path in ply_files:
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    center = pcd.get_center()
    distances = np.linalg.norm(points - center, axis=1)
    scale = 1.0 / np.mean(distances)
    pcd.scale(scale, center=center)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pointclouds.append(pcd)

# 첫 번째 포인트클라우드를 기준으로 시작
fused_pcd = pointclouds[0]

# 정합 및 병합
for i in range(1, len(pointclouds)):
    source = pointclouds[i]
    target = fused_pcd

    # 다운샘플링 및 노멀 재추정
    source_down = preprocess(source, voxel_size)
    target_down = preprocess(target, voxel_size)

    # 특징점 추출
    src_pts = np.asarray(source_down.points)
    tgt_pts = np.asarray(target_down.points)

    # 정합 시도
    try:
        if len(src_pts) == 0 or len(tgt_pts) == 0:
            raise ValueError("정합에 사용할 점이 부족합니다.")
        sim3_transform = umeyama_alignment(src_pts, tgt_pts)
        source.transform(sim3_transform)

        # 정합 후 ICP refinement
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size * 0.4, sim3_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        source.transform(result_icp.transformation)

        # 아웃라이어 제거 후 병합
        source_clean = remove_outliers(source)
        fused_pcd += source_clean

        print(f"✔ 정합 성공: {i}번 포인트클라우드 병합 완료")

    except Exception as e:
        print(f"⚠ 정합 실패: {e}")

# 최종 저장
output_path = "fused_with_outlier_removal.ply"
o3d.io.write_point_cloud(output_path, fused_pcd)
print(f"✔ 정합된 포인트 클라우드 저장됨: {output_path}")
