import open3d as o3d
import numpy as np
from glob import glob
import os

def umeyama_alignment(source_points, target_points):
    assert source_points.shape == target_points.shape
    src_mean = np.mean(source_points, axis=0)
    tgt_mean = np.mean(target_points, axis=0)

    src_centered = source_points - src_mean
    tgt_centered = target_points - tgt_mean

    cov_matrix = np.dot(tgt_centered.T, src_centered) / source_points.shape[0]
    U, _, Vt = np.linalg.svd(cov_matrix)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    scale = np.trace(np.dot(np.diag(_), U.T @ Vt)) / np.sum(src_centered ** 2)
    t = tgt_mean.T - scale * R @ src_mean.T

    T = np.identity(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T

# 설정
input_dir = "output"
voxel_size = 0.01

# 파일 불러오기
ply_files = sorted(glob(os.path.join(input_dir, "*.ply")))
print("사용 가능한 PLY 파일:")
for i, path in enumerate(ply_files):
    print(f"[{i}] {os.path.basename(path)}")

# 기준 인덱스 선택
while True:
    try:
        ref_idx = int(input("기준이 될 포인트 클라우드 파일 번호를 입력하세요: "))
        if 0 <= ref_idx < len(ply_files):
            break
        else:
            print("⚠ 범위 오류: 유효한 인덱스를 입력하세요.")
    except ValueError:
        print("⚠ 숫자를 입력하세요.")

# 기준 파일을 맨 앞으로 정렬
ref_file = ply_files.pop(ref_idx)
ply_files.insert(0, ref_file)

# 포인트 클라우드 로딩 및 정규화
pointclouds = []
for path in ply_files:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        continue

    points = np.asarray(pcd.points)
    center = pcd.get_center()
    distances = np.linalg.norm(points - center, axis=1)
    scale = 1.0 / np.mean(distances)
    pcd.scale(scale, center=center)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pointclouds.append(pcd)

# 첫 번째 파일을 기준으로 설정
base_pcd = pointclouds[0]

for i in range(1, len(pointclouds)):
    source = pointclouds[i]
    target = base_pcd

    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)

    if len(src_pts) == 0 or len(tgt_pts) == 0:
        print(f"⚠ {i}번째 포인트 클라우드: 포인트 없음, 스킵")
        continue

    # Umeyama 정합
    try:
        sim3_transform = umeyama_alignment(src_pts, tgt_pts)
        source.transform(sim3_transform)
    except Exception as e:
        print(f"⚠ Umeyama 정합 실패: {e}")
        continue

    # ICP 정제
    try:
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size * 0.4, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        source.transform(icp_result.transformation)
    except Exception as e:
        print(f"⚠ 정합 실패: {e}")
        continue

    base_pcd += source
    base_pcd = base_pcd.voxel_down_sample(voxel_size)

# outlier 제거
base_pcd, ind = base_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# 저장
output_path = "fused_with_outlier_removal.ply"
o3d.io.write_point_cloud(output_path, base_pcd)
print(f"✔ 정합된 포인트 클라우드 저장됨: {output_path}")
