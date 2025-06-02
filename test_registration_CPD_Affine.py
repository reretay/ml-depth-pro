import open3d as o3d
import numpy as np
from pycpd import AffineRegistration

def preprocess_point_cloud(pcd, voxel_size=0.01):
    # 1. 다운샘플링
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # 2. 중심 정렬
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points -= centroid
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 3. 정규화 (선택)
    scale = np.linalg.norm(points, axis=1).max()
    points /= scale
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

# 1. PLY 파일 로드
source_pcd = o3d.io.read_point_cloud("output/KakaoTalk_20250517_153332333_01.ply")
target_pcd = o3d.io.read_point_cloud("output/KakaoTalk_20250517_153332333_02.ply")

# 2. 전처리
source_pcd = preprocess_point_cloud(source_pcd, voxel_size=0.04)
target_pcd = preprocess_point_cloud(target_pcd, voxel_size=0.04)

# 3. NumPy 배열로 변환
source_points = np.asarray(source_pcd.points)
target_points = np.asarray(target_pcd.points)

# 4. CPD Affine 정합 수행
reg = AffineRegistration(X=source_points, Y=target_points)
transformed_source_points, _ = reg.register()

# 5. 정합된 포인트 클라우드를 Open3D 객체로 변환
aligned_pcd = o3d.geometry.PointCloud()
aligned_pcd.points = o3d.utility.Vector3dVector(transformed_source_points)

# 6. 시각화
source_pcd.paint_uniform_color([1, 0, 0])     # 빨강
target_pcd.paint_uniform_color([0, 1, 0])     # 초록
aligned_pcd.paint_uniform_color([0, 0, 1])    # 파랑

o3d.visualization.draw_geometries(
    [source_pcd, target_pcd, aligned_pcd],
    window_name="CPD Affine Registration (with Preprocessing)",
    width=800, height=600
)
