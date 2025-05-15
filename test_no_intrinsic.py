import numpy as np
import torch
from PIL import Image
import depth_pro
import open3d as o3d

# 모델 및 transform 불러오기
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# 이미지 불러오기
image_path = "subway.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# depth 예측
with torch.no_grad():
    prediction = model.infer(input_tensor)
depth_np = prediction["depth"].squeeze().cpu().numpy()  # (H, W), metric 단위

# 컬러 정보 가져오기
rgb_np = np.asarray(image) / 255.0  # (H, W, 3), float32

# 3D 포인트 생성 (u, v, depth)
H, W = depth_np.shape
u, v = np.meshgrid(np.arange(W), np.arange(H))
points = np.stack((u.flatten(), v.flatten(), depth_np.flatten()), axis=1)

# 컬러 정보 포인트별로 정렬
colors = rgb_np.reshape(-1, 3)

# Open3D 포인트 클라우드 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# ply로 저장
o3d.io.write_point_cloud("output.ply", pcd)
print("✔ 3D 포인트 클라우드를 output.ply로 저장했습니다.")
