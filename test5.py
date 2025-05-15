import time
from PIL import Image
import depth_pro
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import io

# 전체 시작 시간 기록
total_start_time = time.time()

# 1. 모델 초기화 시간 측정
model_load_start = time.time()
yolo_model = YOLO('yolo11s.pt')
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()
model_load_time = time.time() - model_load_start
print(f"✔ 모델 로드 시간: {model_load_time:.2f}초")

# 2. 이미지 처리 및 YOLO 탐지 시간 측정
yolo_start = time.time()
image_path = "data/example.jpg"
yolo_input = cv2.imread(image_path)
results = yolo_model(yolo_input)
person_boxes = []

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    for box, cls in zip(boxes, classes):
        if result.names[int(cls)] == 'person':
            x1, y1, x2, y2 = map(int, box[:4])
            person_boxes.append((x1, y1, x2, y2))
yolo_time = time.time() - yolo_start
print(f"✔ YOLO 탐지 시간: {yolo_time:.2f}초")

# 3. 깊이 추정 시간 측정
depth_start = time.time()
image_data, image_pil, f_px = depth_pro.load_rgb(image_path)

# 만약 image_pil이 bytes 형식이라면 PIL 이미지로 변환
if isinstance(image_pil, bytes):
    image_pil = Image.open(io.BytesIO(image_pil)).convert("RGB")

# 이미 image_pil이 PIL Image 객체라면 아래 코드를 사용
depth_input = transform(image_pil)
prediction = depth_model.infer(depth_input, f_px=f_px)
depth_np = prediction["depth"].squeeze().cpu().numpy()
depth_time = time.time() - depth_start
print(f"✔ 깊이 추정 시간: {depth_time:.2f}초")

# 4. 시각화 및 저장 시간 측정
vis_start = time.time()
# 인물 박스에 깊이 정보 표시 (생략)
depth_np_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
inv_depth_np_normalized = 1.0 - depth_np_normalized
depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

# 최종 저장
cv2.imwrite('person_detection_with_depth.jpg', yolo_input)
cv2.imwrite('inverted_depth_map.jpg', depth_colormap)
vis_time = time.time() - vis_start
print(f"✔ 시각화 및 저장 시간: {vis_time:.2f}초")

# 5. 3D 점 구름 시각화
print("🟢 Visualizing point cloud with Open3D...")
height, width = depth_np.shape
depth_map = np.asarray(depth_np)

# 이미지 크기를 맞추기 위해 depth map을 rgb 이미지로 변환
rgb_img = np.asarray(image_pil.resize(depth_map.shape[::-1], Image.BILINEAR))

# x, y, z 좌표 계산
x, y = np.meshgrid(np.arange(width), np.arange(height))
x = x.flatten()
y = y.flatten()
z = depth_map.flatten()

# RGB 이미지에서 색상 값 가져오기
r = rgb_img[:, :, 0].flatten()
g = rgb_img[:, :, 1].flatten()
b = rgb_img[:, :, 2].flatten()

# 3D 점 구름 생성
points = np.vstack((x, y, z)).T
colors = np.vstack((r, g, b)).T / 255.0  # RGB 색상 범위를 [0, 1]로 정규화

# Open3D를 이용한 시각화
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Open3D를 이용해 3D 시각화
o3d.visualization.draw_geometries([point_cloud])

# 총 실행 시간 계산
total_time = time.time() - total_start_time
print("\n──────────────────────────────")
print(f"✅ 전체 실행 시간: {total_time:.2f}초")
print("──────────────────────────────")
print(f"- 모델 로드: {model_load_time:.2f}초 ({model_load_time/total_time*100:.1f}%)")
print(f"- YOLO 탐지: {yolo_time:.2f}초 ({yolo_time/total_time*100:.1f}%)")
print(f"- 깊이 추정: {depth_time:.2f}초 ({depth_time/total_time*100:.1f}%)")
print(f"- 시각화/저장: {vis_time:.2f}초 ({vis_time/total_time*100:.1f}%)")
