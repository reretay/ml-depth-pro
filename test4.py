import time
from PIL import Image
import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO

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
image, _, f_px = depth_pro.load_rgb(image_path)
depth_input = transform(image)
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

# 총 실행 시간 계산
total_time = time.time() - total_start_time
print("\n──────────────────────────────")
print(f"✅ 전체 실행 시간: {total_time:.2f}초")
print("──────────────────────────────")
print(f"- 모델 로드: {model_load_time:.2f}초 ({model_load_time/total_time*100:.1f}%)")
print(f"- YOLO 탐지: {yolo_time:.2f}초 ({yolo_time/total_time*100:.1f}%)")
print(f"- 깊이 추정: {depth_time:.2f}초 ({depth_time/total_time*100:.1f}%)")
print(f"- 시각화/저장: {vis_time:.2f}초 ({vis_time/total_time*100:.1f}%)")