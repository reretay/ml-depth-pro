import torch
import matplotlib.pyplot as plt
import depth_pro
import time

# 이미지 경로
image_path = "subway.jpg"

# [1] 이미지 로드 (focal length 포함)
image, _, f_px = depth_pro.load_rgb(image_path)

# [2] 모델 로드 및 타임로그
start_model_load = time.time()
model, transform = depth_pro.create_model_and_transforms()
model.eval()
end_model_load = time.time()
print(f"[타임로그] 모델 로드 완료: {end_model_load - start_model_load:.3f}초")

# [3] 이미지 전처리 및 타임로그
start_preprocess = time.time()
input_tensor = transform(image)
end_preprocess = time.time()
print(f"[타임로그] 이미지 전처리 완료: {end_preprocess - start_preprocess:.3f}초")

# [4] 추론 및 타임로그 (infer 사용)
start_inference = time.time()
with torch.no_grad():
    prediction = model.infer(input_tensor, f_px=f_px)
end_inference = time.time()
print(f"[타임로그] 추론(깊이 예측) 완료: {end_inference - start_inference:.3f}초")

# [5] depth 맵 추출 (미터값)
if isinstance(prediction, dict) and "depth" in prediction:
    depth_map_tensor = prediction["depth"]
else:
    depth_map_tensor = prediction

depth_map_np = depth_map_tensor.squeeze().cpu().numpy()

# [6] 인터랙티브 시각화 (클릭 시 미터값 표시)
fig = plt.figure(figsize=(10, 8))
plt.imshow(depth_map_np, cmap='magma', aspect='equal')
plt.colorbar(label='Depth (meters)')

def onclick(event):
    if event.inaxes == plt.gca():
        x = int(event.xdata)
        y = int(event.ydata)
        if 0 <= x < depth_map_np.shape[1] and 0 <= y < depth_map_np.shape[0]:
            depth_value = depth_map_np[y, x]  # 해당 위치의 미터값을 그대로 사용
            if hasattr(onclick, 'text'):
                onclick.text.remove()
            onclick.text = plt.gca().text(
                x, y, f"{depth_value:.2f}m",
                color='white', fontsize=10,
                verticalalignment='bottom',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
            )
            plt.title(f"Clicked at ({x}, {y})")
            fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Click anywhere to measure depth")
plt.show()
