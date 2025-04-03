from PIL import Image
import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize models
yolo_model = YOLO('yolo11s.pt')
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

image_path = "subway.jpg"

# Load and process image for YOLO
yolo_input = cv2.imread(image_path)
if yolo_input is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# YOLO detection
results = yolo_model(yolo_input)
person_boxes = []

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if result.names[int(cls)] == 'person':
            x1, y1, x2, y2 = map(int, box[:4])
            person_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(yolo_input, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Depth processing
image, _, f_px = depth_pro.load_rgb(image_path)
depth_input = transform(image)
prediction = depth_model.infer(depth_input, f_px=f_px)
depth = prediction["depth"]  # Depth in meter
depth_np = depth.squeeze().cpu().numpy()

# Add depth information to boxes
for x1, y1, x2, y2 in person_boxes:
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2  # Fixed: y1 + y2 instead of x1 + y2

    depth_value = depth_np[center_y, center_x]

    text = f'Depth: {depth_value:.2f}m'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # Reduced font size
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    text_x = x1
    text_y = y1 - 10
    rect_x1 = text_x - 5
    rect_y1 = text_y - text_size[1] - 10
    rect_x2 = text_x + text_size[0] + 5
    rect_y2 = text_y + 5

    cv2.rectangle(yolo_input, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.putText(yolo_input, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

# Display final result
cv2.imshow('Person Detection with Depth', yolo_input)
cv2.imwrite('person_detection_with_depth.jpg', yolo_input)

# Create and display depth map
depth_np_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
inv_depth_np_normalized = 1.0 - depth_np_normalized
depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
cv2.imshow('Inverted Depth Map', depth_colormap)
cv2.imwrite('inverted_depth_map.jpg', depth_colormap)

# Wait for key press only once at the end
print("Press any key to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()