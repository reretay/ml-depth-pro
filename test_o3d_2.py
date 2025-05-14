import torch
from PIL import Image
import numpy as np
import depth_pro
import open3d as o3d

# Step 1: Load Depth Pro model and transforms
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Step 2: Load original image without resizing
image_path = "subway.jpg"
image = Image.open(image_path).convert("RGB")

# Step 3: Apply transform directly to original image (no resize)
input_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]

# Step 4: Inference
with torch.no_grad():
    prediction = model.infer(input_tensor)
depth_np = prediction["depth"].squeeze().cpu().numpy()

# Step 5: Convert RGB image to NumPy array (original size)
rgb_np = np.asarray(image) / 255.0

# Step 6: Create Open3D RGBD image
depth_o3d = o3d.geometry.Image((depth_np * 1000).astype(np.uint16))  # millimeters
color_o3d = o3d.geometry.Image((rgb_np * 255).astype(np.uint8))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color=color_o3d,
    depth=depth_o3d,
    convert_rgb_to_intensity=False,
    depth_scale=1000.0,
    depth_trunc=1000.0
)

# Step 7: Camera intrinsics (adjust according to original image size)
height, width = depth_np.shape
fx = fy = 575.0
cx, cy = width / 2.0, height / 2.0
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Step 8: Generate point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

# Flip for Open3D visualization
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Step 9: Visualize
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud from Depth Pro")
