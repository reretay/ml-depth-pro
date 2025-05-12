import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import depth_pro
import open3d as o3d
import matplotlib.pyplot as plt

# Step 1: Load Depth Pro model
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Step 2: Load and preprocess image (resize to model's expected input size)
image_path = "subway.jpg"
image = Image.open(image_path).convert("RGB")

# Resize image to 1536x1536 (Depth Pro requirement)
input_size = 1536
image_resized = image.resize((input_size, input_size))
input_tensor = transform(image_resized).unsqueeze(0)  # [1, 3, H, W]

# Step 3: Inference
with torch.no_grad():
    depth = model(input_tensor)
    if isinstance(depth, tuple):
        depth = depth[0]

# Step 4: Convert to NumPy
depth_np = depth.squeeze().cpu().numpy()
rgb_np = np.asarray(image_resized) / 255.0  # RGB image resized to depth map size

# Step 5: Create Open3D RGBD image
depth_o3d = o3d.geometry.Image((depth_np * 1000).astype(np.uint16))  # in millimeters
color_o3d = o3d.geometry.Image((rgb_np * 255).astype(np.uint8))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color=color_o3d,
    depth=depth_o3d,
    convert_rgb_to_intensity=False,
    depth_scale=1000.0,  # back to meters
    depth_trunc=10.0
)

# Step 6: Camera intrinsics (you can adjust fx/fy based on actual camera)
width, height = depth_np.shape[1], depth_np.shape[0]
fx = fy = 575.0
cx, cy = width / 2.0, height / 2.0
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Step 7: Generate point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

# Flip for Open3D view (optional)
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Step 8: Visualize
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud from Depth Pro")
