import torch
from PIL import Image
import numpy as np
import depth_pro
import open3d as o3d
import os
from glob import glob

# Load model and transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Set paths
input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Collect all image files
image_paths = glob(os.path.join(input_dir, "*.*"))
image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Camera intrinsics (can be adjusted per image if needed)
fx = fy = 0.9 * 4000  # Focal length
depth_scale = 1000.0  # mm

for image_path in image_paths:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}.ply")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    model_device = next(model.parameters()).device
    input_tensor = input_tensor.to(model_device)  # GPU로 이동

    # Inference
    with torch.no_grad():
        prediction = model.infer(input_tensor)
    depth_np = prediction["depth"].squeeze().cpu().numpy()
    rgb_np = np.asarray(image) / 255.0

    # Create Open3D RGBD image
    depth_o3d = o3d.geometry.Image((depth_np * depth_scale).astype(np.uint16))
    color_o3d = o3d.geometry.Image((rgb_np * 255).astype(np.uint8))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=depth_scale,
        depth_trunc=1000.0
    )

    # Camera intrinsics for current image size
    height, width = depth_np.shape
    cx, cy = width / 2.0, height / 2.0
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # Flip for Open3D coordinate alignment
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # Save point cloud
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"✔ Saved: {output_path}")
