import torch
from PIL import Image
from torchvision import transforms
# from depth_pro.models.depth_pro import DepthProModel  # Import the Depth Pro model
import depth_pro

# Step 1: Load the Depth Pro model
# model = DepthProModel(pretrained=True)
# model.eval()
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Step 2: Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((1536, 1536)),  # Resize to model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

image_path = "data/example.jpg"  # Replace with your image path
input_image = preprocess_image(image_path)

# Step 3: Predict depth map using Depth Pro
with torch.no_grad():
    depth_map = model(input_image)

# Step 4: Visualize or process the depth map
depth_map_np = depth_map.squeeze().cpu().numpy()  # Convert to numpy array for visualization or further processing

# Save or display the depth map (optional)
import matplotlib.pyplot as plt
plt.imshow(depth_map_np, cmap='magma')
plt.colorbar()
plt.title("Predicted Depth Map")
plt.show()

# Step 5: Calculate actual distance (if required)
# Depth Pro provides metric depth directly in meters.
# To measure the distance of a specific object:
def measure_distance(depth_map_np, bbox):
    x_min, y_min, x_max, y_max = bbox
    object_depth = depth_map_np[y_min:y_max, x_min:x_max].mean()  # Average depth within bounding box
    return object_depth

# Example bounding box coordinates (x_min, y_min, x_max, y_max)
bbox = (100, 100, 200, 200)  
distance_in_meters = measure_distance(depth_map_np, bbox)
print(f"Measured Distance: {distance_in_meters} meters")
