import numpy as np
import cv2
import torch
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt

# Existing code to get the depth map
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Invert the depth map
output = 1 / prediction.cpu().numpy()

# Normalize the depth map
min_val = output.min()
max_val = output.max()
output = (output - min_val) / (max_val - min_val)


# Make depth plausible by multiplying and adding some values
output = output * 100 + 50

# Create and transform point cloud
height, width = output.shape
x, y = np.meshgrid(np.arange(width), np.arange(height))
z = output

# Flatten the arrays for easy transformation
points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

# Rotation matrix around the Y-axis by 45 degrees
theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
rotation_matrix = np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]])

# Rotate the points
rotated_points = points @ rotation_matrix.T

# Reproject the points back to the 2D image plane
x_rotated = rotated_points[:, 0]
y_rotated = rotated_points[:, 1]
z_rotated = rotated_points[:, 2]

# Normalize the z values to maintain the original depth range
z_rotated_normalized = (z_rotated - z_rotated.min()) / (z_rotated.max() - z_rotated.min())

# Create a new image from the reprojected points
reprojected_img = np.full((height, width, 3), 255, dtype=np.uint8)  # White background for occlusion

# Reproject points back into the image
for i in range(len(x_rotated)):
    x_img = int(x_rotated[i])
    y_img = int(y_rotated[i])
    if 0 <= x_img < width and 0 <= y_img < height:
        reprojected_img[y_img, x_img] = img[int(y[i]), int(x[i])]

# Save the new image back to disk
reprojected_img_pil = Image.fromarray(reprojected_img)
reprojected_img_pil.save("reprojected_image.png")

plt.imshow(reprojected_img)
plt.savefig('rep_image.png')