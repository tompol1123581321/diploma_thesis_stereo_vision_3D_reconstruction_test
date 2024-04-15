
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV library
import open3d as o3d  # Open3D library

# Load calibration parameters
focal_length = 1733.74
baseline = 536.62
ndisp = 170
vmin = 55
vmax = 142

def readPFM(file):
    with open(file, "rb") as f:
        # Read header
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        # Read dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('latin-1'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # Read scale
        scale = float(f.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'  # little endian
            scale = -scale
        else:
            endian = '>'  # big endian

        # Read data
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM files are stored in top-to-bottom order

    return data, scale


# Read the disparity map
disparity, scale = readPFM('data/artroom1/disp0.pfm')

# Calculate the depth map
depth = focal_length * baseline / np.maximum(disparity, 1e-6)  # Avoid division by zero

# Get the width and height from the disparity map
height, width = disparity.shape

# Create meshgrid of pixel coordinates
x = np.tile(np.arange(width), (height, 1))
y = np.repeat(np.arange(height), width).reshape(height, width)

# Re-project points into 3D space
X = (x - 792.27) * depth / focal_length
Y = (y - 541.89) * depth / focal_length
Z = depth

# Mask for valid depth values
valid_mask = (Z > 0) & (disparity > vmin) & (disparity < vmax)

# Apply the valid mask
X_valid = X[valid_mask]
Y_valid = Y[valid_mask]
Z_valid = Z[valid_mask]

# Load the color image
color_image_path = 'data/artroom1/im0.png'  # Replace with your actual image path
img = cv2.imread(color_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack((X_valid, Y_valid, Z_valid)).T)

# Assign colors to valid points
u_valid = np.clip(np.round(X_valid * focal_length / Z_valid + 792.27).astype(int), 0, img.shape[1] - 1)
v_valid = np.clip(np.round(Y_valid * focal_length / Z_valid + 541.89).astype(int), 0, img.shape[0] - 1)
colors = img[v_valid, u_valid]
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

# Visualization using Open3D
o3d.visualization.draw_geometries([pcd], point_show_normal=False)

# Visualization using Matplotlib (Optional, as it might be less informative for dense point clouds)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_valid, Y_valid, Z_valid, c=colors/255.0, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

