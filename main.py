
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV library
import open3d as o3d
import torch

from model.model_training import start_training
from model.read_pfm import readPFM 



def extract_parameters(filepath):
    params = {}
    with open(filepath, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=')
                params[key.strip()] = value.strip()
    
    # Extract specific parameters
    focal_length = float(params.get('cam0').split()[0][1:])
    baseline = float(params.get('baseline'))
    ndisp = int(params.get('ndisp'))
    vmin = int(params.get('vmin'))
    vmax = int(params.get('vmax'))

    return focal_length, baseline, ndisp, vmin, vmax

focal_length, baseline, ndisp, vmin, vmax = extract_parameters("data/bandsaw1/calib.txt")
print(focal_length, baseline, ndisp, vmin, vmax)

def test_visualize_disparity_map(disparity):
    """
    Visualizes a disparity map using a heatmap.
    
    Args:
    disparity (numpy.array or torch.Tensor): The disparity map to visualize.
    """
    if isinstance(disparity, torch.Tensor):
        # Check if the tensor is on GPU, and move it to CPU
        disparity = disparity.cpu()
        
        # Convert torch.Tensor to numpy array
        disparity = disparity.numpy()
    
    plt.imshow(disparity, cmap='hot')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()



def calculate_depth_map(disparity,focal_length,baseline):
    return focal_length * baseline / np.maximum(disparity, 1e-6)  # Avoid division by zero

def create_point_cloud(depth, disparity, focal_length):
    height, width = disparity.shape

    x = np.tile(np.arange(width), (height, 1))
    y = np.repeat(np.arange(height), width).reshape(height, width)

    X = (x - 792.27) * depth / focal_length
    Y = (y - 541.89) * depth / focal_length
    Z = depth

    valid_mask = (Z > 0) & (disparity > vmin) & (disparity < vmax)

    X_valid = X[valid_mask]
    Y_valid = Y[valid_mask]
    Z_valid = Z[valid_mask]
    return X_valid, Y_valid, Z_valid

def read_image_colors(img_path,X_valid,Y_valid,Z_valid):
    color_image_path = img_path  # Replace with your actual image path
    img = cv2.imread(color_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    u_valid = np.clip(np.round(X_valid * focal_length / Z_valid + 792.27).astype(int), 0, img.shape[1] - 1)
    v_valid = np.clip(np.round(Y_valid * focal_length / Z_valid + 541.89).astype(int), 0, img.shape[0] - 1)
    colors = img[v_valid, u_valid]
    return np.asarray(colors)/255
    
def generate_3D_mesh(X_valid,Y_valid,Z_valid,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((X_valid, Y_valid, Z_valid)).T)
    pcd.colors = pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def vizualize_point_cloud(X_valid, Y_valid, Z_valid, colors):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_valid, Y_valid, Z_valid, c=colors, s=1)  # use normalized colors directly
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def vizualize_3D_mesh(pcd):
# Visualization using Open3D
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)

def main():
    disparity = readPFM("data/bandsaw1/disp0.pfm")
    depth = calculate_depth_map(disparity,focal_length,baseline)
    X_valid, Y_valid, Z_valid = create_point_cloud(depth,disparity,focal_length)
    colors = read_image_colors("data/bandsaw1/im0.png", X_valid, Y_valid, Z_valid)
    vizualize_point_cloud(X_valid, Y_valid, Z_valid,colors)

    pcd = generate_3D_mesh(X_valid, Y_valid, Z_valid,colors)
    vizualize_3D_mesh(pcd)


if __name__=='__main__':
    # disparity = readPFM("data/Backpack-imperfect/disp0.pfm")
    # test_vizualize_disperity_map(disparity)
    # load_data("data")
    start_training()
    # main()
