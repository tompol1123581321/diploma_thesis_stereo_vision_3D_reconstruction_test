import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch

import open3d as o3d

from model_module.generate_and_save_disparity import predict_disparity_from_model
from model_module.read_pfm import readPFM
from reconstruction_module.calculate_depth import calculate_depth
from reconstruction_module.generate_3D_mesh import generate_3D_mesh
from reconstruction_module.generate_point_cloud import calculate_point_cloud
from reconstruction_module.read_colors import read_image_colors
from reconstruction_module.read_parameters import read_parameters


def test_visualize_disparity_map(disparity, kernel_size=5):
    """
    Visualizes a disparity map using a heatmap after applying a Gaussian blur.

    Args:
    disparity (numpy.array or torch.Tensor): The disparity map to visualize.
    kernel_size (int): Size of the Gaussian kernel.
    """
    if isinstance(disparity, torch.Tensor):
        disparity = disparity.cpu().numpy()

    # Apply Gaussian Blur
    disparity_smoothed = cv2.GaussianBlur(disparity, (kernel_size, kernel_size), 0)

    plt.imshow(disparity_smoothed, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Smoothed Disparity Map")
    plt.show()


def visualize_point_cloud(X_valid, Y_valid, Z_valid):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize the depth data to use with the colormap
    norm = plt.Normalize(Z_valid.min(), Z_valid.max())

    # Create a colormap
    cmap = plt.get_cmap("hot")

    # Map the normalized Z-values to colors
    colors = cmap(norm(Z_valid))

    scatter = ax.scatter(X_valid, Y_valid, Z_valid, c=colors, s=1)

    # Adding a color bar to show the depth scale
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Depth")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def vizualize_3D_mesh(pcd):
    # Visualization using Open3D
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)


def reconstruct_and_vizualize_mesh(
    left_image_path,
    right_image_path,
    params_file_path,
    model_path,
    disparity_output_path,
    disparity_path=None,
):

    focal_length, cx, cy, baseline, vmin, vmax, width, height, _ = read_parameters(
        params_file_path
    )
    if not disparity_path:
        predict_disparity_from_model(
            model_path, left_image_path, right_image_path, disparity_output_path
        )
    disparity = (
        readPFM(disparity_path, True, vmax, vmin)
        if disparity_path
        else (
            readPFM(disparity_output_path, False, vmax, vmin)
            if disparity_output_path
            else None
        )
    )
    test_visualize_disparity_map(disparity)
    depth = calculate_depth(disparity, focal_length, baseline, vmin, vmax)
    X_valid, Y_valid, Z_valid = calculate_point_cloud(
        depth, disparity, focal_length, vmin, vmax, cx, cy, width, height
    )
    visualize_point_cloud(X_valid, Y_valid, Z_valid)

    colors = read_image_colors(
        left_image_path,
        X_valid,
        Y_valid,
        Z_valid,
        focal_length,
        cx,
        cy,
        (width, height),
    )
    pcd = generate_3D_mesh(X_valid, Y_valid, Z_valid, colors)
    vizualize_3D_mesh(pcd)
