import numpy as np
import open3d as o3d


def generate_3D_mesh(X_valid, Y_valid, Z_valid, colors):
    """
    Generate a 3D mesh from valid X, Y, Z coordinates and corresponding color data.

    Args:
        X_valid (numpy.ndarray): Array of X coordinates.
        Y_valid (numpy.ndarray): Array of Y coordinates.
        Z_valid (numpy.ndarray): Array of Z coordinates.
        colors (numpy.ndarray): Array of RGB color data with values between 0 and 1.

    Returns:
        open3d.geometry.PointCloud: Point cloud object with points and colors.
    """
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    points = np.vstack((X_valid, Y_valid, Z_valid)).T
    pcd.points = o3d.utility.Vector3dVector(points)

    # Ensure colors are within the correct range (0 to 1) and shape matches the points
    if colors.max() > 1:
        colors = colors / 255.0  # Assuming color values range from 0 to 255
    assert (
        colors.shape[0] == len(Z_valid) and colors.shape[1] == 3
    ), "Color array shape mismatch or invalid range."

    # Set colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# Example usage:
# Assuming X_valid, Y_valid, Z_valid are numpy arrays containing your 3D point data
# colors = np.random.rand(len(X_valid), 3)  # Example color data, replace with actual color data
# pcd = generate_3D_mesh(X_valid, Y_valid, Z_valid, colors)
# o3d.visualization.draw_geometries([pcd])  # Visualize the point cloud with color
