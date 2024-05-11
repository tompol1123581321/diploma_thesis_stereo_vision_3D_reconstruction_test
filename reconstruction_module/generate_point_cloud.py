import numpy as np


def calculate_point_cloud(
    depth,
    disparity_normalized,
    focal_length,
    vmin,
    vmax,
    original_cx,
    original_cy,
    original_width,
    original_height,
    resized_width=960,
    resized_height=540,
):
    # Scaling factors for focal length and center coordinates based on resizing
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    # Adjusted focal length and center coordinates
    focal_length_adjusted = (
        focal_length * scale_x
    )  # Assuming similar scaling for x and y
    cx_adjusted = original_cx * scale_x
    cy_adjusted = original_cy * scale_y

    # Rescale the normalized disparity to actual values
    disparity_actual = vmin + disparity_normalized * (vmax - vmin)

    # Calculate the depth if not already calculated
    # depth = focal_length_adjusted * baseline / np.maximum(disparity_actual, 1e-6)

    # Validate dimensions
    height, width = disparity_normalized.shape
    assert depth.shape == (
        height,
        width,
    ), "Depth and disparity maps must have the same dimensions."

    # Create grid of pixel coordinates
    x = np.tile(np.arange(width), (height, 1))
    y = np.repeat(np.arange(height), width).reshape(height, width)

    # Compute 3D coordinates
    X = (x - cx_adjusted) * depth / focal_length_adjusted
    Y = (y - cy_adjusted) * depth / focal_length_adjusted
    Z = depth

    # Validity mask based on Z values and disparity range
    valid_mask = (disparity_actual > vmin) & (disparity_actual < vmax)

    # Extract valid coordinates
    X_valid = X[valid_mask]
    Y_valid = Y[valid_mask]
    Z_valid = Z[valid_mask]

    return X_valid, Y_valid, Z_valid


# Example usage might be commented out in the actual function file
# depth, disparity_normalized, focal_length, baseline, vmin, vmax = obtain_your_parameters_somehow()
# original_width, original_height, resized_width, resized_height = get_your_image_dimensions()
# X_valid, Y_valid, Z_valid = calculate_point_cloud(depth, disparity_normalized, focal_length, baseline, vmin, vmax, original_width, original_height, resized_width, resized_height)
