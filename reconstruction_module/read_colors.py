import cv2
import numpy as np


def load_and_preprocess_image(img_path, target_size=None):
    """Load an image and resize it to the target size."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return img


def read_image_colors(
    img_path,
    X_valid,
    Y_valid,
    Z_valid,
    focal_length,
    cx,
    cy,
    original_size,
    resized_size=(960, 540),
):
    """
    Extract color values from a specific image based on 3D coordinates and camera parameters,
    optionally considering image resizing.

    Args:
        img_path (str): Path to the color image.
        X_valid, Y_valid, Z_valid (numpy.ndarray): Arrays containing the 3D coordinates.
        focal_length (float): Original focal length of the camera.
        cx, cy (float): Original principal points (center) of the camera.
        original_size (tuple): Original dimensions (width, height) of the image.
        resized_size (tuple, optional): New dimensions (width, height) if the image was resized.

    Returns:
        numpy.ndarray: Array of normalized color values.
    """
    if resized_size:
        # Scale the camera parameters based on the resizing
        scale_x = resized_size[0] / original_size[0]
        scale_y = resized_size[1] / original_size[1]
        focal_length *= scale_x  # Assuming similar scaling for x and y
        cx *= scale_x
        cy *= scale_y

    # Load and preprocess image
    img = load_and_preprocess_image(img_path, resized_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate pixel coordinates in the resized image
    u_valid = np.clip(
        np.round(X_valid * focal_length / Z_valid + cx).astype(int), 0, img.shape[1] - 1
    )
    v_valid = np.clip(
        np.round(Y_valid * focal_length / Z_valid + cy).astype(int), 0, img.shape[0] - 1
    )

    # Extract colors from the image at the calculated coordinates
    colors = img[v_valid, u_valid]

    # Normalize colors
    return colors / 255.0
