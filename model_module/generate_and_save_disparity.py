import torch
import numpy as np
import matplotlib.pyplot as plt

from img_preprocessing.load_and_preprocess_image import load_and_preprocess_image
from model_module.CNN_model import EnhancedCustomCNN
from model_module.read_pfm import readPFM


def analyze_disparity(disparity_map):
    """Analyzes and prints statistics for a disparity map."""
    print(f"Min: {np.min(disparity_map)}")
    print(f"Max: {np.max(disparity_map)}")
    print(f"Mean: {np.mean(disparity_map)}")
    print(f"Std Dev: {np.std(disparity_map)}")


def write_pfm(file, image, scale=1):
    """Writes an image array to a PFM file, flipping it vertically first."""
    print("Before saving to PFM:")
    print(f"Min: {np.min(image)}")
    print(f"Max: {np.max(image)}")
    print(f"Mean: {np.mean(image)}")
    print(f"Std Dev: {np.std(image)}")

    with open(file, "wb") as f:
        image = np.flipud(image)
        color = image.shape[-1] == 3 if len(image.shape) == 3 else False

        header = "PF\n" if color else "Pf\n"
        f.write(header.encode("latin-1"))
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("latin-1"))

        endian = "<" if image.dtype.byteorder == "<" else ">"
        scale = scale if endian == "<" else -scale
        f.write(f"{scale}\n".encode("latin-1"))

        image.tofile(f)


def predict_disparity_from_model(
    model_path, left_img_path, right_img_path, output_path
):
    model = EnhancedCustomCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    """Predicts and saves the disparity map from two images using a trained model."""
    left_img = load_and_preprocess_image(left_img_path)
    right_img = load_and_preprocess_image(right_img_path)

    # Add batch dimension
    left_img = left_img.unsqueeze(0)
    right_img = right_img.unsqueeze(0)

    device = next(model.parameters()).device
    left_img = left_img.to(device)
    right_img = right_img.to(device)

    model.eval()
    with torch.no_grad():
        output = model(left_img, right_img).squeeze(0).squeeze(0)

    output_np = output.cpu().numpy()

    # Visualize disparity map
    plt.figure()
    plt.imshow(output_np, cmap="hot")
    plt.colorbar()
    plt.title("Disparity Map")
    plt.show()

    # Check statistics before saving
    print("Before saving to PFM:")
    analyze_disparity(output_np)

    write_pfm(output_path, output_np)

    loaded_map = readPFM(output_path)

    print("After reading PFM:")
    analyze_disparity(loaded_map)


# Assuming you have defined a way to load the model elsewhere in your script, like:
# model = EnhancedCustomCNN().to(device) and have loaded its weights with model.load_state_dict()
# Call predict_disparity for specific left/right images and save path:
