import os
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import cv2
import torch
from sklearn.metrics import mean_squared_error
from math import sqrt


import cv2

from img_preprocessing.load_and_preprocess_image import load_and_preprocess_image
from model_module.CNN_model import EnhancedCustomCNN
from model_module.read_pfm import readPFM
from reconstruction_module.read_parameters import read_parameters


def save_heatmap(data, filename):
    """Save heatmap of disparity data, normalized to range [0, 1] based on percentiles."""
    p2, p98 = np.percentile(data, [2, 98])
    data_clipped = np.clip(data, p2, p98)
    if p98 > p2:
        data_normalized = (data_clipped - p2) / (p98 - p2)
    else:
        data_normalized = data_clipped

    plt.imshow(data_normalized, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Disparity Map")
    plt.savefig(filename)
    plt.close()


def generate_cnn_disparity(model, left_img_path, right_img_path):
    """Generates a disparity map using a CNN model and saves it."""
    left_img = load_and_preprocess_image(left_img_path)
    right_img = load_and_preprocess_image(right_img_path)

    left_img = left_img.unsqueeze(0)
    right_img = right_img.unsqueeze(0)

    device = next(model.parameters()).device
    left_img = left_img.to(device)
    right_img = right_img.to(device)

    model.eval()
    with torch.no_grad():
        output = model(left_img, right_img).squeeze(0).squeeze(0)

    output_np = output.cpu().numpy()

    return output_np


def generate_bm_disparity(
    left_img_path,
    right_img_path,
    target_size=(960, 540),
    numDisparities=64,
    blockSize=15,
    minDisparity=0,
    maxDisparity=1000,
):
    """Generates a disparity map using Stereo Block Matching (StereoBM) with resized images and scaled output."""
    imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        raise FileNotFoundError("One or both image paths are incorrect.")

    # Resize images to the target size
    imgL = cv2.resize(imgL, target_size)
    imgR = cv2.resize(imgR, target_size)

    # Initialize the StereoBM object
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    stereo.setMinDisparity(minDisparity)

    # Compute the disparity map
    disparity = stereo.compute(imgL, imgR)

    # Normalize the disparity map to be between 0 and 1
    min_disp = np.min(disparity)
    max_disp = np.max(disparity)
    if max_disp - min_disp != 0:
        disparity = (disparity - min_disp) / (max_disp - min_disp)
    else:
        disparity = np.zeros_like(disparity)

    return disparity


def calculate_rmse(predicted_map, ground_truth_map):
    """Calculates RMSE between predicted and ground truth disparity maps."""
    if predicted_map.shape != ground_truth_map.shape:
        print("Resizing predicted map to match ground truth dimensions.")
        predicted_map = cv2.resize(
            predicted_map,
            (ground_truth_map.shape[1], ground_truth_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    mask = ground_truth_map > 0  # Exclude invalid areas
    mse = mean_squared_error(predicted_map[mask], ground_truth_map[mask])
    return sqrt(mse)


def compare_disparity_maps(model, validation_folders):
    """Compares CNN-based and traditional disparity maps against ground truth, with visualizations."""
    cnn_rmse_vals = []
    sift_rmse_vals = []
    results_md = "results_comparison/results_table.md"
    with open(results_md, "w") as md_file:
        md_file.write("| Folder | CNN RMSE | BM RMSE |\n")
        md_file.write("|--------|----------|-----------|\n")

        # Limit comparison to the first 30 folders
        for index, folder in enumerate(validation_folders[:30]):
            left_img_path = os.path.join(folder, "im0.png")
            right_img_path = os.path.join(folder, "im1.png")
            ground_truth_path = os.path.join(folder, "disp0.pfm")

            cnn_disparity = generate_cnn_disparity(model, left_img_path, right_img_path)

            _, _, _, _, vmin, vmax, _, _, _ = read_parameters(
                os.path.join(folder, "calib.txt")
            )
            bm_disparity = generate_bm_disparity(
                left_img_path, right_img_path, (960, 540), 48, 17, vmin, vmax
            )

            ground_truth = readPFM(ground_truth_path, True, vmax, vmin)

            # Ensure all disparities are the same dimension before RMSE calculation
            if cnn_disparity.shape != ground_truth.shape:
                cnn_disparity = cv2.resize(
                    cnn_disparity, (ground_truth.shape[1], ground_truth.shape[0])
                )
            if bm_disparity.shape != ground_truth.shape:
                bm_disparity = cv2.resize(
                    bm_disparity, (ground_truth.shape[1], ground_truth.shape[0])
                )

            cnn_rmse = calculate_rmse(cnn_disparity, ground_truth)
            sift_rmse = calculate_rmse(bm_disparity, ground_truth)
            cnn_rmse_vals.append(cnn_rmse)
            sift_rmse_vals.append(sift_rmse)

            # Write individual RMSE values to markdown file
            md_file.write(f"| {folder} | {cnn_rmse:.3f} | {sift_rmse:.3f} |\n")

            # Visualization of the disparity maps
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            norm = Normalize(vmin=0, vmax=1)
            axs[0].imshow(ground_truth, cmap="hot", norm=norm)
            axs[0].set_title("Ground Truth")
            axs[0].axis("off")
            axs[1].imshow(cnn_disparity, cmap="hot", norm=norm)
            axs[1].set_title("CNN Disparity")
            axs[1].axis("off")
            axs[2].imshow(bm_disparity, cmap="hot", norm=norm)
            axs[2].set_title("SIFT Disparity")
            axs[2].axis("off")
            plt.suptitle(f"Disparity Map Comparison for Folder {folder}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(
                os.path.join("results_comparison", f"disparity_comparison_{index}.png")
            )
            plt.close()

        # Calculate and write average RMSE values to markdown file
        cnn_avg_rmse = np.mean(cnn_rmse_vals)
        sift_avg_rmse = np.mean(sift_rmse_vals)
        md_file.write("\n## Average RMSE Values\n")
        md_file.write(f"- **CNN Average RMSE:** {cnn_avg_rmse:.3f}\n")
        md_file.write(f"- **BM Average RMSE:** {sift_avg_rmse:.3f}\n")

        print(f"Average RMSE (CNN): {cnn_avg_rmse}")
        print(f"Average RMSE (SIFT): {sift_avg_rmse}")


def compare(model_path="trained_models/cnn_disparity_generator_model_epoch_20.pth"):
    model = EnhancedCustomCNN()
    model.load_state_dict(torch.load(model_path))

    validation_dir = "data/real-data"  # Update as needed
    validation_folders = [
        os.path.join(validation_dir, folder) for folder in os.listdir(validation_dir)
    ]

    compare_disparity_maps(model, validation_folders)
