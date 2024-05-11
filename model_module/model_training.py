import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model_module.CNN_model import EnhancedCustomCNN
from model_module.data_loader import StereoDataset, load_data


def save_heatmap(data, filename):
    """Save heatmap of disparity data, normalized to range [0, 1] based on percentiles."""
    # Calculate the 2nd and 98th percentiles
    p2, p98 = np.percentile(data, [2, 98])
    # Clip and scale data to [0, 1] based on these percentiles
    data_clipped = np.clip(data, p2, p98)
    if p98 > p2:  # Avoid division by zero if all values are very close or the same
        data_normalized = (data_clipped - p2) / (p98 - p2)
    else:
        data_normalized = data_clipped

    plt.imshow(data_normalized, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Disparity Map")
    plt.savefig(filename)
    plt.close()


def pixel_accuracy(outputs, targets, threshold=0.3):
    """Calculate the percentage of pixels where the disparity is within a threshold."""
    return torch.mean(((torch.abs(outputs - targets) < threshold).float())).item()


def train_custom_cnn(data_dir, num_epochs=50, patience=10):
    (
        train_left,
        train_right,
        train_disp,
        train_params,
        val_left,
        val_right,
        val_disp,
        val_params,
    ) = load_data(data_dir)
    train_dataset = StereoDataset(
        train_left, train_right, train_disp, train_params, augment=True
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_dataset = StereoDataset(val_left, val_right, val_disp, val_params)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedCustomCNN().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(train_loader)
    )

    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_accuracy = 0, 0

        for i, (left_img, right_img, disparity) in enumerate(train_loader):
            left_img, right_img, disparity = [
                x.to(device) for x in [left_img, right_img, disparity]
            ]
            optimizer.zero_grad()
            outputs = model(left_img, right_img)
            disparity = disparity.squeeze(
                1
            )  # Ensure the disparity map has the same shape as the model's output
            loss = criterion(outputs, disparity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            accuracy = pixel_accuracy(outputs, disparity)
            total_accuracy += accuracy
            if i % 10 == 0:
                print(
                    f"Epoch {epoch+1}, Batch {i+1}, Training Loss: {loss.item()}, Accuracy: {accuracy}"
                )

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = total_accuracy / len(train_loader)
        print(
            f"Epoch {epoch+1}, Average Training Loss: {epoch_loss}, Average Accuracy: {epoch_accuracy}"
        )

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (left_img, right_img, disparity) in enumerate(val_loader):
                left_img, right_img, disparity = [
                    x.to(device) for x in [left_img, right_img, disparity]
                ]
                outputs = model(left_img, right_img)
                val_loss = criterion(outputs, disparity.squeeze(1))
                total_val_loss += val_loss.item()

                if i == 1:  # 10% chance to save a visualization
                    save_heatmap(
                        outputs[0].cpu().squeeze().numpy(),
                        f"results/result_disparity_epoch_{epoch+1}.png",
                    )
                    save_heatmap(
                        disparity[0].cpu().squeeze().numpy(),
                        f"results/ground_truth_disparity_epoch.png",
                    )

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")
        scheduler.step()
        torch.save(
            model.state_dict(),
            f"trained_models/cnn_disparity_generator_model_epoch_{epoch+1}.pth",
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            print(f"Saved new best model at epoch {epoch+1}")

        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print("Early stopping triggered!")
            break


def start_training():
    train_custom_cnn("data")
