import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model.CNN_model import EnhancedCustomCNN
from model.data_loader import StereoDataset, load_data

def save_heatmap(data, filename):
    """Save heatmap of disparity data."""
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def pixel_precision(outputs, targets, threshold=1.0):
    """Calculate the percentage of pixels where the disparity is within a threshold."""
    return torch.mean(((torch.abs(outputs - targets) < threshold).float())).item()

def train_custom_cnn(data_dir, num_epochs=25, patience=10):
    device = torch.device("cpu")  # Force using CPU to avoid any CUDA errors

    model = EnhancedCustomCNN().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_left, train_right, train_disp, val_left, val_right, val_disp = load_data(data_dir)
    train_dataset = StereoDataset(train_left, train_right, train_disp, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_dataset = StereoDataset(val_left, val_right, val_disp)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_precision = 0, 0

        for i, (left_img, right_img, disparity) in enumerate(train_loader):
            left_img, right_img, disparity = [x.to(device) for x in [left_img, right_img, disparity]]
            optimizer.zero_grad()
            outputs = model(left_img, right_img)
            loss = criterion(outputs, disparity)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_precision += pixel_precision(outputs, disparity)

            if i % 10 == 0:  # Print progress every 10 batches
                print(f"Epoch {epoch+1}, Batch {i+1}, Training Loss: {loss.item()}, Precision: {pixel_precision(outputs, disparity)}")

        epoch_loss = total_loss / len(train_loader)
        epoch_precision = total_precision / len(train_loader)
        print(f"Epoch {epoch+1}, Average Training Loss: {epoch_loss}, Average Precision: {epoch_precision}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (left_img, right_img, disparity) in enumerate(val_loader):
                left_img, right_img, disparity = [x.to(device) for x in [left_img, right_img, disparity]]
                outputs = model(left_img, right_img)
                val_loss = criterion(outputs, disparity)
                total_val_loss += val_loss.item()

                if i == 0:  # Save the first batch of results as heatmaps for visual inspection
                    save_heatmap(outputs.cpu().squeeze().numpy(), f"result_disparity_epoch_{epoch+1}.png")
                    save_heatmap(disparity.cpu().squeeze().numpy(), f"ground_truth_disparity_epoch_{epoch+1}.png")

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), "best_cnn_disparity_map_generator.pth")
            print(f"Saved new best model at epoch {epoch+1}")

        if epochs_since_improvement >= patience:
            print('Early stopping triggered!')
            break

def start_training():
    train_custom_cnn("data")
