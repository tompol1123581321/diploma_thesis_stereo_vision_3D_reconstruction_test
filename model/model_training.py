import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.CNN_model import StereoCNN
from torch.utils.data import DataLoader

from model.data_loader import get_datasets, get_training_augmentation

def compute_loss(outputs, true_disparity):
    return nn.SmoothL1Loss()(outputs, true_disparity)

def percentage_of_correct_pixels(pred, target, threshold=3):
    abs_diff = torch.abs(pred - target)
    correct_pixels = (abs_diff < threshold).float()
    accuracy = correct_pixels.mean() * 100.0
    return accuracy

def validate(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in data_loader:
            left_imgs = data['left_img'].to(device)
            right_imgs = data['right_img'].to(device)
            true_disparity = data['disparity_map'].to(device)

            outputs = model(left_imgs, right_imgs)
            loss = loss_function(outputs, true_disparity)
            accuracy = percentage_of_correct_pixels(outputs, true_disparity)

            total_loss += loss.item()
            total_accuracy += accuracy

    return total_loss / len(data_loader), total_accuracy / len(data_loader)

def train_model_loop(num_epochs, train_loader, val_loader, device, patience=5):
    model = StereoCNN().to(device)
    loss_function = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0

        for data in train_loader:
            left_imgs = data['left_img'].to(device)
            right_imgs = data['right_img'].to(device)
            true_disparity = data['disparity_map'].to(device)

            optimizer.zero_grad()
            outputs = model(left_imgs, right_imgs)
            loss = compute_loss(outputs, true_disparity)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            accuracy = percentage_of_correct_pixels(outputs, true_disparity)
            total_train_accuracy += accuracy

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        val_loss, val_accuracy = validate(model, val_loader, loss_function, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping triggered.')
                break

def start_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    root_dir = 'data'
    transform = get_training_augmentation()
    train_dataset, val_dataset = get_datasets(root_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    train_model_loop(20, train_loader, val_loader, device)
