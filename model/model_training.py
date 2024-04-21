import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from model.CNN_model import DisparityEstimationNet
from model.data_loader import StereoDataset\

def custom_collate(batch):
    inputs, targets = zip(*batch)
    
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    
    return inputs, targets

def disparity_accuracy(predicted, ground_truth, threshold=0.5):
    abs_diff = torch.abs(predicted - ground_truth)
    correct = abs_diff <= threshold
    return torch.mean(correct.float()) * 100

def save_model(model, path):
    """
    Save the given model to the specified path.
    """
    torch.save(model.state_dict(), path)

def train(model, device, train_loader, val_loader, optimizer, num_epochs=20):
    model.train()
    best_val_loss = float('inf') 
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_accuracy = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy = disparity_accuracy(outputs, targets)
            total_accuracy += accuracy

        print(f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader):.3f}, Train Accuracy: {total_accuracy / len(train_loader):.2f}%')

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                val_loss += loss.item()
                accuracy = disparity_accuracy(outputs, targets)
                val_accuracy += accuracy

        val_loss_avg = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}, Val Loss: {val_loss_avg:.3f}, Val Accuracy: {val_accuracy / len(val_loader):.2f}%')

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            save_model(model, f'best_model_epoch_{epoch+1}.pth')

def start_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StereoDataset(root_dir="data")
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    split = int(np.floor(0.8 * total_size))
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = StereoDataset(root_dir="data", indices=train_indices, transform=Compose([
        Resize((1080, 1920)),
        ToTensor()
        
    ]))
    val_dataset = StereoDataset(root_dir="data", indices=val_indices, transform=Compose([
        Resize((1080, 1920)),
        ToTensor()
    ]))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,num_workers=4)

    model = DisparityEstimationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, device, train_loader, val_loader, optimizer)

