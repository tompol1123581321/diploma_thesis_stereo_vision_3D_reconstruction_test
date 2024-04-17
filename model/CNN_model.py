import torch
import torch.nn as nn
import torch.nn.functional as F

class StereoCNN(nn.Module):
    def __init__(self, image_height=512, image_width=512):
        super(StereoCNN, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1_input_features = 64 * (self.image_height // 4) * (self.image_width // 4)
        self.fc1 = nn.Linear(self.fc1_input_features, 1024)
        self.fc2 = nn.Linear(1024, self.image_height * self.image_width)

    def forward(self, left_img, right_img):
        x = torch.cat((left_img, right_img), dim=1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc1_input_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.image_height, self.image_width)
        return x
