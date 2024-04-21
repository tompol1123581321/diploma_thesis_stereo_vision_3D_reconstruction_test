import torch
import torch.nn as nn
import torch.nn.functional as F

class DisparityEstimationNet(nn.Module):
    def __init__(self):
        super(DisparityEstimationNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.out_conv(x)
        return x
