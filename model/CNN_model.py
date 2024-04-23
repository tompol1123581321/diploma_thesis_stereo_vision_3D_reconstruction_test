import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCustomCNN(nn.Module):
    def __init__(self, disparity_shape=(960,540 )):
        super(EnhancedCustomCNN, self).__init__()
        # Encoder layers
        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(512))
        self.pool = nn.MaxPool2d(2, 2)  # This halves the dimensions

        # Decoder layers
        self.upconv1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(256))  # This doubles the dimensions
        self.upconv2 = nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(128))
        self.upconv3 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(64))
        self.upconv4 = nn.Sequential(nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(32))
        self.upconv5 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(32))

        # Final convolution to produce the disparity map
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.disparity_shape = disparity_shape

    def forward(self, left_img, right_img):
        x = torch.cat((left_img, right_img), dim=1)  # Concatenating stereo images along the channel dimension
        x1 = self.conv1(x)
        x2 = self.pool(self.conv2(x1))
        x3 = self.pool(self.conv3(x2))
        x4 = self.pool(self.conv4(x3))
        x5 = self.pool(self.conv5(x4))
        
        # Decoder with skip connections
        x = self.upconv1(x5)
        x = self.crop_and_concat(x4, x)
        x = self.upconv2(x)
        x = self.crop_and_concat(x3, x)
        x = self.upconv3(x)
        x = self.crop_and_concat(x2, x)
        x = self.upconv4(x)
        x = self.crop_and_concat(x1, x)
        x = self.upconv5(x)
        
        # Producing the final disparity map
        disparity_map = self.final_conv(x)
        
        # Scaling up to the original disparity map size
        disparity_map = F.interpolate(disparity_map, size=self.disparity_shape, mode='bilinear', align_corners=False)
        
        return disparity_map

    def crop_and_concat(self, bypass, upsampled):
        # Dynamically pad the upsampled tensor if its dimensions are smaller than the bypass tensor
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]
        upsampled_padded = F.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        return torch.cat((bypass, upsampled_padded), dim=1)