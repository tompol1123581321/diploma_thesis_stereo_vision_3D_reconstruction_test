import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCustomCNN(nn.Module):
    def __init__(self, disparity_shape=(540, 960)):
        super(EnhancedCustomCNN, self).__init__()
        self.leaky_relu_slope = 0.1
        self.dropout_p = 0.5

        # Encoder layers
        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=3, padding=1), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(128))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(256))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(512))
        self.conv6 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(1024))
        self.pool = nn.MaxPool2d(2, 2)  # This halves the dimensions

        # Decoder layers
        self.upconv1 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(512), nn.Dropout(self.dropout_p))
        self.upconv2 = nn.Sequential(nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(256), nn.Dropout(self.dropout_p))
        self.upconv3 = nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(128), nn.Dropout(self.dropout_p))
        self.upconv4 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(64), nn.Dropout(self.dropout_p))
        self.upconv5 = nn.Sequential(nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(32), nn.Dropout(self.dropout_p))
        self.upconv6 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.LeakyReLU(self.leaky_relu_slope), nn.BatchNorm2d(32), nn.Dropout(self.dropout_p))

        # Final convolution to produce the disparity map
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.disparity_shape = disparity_shape

    def forward(self, left_img, right_img):
        x = torch.cat((left_img, right_img), dim=1)
        # Encoder forward
        x1 = self.conv1(x)
        x2 = self.pool(self.conv2(x1))
        x3 = self.pool(self.conv3(x2))
        x4 = self.pool(self.conv4(x3))
        x5 = self.pool(self.conv5(x4))
        x6 = self.pool(self.conv6(x5))
        
        # Decoder with skip connections
        x = self.upconv1(x6)
        x = self.crop_and_concat(x5, x)  # Assuming x5 was intended here, adjust as necessary
        
        x = self.upconv2(x)
        x = self.crop_and_concat(x4, x)  # Check if x4 needs to be used here

        x = self.upconv3(x)
        x = self.crop_and_concat(x3, x)  # Continue this pattern
        
        x = self.upconv4(x)
        x = self.crop_and_concat(x2, x)
        
        x = self.upconv5(x)
        x = self.crop_and_concat(x1, x)
        
        x = self.upconv6(x)
        
        # Final disparity map
        disparity_map = self.final_conv(x)
        disparity_map = F.interpolate(disparity_map, size=self.disparity_shape, mode='bilinear', align_corners=False)
        return disparity_map

    def crop_and_concat(self, bypass, upsampled):
        _, _, H, W = upsampled.size()
        bypass_cropped = F.interpolate(bypass, size=(H, W), mode='bilinear', align_corners=True)
        return torch.cat((upsampled, bypass_cropped), dim=1)

