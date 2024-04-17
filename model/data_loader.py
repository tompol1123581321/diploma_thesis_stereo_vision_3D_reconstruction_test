import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from model.read_pfm import readPFM  # Ensure this path is correct based on your project structure

class StereoDataset(Dataset):
    def __init__(self, left_images, right_images, disparities, transform=None):
        self.left_images = left_images
        self.right_images = right_images
        self.disparity_maps = disparities
        self.transform = transform

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img = cv2.imread(self.left_images[idx], cv2.IMREAD_COLOR)
        right_img = cv2.imread(self.right_images[idx], cv2.IMREAD_COLOR)
        disparity_map = readPFM(self.disparity_maps[idx])

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=left_img, image0=right_img, mask=disparity_map)
            left_img = augmented['image']
            right_img = augmented['image0']
            disparity_map = augmented['mask']

        return {"left_img": left_img, "right_img": right_img, "disparity_map": disparity_map}

def get_datasets(root_dir, transform, test_size=0.2, random_state=42):
    left_images, right_images, disparities = [], [], []
    for scene_name in os.listdir(root_dir):
        scene_dir = os.path.join(root_dir, scene_name)
        left_path = os.path.join(scene_dir, 'im0.png')
        right_path = os.path.join(scene_dir, 'im1.png')
        disp_path = os.path.join(scene_dir, 'disp0.pfm')
        if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(disp_path):
            left_images.append(left_path)
            right_images.append(right_path)
            disparities.append(disp_path)

    left_train, left_val, right_train, right_val, disp_train, disp_val = train_test_split(
        left_images, right_images, disparities, test_size=test_size, random_state=random_state)

    train_dataset = StereoDataset(left_train, right_train, disp_train, transform)
    val_dataset = StereoDataset(left_val, right_val, disp_val, transform)

    return train_dataset, val_dataset

def get_training_augmentation():
    """Define and return a composition of image transformations for training."""
    return A.Compose([
        A.Resize(512, 512),  # Resize images and masks to 512x512
        A.HorizontalFlip(p=0.5),  # Randomly flip images horizontally
        A.RandomRotate90(),  # Randomly rotate images by 90 degrees
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.75),  # Random shift, scale, and rotate
        A.Blur(blur_limit=3),  # Apply blur to images
        A.OpticalDistortion(p=0.3),  # Apply optical distortion
        A.GridDistortion(p=0.1),  # Apply grid distortion
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.9),  # Randomly change hue, saturation, value
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize images
        ToTensorV2()  # Convert images and masks to PyTorch tensors
    ], additional_targets={'image0': 'image', 'mask': 'mask'})  