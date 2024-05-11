import os
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
    Normalize,
)
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image

from model_module.read_pfm import disparity_read, readPFM
from reconstruction_module.read_parameters import read_parameters


class StereoDataset(Dataset):
    def __init__(
        self,
        left_image_paths,
        right_image_paths,
        disparity_paths,
        params_paths,
        augment=False,
        target_size=(960, 540),
        rotation_range=10,
        horizontal_flip_prob=0.5,
        color_jitter_params=None,
        normalize_params=None,
    ):
        self.left_image_paths = left_image_paths
        self.right_image_paths = right_image_paths
        self.disparity_paths = disparity_paths
        self.params_paths = params_paths
        self.augment = augment
        self.target_size = target_size

        # Basic transformations
        self.to_tensor = ToTensor()
        self.resize = Resize(target_size)
        self.normalize = (
            Normalize(**normalize_params) if normalize_params else lambda x: x
        )

        # Augmentation configurations
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_range = rotation_range
        self.color_jitter = (
            ColorJitter(**color_jitter_params) if color_jitter_params else lambda x: x
        )

    def __len__(self):
        return len(self.left_image_paths)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_image_paths[idx]).convert("RGB")
        right_img = Image.open(self.right_image_paths[idx]).convert("RGB")
        _, _, _, _, vmin, vmax, _, _, _ = read_parameters(self.params_paths[idx])
        disparity = readPFM(
            self.disparity_paths[idx], True, vmax, vmin, self.target_size
        )

        disparity = to_pil_image(disparity)

        if self.augment:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            if random.random() < self.horizontal_flip_prob:
                left_img = TF.hflip(left_img)
                right_img = TF.hflip(right_img)
                disparity = TF.hflip(disparity)
            left_img = TF.rotate(left_img, angle)
            right_img = TF.rotate(right_img, angle)
            disparity = TF.rotate(disparity, angle)
            left_img = self.color_jitter(left_img)
            right_img = self.color_jitter(right_img)

        disparity = to_tensor(disparity)

        left_img = self.to_tensor(self.resize(left_img))
        right_img = self.to_tensor(self.resize(right_img))

        left_img = self.normalize(left_img)
        right_img = self.normalize(right_img)

        return left_img, right_img, disparity.unsqueeze(0)


def load_data(data_dir, test_size=0.2):
    left_image_paths = []
    right_image_paths = []
    disparity_paths = []
    params_paths = []

    # # Synthetic Data
    # training_data_dir = os.path.join(data_dir, "training")
    # for scene_name in os.listdir(os.path.join(training_data_dir, "clean_left")):
    #     scene_dir = os.path.join(training_data_dir,"clean_left" ,scene_name)
    #     for image_name in os.listdir(os.path.join(scene_dir)):
    #         left_image_paths.append(os.path.join(training_data_dir, 'clean_left',scene_name, image_name))
    #         right_image_paths.append(os.path.join(training_data_dir, 'clean_right',scene_name, image_name))
    #         disparity_paths.append(os.path.join(training_data_dir, 'disparities', scene_name,image_name))  # Assuming disparity maps are .pfm files

    # Real Data
    real_data_dir = os.path.join(data_dir, "real-data")
    for scene_name in os.listdir(real_data_dir):
        scene_dir = os.path.join(real_data_dir, scene_name)
        left_path = os.path.join(scene_dir, "im0.png")
        right_path = os.path.join(scene_dir, "im1.png")
        disp_path = os.path.join(
            scene_dir, "disp0.pfm"
        )  # Assuming real data also has .pfm disparity maps
        params_path = os.path.join(scene_dir, "calib.txt")
        if (
            os.path.exists(left_path)
            and os.path.exists(right_path)
            and os.path.exists(disp_path)
            and os.path.exists(params_path)
        ):
            left_image_paths.append(left_path)
            right_image_paths.append(right_path)
            disparity_paths.append(disp_path)
            params_paths.append(params_path)

    #  # Real Data 2
    # real_data_dir_2 = os.path.join(data_dir, "real-data-2")
    # for scene_name in os.listdir(real_data_dir_2):
    #     scene_dir = os.path.join(real_data_dir_2, scene_name)
    #     left_path = os.path.join(scene_dir, 'view1.png')
    #     right_path = os.path.join(scene_dir, 'view5.png')
    #     disp_path = os.path.join(scene_dir, 'disp1.png')  # Assuming real data also has .pfm disparity maps
    #     if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(disp_path):
    #         left_image_paths.append(left_path)
    #         right_image_paths.append(right_path)
    #         disparity_paths.append(disp_path)

    # Split dataset into training and validation sets
    (
        train_params,
        val_params,
        train_left,
        val_left,
        train_right,
        val_right,
        train_disp,
        val_disp,
    ) = train_test_split(
        params_paths,
        left_image_paths,
        right_image_paths,
        disparity_paths,
        test_size=test_size,
        random_state=42,
    )

    return (
        train_left,
        train_right,
        train_disp,
        train_params,
        val_left,
        val_right,
        val_disp,
        val_params,
    )
