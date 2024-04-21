import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from model.read_pfm import readPFM  # Assuming readPFM is properly defined elsewhere
from torchvision.transforms.functional import to_tensor

class StereoDataset(Dataset):
    def __init__(self, root_dir, indices=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform or Compose([Resize((1080, 1920)), ToTensor()])
        self.pairs = self._load_pairs(root_dir, indices)

    def _load_pairs(self, root_dir, indices=None):
        pairs = []
        for scene_name in os.listdir(root_dir):
            scene_dir = os.path.join(root_dir, scene_name)
            left_path = os.path.join(scene_dir, 'im0.png')
            right_path = os.path.join(scene_dir, 'im1.png')
            disp_path = os.path.join(scene_dir, 'disp0.pfm')
            if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(disp_path):
                pairs.append((left_path, right_path, disp_path))
        
        if indices is not None:
            pairs = [pairs[i] for i in indices if i < len(pairs)]
    
        return pairs
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        left_img_path, right_img_path, disp_path = self.pairs[idx]
        left_img = Image.open(left_img_path).convert('L')
        right_img = Image.open(right_img_path).convert('L')
        disparity = Image.open(disp_path).convert("L")

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            disparity = self.transform(disparity)
        
        input_img = torch.cat([left_img, right_img], dim=0)
        return input_img, disparity
