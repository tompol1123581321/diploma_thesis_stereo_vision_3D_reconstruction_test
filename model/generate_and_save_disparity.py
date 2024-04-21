import torch
import numpy as np
import cv2
from torchvision.transforms import Compose, ToTensor, Resize

from model.CNN_model import DisparityEstimationNet
from model.save_pfm import savePFM
from PIL import Image

def generate_and_save_disparity(model_path, left_image_path, right_image_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model
    model = DisparityEstimationNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load images using PIL directly
    left_img = Image.open(left_image_path).convert('L')
    right_img = Image.open(right_image_path).convert('L')

    # Transformation setup
    transform = Compose([Resize((256, 512)), ToTensor()])
    left_img = transform(left_img).unsqueeze(0).to(device)
    right_img = transform(right_img).unsqueeze(0).to(device)
    
    # Prepare model input
    model_input = torch.cat([left_img, right_img], dim=1)
    
    # Generate disparity map
    with torch.no_grad():
        disparity_map = model(model_input).squeeze().cpu().numpy()
    
    # Save the disparity map to a PFM file
    savePFM(output_path, disparity_map)


