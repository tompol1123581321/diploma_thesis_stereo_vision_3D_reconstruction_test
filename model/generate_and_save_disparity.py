import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(960, 540)):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def write_pfm(file, image, scale=1):
    with open(file, "wb") as f:
        color = None
        image = np.flipud(image)  # Flip vertically
        if len(image.shape) == 3 and image.shape[2] == 3:
            color = True
        elif (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[2] == 1):
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        f.write(f'PF\n' if color else 'Pf\n'.encode('latin-1'))
        f.write(f'{image.shape[1]} {image.shape[0]}\n'.encode('latin-1'))
        endian = '>' if image.dtype.byteorder == '<' else '<'
        scale = scale if endian == '<' else -scale
        f.write(f'{scale}\n'.encode('latin-1'))
        image.tofile(f)

def predict_disparity(model, left_img_path, right_img_path, output_path):
    # Load and preprocess the images
    left_img = load_and_preprocess_image(left_img_path)
    right_img = load_and_preprocess_image(right_img_path)

    # Add batch dimension
    left_img = left_img.unsqueeze(0)
    right_img = right_img.unsqueeze(0)

    # Move to the same device as the model
    device = next(model.parameters()).device
    left_img = left_img.to(device)
    right_img = right_img.to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        output = model(left_img, right_img).squeeze(0).squeeze(0)  # remove batch dimension and channel dimension

    # Convert to numpy for saving and visualizing
    output = output.cpu().numpy()

    # Visualize the disparity map
    plt.imshow(output, cmap='plasma')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()

    write_pfm(output_path, output)