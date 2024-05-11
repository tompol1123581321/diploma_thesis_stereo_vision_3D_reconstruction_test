import torchvision.transforms as transforms
from PIL import Image


def load_and_preprocess_image(image_path, target_size=(960, 540)):
    """Loads and preprocesses an image from a specified path."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            # Uncomment and use normalization if necessary
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return transform(img)
