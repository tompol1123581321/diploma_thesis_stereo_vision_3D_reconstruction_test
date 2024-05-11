import numpy as np
import re
from PIL import Image


import numpy as np
import re
from PIL import Image


def readPFM(
    file,
    is_from_dataset=False,
    max_disparity=3000,
    min_disparity=30,
    target_size=(960, 540),
):
    with open(file, "rb") as f:
        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("latin-1"))
        if not dim_match:
            raise Exception("Malformed PFM header.")

        width, height = map(int, dim_match.groups())
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(f, endian + "f")
        shape = (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM files are stored in top-to-bottom order

        if is_from_dataset:

            # Clipping the data
            data = np.clip(data, min_disparity, max_disparity)

            # Resize the image
            img = Image.fromarray(data.squeeze().astype("float32"), "F")
            img = img.resize(target_size, Image.BILINEAR)

            resized_data = np.array(img)

            # Normalize the resized data
            min_val, max_val = resized_data.min(), resized_data.max()
            normalized_data = (
                (resized_data - min_val) / (max_val - min_val)
                if max_val > min_val
                else resized_data
            )

            return normalized_data
        else:
            return data


def readPNGDisparity(filename, target_size):
    """Reads a disparity map from a PNG file, resizing it to the target size and converting it into a float representation normalized between 0 and 1.

    Args:
        filename (str): Path to the PNG file.
        target size (tuple): Target size to resize the image to.

    Returns:
        numpy.ndarray: The normalized loaded and converted disparity data.
    """
    img = Image.open(filename)
    img = img.resize(target_size, Image.BILINEAR)
    depth = np.array(img)

    # Convert RGB values into a disparity value
    d_r = depth[:, :, 0].astype("float64")
    d_g = depth[:, :, 1].astype("float64")
    d_b = depth[:, :, 2].astype("float64")
    depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)

    # Normalize the data to range [0, 1]
    min_val, max_val = depth.min(), depth.max()
    if max_val > min_val:
        depth = (depth - min_val) / (max_val - min_val)

    return depth


def disparity_read(filename, target_size=(960, 540)):
    """Reads disparity data from a given file and resizes it to the target size.

    Args:
        filename (str): Path to the disparity file.
        target_size (tuple): Target size to resize the disparity data to.

    Returns:
        numpy.ndarray: The resized and converted disparity data.
    """
    try:
        if filename.endswith(".pfm"):
            depth = np.array(readPFM(filename, target_size=target_size))

        else:
            depth = readPNGDisparity(filename, target_size)

        # Normalizing disparity map
        depth_max = np.max(depth)
        if depth_max != 0:
            depth = depth / depth_max

        # Additional prints for debugging
        # print(f"After Normalization:")
        # print(f"Shape: {depth.shape}")
        # print(f"Max Value: {np.max(depth)}")
        # print(f"Min Value: {np.min(depth)}")

        return depth
    except Exception as e:
        print(f"Failed to read disparity data from {filename}: {e}")
        return np.zeros(target_size)


# disparity_read("data/training/disparities/alley_1/frame_0001.png")
# disparity_read("data/real-data/Adirondack-perfect/disp0.pfm")
