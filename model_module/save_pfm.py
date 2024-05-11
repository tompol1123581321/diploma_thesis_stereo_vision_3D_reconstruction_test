import sys


def savePFM(file_path, image, scale=1):
    """
    Save a Numpy array to a PFM file.
    """
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    # Open the file for binary write operations
    with open(file_path, 'wb') as file:
        color = image.ndim == 3 and image.shape[2] == 3

        if color:
            file.write(b'PF\n')
        else:
            file.write(b'Pf\n')

        height, width = image.shape[:2]
        file.write(f'{width} {height}\n'.encode())

        endian = image.dtype.byteorder
        if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
            scale = -scale

        file.write(f'{scale}\n'.encode())

        # Ensure image data is written in the correct byte order
        if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
            image = image.byteswap()

        image.tofile(file)
