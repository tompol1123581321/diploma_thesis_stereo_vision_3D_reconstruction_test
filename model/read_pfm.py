import re

import numpy as np


def readPFM(file, max_disparity=2000):
    """ Read a PFM file and return its content after applying a maximum disparity clipping. """
    with open(file, "rb") as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('latin-1'))
        if not dim_match:
            raise Exception('Malformed PFM header.')

        width, height = map(int, dim_match.groups())
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM files are stored in top-to-bottom order

        # Apply clipping to the disparity data
        if color:
            # Assuming the disparity information is in one of the channels, usually the first
            data[:, :, 0] = np.clip(data[:, :, 0], -np.inf, max_disparity)
        else:
            data = np.clip(data, -np.inf, max_disparity)

        return data
