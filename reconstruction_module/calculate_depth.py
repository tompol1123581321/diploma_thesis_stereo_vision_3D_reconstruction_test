import numpy as np


def calculate_depth(disparity_normalized, focal_length, baseline, vmin, vmax):
    # Rescale the normalized disparity to actual disparity values
    disparity_actual = vmin + disparity_normalized * (vmax - vmin)

    # Calculate depth from actual disparity
    depth = focal_length * baseline / np.maximum(disparity_actual, 1e-6)
    print("Depth Average:", depth.mean(), "Disparity Average:", disparity_actual.mean())

    return depth
