def read_parameters(filepath):
    params = {}
    with open(filepath, "r") as file:
        for line in file:
            if "=" in line:
                key, value = line.strip().split("=")
                params[key.strip()] = value.strip()

    # Extracting the focal length, cx, cy from cam0
    cam0 = params.get("cam0")
    if cam0:
        matrix_values = cam0.replace("[", "").replace("]", "").replace(";", "").split()
        focal_length = float(matrix_values[0])
        cx = float(matrix_values[2])
        cy = float(matrix_values[5])
    else:
        focal_length = cx = cy = None

    # Extract other specific parameters
    baseline = float(params.get("baseline", 0))
    vmin = int(params.get("vmin", 0))
    vmax = int(params.get("vmax", 0))
    width = int(params.get("width", 0))
    height = int(params.get("height", 0))
    ndisp = int(params.get("ndisp", 0))

    return focal_length, cx, cy, baseline, vmin, vmax, width, height, ndisp
