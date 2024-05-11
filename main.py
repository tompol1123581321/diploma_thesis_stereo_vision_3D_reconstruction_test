from compare_module.compare import compare
from model_module.model_training import start_training
from reconstruction_module.reconstruct_and_vizualize_mesh import (
    reconstruct_and_vizualize_mesh,
)


def main():
    start_training()
    # compare("trained_models/cnn_disparity_generator_model_epoch_20.pth")
    # reconstruct_and_vizualize_mesh(
    #     "data/real-data/pendulum2/im0.png",
    #     "data/real-data/pendulum2/im1.png",
    #     "data/real-data/pendulum2/calib.txt",
    #     "trained_models/cnn_disparity_generator_model_epoch_26.pth",
    #     "model_output_disparity.pfm",
    #     None,
    # )


if __name__ == "__main__":
    main()
