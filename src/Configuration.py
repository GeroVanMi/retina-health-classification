import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Configuration:
    # In dev mode we only train 3 images for 3 epochs
    DEV_MODE = False

    # Training hyper-parameters
    NUMBER_OF_EPOCHS_TRAINING = 20
    NUMBER_OF_EPOCHS_TESTING = 3

    BATCH_SIZE_TESTING = 3
    BATCH_SIZE_TRAINING = 110
    MULTI_GPU_BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    LEARNING_RATE_FACTOR = 0.1
    LEARNING_RATE_EPOCH_PATIENCE = 2

    # Names / Tags
    EXPERIMENT_NAME = "ResizeThenCrop"
    PROJECT_NAME = "retina-health-classification"
    ENTITY_NAME = "gerovanmi"
    ARCHITECTURE_NAME = (
        "U-NetEncoder"  # Must not contain special characters (except "-")
    )
    DATASET_NAME = "Medical Scan Classification Dataset"

    # Pathes
    PROJECT_DIR = Path(__file__).parents[1]
    DATA_PATH = PROJECT_DIR.joinpath("data/Eyes/")
    MODEL_SAVE_PATH = PROJECT_DIR.joinpath(
        f"models/{ARCHITECTURE_NAME}_{int(time.time())}.pt"
    )

    @property
    def RUN_NAME(self):
        if self.DEV_MODE:
            return f"DEV {self.EXPERIMENT_NAME}"
        return self.EXPERIMENT_NAME

    @property
    def NUMBER_OF_EPOCHS(self):
        if self.DEV_MODE:
            return self.NUMBER_OF_EPOCHS_TESTING

        return self.NUMBER_OF_EPOCHS_TRAINING

    @property
    def NUMBER_OF_GPUS(self):
        return torch.cuda.device_count()

    @property
    def BATCH_SIZE(self):
        if self.DEV_MODE:
            return self.BATCH_SIZE_TESTING

        if self.NUMBER_OF_GPUS > 1:
            return self.MULTI_GPU_BATCH_SIZE

        return self.BATCH_SIZE_TRAINING
