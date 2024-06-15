import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Configuration:
    """
    The configuration for training the model and experiment.
    """

    # In dev mode we only train 3 images for 3 epochs and 3 k-folds
    DEV_MODE = False

    # Training hyper-parameters
    NUMBER_OF_EPOCHS_TRAINING = 20
    NUMBER_OF_EPOCHS_DEV = 3

    BATCH_SIZE_TESTING = 3
    BATCH_SIZE_TRAINING = 64
    MULTI_GPU_BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    LEARNING_RATE_FACTOR = 0.1
    LEARNING_RATE_EPOCH_PATIENCE = 4

    K_FOLDS_TRAINING = 5
    K_FOLDS_DEV = 3

    # Names / Tags
    EXPERIMENT_NAME = "CrossValidation"
    PROJECT_NAME = "retina-health-classification"
    ENTITY_NAME = "gerovanmi"
    ARCHITECTURE_NAME = (
        "U-NetEncoder"  # Must not contain special characters (except "-")
    )
    DATASET_NAME = "Medical Scan Classification Dataset"

    # Pathes
    PROJECT_DIR = Path(__file__).parents[1]
    DATA_PATH = PROJECT_DIR.joinpath("data/Eyes/")

    os.makedirs(PROJECT_DIR.joinpath("models/"), exist_ok=True)
    MODEL_SAVE_PATH = PROJECT_DIR.joinpath(
        f"models/{ARCHITECTURE_NAME}_{int(time.time())}.pt"
    )

    CLASS_INDICES = {
        "Cataract": 0,
        "Diabetic Retinopathy": 1,
        "Glaucoma": 2,
        "Normal": 3,
        "Empty": -1,
    }

    @property
    def RUN_NAME(self):
        if self.DEV_MODE:
            return f"DEV {self.EXPERIMENT_NAME}"
        return self.EXPERIMENT_NAME

    @property
    def NUMBER_OF_EPOCHS(self):
        if self.DEV_MODE:
            return self.NUMBER_OF_EPOCHS_DEV

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

    @property
    def K_FOLDS(self):
        if self.DEV_MODE:
            return self.K_FOLDS_DEV

        return self.K_FOLDS_TRAINING
