import time
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb
from pipeline.evaluate import evaluate_epoch
from pipeline.train import train_epoch
from SimpleClassifier import SimpleClassifier
from utils.data import create_train_validation_loaders, stop_if_data_is_missing
from utils.model import initialize_model, save_torch_model
from utils.quality_of_life import preflight_check

# TODO: The GLOBAL variables here could be moved into a configuration class?
PROJECT_DIR = Path(__file__).parents[1]
DATA_PATH = PROJECT_DIR.joinpath("data/Eyes/")

NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 110
MULTI_GPU_BATCH_SIZE = 512
LEARNING_RATE = 1e-4
LEARNING_RATE_FACTOR = 0.1
LEARNING_RATE_EPOCH_PATIENCE = 2

EXPERIMENT_NAME = "PlateauLR"
PROJECT_NAME = "retina-health-classification"
ENTITY_NAME = "gerovanmi"
ARCHITECTURE_NAME = "U-NetEncoder"  # Must not contain special characters (except "-")
DATASET_NAME = "Medical Scan Classification Dataset"

MODEL_SAVE_PATH = PROJECT_DIR.joinpath(
    f"models/{ARCHITECTURE_NAME}_{int(time.time())}.pt"
)
# In dev mode we only train 3 images for 3 epochs
DEV_MODE = True

if DEV_MODE:
    BATCH_SIZE = 3
    NUMBER_OF_EPOCHS = 3
    EXPERIMENT_NAME = f"DEV {EXPERIMENT_NAME}"


def run_experiment():
    stop_if_data_is_missing(DATA_PATH)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    number_of_gpus = torch.cuda.device_count()
    batch_size = BATCH_SIZE
    if number_of_gpus > 1:
        print("Using", number_of_gpus, "GPUs.")
        batch_size = MULTI_GPU_BATCH_SIZE

    # Give the user information about the run and
    # then ask for confirmation. This allows the user to
    # abort in case they made a mistake in the configuration.
    preflight_check(device, EXPERIMENT_NAME, DEV_MODE)

    # Only initalize W&B after a successful preflight_check
    # to avoid polluting the project with killed runs.
    wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        name=EXPERIMENT_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "learning_rate_factor": LEARNING_RATE_FACTOR,
            "learning_rate_epoch_patience": LEARNING_RATE_EPOCH_PATIENCE,
            "architecture": ARCHITECTURE_NAME,
            "batch_size": batch_size,
            "dataset": DATASET_NAME,
            "epochs": NUMBER_OF_EPOCHS,
            "dev_mode": DEV_MODE,
        },
    )

    # Load Data and Initialize Model, Loss Function & Optimizer
    train_loader, validation_loader = create_train_validation_loaders(
        DATA_PATH, batch_size
    )
    model = initialize_model(SimpleClassifier, device)

    loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LEARNING_RATE_FACTOR,
        patience=LEARNING_RATE_EPOCH_PATIENCE,
    )

    # Training Loop
    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"Started training Epoch {epoch}")
        train_for_one_epoch(
            model,
            train_loader,
            validation_loader,
            loss_function,
            optimizer,
            scheduler,
            device,
        )

    # Saves the model to W&B artifact registry, so that we can
    # download it later, if we want to. Prevents us from loosing
    # the model if we accidentally delete it on our machine.
    if not DEV_MODE:
        save_torch_model(model, MODEL_SAVE_PATH, ARCHITECTURE_NAME)


def train_for_one_epoch(
    model: Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    device: str,
):
    train_loss, train_accuracy, train_f1 = train_epoch(
        model, train_loader, loss_function, optimizer, device, DEV_MODE
    )
    validation_loss, validation_accuracy, validation_f1 = evaluate_epoch(
        model, validation_loader, loss_function, device, DEV_MODE
    )
    scheduler.step(validation_accuracy)

    wandb.log(
        {
            "Training Loss": train_loss,
            "Training Accuracy": train_accuracy,
            "Training F1-Score": train_f1,
            "Testing Loss": validation_loss,
            "Testing Accuracy": validation_accuracy,
            "Testing F1-Score": validation_f1,
        }
    )


if __name__ == "__main__":
    run_experiment()
