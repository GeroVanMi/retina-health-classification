from time import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb
from Configuration import Configuration
from DoubleClassifier import DoubleClassifier
from pipeline.evaluate import evaluate_epoch
from pipeline.train import train_epoch
from utils.data import create_train_validation_loaders, stop_if_data_is_missing
from utils.EyesDataset import EyesDataset
from utils.model import initialize_model, save_torch_model
from utils.quality_of_life import preflight_check

config = Configuration()


def run_experiment():
    """
    Run the experiment as configured in the Configuration class.

    This function is the main entry point for the training pipeline.

    It will:
    - Load the data
    - Initialize the model, loss function and optimizer
    - Run the training loop
    - Log the results on wandb
    - Save the model on wandb as artifact registry
    """
    stop_if_data_is_missing(config.DATA_PATH)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if config.NUMBER_OF_GPUS > 1:
        print("Using", config.NUMBER_OF_GPUS, "GPUs.")

    # Give the user information about the run and
    # then ask for confirmation. This allows the user to
    # abort in case they made a mistake in the configuration.
    preflight_check(device, config.RUN_NAME, config.DEV_MODE)

    # Only initalize W&B after a successful preflight_check
    # to avoid polluting the project with killed runs.

    k_fold = KFold(n_splits=config.K_FOLDS, shuffle=True)
    dataset = EyesDataset(config.DATA_PATH)
    start_time = int(time())
    for fold, (train_indices, validation_indices) in enumerate(k_fold.split(dataset)):
        train_k_fold(
            fold, dataset, train_indices, validation_indices, device, start_time
        )


def train_k_fold(
    fold_number: int,
    dataset: EyesDataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: str,
    start_time: int,
):
    """
    Trains a single fold in the k-fold crossvalidation.
    """
    k_fold_run = wandb.init(
        project=config.PROJECT_NAME,
        entity=config.ENTITY_NAME,
        name=f"fold_{fold_number + 1} {config.RUN_NAME}",
        group=f"{config.RUN_NAME} {start_time}",
        config={
            "learning_rate": config.LEARNING_RATE,
            "learning_rate_factor": config.LEARNING_RATE_FACTOR,
            "learning_rate_epoch_patience": config.LEARNING_RATE_EPOCH_PATIENCE,
            "architecture": config.ARCHITECTURE_NAME,
            "batch_size": config.BATCH_SIZE,
            "dataset": config.DATASET_NAME,
            "epochs": config.NUMBER_OF_EPOCHS,
            "dev_mode": config.DEV_MODE,
            "k-fold": fold_number,
        },
    )
    # Load Data and Initialize Model, Loss Function & Optimizer
    train_loader, validation_loader = create_train_validation_loaders(
        dataset,
        train_indices,
        validation_indices,
        config.BATCH_SIZE,
    )
    model = initialize_model(DoubleClassifier, device)

    loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LEARNING_RATE_FACTOR,
        patience=config.LEARNING_RATE_EPOCH_PATIENCE,
        min_lr=1e-5,
    )

    # Training Loop
    for epoch in range(config.NUMBER_OF_EPOCHS):
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
    if not config.DEV_MODE:
        save_torch_model(model, config.MODEL_SAVE_PATH, config.ARCHITECTURE_NAME)

    k_fold_run.finish()


def train_for_one_epoch(
    model: Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    device: str,
):
    """
    Train the model for one epoch.
    :param model: The model to train.
    :param train_loader: The data loader for training.
    :param validation_loader: The data loader for validation.
    :param loss_function: The loss function to use.
    :param optimizer: The optimizer to use.
    :param scheduler: The learning rate scheduler to use.
    :param device: The device to use.

    This function is responsible for training the model for one epoch.
    It will:
    - Run the training loop
    - Evaluate the model on the validation set
    - Log the results on wandb
    - Update the learning rate scheduler
    """
    train_loss, train_accuracy, train_f1 = train_epoch(
        model, train_loader, loss_function, optimizer, device, config.DEV_MODE
    )
    validation_loss, validation_accuracy, validation_f1 = evaluate_epoch(
        model, validation_loader, loss_function, device, config.DEV_MODE
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
            "current_lr": scheduler.get_last_lr()[-1],
        }
    )


if __name__ == "__main__":
    run_experiment()
