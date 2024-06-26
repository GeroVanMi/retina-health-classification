from pathlib import Path

import torch
from torch.nn import DataParallel

import wandb


def initialize_model(ModelClass, device):
    """
    Creates and returns an instance of the model class.
    Except if there are multiple GPUs available, then the model instance
    is wrapped in an instance of DataParallel.
    """
    model = ModelClass()

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    model.to(device)
    return model


def save_torch_model(model: torch.nn.Module, path: Path, model_name: str):
    """
    Save the model to the given path and log it as an artifact on wandb.
    """
    torch.save(model.state_dict(), path)
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)
