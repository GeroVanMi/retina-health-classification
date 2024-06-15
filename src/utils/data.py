from pathlib import Path

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


def stop_if_data_is_missing(data_path: Path):
    """
    Stop the program and informs the user if the data is missing.
    :param data_path: The path to the data.
    """
    if not data_path.exists():
        print("ERROR: Could not find the retinal imaging data!")
        print("Download the dataset from ")
        print(
            "https://www.kaggle.com/datasets/arjunbasandrai/medical-scan-classification-dataset"
        )
        print("and place it in ./data/Eyes!")
        exit(1)


def create_train_validation_loaders(
    data,
    train_indices,
    validation_indices,
    batch_size=32,
    random_seed=0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create the training and validation data loaders.
    :param folder_path: The path to the folder containing the data.
    :param batch_size: The batch size to use.
    :param random_seed: The random seed to use.
    :return: The training and validation data loaders.
    """

    torch.manual_seed(random_seed)

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(data, batch_size=1, sampler=validation_sampler)

    return train_loader, validation_loader
