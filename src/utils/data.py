from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

from utils.EyesDataset import EyesDataset


def stop_if_data_is_missing(data_path: Path):
    """
    Stop the program if the data is missing.
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
    folder_path: Path, batch_size=32, random_seed=0
) -> tuple[DataLoader, DataLoader]:
    data = EyesDataset(folder_path)
    """
    Create the training and validation data loaders.
    :param folder_path: The path to the folder containing the data.
    :param batch_size: The batch size to use.
    :param random_seed: The random seed to use.
    :return: The training and validation data loaders.
    """
    torch.manual_seed(random_seed)
    train_data, validation_data = random_split(data, (0.8, 0.2))
    print(f"Training data: {len(train_data)}, Validation data: {len(validation_data)}")

    train_sampler = RandomSampler(train_data)
    validation_sampler = SequentialSampler(validation_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(
        validation_data, batch_size=1, sampler=validation_sampler
    )

    return train_loader, validation_loader


if __name__ == "__main__":
    train_dataloader, validation_loader = create_train_validation_loaders(
        Path("../../data/Eyes")
    )
    batch, labels = next(iter(train_dataloader))
    print(labels)
