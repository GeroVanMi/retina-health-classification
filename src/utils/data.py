import torch
import torchvision
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              random_split)


def create_train_validation_loaders(
    folder_path: str, batch_size=32, random_seed=0
) -> tuple[DataLoader, DataLoader]:
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.ToTensor(),  # Convert the image to a pytorch tensor
        ]
    )
    data = torchvision.datasets.ImageFolder(folder_path, transform=image_transform)

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
        "../../data/Eyes"
    )
    batch, labels = next(iter(train_dataloader))
    print(labels)
