import torch
import torchvision
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              random_split)


def create_train_test_loaders(
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
    train_data, test_data = random_split(data, (0.8, 0.2))

    train_sampler = RandomSampler(train_data)
    test_sampler = SequentialSampler(test_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=1, sampler=test_sampler)

    return train_loader, test_loader


if __name__ == "__main__":
    train_dataloader, test_loader = create_train_test_loaders("../../data/Eyes")
    batch, labels = next(iter(train_dataloader))
    print(labels)
