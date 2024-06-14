import re
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from Configuration import Configuration

config = Configuration()


def extract_patients_from_path(path: Path):
    """
    The images of patients eyes sometimes come in pairs (left & right).
    This function extracts all the pairs of images into a dictionary.
    {
        file1: PATH
        file2: PATH | None
        class: string
        id:    number
    }
    :param path: The path to the images.
    :return: The patients as a dictionary.
    """
    patients = {}
    disease_name = path.name
    files = path.glob("*")

    for file in files:
        match = re.search(r"(\d+)", file.name)
        if match is not None:
            patient_id = match.group(0)
            if patient_id not in patients:
                patients[patient_id] = {
                    "file1": file,
                    "file2": None,
                    "disease_name": disease_name,
                    "class_index": config.CLASS_INDICES[disease_name],
                    "id": patient_id,
                }
            else:
                patients[patient_id]["file2"] = file

    return pd.DataFrame(patients.values())


class EyesDataset(Dataset):
    """
    The dataset for the eyes.
    """
    def __init__(self, image_directory: Path):
        """
        Initialize the dataset.
        :param image_directory: The directory containing the images.
        """
        classes = image_directory.glob("*")

        patients_per_class = [extract_patients_from_path(path) for path in classes]
        self.patients = pd.concat(patients_per_class)
        self.image_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((256, 256)),
                # torchvision.transforms.CenterCrop(512),
                torchvision.transforms.ToTensor(),  # Convert the image to a pytorch tensor
            ]
        )

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.patients)

    def __getitem__(self, index):
        """
        Get an item from the dataset.
        :param index: The index of the item to get.
        :return: The item at the given index.
        """
        patient = self.patients.iloc[index]

        image1 = self.image_transforms(Image.open(patient.file1))
        if patient.file2:
            image2 = self.image_transforms(Image.open(patient.file2))
            class2 = patient.class_index
        else:
            image2 = torch.zeros((3, 256, 256))
            class2 = config.CLASS_INDICES["Empty"]

        return (
            image1,
            image2,
            patient.class_index,
            class2,
        )


if __name__ == "__main__":
    dataset = EyesDataset(config.DATA_PATH, None)
    data_loader = DataLoader(dataset, batch_size=1)

    for entry in data_loader:
        print(entry)
