import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from Configuration import Configuration


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
                    "id": patient_id,
                }
            else:
                patients[patient_id]["file2"] = file

    return pd.DataFrame(patients.values())


class EyesDataset(Dataset):
    def __init__(self, image_directory: Path):
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
        return len(self.patients)

    def __getitem__(self, index):
        patient = self.patients.iloc[index]

        image1 = self.image_transforms(Image.open(patient.file1))
        if patient.file2:
            image2 = self.image_transforms(Image.open(patient.file2))
            class2 = patient.disease_name
        else:
            image2 = torch.zeros((256, 256))
            class2 = "Empty"

        return (
            image1,
            image2,
            patient.disease_name,
            class2,
        )


if __name__ == "__main__":
    config = Configuration()
    dataset = EyesDataset(config.DATA_PATH, None)
    data_loader = DataLoader(dataset, batch_size=1)

    for entry in data_loader:
        print(entry)
