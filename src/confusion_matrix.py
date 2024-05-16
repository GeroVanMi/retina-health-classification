from pathlib import Path

import numpy as np
import pandas as pd
import torch
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import wandb
from Configuration import Configuration
from SimpleClassifier import SimpleClassifier
from utils.data import create_train_validation_loaders


def compute_confusion_matrix():
    config = Configuration()
    y_true = []
    y_pred = []
    _, validation_loader = create_train_validation_loaders(config.DATA_PATH, 1)

    api = wandb.Api()
    artifact = api.artifact(
        f"{config.ENTITY_NAME}/{config.PROJECT_NAME}/{config.ARCHITECTURE_NAME}:latest"
    )
    artifact_dir = Path(artifact.download())

    model_weights_path = artifact_dir.joinpath(f"U-NetEncoder_1715862877.pt")
    model = SimpleClassifier()
    model.load_state_dict(
        torch.load(
            model_weights_path,
            map_location=torch.device("cpu"),
        )
    )
    model.eval()

    class_indices = {
        "Cataract": 0,
        "Diabetic Retinopathy": 1,
        "Glaucoma": 2,
        "Normal": 3,
    }
    confusion_matrix_path = Path("confusion_matrix.npy")
    if not confusion_matrix_path.exists():
        with alive_bar(len(validation_loader)) as bar:
            with torch.inference_mode():
                for images, labels in validation_loader:
                    bar()
                    y_pred.append(np.argmax(model(images)))
                    y_true.append(labels[0])

        conf_matrix = confusion_matrix(y_true, y_pred)

        np.save(confusion_matrix_path, conf_matrix)
    else:
        conf_matrix = np.load(confusion_matrix_path)

    print(conf_matrix)

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(20, 14))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=list(class_indices.keys())
    )
    disp.plot(ax=ax)
    fig.savefig("confusion_matrix.png")


if __name__ == "__main__":
    compute_confusion_matrix()
