import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score


def evaluate_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    device: str,
    dev_mode=False,
):
    model.eval()

    running_loss = 0
    running_accuracy = 0
    running_f1 = 0
    count = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            count += 1

            images = images.to(device)
            predictions = model(images)
            labels = labels.to(device)

            loss = loss_function(predictions, labels)

            compute_accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
            compute_f1 = F1Score(task="multiclass", num_classes=4).to(device)
            accuracies = compute_accuracy(predictions, labels)
            f1_scores = compute_f1(predictions, labels)

            running_loss += loss.item()
            running_accuracy += accuracies
            running_f1 += f1_scores

            if dev_mode:
                print(f"Validation Loss: {loss.item()}")
                break

    epoch_loss = running_loss / count
    epoch_accuracy = running_accuracy / count
    epoch_f1 = running_f1 / count
    return epoch_loss, epoch_accuracy, epoch_f1
