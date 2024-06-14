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
    """
    Evaluate the model on the given data loader.
    :param model: The model to evaluate.
    :param data_loader: The data loader to evaluate the model on.
    :param loss_function: The loss function to use.
    :param device: The device to use.
    :param dev_mode: Whether to run in development mode.
    :return: The loss, accuracy, and F1 score for the model.
    """
    model.eval()

    running_loss = 0
    running_accuracy = 0
    running_f1 = 0
    count = 0

    with torch.inference_mode():
        for images1, images2, labels1, labels2 in data_loader:
            count += 1

            images1 = images1.to(device)
            images2 = images2.to(device)
            predictions = model(images1, images2)

            labels1 = labels1.to(device)
            labels2 = labels2.to(device)

            # TODO: Incorporate the second label into the loss?
            loss = loss_function(predictions, labels1)

            compute_accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
            compute_f1 = F1Score(task="multiclass", num_classes=4).to(device)
            accuracies = compute_accuracy(predictions, labels1)
            f1_scores = compute_f1(predictions, labels1)

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
