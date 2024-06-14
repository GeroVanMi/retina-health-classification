import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score


def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    optimizer: Optimizer,
    device: str,
    dev_mode=False,
):
    """
    Train the model on the given data loader.
    :param model: The model to train.
    :param data_loader: The data loader to train the model on.
    :param loss_function: The loss function to use.
    :param optimizer: The optimizer to use.
    :param device: The device to use.
    :param dev_mode: Whether to run in development mode.
    :return: The loss, accuracy, and F1 score for the model.
    """
    model.train()

    running_loss = 0
    running_accuracy = 0
    running_f1 = 0
    count = 0

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

        # What do these lines really do? This would be interesting to know.
        # Obviously I know that they apply the backpropagation, but what does that mean on a technical level?
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if dev_mode:
            print(f"Training Loss: {loss.item()}")
            break

    # Calculate running metrics
    epoch_loss = running_loss / count
    epoch_accuracy = running_accuracy / count
    epoch_f1 = running_f1 / count
    return epoch_loss, epoch_accuracy, epoch_f1
