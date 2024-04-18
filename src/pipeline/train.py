import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    optimizer: Optimizer,
    device: str,
    dev_mode=False,
):
    model.train()
    total_data_length = len(data_loader)
    epoch_loss = []
    # TODO: Adjust this to the classification task!
    for batch_index, (image, label) in enumerate(data_loader):

        image = torch.Tensor.type(image, dtype=torch.float32)
        image = image.to(device)
        prediction = model(image)

        loss = loss_function(prediction, label)
        epoch_loss.append(loss.item())

        # What do these lines really do? This would be interesting to know.
        # Obviously I know that they apply the backpropagation, but what does that mean on a technical level?
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"{batch_index + 1}/{total_data_length} Training Loss: {loss.item()}")

        if dev_mode:
            print(f"Training Loss: {loss.item()}")
            break

    return np.mean(epoch_loss)
