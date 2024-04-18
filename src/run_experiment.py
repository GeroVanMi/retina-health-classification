import time

import torch
from torch.nn import CrossEntropyLoss

import wandb
from pipeline.evaluate import evaluate_epoch
from pipeline.train import train_epoch
from SimpleClassifier import SimpleClassifier
from utils.data import create_train_test_loaders
from utils.model import initialize_model, save_torch_model

DATA_PATH = "../data/Eyes/"

NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-5

EXPERIMENT_NAME = "Larger Images"
PROJECT_NAME = "retina-health-classification"
ENTITY_NAME = "gerovanmi"
ARCHITECTURE_NAME = "U-NetEncoder"  # Must not contain special characters (except "-")
DATASET_NAME = "Medical Scan Classification Dataset"

MODEL_SAVE_PATH = f"../models/{ARCHITECTURE_NAME}_{int(time.time())}.pt"
# In dev mode we only train 3 images for 3 epochs
DEV_MODE = False

if DEV_MODE:
    print("RUNNING IN DEVELOPER TESTING MODE. THIS WILL NOT TRAIN THE MODEL PROPERLY.")
    print("To train the model, set DEV_MODE = False in run_experiment.py!")
    BATCH_SIZE = 3
    NUMBER_OF_EPOCHS = 3
    EXPERIMENT_NAME = f"DEV {EXPERIMENT_NAME}"


def run_experiment():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Run: {EXPERIMENT_NAME}")
    print(f"Training on {device} device!")

    number_of_gpus = torch.cuda.device_count()
    batch_size = BATCH_SIZE
    if number_of_gpus > 1:
        print("Using ", number_of_gpus, "GPUs.")
        batch_size = 512
    input("Confirm with Enter or cancel with Ctrl-C:")

    wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        name=EXPERIMENT_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": ARCHITECTURE_NAME,
            "batch_size": batch_size,
            "dataset": DATASET_NAME,
            "epochs": NUMBER_OF_EPOCHS,
            "dev_mode": DEV_MODE,
        },
    )
    train_loader, test_loader = create_train_test_loaders(DATA_PATH, batch_size)
    model = initialize_model(SimpleClassifier, device)

    loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"#### EPOCH {epoch} ####")
        train_loss, train_accuracy, train_f1 = train_epoch(
            model, train_loader, loss_function, optimizer, device, DEV_MODE
        )
        test_loss, test_accuracy, test_f1 = evaluate_epoch(
            model, test_loader, loss_function, device, DEV_MODE
        )

        wandb.log(
            {
                "Training Loss": train_loss,
                "Training Accuracy": train_accuracy,
                "Training F1-Score": train_f1,
                "Testing Loss": test_loss,
                "Testing Accuracy": test_accuracy,
                "Testing F1-Score": test_f1,
            }
        )

    if not DEV_MODE:
        save_torch_model(model, MODEL_SAVE_PATH, ARCHITECTURE_NAME)


if __name__ == "__main__":
    run_experiment()
