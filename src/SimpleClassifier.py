from torch.nn import (Flatten, Linear, MaxPool2d, Module, ReLU, Sequential,
                      Softmax)

from utils.layers import convolution_layer


class SimpleClassifier(Module):
    def __init__(self):
        super().__init__()
        self.classifier = Sequential(
            convolution_layer(3, 16),
            MaxPool2d(2),
            convolution_layer(16, 32),
            MaxPool2d(2),
            convolution_layer(32, 64),
            MaxPool2d(2),
            convolution_layer(64, 128),
            MaxPool2d(2),
            convolution_layer(128, 256),
            MaxPool2d(2),
            convolution_layer(256, 512),
            Flatten(),
            Linear(25088, 512),
            ReLU(),
            Linear(512, 4),
        )

    def forward(self, x):
        return self.classifier(x)
