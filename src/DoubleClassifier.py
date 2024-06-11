import torch
from torch.nn import Flatten, Linear, MaxPool2d, Module, ReLU, Sequential

from utils.layers import double_convolution_layer


class DoubleClassifier(Module):
    def __init__(self):
        super().__init__()
        self.left_feature_extractor = Sequential(
            double_convolution_layer(3, 16),
            MaxPool2d(4),
            double_convolution_layer(16, 32),
            MaxPool2d(4),
            double_convolution_layer(32, 64),
            MaxPool2d(2),
            double_convolution_layer(64, 128),
            MaxPool2d(2),
            double_convolution_layer(128, 256),
            MaxPool2d(2),
            double_convolution_layer(256, 512),
            Flatten(),
        )

        self.right_feature_extractor = Sequential(
            double_convolution_layer(3, 16),
            MaxPool2d(4),
            double_convolution_layer(16, 32),
            MaxPool2d(4),
            double_convolution_layer(32, 64),
            MaxPool2d(2),
            double_convolution_layer(64, 128),
            MaxPool2d(2),
            double_convolution_layer(128, 256),
            MaxPool2d(2),
            double_convolution_layer(256, 512),
            Flatten(),
        )

        self.classifier = Sequential(
            Linear(4096, 2048),
            ReLU(),
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 4),
        )

    def forward(self, left, right):
        left_features = self.left_feature_extractor(left)
        right_features = self.right_feature_extractor(right)
        return self.classifier(torch.cat((left_features, right_features), axis=1))
