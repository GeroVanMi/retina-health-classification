import torch
from torch.nn import Flatten, Linear, MaxPool2d, Module, ReLU, Sequential

from utils.layers import double_convolution_layer


class DoubleEyeClassifier(Module):
    """
    A double classifier model that takes two images as input and outputs a classification.

    The model consists of two feature extractors that extract features from the input images.
    The output of the feature extractors is concatenated and passed to a classifier that outputs the classification.

    The model consists of the following layers:
    - A left feature extractor that extracts features from the left eye image.
    - A right feature extractor that extracts features from the right eye image.
    - A classifier that classifies the concatenated features.
    """

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
            MaxPool2d(2),
            double_convolution_layer(512, 1024),
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
            MaxPool2d(2),
            double_convolution_layer(512, 1024),
            Flatten(),
        )

        self.classifier = Sequential(
            Linear(8192, 2048),
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
        """
        Forward pass through the model.
        :param left: The left eye image.
        :param right: The right eye image.
        :return: The output of the model.
        """
        left_features = self.left_feature_extractor(left)
        right_features = self.right_feature_extractor(right)
        return self.classifier(torch.cat((left_features, right_features), axis=1))  # type: ignore (torch.cat DOES exist)
