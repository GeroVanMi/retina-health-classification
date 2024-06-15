from torch.nn import Flatten, Linear, MaxPool2d, Module, ReLU, Sequential

from utils.layers import double_convolution_layer


class SingleEyeClassifier(Module):
    """
    A simple classifier model that takes an image as input and outputs a classification.

    The model consists of a feature extractor that extracts features from the input image.
    The output of the feature extractor is passed to a classifier that outputs the classification.

    The model consists of the following layers:
    - A feature extractor that extracts features from the input image.
    - A classifier that classifies the features.
    """

    def __init__(self):
        super().__init__()
        self.classifier = Sequential(
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
            Linear(8192, 4064),
            ReLU(),
            Linear(4064, 256),
            ReLU(),
            Linear(256, 4),
        )

    def forward(self, x):
        return self.classifier(x)
