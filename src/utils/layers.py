from torch.nn import BatchNorm2d, Conv2d, ReLU, Sequential


def double_convolution_layer(input_channels: int, output_channels: int):
    """
    Create a double convolution layer.
    :param input_channels: The number of input channels.
    :param output_channels: The number of output channels.
    :return: The double convolution layer.
    """
    return Sequential(
        Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        BatchNorm2d(output_channels),
        ReLU(),
        Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        BatchNorm2d(output_channels),
        ReLU(),
    )
