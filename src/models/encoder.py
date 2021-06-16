import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_features, out_features, k_size, nb_layers,
                 maxpool=True, batch_norm=True, padding=None, dropout_p=0.2):

        # nb_layers: number of conv layers to use
        # k_size: dimensions of the kernel to use

        super(ConvBlock, self).__init__()
        padding = (k_size - 1) // 2 if padding is None else padding

        self.conv_layers = nn.ModuleList([
                nn.Conv2d(in_features, out_features, k_size, padding=padding),
                nn.ReLU()
        ])

        for i in range(1, nb_layers):

            layer = nn.Conv2d(out_features, out_features,
                              k_size, padding=padding)

            self.conv_layers.append(layer)
            self.conv_layers.append(nn.ReLU())

            if (i+1) % 2 == 0:
                self.conv_layers.append(nn.Dropout(dropout_p))

        if batch_norm:
            self.conv_layers.append(nn.BatchNorm2d(out_features))

        if maxpool:
            self.conv_layers.append(nn.MaxPool2d(2, 2))

    def forward(self, x):
        for module in self.conv_layers:
            x = module(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channels=3):

        super(EncoderBlock, self).__init__()

        features = [in_channels, 32, 64, 128, 256, 1024]
        n_layers = [None, 4, 4, 4, 4, 2]
        k_sizes = [None, 3, 3, 3, 5, 5]
        # None: bcz we won't be having conv layers which output in_channels

        self.conv_modules = nn.ModuleList([])

        for i in range(1, len(features)):
            in_ = features[i-1]
            out_ = features[i]
            nb_layers = n_layers[i]
            k_size = k_sizes[i]

            module = ConvBlock(in_, out_, k_size, nb_layers)

            self.conv_modules.append(module)

    def forward(self, x):
        for module in self.conv_modules:
            x = module(x)
        return x

    def get_output_shape(self, C, H, W):
        shape = self(torch.rand(1, C, H, W)).shape[1:]
        return shape
