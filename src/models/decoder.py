import torch.nn as nn


class UpConvBlock(nn.Module):

    def __init__(self, in_features, out_features, k_size, nb_layers,
                 upsample=True, up_scale=2, batch_norm=True, dropout_p=0.2):

        super(UpConvBlock, self).__init__()

        padding = (k_size - 1) // 2

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

        if upsample:
            self.conv_layers.append(nn.Upsample(scale_factor=up_scale))

    def forward(self, x):
        for module in self.conv_layers:
            x = module(x)
        return x
        pass


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels=3):

        super(DecoderBlock, self).__init__()

        features = [in_channels, 512, 256, 128, 64, out_channels]
        n_layers = [None, 4, 4, 4, 4, 1]
        k_sizes = [None, 3, 3, 3, 5, 3]

        self.conv_modules = nn.ModuleList([])

        for i in range(1, len(features)):
            in_ = features[i-1]
            out_ = features[i]
            nb_layers = n_layers[i]
            k_size = k_sizes[i]

            module = UpConvBlock(in_, out_, k_size, nb_layers)

            self.conv_modules.append(module)

    def forward(self, x):
        for module in self.conv_modules:
            x = module(x)
        return x
