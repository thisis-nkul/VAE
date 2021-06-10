import pytorch_lightning as pl
import torch.nn as nn
from .encoder import EncoderBlock
from .decoder import DecoderBlock
import torch
"""
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics.functional as FM
"""


class VAE(pl.LightningModule):

    def __init__(self, input_shape, latent_dim=256,
                 encoder=None, decoder=None):

        super(VAE, self).__init__()

        C, H, W = input_shape
        self.encoder = encoder if encoder is None else EncoderBlock(C)
        self.latent_dim = latent_dim

        enc_out_shape = self.decoder.get_output_shape(C, H, W)

        self.aux_channels = enc_out_shape[0]
        # the no. of output channels by encoder,
        # these will also be the number of channels we will be feed to decoder

        self.decoder = decoder if decoder is None else\
            DecoderBlock(self.aux_channels)

        # if output is CxHxW, we'll average out along H and W

        self.mu_layer = nn.Linear(enc_out_shape[0], latent_dim)
        self.logvar_layer = nn.Linear(enc_out_shape[0], latent_dim)

        self.decode_latent = nn.Linear(latent_dim, enc_out_shape[0]*4*4)
        # this will be reshaped to Cx4x4

    def forward(self, x):
        x, mu, logvar = self.encode(x)
        # z
        x = self.decode(x)

        return x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        return self.reparameterize(mu, logvar), mu, logvar

    def decode(self, x):
        x = self.decode_latent(x)
        x = x.view(-1, self.aux_channels, 4, 4)
        x = self.decoder(x)

        return x

    def _step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, *args, **kwargs):
        pass
