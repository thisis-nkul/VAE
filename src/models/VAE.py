import pytorch_lightning as pl
from .encoder import EncoderBlock
from .decoder import DecoderBlock
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics.functional as FM
"""


class VAE(pl.LightningModule):

    def __init__(self, latent_dim, encoder=EncoderBlock, decoder=DecoderBlock):
        super(VAE, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.latent_dim = latent_dim

    def forward(self, x):
        pass

    def reparameterize(self, logvar, mu):
        pass

    def _step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, *args, **kwargs):
        pass
