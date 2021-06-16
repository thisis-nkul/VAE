import pytorch_lightning as pl
import torch.nn as nn
from .encoder import EncoderBlock
from .decoder import DecoderBlock
import torch
import torch.optim as optim
import torch.nn.functional as F


class VAE(pl.LightningModule):

    def __init__(self, input_shape, latent_dim=256,
                 encoder=None, decoder=None):

        super(VAE, self).__init__()

        C, H, W = input_shape
        self.encoder = EncoderBlock(C) if encoder is None else encoder
        self.latent_dim = latent_dim

        enc_out_shape = self.encoder.get_output_shape(C, H, W)

        self.aux_C, self.aux_H, self.aux_W = enc_out_shape
        # the no. of output channels by encoder,
        # these will also be the number of channels we will be feed to decoder

        self.decoder = DecoderBlock(self.aux_C, C) if decoder is None\
            else decoder

        # if output is CxHxW from Encoder, we'll average out along H and W

        self.mu_layer = nn.Linear(self.aux_C, latent_dim)
        self.logvar_layer = nn.Linear(self.aux_C, latent_dim)

        self.decode_latent = nn.Linear(latent_dim,
                                       self.aux_C*self.aux_H*self.aux_W)
        # the output of this will be reshaped to aux_C x aux_H x aux_W

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

        # taking mean along H and W, similar to Global Average Pooling
        x = x.mean([2, 3])
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        return self.reparameterize(mu, logvar), mu, logvar

    def decode(self, x):
        x = self.decode_latent(x)
        x = x.view(-1, self.aux_C, self.aux_H, self.aux_W)
        x = self.decoder(x)

        return x

    def _step(self, batch):
        inp, _ = batch
        x, mu, logvar = self(inp)

        loss = self.get_loss(inp, x, mu, logvar)

        return loss

    def get_loss(self, inp, recon, mu, logvar):

        kld_loss = torch.mean(-0.5*torch.sum(1 + logvar - mu ** 2
                                             - logvar.exp(), dim=1),
                              dim=0)

        recon_loss = F.mse_loss(inp, recon)

        loss = recon_loss + kld_loss

        self.log('recon_loss', recon_loss)
        self.log('kld_loss', kld_loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=0.0001)
        return opt
