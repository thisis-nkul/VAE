import torch
import torch.nn as nn


class EncoderBlock(nn.Module):

    def __init__(self, in_channels=3):

        self(EncoderBlock, self).__init__()
        pass

    def forward(self, x):
        pass

    def get_output_shape(self, C, H, W):

        dummy_inp = torch.rand(1, C, H, W)
        shape = self(dummy_inp).shape[1:]
        self.zero_grad()
        del dummy_inp
        return shape
