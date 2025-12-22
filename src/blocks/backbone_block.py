import torch.nn as nn

from src.layers.conv_block import ConvBlock
from src.layers.pooling import MaxPool

class BackboneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()

        layers = [
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        ]

        if pool:
            layers.append(MaxPool())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
