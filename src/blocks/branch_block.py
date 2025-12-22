import torch.nn as nn

from src.layers.conv_block import ConvBlock
from src.layers.pooling import GlobalAvgPool

class BranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            GlobalAvgPool()
        )

    def forward(self, x):
        return self.branch(x)
