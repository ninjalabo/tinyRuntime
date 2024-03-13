# Don't edit this file! This was automatically generated from "model.ipynb".

import torch
from torch import nn
import numpy as np

# one block with skip connection
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(Block, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_connection = nn.Identity()
        
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.block(x)
        y += self.skip_connection(x)
        y = self.act(y)
        return y

# group of blocks
class GroupOfBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride=1):
        super(GroupOfBlocks, self).__init__()

        first_block = Block(in_channels, out_channels, stride)
        other_blocks = [Block(out_channels, out_channels) for _ in range(1, n_blocks)]
        self.group = nn.Sequential(first_block, *other_blocks)

    def forward(self, x):
        return self.group(x)

class ResNetImagenette(nn.Module):
  def __init__(self, n_blocks=2, n_channels=64, n_class=10):
    super(ResNetImagenette, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=7, stride=1, padding=2, bias=False)
    self.bn1 = nn.BatchNorm2d(n_channels)
    self.act = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.group1 = GroupOfBlocks(n_channels, n_channels, n_blocks)
    self.group2 = GroupOfBlocks(n_channels, 2*n_channels, n_blocks, stride=2)
    self.group3 = GroupOfBlocks(2*n_channels, 4*n_channels, n_blocks, stride=2)
    self.group4 = GroupOfBlocks(4*n_channels, 8*n_channels, n_blocks, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc = nn.Linear(8*n_channels, n_class)

    # Initialize weights
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

  def forward(self, x):

    # first conv layer
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act(x)
    x = self.maxpool(x)

    # residual blocks
    x = self.group1(x)
    x = self.group2(x)
    x = self.group3(x)
    x = self.group4(x)

    # final
    x = self.avgpool(x)
    x = x.view(-1, self.fc.in_features)
    x = self.fc(x)

    return x
