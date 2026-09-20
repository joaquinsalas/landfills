import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DoubleConv import DoubleConv

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
