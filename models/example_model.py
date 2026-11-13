"""A compact encoder-decoder for checking segmentation experiments."""

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )


class ExampleModel(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=16):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Model channel counts must be positive")
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.decoder2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 3, base_channels)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, image):
        if min(image.shape[-2:]) < 4:
            raise ValueError("ExampleModel requires image height and width >= 4")
        skip1 = self.encoder1(image)
        skip2 = self.encoder2(self.pool(skip1))
        features = self.bottleneck(self.pool(skip2))
        features = F.interpolate(features, size=skip2.shape[-2:], mode="nearest")
        features = self.decoder2(torch.cat((features, skip2), dim=1))
        features = F.interpolate(features, size=skip1.shape[-2:], mode="nearest")
        features = self.decoder1(torch.cat((features, skip1), dim=1))
        return self.classifier(features)
