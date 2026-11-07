"""Dual-branch CNN with Sobel edge extraction and an edge decoder."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class EdgeTextureCNN(nn.Module):
    """CNN combining RGB texture features and Sobel-derived shape features."""

    classes = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, dropout=0.5):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.texture1 = ConvBlock(3, 64)
        self.texture2 = ConvBlock(64, 128)
        self.texture3 = ConvBlock(128, 256)
        self.shape1 = ConvBlock(1, 64)
        self.shape2 = ConvBlock(64, 128)
        self.shape3 = ConvBlock(128, 256)
        self.decoder1 = nn.Conv2d(128, 64, 3, padding=1)
        self.decoder2 = nn.Conv2d(128, 32, 3, padding=1)
        self.decoder3 = nn.Conv2d(32, 1, 3, padding=1)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.sqrt(edge_x.square() + edge_y.square() + 1e-12)

        t1 = F.max_pool2d(self.texture1(x), 2)
        t2 = F.max_pool2d(self.texture2(t1), 2)
        t3 = F.max_pool2d(self.texture3(t2), 2)

        s1 = F.max_pool2d(self.shape1(edges), 2)
        s2 = self.shape2(s1)
        s_mid = F.max_pool2d(s2, 2)
        s3 = F.max_pool2d(self.shape3(s_mid), 2)

        decoded = F.relu(self.decoder1(s_mid))
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        decoded = torch.cat((decoded, s1), dim=1)
        decoded = F.relu(self.decoder2(decoded))
        reconstructed_edges = self.decoder3(decoded)
        reconstructed_edges = F.interpolate(reconstructed_edges, size=(32, 32), mode="bilinear", align_corners=False)

        features = torch.cat((t3.flatten(1), 0.5 * s3.flatten(1)), dim=1)
        logits = self.classifier(features)
        return logits, edges, reconstructed_edges
