from __future__ import annotations

import torch
import torch.nn as nn


class CNNFFN(nn.Module):
    def __init__(self, in_channels: int = 1, m: int = 50, conv_channels: int = 32, conv_depth: int = 2,
                 dense_width: int = 64, dense_depth: int = 3, dropout: float = 0.25):
        super().__init__()
        self.m = m
        self.conv_channels = conv_channels
        self.conv_depth = conv_depth
        self.dense_width = dense_width
        self.dense_depth = dense_depth
        layers = []
        c = in_channels
        for i in range(conv_depth):
            layers.append(nn.Conv2d(c, conv_channels, kernel_size=3, padding=1 if i == 0 else 0))
            layers.append(nn.ReLU())
            c = conv_channels
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)

        # Build a static head once: no module mutation in forward (required for torch.export/ONNX path).
        h = m
        w = m
        for i in range(conv_depth):
            if i > 0:
                h = h - 2
                w = w - 2
        h = h // 2
        w = w // 2
        flat_dim = conv_channels * h * w
        if flat_dim <= 0:
            raise ValueError(f"Invalid flattened dim ({flat_dim}) for m={m}, conv_depth={conv_depth}")

        layers = []
        in_dim = flat_dim + 2  # h and theta
        for _ in range(self.dense_depth):
            layers += [nn.Linear(in_dim, self.dense_width), nn.ReLU()]
            in_dim = self.dense_width
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x_img, x_h, x_theta):
        z = self.conv(x_img)
        z = z.flatten(1)
        z = torch.cat([z, x_h.unsqueeze(1), x_theta.unsqueeze(1)], dim=1)
        return self.head(z).squeeze(1)
