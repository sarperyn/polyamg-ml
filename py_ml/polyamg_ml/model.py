from __future__ import annotations

import torch
import torch.nn as nn


class CNNFFN(nn.Module):
    def __init__(self, in_channels: int = 1, m: int = 50, conv_channels: int = 32, conv_depth: int = 2,
                 dense_width: int = 64, dense_depth: int = 2, dropout: float = 0.25,
                 conv1_channels: int | None = None, conv1_depth: int | None = None,
                 conv1_dropout: float | None = None, conv2_channels: int | None = None,
                 conv2_depth: int = 0, conv2_dropout: float = 0.0,
                 cnn_out_width: int = 128):
        super().__init__()
        self.m = m
        self.conv1_channels = conv_channels if conv1_channels is None else conv1_channels
        self.conv1_depth = conv_depth if conv1_depth is None else conv1_depth
        self.conv1_dropout = dropout if conv1_dropout is None else conv1_dropout
        self.conv2_channels = conv2_channels
        self.conv2_depth = conv2_depth
        self.conv2_dropout = conv2_dropout
        self.cnn_out_width = cnn_out_width
        self.dense_width = dense_width
        self.dense_depth = dense_depth

        layers: list[nn.Module] = []
        c = in_channels

        def add_conv_block(width: int, depth: int, drop: float) -> None:
            nonlocal c
            if depth <= 0:
                return
            for i in range(depth):
                layers.append(nn.Conv2d(c, width, kernel_size=3, padding=1 if i == 0 else 0))
                layers.append(nn.ReLU())
                c = width
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout(drop))

        add_conv_block(self.conv1_channels, self.conv1_depth, self.conv1_dropout)
        if self.conv2_channels is not None and self.conv2_depth > 0:
            add_conv_block(self.conv2_channels, self.conv2_depth, self.conv2_dropout)
        self.conv = nn.Sequential(*layers)

        # Build a static head once: no module mutation in forward (required for torch.export/ONNX path).
        h = m
        w = m
        for block_depth in (self.conv1_depth, self.conv2_depth if self.conv2_channels is not None else 0):
            if block_depth <= 0:
                continue
            for i in range(block_depth):
                if i == 0:
                    continue
                h = h - 2
                w = w - 2
            h = h // 2
            w = w // 2
        final_channels = self.conv2_channels if self.conv2_channels is not None and self.conv2_depth > 0 else self.conv1_channels
        flat_dim = final_channels * h * w
        if flat_dim <= 0:
            raise ValueError(f"Invalid flattened dim ({flat_dim}) for m={m}")

        self.cnn_head = nn.Sequential(
            nn.Linear(flat_dim, cnn_out_width),
            nn.ReLU(),
        )

        layers = []
        in_dim = cnn_out_width + 2  # -log2(h) and theta
        for _ in range(self.dense_depth):
            layers += [nn.Linear(in_dim, self.dense_width), nn.ReLU()]
            in_dim = self.dense_width
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x_img, x_h, x_theta):
        z = self.conv(x_img)
        z = z.flatten(1)
        z = self.cnn_head(z)
        z = torch.cat([z, x_h.unsqueeze(1), x_theta.unsqueeze(1)], dim=1)
        return self.head(z).squeeze(1)
