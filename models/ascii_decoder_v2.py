"""Viewpoint-Conditioned ASCII Decoder v2.

Takes 2D projected features from tri-plane module and generates ASCII art.
Can produce multiple frames from different viewpoints for animation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import V2_TRIPLANE_FEAT, V2_ASCII_GRID, NUM_ASCII_CHARS


class ASCIIDecoderV2(nn.Module):
    """Generates ASCII art from viewpoint-projected 2D features.

    Input:  projected features [B, 64, 32, 32]
    Output: ASCII logits [B, num_chars, grid_size, grid_size]
    """

    def __init__(self, in_channels=V2_TRIPLANE_FEAT,
                 grid_size=V2_ASCII_GRID, num_chars=NUM_ASCII_CHARS):
        super().__init__()
        self.grid_size = grid_size

        self.decoder = nn.Sequential(
            # 32×32 → 48×48 (or keep 32 if grid_size=32)
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, num_chars, 1),  # per-cell logits over ASCII palette
        )

        # Adaptive to target grid size
        self.adapt = nn.AdaptiveAvgPool2d(grid_size) if grid_size != 32 else nn.Identity()

    def forward(self, projected_features):
        """
        projected_features: [B, 64, 32, 32]
        Returns: logits [B, num_chars, grid_size, grid_size]
        """
        x = self.decoder(projected_features)
        x = self.adapt(x)
        return x

    def decode_to_indices(self, projected_features):
        logits = self.forward(projected_features)
        return logits.argmax(dim=1)  # [B, H, W]
