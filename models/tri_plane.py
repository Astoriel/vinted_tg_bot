"""Tri-Plane 3D Representation Module.

Represents a 3D scene with three orthogonal feature planes (XY, XZ, YZ).
Supports viewpoint conditioning for rendering from different angles.

Much more efficient than a full 3D volume:
  Full volume: [C, D, H, W] = 64 × 32 × 32 × 32 = 2M values
  Tri-plane:   3 × [C, H, W] = 3 × 64 × 32 × 32 = 196K values (10× smaller)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    ENCODER_OUT_CHANNELS, ENCODER_SPATIAL,
    V2_TRIPLANE_RES, V2_TRIPLANE_FEAT, V2_LATENT_DIM,
)


class ViewpointEncoder(nn.Module):
    """Encodes camera viewpoint (azimuth, elevation, distance) into embedding."""

    def __init__(self, out_dim=128):
        super().__init__()
        # Input: (azimuth, elevation, distance) — 3 values
        # Use sinusoidal positional encoding for angles
        self.freq_bands = 8  # 8 frequency bands
        input_dim = 3 * (1 + 2 * self.freq_bands)  # raw + sin/cos per freq
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, viewpoint):
        """viewpoint: [B, 3] — (azimuth_rad, elevation_rad, distance)"""
        # Sinusoidal encoding
        freqs = 2.0 ** torch.arange(self.freq_bands, device=viewpoint.device).float()
        encoded = [viewpoint]
        for freq in freqs:
            encoded.append(torch.sin(viewpoint * freq))
            encoded.append(torch.cos(viewpoint * freq))
        encoded = torch.cat(encoded, dim=-1)  # [B, 3*(1+2*8)] = [B, 51]
        return self.mlp(encoded)  # [B, 128]


class TriPlaneGenerator(nn.Module):
    """Generates tri-plane features from 2D image features.

    Input:  image features [B, C, H, W] from encoder
    Output: three planes [B, 3, feat_dim, res, res]
    """

    def __init__(self, in_channels=ENCODER_OUT_CHANNELS, in_spatial=ENCODER_SPATIAL,
                 feat_dim=V2_TRIPLANE_FEAT, plane_res=V2_TRIPLANE_RES):
        super().__init__()
        self.feat_dim = feat_dim
        self.plane_res = plane_res

        # Compress channel dim
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, 128, 1),
            nn.GELU(),
        )

        # Upscale from 7×7 to plane_res×plane_res (32×32)
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1),  # 7→14
            nn.GELU(),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),  # 14→28
            nn.GELU(),
            nn.Conv2d(64, feat_dim * 3, 5, padding=2),  # 28→28, channels=192
        )

        # Adaptive pool to exact resolution
        self.adapt = nn.AdaptiveAvgPool2d(plane_res)

    def forward(self, image_features):
        """
        image_features: [B, 576, 7, 7]
        Returns: [B, 3, feat_dim, res, res]
        """
        B = image_features.shape[0]
        x = self.compress(image_features)    # [B, 128, 7, 7]
        x = self.upscale(x)                  # [B, 192, 28, 28]
        x = self.adapt(x)                    # [B, 192, 32, 32]
        # Reshape to 3 planes
        x = x.view(B, 3, self.feat_dim, self.plane_res, self.plane_res)
        return x


class TriPlaneProjector(nn.Module):
    """Projects tri-plane features to 2D from a given viewpoint.

    Samples from the three planes based on viewpoint, producing a
    2D feature map that can be decoded into ASCII art.
    """

    def __init__(self, feat_dim=V2_TRIPLANE_FEAT, view_dim=128,
                 out_dim=V2_TRIPLANE_FEAT):
        super().__init__()
        # Viewpoint modulation: generates per-plane attention weights
        self.view_to_weights = nn.Sequential(
            nn.Linear(view_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # weight per plane
        )

        # Viewpoint-dependent spatial transform
        self.view_to_offset = nn.Sequential(
            nn.Linear(view_dim, 64),
            nn.GELU(),
            nn.Linear(64, 6),  # 2D affine params for grid sampling
        )

        # Fuse planes after viewpoint-dependent selection
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )

    def forward(self, planes, view_embed):
        """
        planes: [B, 3, C, H, W]
        view_embed: [B, 128]
        Returns: [B, C, H, W] projected 2D features
        """
        B, _, C, H, W = planes.shape

        # Viewpoint-dependent plane weighting
        weights = F.softmax(self.view_to_weights(view_embed), dim=-1)  # [B, 3]
        weights = weights.view(B, 3, 1, 1, 1)

        # Weighted sum of planes
        projected = (planes * weights).sum(dim=1)  # [B, C, H, W]

        # Viewpoint-dependent spatial shift (affine transform)
        affine = self.view_to_offset(view_embed)  # [B, 6]
        # Construct 2×3 affine matrix
        theta = torch.zeros(B, 2, 3, device=planes.device)
        theta[:, 0, 0] = 1.0 + affine[:, 0] * 0.2  # scale x
        theta[:, 1, 1] = 1.0 + affine[:, 1] * 0.2  # scale y
        theta[:, 0, 1] = affine[:, 2] * 0.3          # shear
        theta[:, 1, 0] = affine[:, 3] * 0.3          # shear
        theta[:, 0, 2] = affine[:, 4] * 0.5          # translate x
        theta[:, 1, 2] = affine[:, 5] * 0.5          # translate y

        grid = F.affine_grid(theta, projected.shape, align_corners=False)
        projected = F.grid_sample(projected, grid, align_corners=False,
                                  mode="bilinear", padding_mode="zeros")

        projected = self.fuse(projected)
        return projected


class TriPlane3D(nn.Module):
    """Complete 3D-aware module: image features → viewpoint-conditioned 2D features.

    Composes: TriPlaneGenerator + ViewpointEncoder + TriPlaneProjector
    """

    def __init__(self):
        super().__init__()
        self.view_encoder = ViewpointEncoder(out_dim=128)
        self.plane_generator = TriPlaneGenerator()
        self.projector = TriPlaneProjector()

        # Also produce a global latent for text decoder
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(V2_TRIPLANE_FEAT, V2_LATENT_DIM),
        )

    def forward(self, image_features, viewpoint=None):
        """
        image_features: [B, 576, 7, 7] from encoder
        viewpoint: [B, 3] — (azimuth, elevation, distance) or None for default

        Returns:
            projected_2d: [B, 64, 32, 32] — viewpoint-dependent features
            latent: [B, 256] — global latent for text decoder
            planes: [B, 3, 64, 32, 32] — stored for multi-view rendering
        """
        B = image_features.shape[0]
        dev = image_features.device

        if viewpoint is None:
            # Default: front view
            viewpoint = torch.tensor([[0.0, 0.3, 2.0]], device=dev).expand(B, -1)

        planes = self.plane_generator(image_features)      # [B, 3, 64, 32, 32]
        view_embed = self.view_encoder(viewpoint)           # [B, 128]
        projected = self.projector(planes, view_embed)      # [B, 64, 32, 32]
        latent = self.to_latent(projected)                  # [B, 256]

        return projected, latent, planes

    def render_from_viewpoint(self, planes, viewpoint):
        """Render from stored planes with a new viewpoint (for animation).

        planes: [B, 3, 64, 32, 32] — previously generated
        viewpoint: [B, 3]
        Returns: projected_2d [B, 64, 32, 32]
        """
        view_embed = self.view_encoder(viewpoint)
        projected = self.projector(planes, view_embed)
        return projected
