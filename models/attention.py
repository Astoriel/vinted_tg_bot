import torch
import torch.nn as nn
from config import ENCODER_OUT_CHANNELS, ATTENTION_HEAD_DIM, LATENT_DIM


class SpatialSelfAttention(nn.Module):
    """Single-head self-attention over spatial patches.

    Input:  [B, 576, 7, 7] (from encoder)
    Output: [B, 128] (pooled latent vector)
    """

    def __init__(self, in_channels=ENCODER_OUT_CHANNELS, head_dim=ATTENTION_HEAD_DIM,
                 latent_dim=LATENT_DIM):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(in_channels, head_dim)
        self.k_proj = nn.Linear(in_channels, head_dim)
        self.v_proj = nn.Linear(in_channels, head_dim)
        self.out_proj = nn.Linear(head_dim, in_channels)
        self.pool_proj = nn.Linear(in_channels, latent_dim)
        self.scale = head_dim ** -0.5

    def forward(self, feature_maps):
        B, C, H, W = feature_maps.shape
        # [B, H*W, C] = [B, 49, 576]
        x = feature_maps.flatten(2).transpose(1, 2)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out = attn @ V
        out = self.out_proj(out)
        out = out + x  # residual

        pooled = out.mean(dim=1)  # [B, 576]
        return self.pool_proj(pooled)  # [B, 128]
