import torch
import torch.nn as nn
from config import LATENT_DIM, TEMPORAL_HIDDEN, NUM_TRANSFORMS, EMBED_DIM


class TemporalRNN(nn.Module):
    """Frame-to-frame animation predictor (Stage 3).

    Given current frame latent + transform type, predicts delta for next frame.
    Input:  current_latent [B, 128], hidden [B, 192], transform_idx [B, 1]
    Output: latent_delta [B, 128], new_hidden [B, 192]
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=TEMPORAL_HIDDEN,
                 num_transforms=NUM_TRANSFORMS, transform_dim=32):
        super().__init__()
        self.transform_embedding = nn.Embedding(num_transforms, transform_dim)
        self.input_proj = nn.Linear(latent_dim + transform_dim, latent_dim)
        self.gru_cell = nn.GRUCell(latent_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.hidden_dim = hidden_dim

    def forward(self, latent, hidden, transform_idx):
        transform_emb = self.transform_embedding(transform_idx.squeeze(-1))
        fused = torch.cat([latent, transform_emb], dim=-1)
        inp = self.input_proj(fused)

        if hidden is None:
            hidden = torch.zeros(latent.size(0), self.hidden_dim, device=latent.device)

        hidden = self.gru_cell(inp, hidden)
        delta = self.output_proj(hidden)
        return delta, hidden
