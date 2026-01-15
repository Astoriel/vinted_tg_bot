import torch
import torch.nn as nn
from config import EMBED_DIM, LATENT_DIM, BRIDGE_MAX_WORDS, BRIDGE_HIDDEN


class TextToVisualBridge(nn.Module):
    """Maps word embeddings into visual latent space.

    Input:  [B, num_words, 32] word embeddings (2-3 words)
    Output: [B, 128] latent vector
    """

    def __init__(self, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
                 max_words=BRIDGE_MAX_WORDS, hidden_dim=BRIDGE_HIDDEN):
        super().__init__()
        self.max_words = max_words
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * max_words, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, word_embeddings):
        B, N, D = word_embeddings.shape
        if N < self.max_words:
            pad = torch.zeros(B, self.max_words - N, D, device=word_embeddings.device)
            word_embeddings = torch.cat([word_embeddings, pad], dim=1)
        elif N > self.max_words:
            word_embeddings = word_embeddings[:, :self.max_words, :]

        flat = word_embeddings.reshape(B, -1)  # [B, 96]
        return self.mlp(flat)  # [B, 128]
