import torch
import torch.nn as nn
from config import LATENT_DIM, VOCAB_SIZE, EMBED_DIM


class TextHead(nn.Module):
    """Predicts words from latent vector using shared embeddings.

    Input:  [B, 128]
    Output: logits [B, 3000]
    """

    def __init__(self, latent_dim=LATENT_DIM, vocab_size=VOCAB_SIZE,
                 embed_dim=EMBED_DIM):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, latent):
        hidden = self.proj(latent)  # [B, 128]
        logits = hidden @ self.word_embeddings.weight.T  # [B, 3000]
        return logits

    def get_top_words(self, latent, vocab, k=5):
        logits = self.forward(latent)
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k, dim=-1)
        words = [[vocab[idx] for idx in row] for row in topk.indices.tolist()]
        confidences = topk.values.tolist()
        return words, confidences

    def get_embedding(self, word_indices):
        return self.word_embeddings(word_indices)
