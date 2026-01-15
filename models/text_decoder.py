"""Mini-GPT Text Decoder: autoregressive phrase generation with doubt/confirm.

Generates text token by token, attending to image features.
Uses special tokens [DOUBT], [CONFIRM], [QUESTION] to express uncertainty.

Architecture: 4-layer transformer decoder with cross-attention to visual features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    V2_VOCAB_SIZE, V2_TEXT_DIM, V2_TEXT_HEADS, V2_TEXT_LAYERS,
    V2_TEXT_FFN, V2_MAX_TEXT_LEN, VOCAB_SIZE, V2_SPECIAL_TOKENS,
)

# Token IDs for special tokens (after vocab)
BOS_ID = VOCAB_SIZE + 0
EOS_ID = VOCAB_SIZE + 1
DOUBT_ID = VOCAB_SIZE + 2
CONFIRM_ID = VOCAB_SIZE + 3
QUESTION_ID = VOCAB_SIZE + 4


class CrossAttentionLayer(nn.Module):
    """Single transformer decoder layer with self-attn + cross-attn + FFN."""

    def __init__(self, dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, memory, causal_mask=None):
        """
        x: [B, T, dim] — token sequence
        memory: [B, S, dim] — visual features to attend to
        causal_mask: [T, T] — upper triangular mask for autoregressive
        """
        # Self-attention (causal)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask)
        x = x + h

        # Cross-attention to visual features
        h = self.norm2(x)
        h, _ = self.cross_attn(h, memory, memory)
        x = x + h

        # FFN
        h = self.norm3(x)
        x = x + self.ffn(h)

        return x


class MiniGPTDecoder(nn.Module):
    """Autoregressive text decoder that generates phrases with doubt/confirm.

    Input:  visual features [B, S, dim] from image encoder
    Output: token sequence with words + special tokens

    Example output: "[BOS] mountain [DOUBT] ocean [QUESTION] cliff [CONFIRM] cliff ocean [EOS]"
    Decoded: "I see mountain... but maybe ocean? Is it cliff? Yes — cliff above ocean."
    """

    def __init__(self, vocab_size=V2_VOCAB_SIZE, dim=V2_TEXT_DIM,
                 n_heads=V2_TEXT_HEADS, n_layers=V2_TEXT_LAYERS,
                 ffn_dim=V2_TEXT_FFN, max_len=V2_MAX_TEXT_LEN):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)

        # Project visual features to text dim
        self.visual_proj = nn.Linear(dim, dim)

        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Tie input/output embeddings
        self.head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, T, device):
        """Upper triangular mask: position i can only attend to positions <= i."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    def forward(self, tokens, visual_features):
        """Teacher-forced forward pass (for training).

        tokens: [B, T] token indices (with BOS prepended)
        visual_features: [B, S, dim] encoded image features
        Returns: logits [B, T, vocab_size]
        """
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)

        x = self.token_embed(tokens) + self.pos_embed(positions)
        memory = self.visual_proj(visual_features)
        causal_mask = self._causal_mask(T, tokens.device)

        for layer in self.layers:
            x = layer(x, memory, causal_mask)

        x = self.norm_out(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, visual_features, max_len=None, temperature=0.8,
                 top_k=50):
        """Autoregressive generation — token by token.

        visual_features: [1, S, dim]
        Returns: list of token IDs, list of token strings
        """
        max_len = max_len or self.max_len
        device = visual_features.device
        memory = self.visual_proj(visual_features)

        tokens = [BOS_ID]
        for step in range(max_len - 1):
            idx = torch.tensor([tokens], device=device)  # [1, t]
            positions = torch.arange(len(tokens), device=device).unsqueeze(0)
            x = self.token_embed(idx) + self.pos_embed(positions)
            causal_mask = self._causal_mask(len(tokens), device)

            for layer in self.layers:
                x = layer(x, memory, causal_mask)

            x = self.norm_out(x)
            logits = self.head(x[:, -1, :]) / temperature  # [1, vocab]

            # Top-k sampling
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, top_k)
                probs = F.softmax(topk_vals, dim=-1)
                sampled = torch.multinomial(probs, 1)
                next_token = topk_idx[0, sampled[0, 0]].item()
            else:
                next_token = logits.argmax(dim=-1).item()

            tokens.append(next_token)

            if next_token == EOS_ID:
                break

        return tokens

    def tokens_to_text(self, token_ids, vocab):
        """Convert token IDs to human-readable text with formatting."""
        parts = []
        for tid in token_ids:
            if tid == BOS_ID:
                continue
            elif tid == EOS_ID:
                break
            elif tid == DOUBT_ID:
                parts.append("[DOUBT]")
            elif tid == CONFIRM_ID:
                parts.append("[CONFIRM]")
            elif tid == QUESTION_ID:
                parts.append("[QUESTION]")
            elif 0 <= tid < len(vocab):
                parts.append(vocab[tid])

        # Format into natural-sounding phrase
        text = ""
        for i, p in enumerate(parts):
            if p == "[DOUBT]":
                text += "... but maybe "
            elif p == "[CONFIRM]":
                text += ". Yes — "
            elif p == "[QUESTION]":
                text += "? Could it be "
            else:
                if i > 0 and parts[i - 1] not in ("[DOUBT]", "[CONFIRM]", "[QUESTION]"):
                    text += ", "
                text += p

        if not text:
            text = "(no output)"
        return text.strip(", ")

    def get_confidence_trajectory(self, token_ids):
        """Analyze the doubt/confirm pattern in generated tokens."""
        doubts = sum(1 for t in token_ids if t == DOUBT_ID)
        confirms = sum(1 for t in token_ids if t == CONFIRM_ID)
        questions = sum(1 for t in token_ids if t == QUESTION_ID)
        total_special = doubts + confirms + questions + 1e-8
        return {
            "doubts": doubts,
            "confirms": confirms,
            "questions": questions,
            "confidence": confirms / total_special,
            "uncertainty": doubts / total_special,
        }
