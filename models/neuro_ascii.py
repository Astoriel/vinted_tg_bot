import torch
import torch.nn as nn
from config import GRU_MIN_ITERS, GRU_MAX_ITERS
from models.encoder import ImageEncoder
from models.attention import SpatialSelfAttention
from models.gru_refiner import IterativeGRURefiner
from models.text_head import TextHead
from models.ascii_decoder import ASCIIDecoder
from models.bridge import TextToVisualBridge


class NeuroASCII(nn.Module):
    """Main model orchestrating all components.

    Two paths:
    - forward_image: image → latent → ASCII + words (with iterative refinement)
    - forward_words: word_indices → bridge → latent → ASCII
    """

    def __init__(self, vocab):
        super().__init__()
        self.encoder = ImageEncoder()
        self.attention = SpatialSelfAttention()
        self.gru = IterativeGRURefiner()
        self.text_head = TextHead()
        self.ascii_decoder = ASCIIDecoder()
        self.bridge = TextToVisualBridge()
        self.vocab = vocab

    def forward_image(self, image_tensor, surprise=0.0):
        """Full forward pass: image → encoded features → iterative GRU with
        text feedback loop → ASCII grid + word predictions.

        Returns dict with all intermediate results for streaming.
        """
        # 1. Encode image
        features = self.encoder(image_tensor)  # [B, 576, 7, 7]

        # 2. Attend & pool
        latent = self.attention(features)  # [B, 128]

        # 3. Determine iterations from surprise
        num_iters = min(GRU_MAX_ITERS,
                        max(GRU_MIN_ITERS, int(2 + surprise * 3)))

        # 4. Iterative refinement with internal dialogue
        text_feedback = None
        all_thoughts = []

        for i in range(num_iters):
            latent, _ = self.gru(latent, num_iters=1, text_feedback=text_feedback)

            word_logits = self.text_head(latent)
            words, confs = self.text_head.get_top_words(latent, self.vocab)
            all_thoughts.append({
                "iteration": i,
                "words": words[0],
                "confidences": confs[0],
            })

            # Feed text back for next iteration
            top_word_idx = word_logits.argmax(dim=-1)
            text_feedback = self.text_head.get_embedding(
                top_word_idx.unsqueeze(1)
            ).squeeze(1)  # [B, 32]

        # 5. Generate ASCII
        ascii_logits = self.ascii_decoder(latent)
        ascii_grid = ascii_logits.argmax(dim=1)  # [B, 32, 32]

        return {
            "latent": latent,
            "thoughts": all_thoughts,
            "word_logits": word_logits,
            "ascii_logits": ascii_logits,
            "ascii_grid": ascii_grid,
            "num_iterations": num_iters,
        }

    def forward_words(self, word_indices):
        """Forward pass from words to ASCII (via bridge)."""
        embeddings = self.text_head.get_embedding(word_indices)  # [B, N, 32]
        latent = self.bridge(embeddings)  # [B, 128]
        ascii_logits = self.ascii_decoder(latent)
        ascii_grid = ascii_logits.argmax(dim=1)
        return {
            "latent": latent,
            "ascii_logits": ascii_logits,
            "ascii_grid": ascii_grid,
        }
