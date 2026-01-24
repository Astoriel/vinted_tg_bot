import torch
import torch.nn.functional as F
from config import GRU_MIN_ITERS, GRU_MAX_ITERS


class SurpriseModule:
    """Computes surprise as prediction error.

    surprise = 1.0 - cosine_sim(predicted_embedding, actual_embedding)

    Modulates:
    - GRU iterations: more surprise = more thinking
    - Learning rate: more surprise = larger gradient steps
    - Memory write priority: high-surprise memories stored preferentially
    """

    def __init__(self, text_head):
        self.text_head = text_head
        self.running_mean_surprise = 0.5
        self.ema_alpha = 0.01

    def compute(self, predicted_logits, actual_word_indices):
        """
        predicted_logits: [B, 3000]
        actual_word_indices: [B, num_words]
        Returns: surprise scalar in [0, 1]
        """
        pred_idx = predicted_logits.argmax(dim=-1)
        pred_emb = self.text_head.word_embeddings(pred_idx)  # [B, 32]

        actual_emb = self.text_head.word_embeddings(actual_word_indices)  # [B, N, 32]
        actual_emb = actual_emb.mean(dim=1)  # [B, 32]

        cos_sim = F.cosine_similarity(pred_emb, actual_emb, dim=-1)
        surprise = (1.0 - cos_sim).mean().item()
        surprise = max(0.0, min(1.0, surprise))

        self.running_mean_surprise = (
            self.ema_alpha * surprise
            + (1 - self.ema_alpha) * self.running_mean_surprise
        )

        return surprise

    def get_gru_iterations(self, surprise):
        return min(GRU_MAX_ITERS, max(GRU_MIN_ITERS, int(2 + surprise * 3)))

    def get_learning_rate_multiplier(self, surprise):
        return 0.5 + surprise  # range [0.5, 1.5]
