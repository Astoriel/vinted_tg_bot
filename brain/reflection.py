import torch
import torch.nn.functional as F
from config import LEARNING_RATE


class ReflectionModule:
    """Self-evaluation and online learning.

    Compares predictions with reality, computes loss, runs one SGD step.
    Trains text_head, attention, gru, ascii_decoder, bridge (NOT encoder).
    """

    def __init__(self, model, learning_rate=LEARNING_RATE):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        self.model = model

    def reflect(self, predicted_logits, actual_word_indices, surprise,
                latent_image=None, latent_words=None):
        """One step of online learning. Returns loss dict for logging."""
        lr_mult = 0.5 + surprise
        for pg in self.optimizer.param_groups:
            pg["lr"] = LEARNING_RATE * lr_mult

        # Cross-entropy loss: did text head predict the right words?
        ce_loss = torch.tensor(0.0)
        for i in range(actual_word_indices.size(1)):
            ce_loss = ce_loss + F.cross_entropy(predicted_logits, actual_word_indices[:, i])
        ce_loss = ce_loss / actual_word_indices.size(1)

        # Latent alignment loss (if both paths were run)
        align_loss = torch.tensor(0.0)
        if latent_image is not None and latent_words is not None:
            align_loss = 1.0 - F.cosine_similarity(
                latent_image, latent_words, dim=-1
            ).mean()

        total_loss = ce_loss + 0.3 * align_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()

        return {
            "ce_loss": ce_loss.item(),
            "align_loss": align_loss.item(),
            "total_loss": total_loss.item(),
            "lr": LEARNING_RATE * lr_mult,
        }
