import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ASCII_GRID_SIZE, NUM_ASCII_CHARS


def image_to_ascii_target(images, grid_size=ASCII_GRID_SIZE, num_chars=NUM_ASCII_CHARS):
    """Convert batch of images to ASCII brightness targets.

    images: [B, 3, 224, 224] normalized ImageNet tensors
    Returns: [B, grid_size, grid_size] LongTensor with values in [0, num_chars-1]
    """
    # Undo ImageNet normalization approximately → [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    img = images * std + mean
    img = img.clamp(0, 1)

    # Grayscale
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]  # [B, 224, 224]

    # Downsample to grid_size
    gray = F.interpolate(
        gray.unsqueeze(1), size=(grid_size, grid_size), mode='bilinear', align_corners=False
    ).squeeze(1)  # [B, 32, 32]

    # Quantize to num_chars levels
    target = (gray * (num_chars - 1)).long().clamp(0, num_chars - 1)
    return target


class NeuroASCIILoss(nn.Module):
    """Combined loss for pre-training.

    Components:
    1. Word prediction loss (CE): text_head should predict image labels
    2. ASCII reconstruction loss (CE): ASCII decoder should reproduce image brightness
    3. Latent alignment loss: cosine similarity between image and bridge latents
    """

    def __init__(self, alpha_word=1.0, alpha_ascii=2.0, alpha_align=0.3):
        super().__init__()
        self.alpha_word = alpha_word
        self.alpha_ascii = alpha_ascii
        self.alpha_align = alpha_align

    def forward(self, predictions, targets):
        word_loss = F.cross_entropy(predictions["word_logits"], targets["word_indices"])

        ascii_loss = torch.tensor(0.0)
        if "ascii_logits" in predictions and "ascii_target" in targets:
            # ascii_logits: [B, num_chars, 32, 32], ascii_target: [B, 32, 32]
            ascii_loss = F.cross_entropy(predictions["ascii_logits"], targets["ascii_target"])

        align_loss = torch.tensor(0.0)
        if "latent_image" in predictions and "latent_words" in predictions:
            align_loss = 1.0 - F.cosine_similarity(
                predictions["latent_image"],
                predictions["latent_words"],
                dim=-1,
            ).mean()

        total = (
            self.alpha_word * word_loss
            + self.alpha_ascii * ascii_loss
            + self.alpha_align * align_loss
        )

        return total, {
            "word_loss": word_loss.item(),
            "ascii_loss": ascii_loss.item(),
            "align_loss": align_loss.item(),
            "total": total.item(),
        }
