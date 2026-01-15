import torch
import torch.nn as nn
from config import LATENT_DIM, ASCII_GRID_SIZE, NUM_ASCII_CHARS


class ASCIIDecoder(nn.Module):
    """Generates ASCII art grid from latent vector.

    Input:  [B, 128]
    Output: [B, num_ascii_chars, 32, 32] logits over ASCII palette per cell
    """

    def __init__(self, latent_dim=LATENT_DIM, grid_size=ASCII_GRID_SIZE,
                 num_chars=NUM_ASCII_CHARS):
        super().__init__()
        self.grid_size = grid_size
        self.initial_size = 8
        self.channels = 32

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.channels * self.initial_size * self.initial_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(self.channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(16, num_chars, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latent):
        B = latent.size(0)
        x = self.fc(latent)
        x = x.view(B, self.channels, self.initial_size, self.initial_size)
        x = self.decoder(x)
        return x  # [B, num_chars, 32, 32]

    def decode_to_indices(self, latent):
        logits = self.forward(latent)
        return logits.argmax(dim=1)  # [B, 32, 32]
