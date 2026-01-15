import torch
import torch.nn as nn
from config import LATENT_DIM, GRU_HIDDEN, EMBED_DIM


class IterativeGRURefiner(nn.Module):
    """Iterative refinement: runs 2-5 GRU steps based on surprise.

    Input:  feature_vec [B, 128], optional text_feedback [B, 32]
    Output: final_hidden [B, 128], intermediates list of [B, 128]
    """

    def __init__(self, input_dim=LATENT_DIM, hidden_dim=GRU_HIDDEN,
                 feedback_dim=EMBED_DIM):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.text_feedback_proj = nn.Linear(feedback_dim, input_dim)

    def forward(self, x, num_iters=2, text_feedback=None):
        """
        Args:
            x: [B, 128] initial feature vector
            num_iters: how many refinement steps
            text_feedback: optional [B, 32] from text head
        Returns:
            final_hidden: [B, 128]
            intermediates: list of [B, 128] at each step
        """
        h = torch.zeros(x.size(0), self.gru_cell.hidden_size, device=x.device)
        intermediates = []

        for i in range(num_iters):
            inp = x
            if text_feedback is not None and i > 0:
                inp = inp + self.text_feedback_proj(text_feedback)
            h = self.gru_cell(inp, h)
            intermediates.append(h)

        return h, intermediates
