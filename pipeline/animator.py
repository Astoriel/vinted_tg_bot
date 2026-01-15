import torch
from models.temporal import TemporalRNN
from pipeline.ascii_renderer import ASCIIRenderer
from models.ascii_decoder import ASCIIDecoder


class ASCIIAnimator:
    """Generates multi-frame ASCII animations using Temporal RNN (Stage 3).

    Transform types: rotate, zoom_in, zoom_out, pan_left,
                     pan_right, dissolve, pulse, wave
    """

    TRANSFORMS = [
        "rotate", "zoom_in", "zoom_out", "pan_left",
        "pan_right", "dissolve", "pulse", "wave",
    ]

    def __init__(self, temporal_rnn, ascii_decoder, renderer):
        self.temporal = temporal_rnn
        self.decoder = ascii_decoder
        self.renderer = renderer

    def generate_animation(self, initial_latent, num_frames=30,
                           transform="rotate", target_width=80, target_height=40):
        """Generate a sequence of ASCII art frames.
        Returns: list of ASCII strings (one per frame).
        """
        transform_idx = (
            self.TRANSFORMS.index(transform)
            if transform in self.TRANSFORMS
            else 0
        )
        transform_tensor = torch.tensor([[transform_idx]], dtype=torch.long)

        frames = []
        latent = initial_latent
        hidden = None

        with torch.no_grad():
            for _ in range(num_frames):
                grid = self.decoder.decode_to_indices(latent)
                frame_str = self.renderer.render(
                    grid[0].numpy(), target_width, target_height
                )
                frames.append(frame_str)

                if self.temporal is not None:
                    delta, hidden = self.temporal(latent, hidden, transform_tensor)
                    latent = latent + delta * 0.1

        return frames
