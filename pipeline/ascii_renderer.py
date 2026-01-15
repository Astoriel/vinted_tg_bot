import json
import numpy as np
from PIL import Image
from config import DATA_DIR, ASCII_PALETTE


class ASCIIRenderer:
    """Converts model output grid to displayable ASCII art at requested resolution.

    The model outputs 32x32 grid of character indices.
    This renderer upsamples to requested size and maps indices to chars.
    """

    def __init__(self, palette_name="standard"):
        palettes_path = DATA_DIR / "ascii_palettes.json"
        if palettes_path.exists():
            with open(palettes_path) as f:
                palettes = json.load(f)
            self.palette = palettes.get(palette_name, ASCII_PALETTE)
        else:
            self.palette = ASCII_PALETTE

    def render(self, grid_indices, target_width=80, target_height=40):
        """
        grid_indices: numpy array [32, 32] of ints
        target_width, target_height: character dimensions
        Returns: string of ASCII art with newlines.
        """
        if hasattr(grid_indices, 'detach'):
            grid_indices = grid_indices.detach().cpu().numpy()
        grid_img = Image.fromarray(grid_indices.astype(np.uint8), mode="L")
        grid_resized = grid_img.resize((target_width, target_height), Image.NEAREST)
        grid_array = np.array(grid_resized)

        lines = []
        for row in grid_array:
            line = ""
            for val in row:
                idx = min(int(val), len(self.palette) - 1)
                line += self.palette[idx]
            lines.append(line)

        return "\n".join(lines)
