import json
import time
import numpy as np
import torch
from pathlib import Path
from config import DREAM_LOG_DIR


def slerp(v0, v1, t):
    """Spherical linear interpolation between two vectors."""
    v0_n = v0 / (torch.norm(v0, dim=-1, keepdim=True) + 1e-8)
    v1_n = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    dot = (v0_n * v1_n).sum(dim=-1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega) + 1e-8
    s0 = torch.sin((1 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega
    return s0 * v0 + s1 * v1


class DreamPhase:
    """Background consolidation — replays memories and cross-pollinates.

    Every N minutes (or on manual trigger):
    1. Sample random memory pairs
    2. SLERP interpolate between their latents
    3. Run text head on interpolated latents → discover new word associations
    4. Log the dream
    """

    def __init__(self, model, memory_bank):
        self.model = model
        self.memory = memory_bank
        self.dream_count = 0

    async def dream(self, num_episodes=20, interpolation_steps=5):
        log = {
            "episode": self.dream_count,
            "timestamp": time.time(),
            "memories_replayed": 0,
            "new_associations": [],
        }

        if len(self.memory.values) < 4:
            return log

        for _ in range(num_episodes):
            idx1, idx2 = np.random.choice(len(self.memory.values), 2, replace=False)
            latent1 = torch.tensor(self.memory.keys[idx1]).unsqueeze(0)
            latent2 = torch.tensor(self.memory.keys[idx2]).unsqueeze(0)

            for t in np.linspace(0, 1, interpolation_steps):
                interp = slerp(latent1, latent2, t)

                with torch.no_grad():
                    words, confs = self.model.text_head.get_top_words(
                        interp, self.model.vocab, k=3
                    )

                log["new_associations"].append({
                    "words": words[0],
                    "confidences": confs[0],
                    "t": float(t),
                    "source_words_1": self.memory.values[idx1].get("words", []),
                    "source_words_2": self.memory.values[idx2].get("words", []),
                })

            log["memories_replayed"] += 1

        self.dream_count += 1
        self._save_log(log)
        return log

    def _save_log(self, log):
        log_dir = Path(DREAM_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"dream_{self.dream_count:05d}.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
