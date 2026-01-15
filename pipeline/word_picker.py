import json
import random
import numpy as np
import torch
from config import DATA_DIR


class WordPicker:
    """Picks 2-3 random words from vocabulary with frequency weighting."""

    def __init__(self, vocab_path=None):
        vocab_path = vocab_path or (DATA_DIR / "vocab.json")
        with open(vocab_path) as f:
            self.vocab_data = json.load(f)
        self.words = [entry["word"] for entry in self.vocab_data]
        self.weights = np.array([entry.get("weight", 1.0) for entry in self.vocab_data])
        self.weights = self.weights / self.weights.sum()

    def pick(self, n=None):
        """Pick n random words (default: random 2-3).
        Returns list of (index, word) tuples.
        """
        if n is None:
            n = random.choice([2, 2, 2, 3])  # bias toward 2 words
        indices = np.random.choice(len(self.words), size=n, replace=False, p=self.weights)
        return [(int(idx), self.words[idx]) for idx in indices]

    def word_to_index(self, word):
        try:
            return self.words.index(word)
        except ValueError:
            return 0

    def indices_tensor(self, word_index_pairs):
        """Convert picker output to tensor [1, num_words]."""
        indices = [pair[0] for pair in word_index_pairs]
        return torch.tensor([indices], dtype=torch.long)
