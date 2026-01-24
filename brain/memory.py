import json
import time
import numpy as np
from pathlib import Path
from config import LATENT_DIM, MEMORY_BANK_SIZE, MEMORY_DIR


class AssociativeMemory:
    """Key-value store with nearest-neighbor lookup.

    Keys: latent vectors (128-dim numpy arrays)
    Values: dicts with words, surprise, timestamp, access_count
    Max entries: 1000 (evicts lowest-surprise when full)
    """

    def __init__(self, capacity=MEMORY_BANK_SIZE, storage_dir=None, latent_dim=None):
        self.capacity = capacity
        self.storage_dir = Path(storage_dir or MEMORY_DIR)
        self.latent_dim = latent_dim or LATENT_DIM
        self.keys = np.zeros((0, self.latent_dim), dtype=np.float32)
        self.values = []
        self.load()

    def _to_numpy(self, latent_vec):
        if hasattr(latent_vec, 'detach'):
            return latent_vec.detach().cpu().numpy().reshape(1, -1)
        return np.asarray(latent_vec, dtype=np.float32).reshape(1, -1)

    def store(self, latent_vec, metadata, surprise=0.0):
        vec = self._to_numpy(latent_vec)
        entry = {
            **metadata,
            "surprise": float(surprise),
            "timestamp": time.time(),
            "access_count": 0,
        }

        if len(self.values) >= self.capacity:
            surprises = np.array([v["surprise"] for v in self.values])
            evict_idx = int(surprises.argmin())
            self.keys = np.delete(self.keys, evict_idx, axis=0)
            self.values.pop(evict_idx)

        if len(self.keys) > 0:
            self.keys = np.vstack([self.keys, vec])
        else:
            self.keys = vec
        self.values.append(entry)

    def recall(self, latent_vec, k=5):
        if len(self.values) == 0:
            return []

        vec = self._to_numpy(latent_vec)
        norms_keys = np.linalg.norm(self.keys, axis=1, keepdims=True) + 1e-8
        norm_query = np.linalg.norm(vec) + 1e-8
        similarities = (self.keys @ vec.T).squeeze() / (norms_keys.squeeze() * norm_query)

        if similarities.ndim == 0:
            similarities = np.array([similarities])

        top_k_idx = np.argsort(-similarities)[:k]
        results = []
        for idx in top_k_idx:
            idx = int(idx)
            self.values[idx]["access_count"] += 1
            results.append({
                "similarity": float(similarities[idx]),
                "metadata": self.values[idx],
                "latent": self.keys[idx],
            })
        return results

    def save(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.storage_dir / "keys.npy", self.keys)
        serializable = []
        for v in self.values:
            entry = {}
            for key, val in v.items():
                if isinstance(val, (np.integer, np.floating)):
                    entry[key] = val.item()
                else:
                    entry[key] = val
            serializable.append(entry)
        with open(self.storage_dir / "values.json", "w") as f:
            json.dump(serializable, f)

    def load(self):
        keys_path = self.storage_dir / "keys.npy"
        vals_path = self.storage_dir / "values.json"
        if keys_path.exists() and vals_path.exists():
            loaded_keys = np.load(keys_path)
            with open(vals_path) as f:
                loaded_values = json.load(f)
            # Check dimension compatibility
            if loaded_keys.ndim == 2 and loaded_keys.shape[1] == self.latent_dim:
                self.keys = loaded_keys
                self.values = loaded_values
            else:
                # Dimension mismatch — reset (v1→v2 migration)
                self.keys = np.zeros((0, self.latent_dim), dtype=np.float32)
                self.values = []
