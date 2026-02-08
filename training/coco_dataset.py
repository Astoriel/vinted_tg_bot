"""COCO Captions dataset for NeuroASCII v2 phrase training.

Downloads images on-the-fly from COCO URLs and caches to disk.
Tokenizes captions into vocab word sequences with [DOUBT]/[CONFIRM] injection.
"""

import json
import random
import re
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen, Request

import torch
from PIL import Image
from torch.utils.data import Dataset

from models.text_decoder import BOS_ID, EOS_ID, DOUBT_ID, CONFIRM_ID

# Reuse transforms from existing dataset module
from training.dataset import TRAIN_TRANSFORM, VAL_TRANSFORM


def _tokenize_caption(caption, vocab, max_words=8):
    """Tokenize a COCO caption into vocab word indices.

    Extracts words from caption that exist in our 3000-word vocab.
    Returns list of vocab indices (up to max_words).
    """
    # Lowercase, strip punctuation, split
    words = re.sub(r"[^a-z\s]", "", caption.lower()).split()

    # Build a set for O(1) lookup
    vocab_set = set(vocab) if not isinstance(vocab, set) else vocab

    token_ids = []
    seen = set()
    for w in words:
        if w in vocab_set and w not in seen:
            token_ids.append(vocab.index(w) if isinstance(vocab, list) else vocab[w])
            seen.add(w)
            if len(token_ids) >= max_words:
                break

    return token_ids


def inject_doubt_confirm(token_ids, doubt_prob=0.15, confirm_prob=0.10):
    """Inject [DOUBT] and [CONFIRM] tokens into a word sequence.

    Simulates the model's internal reasoning process:
    - [DOUBT] inserted before a word = "I'm unsure about this next word"
    - [CONFIRM] inserted after a word = "Yes, that word is right"

    This teaches the model to produce doubt/confirm during generation.
    """
    if len(token_ids) < 2:
        return token_ids

    result = []
    for i, tid in enumerate(token_ids):
        # Maybe doubt before this word (not the first)
        if i > 0 and random.random() < doubt_prob:
            result.append(DOUBT_ID)
        result.append(tid)
        # Maybe confirm after this word (not the last)
        if i < len(token_ids) - 1 and random.random() < confirm_prob:
            result.append(CONFIRM_ID)

    return result


class COCOCaptionsDataset(Dataset):
    """COCO Captions with on-the-fly image download + vocab tokenization.

    Each sample returns (image_tensor, word_indices) where word_indices
    is a tensor of vocab indices extracted from a random caption.
    """

    def __init__(self, vocab, train=True, data_root="./data",
                 max_words=6, inject_special=True, max_samples=None):
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.vocab_set = set(self.vocab)
        # Build word->index map for fast lookup
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.transform = TRAIN_TRANSFORM if train else VAL_TRANSFORM
        self.max_words = max_words
        self.inject_special = inject_special

        data_root = Path(data_root)
        self.cache_dir = data_root / "coco" / "images"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load annotations — prefer pre-filtered subset > pickle > JSON
        import pickle
        subset_file = data_root / "coco" / "train_10k.pkl"
        pkl_file = data_root / "coco" / "captions.pkl"

        if subset_file.exists() and max_samples and max_samples <= 10000:
            print("Loading pre-filtered 10K subset...", flush=True)
            with open(subset_file, "rb") as f:
                raw_samples = pickle.load(f)
            self.id_to_url = {}  # not needed for cached images
            # Filter: keep images with >= 2 vocab words
            self.samples = []
            for img_id, caps in raw_samples:
                for cap in caps:
                    tokens = self._fast_tokenize(cap)
                    if len(tokens) >= 2:
                        self.samples.append((img_id, caps))
                        break
            del raw_samples
        elif pkl_file.exists():
            print("Loading captions from pickle...", flush=True)
            with open(pkl_file, "rb") as f:
                entries = pickle.load(f)
            self.id_to_url = {}
            captions_by_img = {}
            for img_id, url, caps in entries:
                self.id_to_url[img_id] = url
                captions_by_img[img_id] = caps
            del entries
            self.samples = []
            for img_id, caps in captions_by_img.items():
                for cap in caps:
                    tokens = self._fast_tokenize(cap)
                    if len(tokens) >= 2:
                        self.samples.append((img_id, caps))
                        break
        else:
            raise FileNotFoundError(
                "No COCO data found. Run: python -m training.coco_dataset to prepare data."
            )

        split = "train" if train else "val"
        n_total = len(self.samples)

        # Optional limit (for faster debugging)
        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"COCO {split}: {len(self.samples)} images with valid captions "
              f"(from {n_total} filtered)", flush=True)

    def _fast_tokenize(self, caption):
        """Fast tokenization without building full result."""
        words = re.sub(r"[^a-z\s]", "", caption.lower()).split()
        ids = []
        seen = set()
        for w in words:
            if w in self.vocab_set and w not in seen:
                ids.append(self.word_to_idx[w])
                seen.add(w)
                if len(ids) >= self.max_words:
                    break
        return ids

    def _download_image(self, image_id):
        """Download image from COCO URL, cache to disk."""
        cache_path = self.cache_dir / f"{image_id:012d}.jpg"
        if cache_path.exists():
            return Image.open(cache_path).convert("RGB")

        url = self.id_to_url[image_id]
        try:
            req = Request(url, headers={"User-Agent": "NeuroASCII/1.0"})
            resp = urlopen(req, timeout=10)
            img_data = resp.read()
            # Save to cache
            cache_path.write_bytes(img_data)
            return Image.open(BytesIO(img_data)).convert("RGB")
        except Exception:
            return None

    def __getitem__(self, idx):
        image_id, captions = self.samples[idx]

        # Download/load image
        img = self._download_image(image_id)
        if img is None:
            # Fallback: return a blank image + "the" token
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            return self.transform(img), torch.tensor([0], dtype=torch.long)

        img_tensor = self.transform(img)

        # Pick a random caption and tokenize
        cap = random.choice(captions)
        token_ids = self._fast_tokenize(cap)

        if len(token_ids) == 0:
            token_ids = [0]  # fallback to "the"

        # Inject doubt/confirm tokens during training
        if self.inject_special:
            token_ids = inject_doubt_confirm(token_ids)

        return img_tensor, torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.samples)


def coco_collate_fn(batch):
    """Custom collate for variable-length word sequences.

    Pads word_indices to the same length within the batch.
    """
    images = torch.stack([b[0] for b in batch])
    word_lists = [b[1] for b in batch]

    # Pad to max length in batch
    max_len = max(w.shape[0] for w in word_lists)
    padded = torch.full((len(word_lists), max_len), EOS_ID, dtype=torch.long)
    for i, w in enumerate(word_lists):
        padded[i, :w.shape[0]] = w

    return images, padded
