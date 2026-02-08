import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CIFAR100, Food101, Flowers102, DTD
import torchvision.transforms as T

# --- CIFAR-100 label -> vocab word mapping ---
CIFAR100_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter",
    "palm_tree", "pear", "pickup_truck", "pine_tree", "plain",
    "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon",
    "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]

# Better mapping: CIFAR label -> primary vocab word + related words
CIFAR100_SYNONYMS = {
    "aquarium_fish": ["fish", "aquarium", "water"],
    "lawn_mower": ["grass", "garden", "machine"],
    "maple_tree": ["tree", "maple", "leaf"],
    "oak_tree": ["tree", "oak", "wood"],
    "palm_tree": ["tree", "palm", "tropical"],
    "pine_tree": ["tree", "pine", "forest"],
    "willow_tree": ["tree", "willow", "nature"],
    "pickup_truck": ["truck", "vehicle", "road"],
    "sweet_pepper": ["pepper", "vegetable", "food"],
    "flatfish": ["fish", "ocean", "flat"],
    "streetcar": ["train", "street", "city"],
}

# Food101 class names (subset - mapped at runtime)
FOOD101_EXTRA_WORDS = {
    "pizza": ["pizza", "cheese", "food"],
    "sushi": ["sushi", "fish", "rice"],
    "ice_cream": ["ice", "cream", "sweet"],
    "chocolate_cake": ["chocolate", "cake", "sweet"],
    "french_fries": ["fries", "potato", "food"],
    "hamburger": ["hamburger", "meat", "bread"],
    "hot_dog": ["sausage", "bread", "food"],
    "steak": ["steak", "meat", "beef"],
    "soup": ["soup", "bowl", "warm"],
}

# Flowers102 name list (first 20 common ones for mapping)
FLOWERS102_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily",
]

TRAIN_TRANSFORM = T.Compose([
    T.Resize(232),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize(232),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _find_vocab_idx(word, vocab):
    """Find word in vocab, trying multiple formats."""
    word = word.lower().strip()
    if word in vocab:
        return vocab.index(word)
    # Try without underscores
    clean = word.replace("_", " ").split()[0]
    if clean in vocab:
        return vocab.index(clean)
    return None


def _build_label_map(labels, vocab, synonyms=None):
    """Build mapping from dataset label index to vocab index."""
    mapping = {}
    for idx, label in enumerate(labels):
        # Try synonyms first
        if synonyms and label in synonyms:
            for syn in synonyms[label]:
                vi = _find_vocab_idx(syn, vocab)
                if vi is not None:
                    mapping[idx] = vi
                    break
        if idx not in mapping:
            vi = _find_vocab_idx(label, vocab)
            if vi is not None:
                mapping[idx] = vi
    return mapping


class CIFAR100Dataset(Dataset):
    """CIFAR-100 with proper label-to-vocab mapping."""

    def __init__(self, vocab, train=True, data_root="./data"):
        self.transform = TRAIN_TRANSFORM if train else VAL_TRANSFORM
        self.dataset = CIFAR100(root=data_root, train=train, download=True)
        self.vocab = vocab
        self.label_map = _build_label_map(CIFAR100_LABELS, vocab, CIFAR100_SYNONYMS)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_tensor = self.transform(img)
        vocab_idx = self.label_map.get(label, 0)
        return img_tensor, torch.tensor([vocab_idx], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


class Food101Dataset(Dataset):
    """Food-101: 101K images of food in real resolution."""

    def __init__(self, vocab, train=True, data_root="./data"):
        self.transform = TRAIN_TRANSFORM if train else VAL_TRANSFORM
        split = "train" if train else "test"
        self.dataset = Food101(root=data_root, split=split, download=True)
        self.vocab = vocab
        self._build_map()

    def _build_map(self):
        self.label_map = {}
        # Food101 classes are like "french_fries", "chocolate_cake" etc
        classes = self.dataset.classes if hasattr(self.dataset, 'classes') else []
        for idx, cls_name in enumerate(classes):
            # Try synonyms
            if cls_name in FOOD101_EXTRA_WORDS:
                for word in FOOD101_EXTRA_WORDS[cls_name]:
                    vi = _find_vocab_idx(word, self.vocab)
                    if vi is not None:
                        self.label_map[idx] = vi
                        break
            if idx not in self.label_map:
                vi = _find_vocab_idx(cls_name, self.vocab)
                if vi is not None:
                    self.label_map[idx] = vi

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_tensor = self.transform(img)
        vocab_idx = self.label_map.get(label, 0)
        return img_tensor, torch.tensor([vocab_idx], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


class Flowers102Dataset(Dataset):
    """Flowers-102: 8K flower images."""

    def __init__(self, vocab, train=True, data_root="./data"):
        self.transform = TRAIN_TRANSFORM if train else VAL_TRANSFORM
        split = "train" if train else "test"
        self.dataset = Flowers102(root=data_root, split=split, download=True)
        self.vocab = vocab
        # Flowers102 labels are 0-101, map to "flower" as default
        flower_idx = _find_vocab_idx("flower", vocab)
        self.default_idx = flower_idx if flower_idx is not None else 0
        # Some specific flowers in our vocab
        self._flower_words = {}
        for word in ["rose", "tulip", "orchid", "sunflower", "lily", "daisy", "poppy", "lotus", "iris", "violet"]:
            vi = _find_vocab_idx(word, vocab)
            if vi is not None:
                self._flower_words[word] = vi

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_tensor = self.transform(img)
        # Try to match specific flower, otherwise use "flower"
        vocab_idx = self.default_idx
        if label < len(FLOWERS102_NAMES):
            name = FLOWERS102_NAMES[label].lower()
            for word, vi in self._flower_words.items():
                if word in name:
                    vocab_idx = vi
                    break
        return img_tensor, torch.tensor([vocab_idx], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


class DTDDataset(Dataset):
    """Describable Textures Dataset: 5.6K texture images."""

    def __init__(self, vocab, train=True, data_root="./data"):
        self.transform = TRAIN_TRANSFORM if train else VAL_TRANSFORM
        split = "train" if train else "test"
        self.dataset = DTD(root=data_root, split=split, download=True)
        self.vocab = vocab
        self._build_map()

    def _build_map(self):
        self.label_map = {}
        classes = self.dataset.classes if hasattr(self.dataset, 'classes') else []
        for idx, cls_name in enumerate(classes):
            vi = _find_vocab_idx(cls_name, self.vocab)
            if vi is not None:
                self.label_map[idx] = vi

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_tensor = self.transform(img)
        vocab_idx = self.label_map.get(label, 0)
        return img_tensor, torch.tensor([vocab_idx], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


# Keep backward compat name
NeuroASCIIDataset = CIFAR100Dataset


def build_combined_dataset(vocab, datasets=("cifar100",), train=True, data_root="./data"):
    """Build a combined dataset from multiple sources."""
    parts = []
    for name in datasets:
        if name == "cifar100":
            parts.append(CIFAR100Dataset(vocab, train=train, data_root=data_root))
        elif name == "food101":
            parts.append(Food101Dataset(vocab, train=train, data_root=data_root))
        elif name == "flowers102":
            parts.append(Flowers102Dataset(vocab, train=train, data_root=data_root))
        elif name == "dtd":
            parts.append(DTDDataset(vocab, train=train, data_root=data_root))

    if len(parts) == 1:
        return parts[0]
    return ConcatDataset(parts)
