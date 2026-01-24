import os
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# --- Model Architecture (v1 — kept for backward compat) ---
LATENT_DIM = 128
VOCAB_SIZE = 3000
EMBED_DIM = 128
ASCII_GRID_SIZE = 32
ASCII_PALETTE = " .:-=+*#%@"
NUM_ASCII_CHARS = len(ASCII_PALETTE)
IMAGE_SIZE = 224

# Encoder
ENCODER_OUT_CHANNELS = 960
ENCODER_SPATIAL = 7

# Attention
ATTENTION_HEAD_DIM = 64

# GRU Refiner
GRU_HIDDEN = 128
GRU_MIN_ITERS = 2
GRU_MAX_ITERS = 5

# Bridge
BRIDGE_HIDDEN = 128
BRIDGE_MAX_WORDS = 3

# Temporal
TEMPORAL_HIDDEN = 192
NUM_TRANSFORMS = 8
MAX_ANIMATION_FRAMES = 60

# --- v2 Architecture ---
V2_LATENT_DIM = 256
V2_TRIPLANE_RES = 32        # tri-plane spatial resolution
V2_TRIPLANE_FEAT = 64       # feature channels per plane
V2_ASCII_GRID = 48          # larger ASCII output
V2_TEXT_DIM = 128            # text decoder hidden dim
V2_TEXT_HEADS = 4            # attention heads in mini-GPT
V2_TEXT_LAYERS = 4           # transformer decoder layers
V2_TEXT_FFN = 512            # feedforward dim
V2_MAX_TEXT_LEN = 48         # max generated phrase length
V2_NUM_VIEWS = 24            # frames per 360° rotation

# Special text tokens (appended after vocab)
V2_SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[DOUBT]", "[CONFIRM]", "[QUESTION]"]
V2_VOCAB_SIZE = VOCAB_SIZE + len(V2_SPECIAL_TOKENS)  # 3005

# --- Training ---
LEARNING_RATE = 1e-4
SURPRISE_THRESHOLD = 0.5

# --- Memory ---
MEMORY_BANK_SIZE = 1000

# --- API ---
SSE_HEARTBEAT_SEC = 1.0
GENERATION_COOLDOWN_SEC = 5.0

# --- External APIs ---
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY", "")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

# --- Paths ---
BASE_DIR = Path(__file__).parent
if os.environ.get("SPACE_ID") and Path("/data").exists():
    STORAGE_DIR = Path("/data/storage")
else:
    STORAGE_DIR = BASE_DIR / "storage"

WEIGHTS_DIR = STORAGE_DIR / "weights"
MEMORY_DIR = STORAGE_DIR / "memory_bank"
DREAM_LOG_DIR = STORAGE_DIR / "dream_logs"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

# --- Device ---
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
