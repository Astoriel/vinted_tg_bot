"""Quick test of trained v2 model quality."""
import torch, json, sys
sys.path.insert(0, '.')
from config import DATA_DIR, WEIGHTS_DIR
from models.neuro_ascii_v2 import NeuroASCIIv2

with open(DATA_DIR / 'vocab.json') as f:
    vocab = [e['word'] for e in json.load(f)]

model = NeuroASCIIv2(vocab)
loaded = model.load_weights(WEIGHTS_DIR / 'neuroascii_v2_best.pt')
model.eval()
print(f"Loaded {loaded} tensors", flush=True)

img = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    result = model.forward_image(img)
    grid = result['ascii_grid'][0]
    unique = len(torch.unique(grid))
    ids, text, conf = model.generate_text(result['visual_seq'], temperature=0.8, top_k=50)

print(f"Unique ASCII: {unique}/10", flush=True)
print(f"Text: {text}", flush=True)
print(f"Confidence: {conf['confidence']:.3f}", flush=True)
print(f"Doubts: {conf['doubts']}, Confirms: {conf['confirms']}", flush=True)

# Histogram
vals, counts = torch.unique(grid, return_counts=True)
for v, c in zip(vals.tolist(), counts.tolist()):
    pct = c / grid.numel() * 100
    print(f"  char {v}: {c} ({pct:.1f}%)", flush=True)
