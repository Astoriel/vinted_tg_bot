"""Pre-training script for NeuroASCII.

Trains: attention, gru_refiner, text_head, ascii_decoder, bridge
Frozen: encoder (MobileNetV3)

Usage:
  python -m training.pretrain --epochs 5 --batch_size 32
  python -m training.pretrain --epochs 10 --datasets cifar100 food101 --batch_size 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, WEIGHTS_DIR
from models.neuro_ascii import NeuroASCII
from training.dataset import build_combined_dataset
from training.losses import NeuroASCIILoss, image_to_ascii_target


def train(epochs=5, batch_size=32, lr=1e-3, resume=False, datasets=("cifar100",)):
    with open(DATA_DIR / "vocab.json") as f:
        vocab = [entry["word"] for entry in json.load(f)]

    print(f"Vocab: {len(vocab)} words", flush=True)
    print(f"Datasets: {', '.join(datasets)}", flush=True)

    model = NeuroASCII(vocab)

    if resume:
        ckpt = WEIGHTS_DIR / "neuroascii_latest.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            # Only load matching keys (architecture may have changed)
            model_state = model.state_dict()
            loaded = {k: v for k, v in state.items()
                      if k in model_state and v.shape == model_state[k].shape}
            model_state.update(loaded)
            model.load_state_dict(model_state)
            print(f"Resumed {len(loaded)}/{len(model_state)} layers from checkpoint", flush=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable_params:,} ({trainable_params*4/1e6:.1f}MB)", flush=True)
    print(f"Frozen: {frozen_params:,}", flush=True)

    dataset = build_combined_dataset(vocab, datasets=datasets, train=True, data_root=str(DATA_DIR))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Training on {len(dataset)} images, {len(loader)} batches/epoch", flush=True)

    loss_fn = NeuroASCIILoss(alpha_word=1.0, alpha_ascii=3.0, alpha_align=0.3)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, (images, word_indices) in enumerate(loader):
            result = model.forward_image(images, surprise=0.3)
            word_result = model.forward_words(word_indices)

            ascii_logits = model.ascii_decoder(result["latent"])
            ascii_target = image_to_ascii_target(images)

            predictions = {
                "word_logits": result["word_logits"],
                "ascii_logits": ascii_logits,
                "latent_image": result["latent"],
                "latent_words": word_result["latent"],
            }
            targets = {
                "word_indices": word_indices.squeeze(1),
                "ascii_target": ascii_target,
            }

            loss, loss_dict = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            epoch_losses.append(loss_dict)

            if batch_idx % 100 == 0:
                elapsed = time.time() - t0
                print(
                    f"E{epoch+1}/{epochs} [{batch_idx}/{len(loader)}] "
                    f"loss={loss_dict['total']:.3f} "
                    f"word={loss_dict['word_loss']:.3f} "
                    f"ascii={loss_dict['ascii_loss']:.3f} "
                    f"align={loss_dict['align_loss']:.3f} "
                    f"[{elapsed:.0f}s]",
                    flush=True,
                )

        scheduler.step()

        avg = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in epoch_losses[0]}
        elapsed = time.time() - t0
        print(
            f"--- Epoch {epoch+1} done ({elapsed:.0f}s) "
            f"avg_loss={avg['total']:.3f} "
            f"word={avg['word_loss']:.3f} "
            f"ascii={avg['ascii_loss']:.3f} "
            f"align={avg['align_loss']:.3f}",
            flush=True,
        )

        torch.save(model.state_dict(), WEIGHTS_DIR / f"pretrain_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), WEIGHTS_DIR / "neuroascii_latest.pt")

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            torch.save(model.state_dict(), WEIGHTS_DIR / "neuroascii_best.pt")
            print(f"  New best! Saved neuroascii_best.pt", flush=True)

        print(f"  Saved pretrain_epoch{epoch+1}.pt", flush=True)

    print(f"\nTraining complete! Best loss: {best_loss:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["cifar100"],
                        choices=["cifar100", "food101", "flowers102", "dtd"])
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          resume=args.resume, datasets=tuple(args.datasets))
