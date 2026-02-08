"""Pre-training script for NeuroASCII v2.

Trains: tri_plane, ascii_decoder, text_decoder, visual_to_text
Frozen: encoder (MobileNetV3)

Usage:
  python -m training.pretrain_v2 --epochs 5 --batch_size 16
  python -m training.pretrain_v2 --epochs 10 --datasets cifar100 food101
  python -m training.pretrain_v2 --epochs 20 --coco --resume
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, WEIGHTS_DIR, V2_ASCII_GRID, NUM_ASCII_CHARS, DEVICE
from models.neuro_ascii_v2 import NeuroASCIIv2
from models.text_decoder import BOS_ID, EOS_ID
from training.dataset import build_combined_dataset
from training.coco_dataset import COCOCaptionsDataset, coco_collate_fn


def image_to_ascii_target_v2(images, grid_size=V2_ASCII_GRID, num_chars=NUM_ASCII_CHARS):
    """Convert images to ASCII brightness targets at v2 resolution (48x48)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    img = (images * std + mean).clamp(0, 1)

    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    gray = F.interpolate(
        gray.unsqueeze(1), size=(grid_size, grid_size),
        mode="bilinear", align_corners=False,
    ).squeeze(1)

    return (gray * (num_chars - 1)).long().clamp(0, num_chars - 1)


def build_text_targets(word_indices, max_len=12):
    """Build autoregressive text targets from word indices.

    word_indices: [B, N] vocab indices (may include DOUBT/CONFIRM tokens)
    Returns: input_tokens [B, T-1], labels [B, T-1]
    """
    B = word_indices.shape[0]
    sequences = []
    for b in range(B):
        words = word_indices[b].tolist()
        # Remove padding (EOS used as pad in coco_collate_fn)
        words = [w for w in words if w != EOS_ID]
        seq = [BOS_ID] + words + [EOS_ID]
        # Pad or truncate
        if len(seq) > max_len:
            seq = seq[:max_len - 1] + [EOS_ID]
        else:
            seq = seq + [EOS_ID] * (max_len - len(seq))
        sequences.append(seq)

    tokens = torch.tensor(sequences, dtype=torch.long, device=word_indices.device)
    return tokens[:, :-1], tokens[:, 1:]


def train(epochs=5, batch_size=16, lr=5e-4, resume=False, datasets=("cifar100",),
          use_coco=False, coco_max_samples=None):
    device = DEVICE
    print(f"Device: {device}", flush=True)

    with open(DATA_DIR / "vocab.json") as f:
        vocab = [entry["word"] for entry in json.load(f)]

    print(f"Vocab: {len(vocab)} words", flush=True)

    model = NeuroASCIIv2(vocab)

    if resume:
        ckpt = WEIGHTS_DIR / "neuroascii_v2.pt"
        if ckpt.exists():
            loaded = model.load_weights(ckpt)
            print(f"Resumed {loaded} tensors from checkpoint", flush=True)

    counts = model.count_params()
    print(f"Trainable: {counts['trainable']:,} ({counts['trainable']*2/1e6:.1f}MB fp16)", flush=True)
    print(f"Frozen: {counts['frozen']:,}", flush=True)

    model.to(device)

    # Build dataset
    collate_fn = None
    if use_coco:
        print("Using COCO Captions dataset", flush=True)
        dataset = COCOCaptionsDataset(
            vocab, train=True, data_root=str(DATA_DIR),
            max_words=6, inject_special=True,
            max_samples=coco_max_samples,
        )
        collate_fn = coco_collate_fn
    else:
        print(f"Datasets: {', '.join(datasets)}", flush=True)
        dataset = build_combined_dataset(vocab, datasets=datasets, train=True, data_root=str(DATA_DIR))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=(device != "cpu"),
                         collate_fn=collate_fn)
    print(f"Training on {len(dataset)} images, {len(loader)} batches/epoch", flush=True)

    trainable = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, (images, word_indices) in enumerate(loader):
            images = images.to(device)
            word_indices = word_indices.to(device)

            # 1. Forward through tri-plane
            result = model.forward_image(images)

            # 2. ASCII reconstruction loss
            ascii_target = image_to_ascii_target_v2(images)
            ascii_loss = F.cross_entropy(result["ascii_logits"], ascii_target)

            # 3. Text loss (teacher-forced)
            input_tokens, labels = build_text_targets(word_indices)
            text_logits = model.forward_text_training(
                input_tokens, result["visual_seq"].detach(),
            )
            text_loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                labels.reshape(-1),
            )

            # 4. Multi-view consistency: two random viewpoints should
            #    produce latents that point in similar directions
            az1 = torch.rand(images.shape[0], 1, device=device) * 2 * math.pi
            az2 = az1 + (torch.rand(images.shape[0], 1, device=device) * 0.5 + 0.1)
            vp1 = torch.cat([az1, torch.full_like(az1, 0.3), torch.full_like(az1, 2.0)], dim=1)
            vp2 = torch.cat([az2, torch.full_like(az2, 0.3), torch.full_like(az2, 2.0)], dim=1)

            proj1 = model.tri_plane.render_from_viewpoint(result["planes"], vp1)
            proj2 = model.tri_plane.render_from_viewpoint(result["planes"], vp2)
            lat1 = model.tri_plane.to_latent(proj1)
            lat2 = model.tri_plane.to_latent(proj2)

            # Close views should have similar (but not identical) latents
            view_consistency = 1.0 - F.cosine_similarity(lat1, lat2, dim=-1).mean()

            # Total loss
            total = 3.0 * ascii_loss + 1.0 * text_loss + 0.3 * view_consistency

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            loss_dict = {
                "ascii": ascii_loss.item(),
                "text": text_loss.item(),
                "view": view_consistency.item(),
                "total": total.item(),
            }
            epoch_losses.append(loss_dict)

            if batch_idx % 100 == 0:
                elapsed = time.time() - t0
                print(
                    f"E{epoch+1}/{epochs} [{batch_idx}/{len(loader)}] "
                    f"total={loss_dict['total']:.3f} "
                    f"ascii={loss_dict['ascii']:.3f} "
                    f"text={loss_dict['text']:.3f} "
                    f"view={loss_dict['view']:.3f} "
                    f"[{elapsed:.0f}s]",
                    flush=True,
                )

        scheduler.step()

        avg = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in epoch_losses[0]}
        elapsed = time.time() - t0
        print(
            f"--- Epoch {epoch+1} done ({elapsed:.0f}s) "
            f"avg_total={avg['total']:.3f} "
            f"ascii={avg['ascii']:.3f} "
            f"text={avg['text']:.3f} "
            f"view={avg['view']:.3f}",
            flush=True,
        )

        model.save_weights(WEIGHTS_DIR / f"v2_epoch{epoch+1}.pt")
        model.save_weights(WEIGHTS_DIR / "neuroascii_v2.pt")

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            model.save_weights(WEIGHTS_DIR / "neuroascii_v2_best.pt")
            print(f"  New best! Saved neuroascii_v2_best.pt", flush=True)

        print(f"  Saved v2_epoch{epoch+1}.pt", flush=True)

    print(f"\nTraining complete! Best loss: {best_loss:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coco", action="store_true", help="Train on COCO Captions")
    parser.add_argument("--coco_max", type=int, default=None, help="Limit COCO samples")
    parser.add_argument("--datasets", nargs="+", default=["cifar100"],
                        choices=["cifar100", "food101", "flowers102", "dtd"])
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          resume=args.resume, datasets=tuple(args.datasets),
          use_coco=args.coco, coco_max_samples=args.coco_max)
