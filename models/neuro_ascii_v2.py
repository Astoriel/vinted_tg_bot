"""NeuroASCII v2 — 3D-aware orchestrator with tri-plane representation.

Wires together:
  ImageEncoder (frozen MobileNetV3) → TriPlane3D → ASCIIDecoderV2 + MiniGPTDecoder

Supports:
  - Single image → ASCII + text with doubt/confirm
  - Multi-view animation from cached tri-planes
  - Teacher-forced training and autoregressive generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    V2_LATENT_DIM, V2_TRIPLANE_FEAT, V2_ASCII_GRID,
    V2_NUM_VIEWS, V2_TEXT_DIM, NUM_ASCII_CHARS,
)
from models.encoder import ImageEncoder
from models.tri_plane import TriPlane3D
from models.ascii_decoder_v2 import ASCIIDecoderV2
from models.text_decoder import MiniGPTDecoder, BOS_ID, EOS_ID


class NeuroASCIIv2(nn.Module):
    """v2 model: 3D-aware ASCII generation + autoregressive text.

    Flow:
        image [B,3,224,224]
          → encoder → [B,576,7,7]
          → tri_plane → projected [B,64,32,32], latent [B,256], planes
          → ascii_decoder(projected) → ASCII logits [B, num_chars, 48, 48]
          → text_decoder(visual_seq) → token sequence with DOUBT/CONFIRM
    """

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

        # Frozen image encoder
        self.encoder = ImageEncoder()

        # 3D-aware representation
        self.tri_plane = TriPlane3D()

        # ASCII output
        self.ascii_decoder = ASCIIDecoderV2()

        # Text decoder needs visual features as sequence
        # Project from tri-plane latent to text decoder dim
        self.visual_to_text = nn.Sequential(
            nn.Linear(V2_TRIPLANE_FEAT, V2_TEXT_DIM),
            nn.GELU(),
        )

        self.text_decoder = MiniGPTDecoder()

    def encode_image(self, image_tensor):
        """Encode image through frozen backbone.

        image_tensor: [B, 3, 224, 224]
        Returns: image_features [B, 576, 7, 7]
        """
        return self.encoder(image_tensor)

    def forward_image(self, image_tensor, viewpoint=None):
        """Full forward: image → 3D representation → ASCII + text features.

        Returns dict with all intermediates for streaming.
        """
        # 1. Encode
        image_features = self.encode_image(image_tensor)

        # 2. Tri-plane 3D
        projected, latent, planes = self.tri_plane(image_features, viewpoint)

        # 3. ASCII decode
        ascii_logits = self.ascii_decoder(projected)
        ascii_grid = ascii_logits.argmax(dim=1)  # [B, H, W]

        # 4. Prepare visual features for text decoder
        # Flatten projected features into a sequence: [B, 64, 32, 32] → [B, 1024, 64] → [B, 1024, 128]
        B = projected.shape[0]
        vis_seq = projected.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        vis_seq = self.visual_to_text(vis_seq)  # [B, 1024, text_dim]

        return {
            "image_features": image_features,
            "projected": projected,
            "latent": latent,
            "planes": planes,
            "ascii_logits": ascii_logits,
            "ascii_grid": ascii_grid,
            "visual_seq": vis_seq,
        }

    def generate_text(self, visual_seq, max_len=None, temperature=0.8, top_k=50):
        """Autoregressive text generation from visual features.

        visual_seq: [B, S, text_dim] from forward_image
        Returns: token_ids (list), text (str), confidence_info (dict)
        """
        token_ids = self.text_decoder.generate(
            visual_seq, max_len=max_len,
            temperature=temperature, top_k=top_k,
        )
        text = self.text_decoder.tokens_to_text(token_ids, self.vocab)
        confidence = self.text_decoder.get_confidence_trajectory(token_ids)
        return token_ids, text, confidence

    def refine_projected(self, projected, target_word_ids, num_steps=4, lr=0.01):
        """Optimize projected features toward target word representations.

        Gradient-descent on the projected bottleneck [B,64,32,32] to maximize
        target word logits in the text decoder. This steers both ASCII output
        and text output simultaneously since projected feeds both paths.

        Args:
            projected: [B, 64, 32, 32] from forward_image
            target_word_ids: list of int — vocab indices for target words
            num_steps: gradient steps on projected
            lr: learning rate for projected optimizer

        Returns:
            refined_projected: [B, 64, 32, 32] (detached)
            loss_history: list of float loss values per step
        """
        device = projected.device

        # Freeze model params — only projected gets gradients
        orig_requires_grad = {}
        for name, p in self.named_parameters():
            orig_requires_grad[name] = p.requires_grad
            p.requires_grad_(False)

        # Optimizable copy of projected
        proj = projected.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([proj], lr=lr)

        # Target: [BOS, word1, word2, ..., EOS]
        seq = [BOS_ID] + list(target_word_ids) + [EOS_ID]
        target_tokens = torch.tensor([seq], dtype=torch.long, device=device)
        input_tokens = target_tokens[:, :-1]
        labels = target_tokens[:, 1:]

        loss_history = []
        for _ in range(num_steps):
            vis_seq = self.visual_to_text(proj.flatten(2).permute(0, 2, 1))
            logits = self.text_decoder(input_tokens, vis_seq)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_history.append(loss.item())

        # Restore model param grad tracking
        for name, p in self.named_parameters():
            p.requires_grad_(orig_requires_grad[name])

        return proj.detach(), loss_history

    def forward_text_training(self, tokens, visual_seq):
        """Teacher-forced text training.

        tokens: [B, T] — target token sequence (BOS-prefixed)
        visual_seq: [B, S, text_dim] — from forward_image
        Returns: logits [B, T, vocab_size]
        """
        return self.text_decoder(tokens, visual_seq)

    def forward_animate(self, image_tensor, num_views=V2_NUM_VIEWS,
                        elevation=0.3, distance=2.0):
        """Generate multi-view ASCII animation from a single image.

        Produces `num_views` frames rotating 360° around the object.

        Returns: list of ascii_grid tensors [B, H, W], one per viewpoint
        """
        # 1. Encode once
        image_features = self.encode_image(image_tensor)

        # 2. Generate planes once
        projected, latent, planes = self.tri_plane(image_features)

        # 3. Render from different viewpoints
        frames = []
        device = image_tensor.device
        B = image_tensor.shape[0]

        for i in range(num_views):
            azimuth = 2.0 * math.pi * i / num_views
            viewpoint = torch.tensor(
                [[azimuth, elevation, distance]],
                device=device,
            ).expand(B, -1)

            proj_i = self.tri_plane.render_from_viewpoint(planes, viewpoint)
            logits_i = self.ascii_decoder(proj_i)
            grid_i = logits_i.argmax(dim=1)  # [B, H, W]
            frames.append(grid_i)

        return frames, latent, planes

    def get_trainable_params(self):
        """Return only trainable parameters (excludes frozen encoder)."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def count_params(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def save_weights(self, path):
        """Save only trainable weights (skip frozen encoder)."""
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.data.half()  # fp16 for smaller files
        for name, buf in self.named_buffers():
            state[name] = buf
        torch.save(state, path)

    def load_weights(self, path):
        """Load trainable weights, converting fp16 back to fp32."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        current = self.state_dict()
        loaded = 0
        for key, val in state.items():
            if key in current:
                if val.shape == current[key].shape:
                    current[key] = val.float()
                    loaded += 1
        self.load_state_dict(current, strict=False)
        return loaded
