"""R2Gen: Memory-Driven Transformer for radiology report generation.

Architecture follows Chen et al. (EMNLP 2020):
  - VisualExtractor:   ResNet-101 (ImageNet) → (B, 49, 2048) patch features
  - RelationalMemory:  Learnable memory slots updated by image features
  - MeshedDecoder:     Multi-layer cross-attention decoder (image + memory)
  - Output:            Linear projection to vocabulary logits

Reference: https://github.com/cuhksz-nlp/R2Gen
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Extractor
# ─────────────────────────────────────────────────────────────────────────────

class VisualExtractor(nn.Module):
    """ResNet-101 backbone that outputs spatial patch features.

    The final global average pooling and FC layers are removed; instead
    the 7×7 spatial feature map from the last residual block is returned
    as a sequence of 49 patch feature vectors.

    Args:
        pretrained:   Load ImageNet weights (default True).
        d_model:      Model hidden dimension.  A linear layer projects
                      2048-dim ResNet features to ``d_model``.
    """

    def __init__(self, pretrained: bool = True, d_model: int = 512):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet101(weights=weights)

        # Keep everything up to (and including) layer4; remove avgpool + fc
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Linear(2048, d_model)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "VisualExtractor — pretrained=%s d_model=%d params=%.2fM",
            pretrained, d_model, n_params / 1e6,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: ``(B, 3, H, W)``
        Returns:
            ``(B, 49, d_model)`` patch feature sequence.
        """
        feats = self.feature_extractor(images)      # (B, 2048, 7, 7)
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1)           # (B, 7, 7, 2048)
        feats = feats.reshape(B, H * W, C)          # (B, 49, 2048)
        feats = self.projection(feats)              # (B, 49, d_model)
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# Relational Memory
# ─────────────────────────────────────────────────────────────────────────────

class RelationalMemory(nn.Module):
    """Memory module that maintains global context across visual patches.

    Learnable memory slots are updated by attending over the visual patch
    features.  The resulting memory is then used to condition the decoder
    via a second attention layer.

    Args:
        num_slots: Number of memory slots (default 3).
        d_model:   Hidden dimension.
        num_heads: Attention heads for memory update (default 8).
        dropout:   Dropout probability.
    """

    def __init__(
        self,
        num_slots: int = 3,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.d_model   = d_model

        # Learnable initial memory
        self.memory_init = nn.Parameter(
            torch.zeros(1, num_slots, d_model)
        )
        nn.init.xavier_uniform_(self.memory_init)

        # Memory update: memory attends over visual patches
        self.update_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.update_norm = nn.LayerNorm(d_model)
        self.update_ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

        logger.info(
            "RelationalMemory — num_slots=%d d_model=%d num_heads=%d",
            num_slots, d_model, num_heads,
        )

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: ``(B, 49, d_model)`` patch features.
        Returns:
            ``(B, num_slots, d_model)`` updated memory.
        """
        B = visual_feats.shape[0]
        memory = self.memory_init.expand(B, -1, -1)           # (B, S, D)

        # Memory slots attend over visual patches (cross-attention)
        mem_out, _ = self.update_attn(memory, visual_feats, visual_feats)
        memory = self.update_norm(memory + mem_out)           # residual + norm

        # Feed-forward
        memory = self.ff_norm(memory + self.update_ff(memory))

        return memory   # (B, S, D)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Encoder (visual)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """Standard pre-norm transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop(x2)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Meshed Decoder
# ─────────────────────────────────────────────────────────────────────────────

class MeshedDecoderLayer(nn.Module):
    """Decoder layer with self-attention, cross-attention to visual patches,
    and cross-attention to relational memory.

    The two cross-attention outputs are combined with learnable weights
    (the "mesh" in meshed-memory).
    """

    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn    = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_vis    = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_mem    = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Learnable gating weights for mesh combination
        self.vis_gate = nn.Linear(d_model, d_model)
        self.mem_gate = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        visual: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt:                  ``(B, T, D)`` decoder input.
            visual:               ``(B, 49, D)`` visual patch features.
            memory:               ``(B, S, D)`` relational memory.
            tgt_mask:             Causal mask ``(T, T)``.
            tgt_key_padding_mask: Padding mask ``(B, T)``.
        Returns:
            ``(B, T, D)`` decoder output.
        """
        # Self-attention (causal)
        x2, _ = self.self_attn(
            self.norm1(tgt), self.norm1(tgt), self.norm1(tgt),
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.drop(x2)

        # Cross-attention to visual features
        v2, _ = self.cross_vis(self.norm2(tgt), visual, visual)
        # Cross-attention to memory
        m2, _ = self.cross_mem(self.norm2(tgt), memory, memory)

        # Meshed gated combination
        alpha = torch.sigmoid(self.vis_gate(self.norm2(tgt)))
        beta  = torch.sigmoid(self.mem_gate(self.norm2(tgt)))
        tgt   = tgt + self.drop(alpha * v2 + beta * m2)

        # Feed-forward
        tgt = tgt + self.drop(self.ff(self.norm3(tgt)))
        return tgt


# ─────────────────────────────────────────────────────────────────────────────
# R2Gen Model
# ─────────────────────────────────────────────────────────────────────────────

class R2GenModel(nn.Module):
    """Memory-Driven Transformer for radiology report generation.

    Args:
        vocab_size:    Size of report vocabulary.
        d_model:       Hidden dimension (default 512).
        num_heads:     Attention heads (default 8).
        num_enc_layers: Transformer encoder layers for visual features.
        num_dec_layers: Meshed decoder layers.
        dim_ff:        Feed-forward inner dimension (default 2048).
        dropout:       Dropout probability (default 0.1).
        num_mem_slots: Relational memory slots (default 3).
        max_seq_len:   Maximum generated sequence length (default 100).
        pretrained_image: Use ImageNet weights for ResNet-101.
        pad_id:        Padding token ID (default 0).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_enc_layers: int = 3,
        num_dec_layers: int = 3,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        num_mem_slots: int = 3,
        max_seq_len: int = 100,
        pretrained_image: bool = True,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.pad_id      = pad_id

        # ── Encoder side ────────────────────────────────────────────────────
        self.visual_extractor = VisualExtractor(pretrained=pretrained_image, d_model=d_model)
        self.relational_memory = RelationalMemory(
            num_slots=num_mem_slots, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_enc_layers)
        ])

        # ── Decoder side ────────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding    = PositionalEncoding(d_model, max_len=max_seq_len + 2, dropout=dropout)
        self.decoder_layers  = nn.ModuleList([
            MeshedDecoderLayer(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_dec_layers)
        ])

        # ── Output projection ────────────────────────────────────────────────
        self.output_norm  = nn.LayerNorm(d_model)
        self.output_proj  = nn.Linear(d_model, vocab_size)

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "R2GenModel — vocab_size=%d d_model=%d heads=%d "
            "enc_layers=%d dec_layers=%d mem_slots=%d total_params=%.2fM",
            vocab_size, d_model, num_heads, num_enc_layers, num_dec_layers,
            num_mem_slots, total / 1e6,
        )

    def _init_weights(self):
        for p in self.output_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Encoding ────────────────────────────────────────────────────────────

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images into visual patch features and relational memory.

        Args:
            images: ``(B, 3, H, W)``
        Returns:
            Tuple ``(visual_feats, memory)``
            - visual_feats: ``(B, 49, d_model)``
            - memory:       ``(B, num_slots, d_model)``
        """
        visual = self.visual_extractor(images)      # (B, 49, d_model)

        # Apply encoder layers
        enc_out = visual
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)                # (B, 49, d_model)

        memory = self.relational_memory(enc_out)    # (B, S, d_model)

        logger.debug(
            "encode — images=%s visual=%s memory=%s",
            list(images.shape), list(enc_out.shape), list(memory.shape),
        )
        return enc_out, memory

    # ── Decoding (teacher-forced, training) ──────────────────────────────────

    def decode(
        self,
        input_ids: torch.Tensor,
        visual: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced decoding for training.

        Args:
            input_ids: ``(B, T)`` token IDs (BOS + tokens).
            visual:    ``(B, 49, d_model)`` from :meth:`encode`.
            memory:    ``(B, S, d_model)`` from :meth:`encode`.
        Returns:
            ``(B, T, vocab_size)`` logits.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

        # Padding mask
        pad_mask = (input_ids == self.pad_id)   # (B, T) True = ignore

        # Embed + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            x = layer(
                tgt=x,
                visual=visual,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=pad_mask,
            )

        logits = self.output_proj(self.output_norm(x))   # (B, T, vocab_size)
        return logits

    # ── Forward (teacher-forced) ─────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images:    ``(B, 3, H, W)``
            input_ids: ``(B, T)`` decoder input (BOS + tokens).
        Returns:
            ``(B, T, vocab_size)`` logits.
        """
        visual, memory = self.encode(images)
        return self.decode(input_ids, visual, memory)

    # ── Stochastic sampling (SCST training) ─────────────────────────────────

    def sample(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stochastic (multinomial) sampling for SCST policy-gradient training.

        Decodes step-by-step; at each step the next token is drawn from the
        categorical distribution defined by the decoder output.  The full
        log-softmax distribution at every step is retained so the caller can
        compute ``sum_t log p(a_t)`` for the REINFORCE gradient.

        Args:
            images:      ``(B, 3, H, W)``
            bos_id:      Beginning-of-sequence token ID.
            eos_id:      End-of-sequence token ID.
            max_length:  Maximum tokens to generate.
            temperature: Softmax temperature (1.0 = unmodified).

        Returns:
            Tuple ``(seq, log_probs)``
            - ``seq``       ``(B, max_length)`` sampled token IDs,
                            ``pad_id`` after the first EOS per row.
            - ``log_probs`` ``(B, max_length, vocab_size)`` log-softmax
                            distribution at each step (differentiable).
        """
        if max_length is None:
            max_length = self.max_seq_len

        B      = images.size(0)
        device = images.device
        visual, memory = self.encode(images)

        seqs          = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        all_log_probs: list[torch.Tensor] = []
        done          = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length):
            logits    = self.decode(seqs, visual, memory)           # (B, t, V)
            step_log  = F.log_softmax(
                logits[:, -1, :] / max(temperature, 1e-8), dim=-1
            )                                                        # (B, V)
            all_log_probs.append(step_log.unsqueeze(1))             # (B, 1, V)

            # Sample next token (no gradient through discrete choice)
            with torch.no_grad():
                next_tok = torch.multinomial(
                    step_log.exp(), num_samples=1
                )                                                    # (B, 1)
                next_tok[done] = self.pad_id

            seqs = torch.cat([seqs, next_tok], dim=1)
            done = done | (next_tok.squeeze(-1) == eos_id)
            if done.all():
                break

        seq       = seqs[:, 1:]                                      # remove BOS
        log_probs = torch.cat(all_log_probs, dim=1)                 # (B, L, V)

        # Pad to max_length if decoding stopped early
        L = seq.size(1)
        if L < max_length:
            seq       = F.pad(seq,       (0, max_length - L),        value=self.pad_id)
            log_probs = F.pad(log_probs, (0, 0, 0, max_length - L), value=0.0)

        return seq[:, :max_length], log_probs[:, :max_length]

    # ── Greedy decoding (SCST baseline) ─────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Deterministic greedy (argmax) decoding for the SCST baseline.

        Used as the ``reward_baseline`` in Self-Critical Sequence Training:
        the stochastic sample is compared against the greedy decode of the
        *same* model state to compute the advantage.

        Args:
            images:     ``(B, 3, H, W)``
            bos_id:     Beginning-of-sequence token ID.
            eos_id:     End-of-sequence token ID.
            max_length: Maximum tokens to decode.

        Returns:
            ``(B, max_length)`` greedy token IDs, ``pad_id`` after EOS.
        """
        if max_length is None:
            max_length = self.max_seq_len

        B      = images.size(0)
        device = images.device
        visual, memory = self.encode(images)

        seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length):
            logits   = self.decode(seqs, visual, memory)             # (B, t, V)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True) # (B, 1)
            next_tok[done] = self.pad_id
            seqs = torch.cat([seqs, next_tok], dim=1)
            done = done | (next_tok.squeeze(-1) == eos_id)
            if done.all():
                break

        seq = seqs[:, 1:]                                            # remove BOS
        L   = seq.size(1)
        if L < max_length:
            seq = F.pad(seq, (0, max_length - L), value=self.pad_id)

        return seq[:, :max_length]

    # ── Beam search (inference) ──────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        beam_size: int = 3,
        max_length: Optional[int] = None,
    ) -> list[list[int]]:
        """Generate reports for a batch of images using beam search.

        Args:
            images:    ``(B, 3, H, W)``
            bos_id:    Beginning-of-sequence token ID.
            eos_id:    End-of-sequence token ID.
            beam_size: Beam width (default 3).
            max_length: Maximum tokens to generate (default ``self.max_seq_len``).
        Returns:
            List of B token ID lists (best beam for each image).
        """
        if max_length is None:
            max_length = self.max_seq_len

        device = images.device
        B = images.shape[0]
        visual, memory = self.encode(images)

        # --- simple greedy decode for speed; beam search below is per-image ---
        results: list[list[int]] = []

        for b in range(B):
            vis_b = visual[b : b + 1]    # (1, 49, D)
            mem_b = memory[b : b + 1]    # (1, S, D)

            # Beam: list of (log_prob, token_ids)
            beams: list[tuple[float, list[int]]] = [(0.0, [bos_id])]
            completed: list[tuple[float, list[int]]] = []

            for step in range(max_length):
                new_beams: list[tuple[float, list[int]]] = []

                for log_p, seq in beams:
                    if seq[-1] == eos_id:
                        completed.append((log_p, seq))
                        continue

                    ids_t = torch.tensor([seq], dtype=torch.long, device=device)
                    logits = self.decode(ids_t, vis_b, mem_b)    # (1, T, V)
                    log_probs = F.log_softmax(logits[0, -1], dim=-1)  # (V,)

                    top_lp, top_ids = log_probs.topk(beam_size)
                    for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                        new_beams.append((log_p + lp, seq + [tid]))

                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_size]

                if all(seq[-1] == eos_id for _, seq in beams):
                    completed.extend(beams)
                    break

            if not completed:
                completed = beams

            best_seq = max(completed, key=lambda x: x[0])[1]
            results.append(best_seq)

            if (b + 1) % 50 == 0:
                logger.debug("Beam search — generated %d/%d", b + 1, B)

        return results
