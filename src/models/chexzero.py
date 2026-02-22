"""CheXzero model: ResNet-50 image encoder + BioClinicalBERT text encoder.

Architecture follows Tiu et al. (Nature BME 2022):
  - Image branch: ResNet-50 (ImageNet pretrained) → 2048-d → Linear → 512-d → L2-norm
  - Text branch:  Bio_ClinicalBERT CLS token → 768-d → Linear → 512-d → L2-norm
  - Learnable temperature parameter (logit_scale)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """ResNet-50 backbone with a linear projection head.

    Output: L2-normalised embeddings of shape ``(B, embed_dim)``.

    Args:
        embed_dim:  Output embedding dimension (default 512).
        pretrained: Load ImageNet-1K weights (default True).
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()          # keep 2048-dim global avg-pool output
        self.backbone = backbone
        self.projection = nn.Linear(2048, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "ImageEncoder — embed_dim=%d pretrained=%s params=%.2fM",
            embed_dim, pretrained, n_params / 1e6,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: ``(B, 3, H, W)``
        Returns:
            ``(B, embed_dim)`` L2-normalised embeddings.
        """
        features = self.backbone(images)                    # (B, 2048)
        projected = self.norm(self.projection(features))    # (B, embed_dim)
        return F.normalize(projected, dim=-1)


class TextEncoder(nn.Module):
    """BioClinicalBERT text encoder with a linear projection head.

    Uses the ``[CLS]`` token from the last hidden state as the sentence
    representation.

    Output: L2-normalised embeddings of shape ``(B, embed_dim)``.

    Args:
        embed_dim: Output embedding dimension (default 512).
    """

    BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        logger.info("TextEncoder — loading %s …", self.BERT_MODEL)
        self.bert = AutoModel.from_pretrained(self.BERT_MODEL)
        bert_hidden = self.bert.config.hidden_size          # 768 for BERT-base

        self.projection = nn.Linear(bert_hidden, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "TextEncoder — bert_hidden=%d embed_dim=%d params=%.2fM",
            bert_hidden, embed_dim, n_params / 1e6,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      ``(B, seq_len)``
            attention_mask: ``(B, seq_len)``
        Returns:
            ``(B, embed_dim)`` L2-normalised embeddings.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]      # (B, 768)
        projected = self.norm(self.projection(cls_token))   # (B, embed_dim)
        return F.normalize(projected, dim=-1)


class CheXzero(nn.Module):
    """CLIP-style contrastive VLM for chest X-ray zero-shot classification.

    Composes :class:`ImageEncoder` and :class:`TextEncoder` with a learnable
    temperature parameter ``logit_scale``.

    Args:
        embed_dim:        Shared embedding dimension (default 512).
        pretrained_image: Use ImageNet pretrained ResNet-50 (default True).
    """

    def __init__(self, embed_dim: int = 512, pretrained_image: bool = True):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained_image)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        # Learnable log temperature; initialised to log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "CheXzero — embed_dim=%d total_params=%.2fM "
            "initial_temperature=%.4f",
            embed_dim, total / 1e6, 1 / math.exp(math.log(1 / 0.07)),
        )

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised image embeddings ``(B, embed_dim)``."""
        return self.image_encoder(images)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return L2-normalised text embeddings ``(B, embed_dim)``."""
        return self.text_encoder(input_ids, attention_mask)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images:         ``(B, 3, H, W)``
            input_ids:      ``(B, seq_len)``
            attention_mask: ``(B, seq_len)``
        Returns:
            Tuple ``(logits_per_image, logits_per_text)`` each ``(B, B)``.
        """
        image_features = self.encode_image(images)           # (B, D)
        text_features = self.encode_text(input_ids, attention_mask)  # (B, D)

        scale = self.logit_scale.exp()
        logits_per_image = scale * image_features @ text_features.T   # (B, B)
        logits_per_text = logits_per_image.T                           # (B, B)

        logger.debug(
            "CheXzero forward — batch=%d scale=%.3f "
            "img_norm_mean=%.3f txt_norm_mean=%.3f",
            images.shape[0], scale.item(),
            image_features.norm(dim=-1).mean().item(),
            text_features.norm(dim=-1).mean().item(),
        )

        return logits_per_image, logits_per_text
