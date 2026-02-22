import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel


class ImageEncoder(nn.Module):
    """ResNet-50 backbone with a linear projection head.

    Output: L2-normalised embeddings of shape (B, embed_dim).
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        # Remove the final classification layer; keep 2048-dim pool output.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projection = nn.Linear(2048, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)          # (B, 2048)
        projected = self.norm(self.projection(features))   # (B, embed_dim)
        return F.normalize(projected, dim=-1)


class TextEncoder(nn.Module):
    """BioClinicalBERT text encoder with a linear projection head.

    Output: L2-normalised embeddings of shape (B, embed_dim).
    """

    BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(self.BERT_MODEL)
        bert_hidden = self.bert.config.hidden_size   # 768 for BERT-base

        self.projection = nn.Linear(bert_hidden, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        projected = self.norm(self.projection(cls_token))  # (B, embed_dim)
        return F.normalize(projected, dim=-1)


class CheXzero(nn.Module):
    """CLIP-style contrastive VLM for chest X-ray zero-shot classification.

    Combines ImageEncoder + TextEncoder with a learnable temperature parameter.
    """

    def __init__(self, embed_dim: int = 512, pretrained_image: bool = True):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained_image)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        # Learnable log temperature; initialised to log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images)   # (B, D)
        text_features = self.encode_text(input_ids, attention_mask)  # (B, D)

        scale = self.logit_scale.exp()
        logits_per_image = scale * image_features @ text_features.T   # (B, B)
        logits_per_text = logits_per_image.T                           # (B, B)

        return logits_per_image, logits_per_text
