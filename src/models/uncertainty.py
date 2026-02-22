"""Extension 1: Uncertainty-Aware Classification.

Wraps CheXzero with Monte Carlo Dropout for epistemic uncertainty
estimation, plus conformal prediction calibration for coverage guarantees.

Key ideas:
- MC Dropout: Enable dropout at inference time; run N forward passes to
  estimate prediction mean and variance.
- Conformal Prediction: Use a held-out calibration set to find a score
  threshold that guarantees (1-α) coverage of true labels.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _enable_dropout(model: nn.Module):
    """Set all Dropout layers to train mode (enables stochastic dropout at inference)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class MCDropoutCheXzero(nn.Module):
    """Monte Carlo Dropout wrapper for CheXzero.

    Injects a configurable dropout layer after the image projection head
    and enables dropout at inference time to obtain stochastic embeddings.

    Args:
        chexzero:     A trained CheXzero model.
        dropout_rate: Dropout probability for MC sampling (default 0.1).
        n_samples:    Default number of MC forward passes.
    """

    def __init__(
        self,
        chexzero: nn.Module,
        dropout_rate: float = 0.1,
        n_samples: int = 20,
    ):
        super().__init__()
        self.model      = chexzero
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.n_samples  = n_samples
        self.dropout_rate = dropout_rate

        logger.info(
            "MCDropoutCheXzero — dropout_rate=%.2f default_n_samples=%d",
            dropout_rate, n_samples,
        )

    def encode_image_stochastic(self, images: torch.Tensor) -> torch.Tensor:
        """Return a single stochastic image embedding with dropout applied."""
        feats = self.model.image_encoder.backbone(images)   # (B, 2048)
        feats = self.dropout(feats)                          # stochastic
        feats = self.model.image_encoder.norm(
            self.model.image_encoder.projection(feats)
        )
        import torch.nn.functional as F
        return F.normalize(feats, dim=-1)                    # (B, embed_dim)

    @torch.no_grad()
    def mc_encode_image(
        self,
        images: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample N stochastic embeddings and return mean + variance.

        Args:
            images:   ``(B, 3, H, W)``
            n_samples: Number of forward passes (defaults to ``self.n_samples``).

        Returns:
            Tuple ``(mean_emb, var_emb)`` each ``(B, embed_dim)``.
        """
        n = n_samples or self.n_samples
        _enable_dropout(self)   # ensure dropout is active

        samples = []
        for i in range(n):
            emb = self.encode_image_stochastic(images)  # (B, D)
            samples.append(emb.unsqueeze(0))             # (1, B, D)

        stacked = torch.cat(samples, dim=0)   # (n, B, D)
        mean_emb = stacked.mean(dim=0)        # (B, D)
        var_emb  = stacked.var(dim=0)         # (B, D)

        logger.debug(
            "mc_encode_image — n_samples=%d mean_var=%.4f max_var=%.4f",
            n, var_emb.mean().item(), var_emb.max().item(),
        )
        return mean_emb, var_emb

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute per-class scores and uncertainty estimates.

        Args:
            images:          ``(B, 3, H, W)``
            text_embeddings: ``(C, embed_dim)`` pre-computed text embeddings
                             for C classes.
            n_samples:       MC samples.

        Returns:
            Dict with:
            - ``mean_scores``   ``(B, C)`` — mean cosine similarity scores
            - ``uncertainty``   ``(B, C)`` — predictive variance per class
            - ``mean_img_emb``  ``(B, D)`` — mean image embedding
        """
        mean_emb, var_emb = self.mc_encode_image(images, n_samples)

        # mean scores: dot product of mean embedding with text embeddings
        mean_scores = mean_emb @ text_embeddings.T    # (B, C)

        # uncertainty: variance of the dot product (first-order approximation)
        # Var[x·w] ≈ w^T Var[x] w  (diagonal approximation: sum(var_x * w^2))
        uncertainty = var_emb @ (text_embeddings ** 2).T   # (B, C)

        logger.debug(
            "predict_with_uncertainty — B=%d C=%d "
            "mean_uncertainty=%.4f",
            images.shape[0], text_embeddings.shape[0],
            uncertainty.mean().item(),
        )

        return {
            "mean_scores": mean_scores,
            "uncertainty": uncertainty,
            "mean_img_emb": mean_emb,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Conformal Prediction
# ─────────────────────────────────────────────────────────────────────────────

class ConformalPredictor:
    """Split conformal prediction for multi-label classification.

    Uses a calibration set of (score, label) pairs to find per-class
    non-conformity thresholds that guarantee marginal coverage.

    Algorithm (RAPS / threshold conformal):
    1. Compute non-conformity scores on calibration set:
           s_i = 1 - score_i  (for the true class)
    2. Set threshold τ = (1 - α) quantile of calibration scores
    3. At test time: predict positive if score > 1 - τ

    Args:
        alpha: Desired error rate (default 0.1 → 90% coverage).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha      = alpha
        self.thresholds: Optional[np.ndarray] = None   # (C,) per-class thresholds
        self.n_labels: int = 0

        logger.info("ConformalPredictor — alpha=%.2f", alpha)

    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ):
        """Fit thresholds from calibration data.

        Args:
            scores: ``(N, C)`` predicted scores for N samples, C classes.
            labels: ``(N, C)`` binary ground-truth labels.
        """
        N, C = scores.shape
        self.n_labels = C
        thresholds = np.zeros(C)

        for c in range(C):
            pos_mask = labels[:, c] == 1
            if pos_mask.sum() == 0:
                logger.warning("Class %d has no positive calibration examples", c)
                thresholds[c] = 0.0
                continue

            # Non-conformity score: 1 - predicted_score for positives
            nc_scores = 1.0 - scores[pos_mask, c]
            n_pos = pos_mask.sum()
            level = np.ceil((n_pos + 1) * (1 - self.alpha)) / n_pos
            level = min(level, 1.0)
            thresholds[c] = np.quantile(nc_scores, level)

        self.thresholds = thresholds
        logger.info(
            "Conformal calibration done — N=%d C=%d alpha=%.2f "
            "mean_threshold=%.4f",
            N, C, self.alpha, thresholds.mean(),
        )

    def predict_set(self, scores: np.ndarray) -> np.ndarray:
        """Return binary prediction sets.

        Args:
            scores: ``(N, C)`` test scores.

        Returns:
            ``(N, C)`` binary array — 1 means the label is included in the
            prediction set.
        """
        if self.thresholds is None:
            raise RuntimeError("Call calibrate() before predict_set()")

        # Include label c if score > 1 - threshold_c
        return (scores > (1.0 - self.thresholds[None, :])).astype(np.float32)

    def compute_coverage(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute empirical coverage and average set size.

        Args:
            scores: ``(N, C)`` test scores.
            labels: ``(N, C)`` binary ground-truth.

        Returns:
            Dict with ``coverage`` and ``avg_set_size``.
        """
        pred_sets = self.predict_set(scores)
        # Coverage: fraction of samples where all true labels are predicted
        covered = np.all((labels == 0) | (pred_sets == 1), axis=1)
        coverage = covered.mean()
        avg_size = pred_sets.sum(axis=1).mean()

        logger.info(
            "Coverage=%.4f (target=%.4f) avg_set_size=%.2f",
            coverage, 1 - self.alpha, avg_size,
        )
        return {"coverage": float(coverage), "avg_set_size": float(avg_size)}

    def compute_ece_reduction(
        self,
        scores_before: np.ndarray,
        scores_after: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Compare ECE before/after conformal calibration.

        Args:
            scores_before: ``(N, C)`` raw model scores.
            scores_after:  ``(N, C)`` calibrated (threshold-shifted) scores.
            labels:        ``(N, C)`` binary ground-truth.

        Returns:
            Dict with ``ece_before``, ``ece_after``, ``ece_reduction``.
        """
        from src.evaluation.metrics import compute_ece
        import torch.nn.functional as F
        import torch

        def mean_ece(s, y):
            # Convert scores to probabilities via sigmoid
            probs = torch.sigmoid(torch.tensor(s)).numpy()
            eces = []
            for c in range(s.shape[1]):
                eces.append(compute_ece(probs[:, c], y[:, c]))
            return float(np.mean(eces))

        ece_before = mean_ece(scores_before, labels)
        ece_after  = mean_ece(scores_after,  labels)
        reduction  = (ece_before - ece_after) / max(ece_before, 1e-8)

        logger.info(
            "ECE before=%.4f after=%.4f reduction=%.1f%%",
            ece_before, ece_after, reduction * 100,
        )
        return {
            "ece_before": ece_before,
            "ece_after":  ece_after,
            "ece_reduction": reduction,
        }
