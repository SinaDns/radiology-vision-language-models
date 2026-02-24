"""Uncertainty estimation for radiology VLMs.

Covers two model families:

1. **CheXzero classification** (Extension 1):
   MC Dropout wrapper + conformal prediction calibration.

2. **R2Gen generation** (Extension from r2gen_test.ipynb):
   MC Dropout over the decoder produces stochastic report variants.
   Two uncertainty metrics are computed per sample:
   - *Disagreement*: fraction of unique decoded texts across T passes.
   - *Token entropy*: mean Shannon entropy of the output distribution
     at each token position, averaged over T passes.

Key idea (both):
  Enable Dropout layers at inference time (``m.train()`` while the rest
  of the model remains in ``eval()``), run T stochastic forward passes,
  and aggregate statistics.
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
# MC Dropout for R2Gen generation
# ─────────────────────────────────────────────────────────────────────────────

def _enable_dropout_only(model: nn.Module):
    """Set model to eval but keep Dropout layers in train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


class MCDropoutR2Gen:
    """MC Dropout uncertainty estimation for R2Gen report generation.

    Runs T stochastic decoder passes (dropout active) and computes two
    uncertainty metrics per sample:

    - **Disagreement**: fraction of unique decoded reports out of T passes.
      High disagreement = the model is unsure *what* to generate.
    - **Token entropy**: mean Shannon entropy of the output vocabulary
      distribution at each non-padding position, averaged over T passes.
      High entropy = the model is unsure *which word* to choose at each step.

    Args:
        model:     Trained :class:`~src.models.r2gen.R2GenModel`.
        tokenizer: :class:`~src.data_loaders.report_tokenizer.ReportTokenizer`.
        T:         Default number of stochastic forward passes (default 10).
    """

    def __init__(self, model: nn.Module, tokenizer, T: int = 10):
        self.model     = model
        self.tokenizer = tokenizer
        self.T         = T
        logger.info("MCDropoutR2Gen — T=%d", T)

    @torch.no_grad()
    def mc_generate(
        self,
        images: torch.Tensor,
        T: Optional[int] = None,
        max_length: int = 60,
        temperature: float = 1.0,
    ) -> dict:
        """Run T stochastic passes and return uncertainty metrics.

        Args:
            images:     ``(B, 3, H, W)`` image batch.
            T:          Number of MC passes (overrides default).
            max_length: Maximum generated sequence length.
            temperature: Softmax temperature for sampling.

        Returns:
            Dict with:
            - ``texts``        ``(T, B)`` decoded report strings per pass.
            - ``disagreement`` ``(B,)``  fraction of unique texts / T.
            - ``entropy``      ``(B,)``  mean token entropy across T passes.
        """
        n      = T or self.T
        device = next(self.model.parameters()).device

        _enable_dropout_only(self.model)
        logger.debug("mc_generate — T=%d B=%d max_length=%d", n, images.size(0), max_length)

        seqs_list: list[torch.Tensor] = []
        ents_list: list[torch.Tensor] = []

        for pass_idx in range(n):
            # sample() maintains computation graph, but we use no_grad here
            # because we only need entropy (no SCST gradient needed)
            seq, log_probs = self.model.sample(
                images,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_length=max_length,
                temperature=temperature,
            )                                          # (B, L), (B, L, V)

            seqs_list.append(seq.cpu())

            # Token entropy: H = -sum_v p_v * log(p_v)
            p   = log_probs.exp()                      # (B, L, V)
            ent = -(p * log_probs).sum(dim=-1)         # (B, L)  entropy per position
            mask    = (seq != self.tokenizer.pad_id).float().cpu()
            avg_ent = (ent.cpu() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            ents_list.append(avg_ent)                  # (B,)

            if (pass_idx + 1) % 5 == 0:
                logger.debug("MC pass %d/%d done", pass_idx + 1, n)

        # Restore full eval mode
        self.model.eval()

        # Decode all passes to text
        texts = np.array([
            [self.tokenizer.decode(row.tolist()) for row in seqs_list[t]]
            for t in range(n)
        ])                                             # (T, B)

        B = texts.shape[1]
        disagreement = np.array(
            [len(set(texts[:, i])) / n for i in range(B)]
        )                                              # (B,)

        entropy = torch.stack(ents_list, dim=0).mean(dim=0).numpy()  # (B,)

        logger.info(
            "MCDropoutR2Gen — mean_disagreement=%.4f mean_entropy=%.4f",
            disagreement.mean(), entropy.mean(),
        )
        return {
            "texts":        texts,
            "disagreement": disagreement,
            "entropy":      entropy,
        }

    def run_on_loader(
        self,
        loader,
        T: Optional[int] = None,
        max_length: int = 60,
    ) -> dict:
        """Run MC Dropout uncertainty estimation over a full DataLoader.

        Args:
            loader:     DataLoader yielding ``{image, study_id, ...}`` dicts.
            T:          MC passes per batch.
            max_length: Decoding length.

        Returns:
            Dict with ``study_ids``, ``disagreement`` ``(N,)``,
            ``entropy`` ``(N,)``, ``representative_texts`` ``(N,)``
            (first pass output for each sample).
        """
        import torch
        device = next(self.model.parameters()).device

        all_ids, all_disag, all_ent, all_texts = [], [], [], []

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            out    = self.mc_generate(images, T=T, max_length=max_length)

            all_ids.extend(list(batch.get("study_id", ["?"] * images.size(0))))
            all_disag.extend(out["disagreement"].tolist())
            all_ent.extend(out["entropy"].tolist())
            all_texts.extend([out["texts"][0, i] for i in range(images.size(0))])

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    "MC uncertainty — batch %d/%d processed",
                    batch_idx + 1, len(loader),
                )

        return {
            "study_ids":          all_ids,
            "disagreement":       np.array(all_disag),
            "entropy":            np.array(all_ent),
            "representative_texts": all_texts,
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
        labels_before: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compare ECE before/after conformal calibration.

        Args:
            scores_before:  ``(N_calib, C)`` raw model scores on calibration set.
            scores_after:   ``(N_test,  C)`` calibrated scores on test set.
            labels:         ``(N_test,  C)`` binary ground-truth for test set.
            labels_before:  ``(N_calib, C)`` binary ground-truth for calibration set.
                            If None, ``labels`` is reused (only valid when
                            N_calib == N_test).
        """
        from src.evaluation.metrics import compute_ece

        if labels_before is None:
            if scores_before.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"scores_before has {scores_before.shape[0]} rows but labels has "
                    f"{labels.shape[0]} rows. Pass labels_before for the calibration set."
                )
            labels_before = labels

        def mean_ece(s: np.ndarray, y: np.ndarray) -> float:
            probs = torch.sigmoid(torch.tensor(s)).numpy()
            eces = []
            for c in range(s.shape[1]):
                eces.append(compute_ece(probs[:, c], y[:, c]))
            return float(np.mean(eces))

        ece_before = mean_ece(scores_before, labels_before)
        ece_after  = mean_ece(scores_after,  labels)
        reduction  = (ece_before - ece_after) / max(ece_before, 1e-8)

        logger.info(
            "ECE before=%.4f after=%.4f reduction=%.1f%%",
            ece_before, ece_after, reduction * 100,
        )
        return {
            "ece_before":    ece_before,
            "ece_after":     ece_after,
            "ece_reduction": reduction,
        }

