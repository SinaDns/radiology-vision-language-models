"""Extension 2: Factuality-Constrained Generation.

Implements a RadGraph-based factuality reward that can be used to
fine-tune R2Gen reports towards higher clinical accuracy.

Two modes:
1. **Direct RadGraph F1** (requires ``radgraph`` package + PhysioNet access):
   Computes full entity/relation F1 using the RadGraph model.
2. **Proxy factuality loss** (always available):
   Penalises the model for predicting tokens that don't appear in the
   reference at all — a lightweight bag-of-clinically-important-words
   approximation.

The proxy loss is used for training when RadGraph is unavailable;
the full RadGraph F1 is used for final evaluation.
"""

import logging
import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Clinical keywords list used by the proxy factuality reward.
# Covers disease names, anatomy, and key radiological descriptors.
CLINICAL_KEYWORDS = {
    "atelectasis", "consolidation", "effusion", "pneumothorax", "pneumonia",
    "edema", "cardiomegaly", "emphysema", "fibrosis", "nodule", "mass",
    "infiltrate", "opacity", "hilar", "mediastinum", "pleural", "cardiac",
    "lung", "lobe", "bilateral", "unilateral", "normal", "clear", "enlarged",
    "calcification", "diaphragm", "costophrenic", "sinus", "trachea",
    "widening", "blunting", "haziness", "airspace", "interstitial",
}


def _tokenise(text: str) -> set[str]:
    """Lower-case word tokeniser → set of words."""
    return set(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


# ─────────────────────────────────────────────────────────────────────────────
# Proxy Factuality Loss
# ─────────────────────────────────────────────────────────────────────────────

class ProxyFactualityLoss(nn.Module):
    """Lightweight factuality loss that encourages generation of reference terms.

    For each sample in a batch, computes the cross-entropy loss on positions
    where the target token is a clinical keyword, down-weighted by a
    ``coverage_weight`` factor.  This steers generation towards including
    clinically relevant terms without requiring RadGraph.

    The total loss is a weighted sum of the standard CE loss and the
    factuality penalty:

    .. code-block:: text

        L = (1 - λ) * CE_loss + λ * FactualityPenalty

    Args:
        vocab:           Dict ``{word: id}`` from the ReportTokenizer.
        coverage_weight: Weight λ ∈ [0, 1] for factuality term (default 0.3).
        pad_id:          Padding token ID.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        coverage_weight: float = 0.3,
        pad_id: int = 0,
    ):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.pad_id = pad_id

        # Build set of clinical keyword token IDs from vocabulary
        clinical_ids = set()
        for kw in CLINICAL_KEYWORDS:
            if kw in vocab:
                clinical_ids.add(vocab[kw])

        self.register_buffer(
            "clinical_mask",
            torch.zeros(len(vocab), dtype=torch.bool),
        )
        for cid in clinical_ids:
            if cid < len(vocab):
                self.clinical_mask[cid] = True

        logger.info(
            "ProxyFactualityLoss — coverage_weight=%.2f "
            "clinical_tokens_in_vocab=%d / %d",
            coverage_weight, len(clinical_ids), len(vocab),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits:  ``(B, T, V)`` model output logits.
            targets: ``(B, T)`` ground-truth token IDs.

        Returns:
            Tuple ``(total_loss, ce_loss, factuality_loss)``.
        """
        B, T, V = logits.shape
        logits_flat  = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)

        # Standard cross-entropy (ignore pad)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.pad_id)

        # Factuality loss: CE restricted to clinical keyword positions
        clinical_pos = self.clinical_mask[targets_flat]         # (B*T,) bool
        non_pad_pos  = targets_flat != self.pad_id

        active = clinical_pos & non_pad_pos
        if active.sum() == 0:
            # No clinical keywords in this batch — no extra loss
            factuality_loss = torch.zeros_like(ce_loss)
        else:
            factuality_loss = F.cross_entropy(
                logits_flat[active],
                targets_flat[active],
            )

        total = (1 - self.coverage_weight) * ce_loss + self.coverage_weight * factuality_loss

        logger.debug(
            "ProxyFactualityLoss — ce=%.4f factuality=%.4f total=%.4f "
            "clinical_positions=%d/%d",
            ce_loss.item(), factuality_loss.item(), total.item(),
            active.sum().item(), B * T,
        )

        return total, ce_loss, factuality_loss


# ─────────────────────────────────────────────────────────────────────────────
# RadGraph F1 (requires radgraph package)
# ─────────────────────────────────────────────────────────────────────────────

def compute_radgraph_f1(
    hypotheses: list[str],
    references: list[str],
    reward_level: str = "partial",
) -> dict[str, float]:
    """Compute RadGraph F1 between generated and reference reports.

    Requires the ``radgraph`` package (PhysioNet credentialed access).
    Falls back to a simple token-overlap F1 if radgraph is not installed.

    Args:
        hypotheses:   List of generated report strings.
        references:   List of ground-truth report strings.
        reward_level: ``"exact"`` or ``"partial"`` (default ``"partial"``).

    Returns:
        Dict with ``radgraph_f1`` (mean over all samples).
    """
    try:
        from src.utils.radgraph_compat import patch_transformers_for_radgraph
        patch_transformers_for_radgraph()

        from radgraph import F1RadGraph
        scorer = F1RadGraph(reward_level=reward_level)

        mean_f1, *_ = scorer(
            refs=references,
            hyps=hypotheses,
        )
        logger.info("RadGraph F1 (level=%s): %.4f", reward_level, mean_f1)
        return {"radgraph_f1": float(mean_f1)}

    except Exception as exc:
        logger.warning(
            "RadGraph F1 unavailable (%s: %s) — falling back to proxy token-overlap F1. "
            "Common cause: transformers>=4.46 removed encode_plus, which the bundled "
            "allennlp inside radgraph still calls.",
            type(exc).__name__, exc,
        )
        return _proxy_radgraph_f1(hypotheses, references)


def _proxy_radgraph_f1(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Token-overlap F1 restricted to clinical keywords — RadGraph proxy."""
    f1_scores = []

    for hyp, ref in zip(hypotheses, references):
        hyp_toks = _tokenise(hyp) & CLINICAL_KEYWORDS
        ref_toks = _tokenise(ref) & CLINICAL_KEYWORDS

        if not ref_toks:
            continue

        tp = len(hyp_toks & ref_toks)
        precision = tp / max(len(hyp_toks), 1)
        recall    = tp / max(len(ref_toks), 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores.append(f1)

    mean_f1 = float(sum(f1_scores) / max(len(f1_scores), 1))
    logger.info("Proxy RadGraph F1 (clinical token overlap): %.4f", mean_f1)
    return {"radgraph_f1_proxy": mean_f1}
