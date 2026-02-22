"""Extension 3: Cross-Dataset Robustness Evaluation.

Provides tools for systematic IU X-Ray → NIH-14 evaluation:
- Domain gap quantification (mean feature cosine distance)
- Per-class AUROC comparison (trained vs zero-shot)
- Few-shot linear probe adaptation on NIH-14

The few-shot adapter trains a small linear classifier on top of frozen
CheXzero image embeddings, using only k labelled NIH-14 images per class.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.evaluation.metrics import compute_auroc

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain Gap Analysis
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Extract L2-normalised image embeddings for an entire dataset.

    Args:
        model:  CheXzero model (uses ``encode_image``).
        loader: DataLoader yielding ``{image: Tensor}``.
        device: Compute device.

    Returns:
        ``(N, embed_dim)`` CPU tensor of all embeddings.
    """
    model.eval()
    all_embs = []
    start = time.time()

    logger.info("Extracting embeddings — %d batches …", len(loader))

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        emb    = model.encode_image(images).cpu()
        all_embs.append(emb)

        if (batch_idx + 1) % 100 == 0:
            logger.info(
                "  Extracted %d batches — elapsed=%.1fs",
                batch_idx + 1, time.time() - start,
            )

    result = torch.cat(all_embs, dim=0)
    logger.info(
        "Embedding extraction done — shape=%s time=%.1fs",
        list(result.shape), time.time() - start,
    )
    return result


def compute_domain_gap(
    src_embeddings: torch.Tensor,
    tgt_embeddings: torch.Tensor,
    n_samples: int = 5000,
) -> dict[str, float]:
    """Quantify domain gap between source and target embeddings.

    Metrics:
    - ``mean_cosine_sim``: Average cosine similarity between random src/tgt pairs.
      Values close to 0 indicate high domain gap.
    - ``centroid_distance``: L2 distance between distribution centroids (on unit sphere).
    - ``std_ratio``: Ratio of embedding norms std (src / tgt).

    Args:
        src_embeddings: ``(N_src, D)`` source domain embeddings (IU X-Ray).
        tgt_embeddings: ``(N_tgt, D)`` target domain embeddings (NIH-14).
        n_samples:      Number of random pairs for cosine similarity estimate.

    Returns:
        Dict with domain gap metrics.
    """
    logger.info(
        "Computing domain gap — src=%d tgt=%d n_samples=%d",
        len(src_embeddings), len(tgt_embeddings), n_samples,
    )

    # Sample random pairs
    n_src = min(n_samples, len(src_embeddings))
    n_tgt = min(n_samples, len(tgt_embeddings))

    idx_src = torch.randperm(len(src_embeddings))[:n_src]
    idx_tgt = torch.randperm(len(tgt_embeddings))[:n_tgt]

    s = F.normalize(src_embeddings[idx_src], dim=-1)
    t = F.normalize(tgt_embeddings[idx_tgt], dim=-1)

    # Mean pairwise cosine similarity (subsample to avoid O(N^2) cost)
    n_pairs = min(n_src, n_tgt, 1000)
    cos_sims = (s[:n_pairs] * t[:n_pairs]).sum(dim=-1)
    mean_cos = cos_sims.mean().item()

    # Centroid distance
    src_centroid = F.normalize(s.mean(dim=0, keepdim=True), dim=-1)
    tgt_centroid = F.normalize(t.mean(dim=0, keepdim=True), dim=-1)
    centroid_cos = (src_centroid * tgt_centroid).sum().item()
    centroid_dist = (1 - centroid_cos)   # range [0, 2]

    # Norm std ratio
    src_norms = src_embeddings[idx_src].norm(dim=-1)
    tgt_norms = tgt_embeddings[idx_tgt].norm(dim=-1)
    std_ratio = (src_norms.std() / tgt_norms.std().clamp(min=1e-8)).item()

    metrics = {
        "mean_cosine_sim":  round(mean_cos, 4),
        "centroid_distance": round(centroid_dist, 4),
        "std_ratio":         round(std_ratio, 4),
    }
    logger.info("Domain gap metrics: %s", metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Few-Shot Linear Probe
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Single linear layer on top of frozen embeddings for few-shot adaptation.

    Args:
        embed_dim: Input embedding dimension.
        n_classes: Number of output classes.
    """

    def __init__(self, embed_dim: int = 512, n_classes: int = 14):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        logger.info("LinearProbe — embed_dim=%d n_classes=%d", embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)   # (B, n_classes)


def few_shot_adapt(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_shot: int,
    n_classes: int = 14,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> LinearProbe:
    """Train a linear probe on k labelled samples per class.

    Args:
        embeddings: ``(N, D)`` pre-computed embeddings (frozen encoder).
        labels:     ``(N, C)`` multi-hot binary labels.
        k_shot:     Number of labelled examples per class (for sampling).
        n_classes:  Number of classes (default 14 for NIH-14).
        n_epochs:   Training epochs for the probe (default 100).
        lr:         Learning rate (default 1e-3).
        device:     Compute device.

    Returns:
        Trained :class:`LinearProbe`.
    """
    logger.info(
        "Few-shot linear probe — k_shot=%d n_classes=%d n_epochs=%d",
        k_shot, n_classes, n_epochs,
    )

    # Sample k examples per class (multi-label: pick by label presence)
    selected_idx = set()
    for c in range(n_classes):
        pos_idx = (labels[:, c] == 1).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            logger.warning("Class %d has no positive samples for few-shot sampling", c)
            continue
        chosen = pos_idx[torch.randperm(len(pos_idx))[:k_shot]]
        selected_idx.update(chosen.tolist())

    if not selected_idx:
        raise ValueError("No samples selected for few-shot adaptation.")

    sel = sorted(selected_idx)
    x_train = embeddings[sel].to(device)
    y_train = labels[sel].to(device, dtype=torch.float32)

    logger.info("Few-shot training set size: %d samples", len(sel))

    probe = LinearProbe(embed_dim=embeddings.shape[1], n_classes=n_classes).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        probe.train()
        logits = probe(x_train)
        loss   = criterion(logits, y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 20 == 0:
            logger.debug("Few-shot probe epoch %d/%d loss=%.4f", epoch, n_epochs, loss.item())

    logger.info("Few-shot probe training done — final loss=%.4f", loss.item())
    probe.eval()
    return probe


def evaluate_probe(
    probe: LinearProbe,
    embeddings: torch.Tensor,
    labels: np.ndarray,
    device: torch.device,
    label_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Evaluate a linear probe on test embeddings.

    Args:
        probe:       Trained :class:`LinearProbe`.
        embeddings:  ``(N, D)`` test embeddings.
        labels:      ``(N, C)`` binary ground-truth.
        device:      Compute device.
        label_names: Optional list of class names for logging.

    Returns:
        Dict with per-class and mean AUROC.
    """
    probe.eval()
    with torch.no_grad():
        logits = probe(embeddings.to(device)).cpu().numpy()   # (N, C)

    n_classes = logits.shape[1]
    if label_names is None:
        label_names = [f"class_{i}" for i in range(n_classes)]

    scores_dict = {name: logits[:, i] for i, name in enumerate(label_names)}
    labels_dict = {name: labels[:, i] for i, name in enumerate(label_names)}

    auroc = compute_auroc(scores_dict, labels_dict)
    logger.info("Probe AUROC: mean=%.4f", auroc.get("mean_auroc", float("nan")))
    return auroc


def cross_dataset_report(
    zero_shot_auroc: dict[str, float],
    few_shot_auroc: dict[str, float],
    label_names: list[str],
) -> str:
    """Return a formatted comparison table.

    Args:
        zero_shot_auroc: AUROC dict from zero-shot evaluation.
        few_shot_auroc:  AUROC dict from few-shot probe evaluation.
        label_names:     List of class names.

    Returns:
        Formatted string table.
    """
    lines = [
        f"{'Label':<25} {'Zero-Shot':>10} {'Few-Shot':>10} {'Δ (k-shot)':>12}",
        "-" * 60,
    ]

    deltas = []
    for name in label_names:
        zs = zero_shot_auroc.get(name, float("nan"))
        fs = few_shot_auroc.get(name, float("nan"))
        delta = fs - zs if not (np.isnan(zs) or np.isnan(fs)) else float("nan")
        if not np.isnan(delta):
            deltas.append(delta)
        lines.append(f"{name:<25} {zs:>10.4f} {fs:>10.4f} {delta:>+12.4f}")

    lines.append("-" * 60)
    zs_mean = zero_shot_auroc.get("mean_auroc", float("nan"))
    fs_mean = few_shot_auroc.get("mean_auroc", float("nan"))
    d_mean  = float(np.mean(deltas)) if deltas else float("nan")
    lines.append(f"{'Mean':<25} {zs_mean:>10.4f} {fs_mean:>10.4f} {d_mean:>+12.4f}")

    recovery = d_mean / max(1 - zs_mean, 1e-8) if not np.isnan(d_mean) else float("nan")
    lines.append(f"\nFew-shot recovery of domain gap: {recovery:.1%}")
    lines.append("Target: recover 50-70% of cross-dataset performance drop")

    return "\n".join(lines)
