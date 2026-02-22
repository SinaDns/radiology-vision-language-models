from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_auroc(
    scores_dict: dict[str, np.ndarray],
    labels_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute per-class and mean AUROC.

    Args:
        scores_dict: {label: score_array}  (higher score → more likely positive)
        labels_dict: {label: binary_label_array}

    Returns:
        Dict with per-class AUROC values and a "mean_auroc" entry.
    """
    results = {}
    valid = []

    for label in scores_dict:
        gt = labels_dict[label]
        sc = scores_dict[label]
        if gt.sum() == 0 or (1 - gt).sum() == 0:
            continue  # skip labels with no positive or no negative examples
        auroc = roc_auc_score(gt, sc)
        results[label] = auroc
        valid.append(auroc)

    results["mean_auroc"] = float(np.mean(valid)) if valid else float("nan")
    return results


def compute_auprc(
    scores_dict: dict[str, np.ndarray],
    labels_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute per-class and mean AUPRC (area under precision-recall curve)."""
    results = {}
    valid = []

    for label in scores_dict:
        gt = labels_dict[label]
        sc = scores_dict[label]
        if gt.sum() == 0:
            continue
        auprc = average_precision_score(gt, sc)
        results[label] = auprc
        valid.append(auprc)

    results["mean_auprc"] = float(np.mean(valid)) if valid else float("nan")
    return results


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    Measures the average absolute difference between confidence and accuracy
    across equal-width probability bins.

    Args:
        probs:  1-D array of predicted probabilities in [0, 1].
        labels: 1-D binary array of ground-truth labels.
        n_bins: Number of equal-width bins.

    Returns:
        ECE as a scalar float.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        accuracy = labels[mask].mean()
        confidence = probs[mask].mean()
        ece += (mask.sum() / n) * abs(confidence - accuracy)

    return float(ece)


def classification_report(
    scores: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[list[str]] = None,
    threshold: float = 0.0,
) -> str:
    """Human-readable per-class metrics table.

    Args:
        scores:      (N, C) score matrix or (N,) for binary.
        labels:      (N, C) or (N,) binary ground-truth.
        label_names: List of C class names.
        threshold:   Decision threshold for binary predictions.

    Returns:
        Formatted string report.
    """
    if scores.ndim == 1:
        scores = scores[:, None]
        labels = labels[:, None]

    n_classes = scores.shape[1]
    if label_names is None:
        label_names = [f"class_{i}" for i in range(n_classes)]

    lines = [
        f"{'Label':<25} {'AUROC':>8} {'AUPRC':>8} {'Prevalence':>12}",
        "-" * 57,
    ]

    auroc_vals = []
    auprc_vals = []

    for i, name in enumerate(label_names):
        gt = labels[:, i]
        sc = scores[:, i]
        prevalence = gt.mean()

        if gt.sum() > 0 and (1 - gt).sum() > 0:
            auroc = roc_auc_score(gt, sc)
            auprc = average_precision_score(gt, sc)
            auroc_vals.append(auroc)
            auprc_vals.append(auprc)
            lines.append(f"{name:<25} {auroc:>8.4f} {auprc:>8.4f} {prevalence:>12.4f}")
        else:
            lines.append(f"{name:<25} {'N/A':>8} {'N/A':>8} {prevalence:>12.4f}")

    if auroc_vals:
        lines.append("-" * 57)
        lines.append(
            f"{'Mean':<25} {np.mean(auroc_vals):>8.4f} {np.mean(auprc_vals):>8.4f}"
        )

    return "\n".join(lines)
