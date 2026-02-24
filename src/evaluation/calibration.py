"""Calibration analysis for radiology report generation.

Computes Expected Calibration Error (ECE) from model uncertainty estimates
and generation quality scores, producing calibration curves and summary
statistics.

Usage::

    from src.evaluation.calibration import compute_ece, calibration_report

    ece, bin_stats = compute_ece(confidence, correctness, n_bins=10)
    report = calibration_report(ece, bin_stats)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ECE
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, dict]:
    """Expected Calibration Error.

    Bins predictions by confidence and measures the gap between confidence
    and empirical accuracy within each bin.

    Args:
        confidence:  ``(N,)`` float array of confidence scores in [0, 1].
        correctness: ``(N,)`` float or binary array of ground-truth quality.
        n_bins:      Number of equal-width bins (default 10).

    Returns:
        Tuple ``(ece, bin_stats)`` where ``bin_stats`` is a dict with keys
        ``bin_conf``, ``bin_acc``, ``bin_count`` (all ``(k,)`` arrays for
        k non-empty bins).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        bin_acc.append(float(correctness[mask].mean()))
        bin_conf.append(float(confidence[mask].mean()))
        bin_count.append(int(mask.sum()))

    if not bin_count:
        logger.warning("compute_ece: all bins empty — returning ECE=0")
        return 0.0, {"bin_conf": np.array([]), "bin_acc": np.array([]),
                     "bin_count": np.array([])}

    bin_acc_arr   = np.array(bin_acc)
    bin_conf_arr  = np.array(bin_conf)
    bin_count_arr = np.array(bin_count)

    ece = float(
        np.sum(bin_count_arr * np.abs(bin_acc_arr - bin_conf_arr))
        / bin_count_arr.sum()
    )

    logger.info(
        "ECE = %.4f  (n=%d bins=%d non-empty=%d)",
        ece, len(confidence), n_bins, len(bin_count),
    )
    return ece, {
        "bin_conf":  bin_conf_arr,
        "bin_acc":   bin_acc_arr,
        "bin_count": bin_count_arr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty → quality analysis
# ─────────────────────────────────────────────────────────────────────────────

def uncertainty_to_confidence(entropy: np.ndarray) -> np.ndarray:
    """Convert token entropy to a confidence proxy in [0, 1].

    Confidence = 1 − normalised_entropy, so high entropy → low confidence.

    Args:
        entropy: ``(N,)`` mean per-token entropy values.

    Returns:
        ``(N,)`` confidence scores in [0, 1].
    """
    e_min, e_max = entropy.min(), entropy.max()
    if e_max - e_min < 1e-9:
        return np.ones_like(entropy)
    normalised = (entropy - e_min) / (e_max - e_min)
    return np.clip(1.0 - normalised, 0.0, 1.0)


def calibration_analysis(
    hypotheses: list[str],
    references: list[str],
    entropy: np.ndarray,
    disagreement: Optional[np.ndarray] = None,
    quality_threshold: float = 0.30,
    n_bins: int = 10,
) -> dict:
    """Full calibration pipeline: quality → correctness → ECE.

    Computes RadGraph F1 (or BLEU-2 fallback) as the quality metric,
    thresholds to get binary correctness, converts entropy to confidence,
    then computes ECE.

    Args:
        hypotheses:         Generated report strings (N,).
        references:         Ground-truth report strings (N,).
        entropy:            ``(N,)`` mean token entropy per sample.
        disagreement:       Optional ``(N,)`` MC Dropout disagreement scores.
        quality_threshold:  Threshold for binarising quality into correctness.
                            Default 0.30 for RadGraph F1; use 0.50 for BLEU.
        n_bins:             ECE bins.

    Returns:
        Dict with keys:
        - ``ece``              float
        - ``bin_stats``        dict (bin_conf, bin_acc, bin_count arrays)
        - ``confidence``       (N,) array
        - ``quality``          (N,) array of per-sample quality scores
        - ``correctness``      (N,) binary array
        - ``metric_type``      str ("RadGraph" or "BLEU-2")
        - ``quality_threshold``float
        - ``mean_quality``     float
        - ``pct_correct``      float
    """
    # ── Compute quality scores ─────────────────────────────────────────────
    quality    = None
    metric_type = None

    # Try RadGraph F1 first
    for rg_model in ["radgraph-xl", "radgraph"]:
        try:
            # Compatibility shim: transformers>=4.46 removed encode_plus,
            # which radgraph's vendored allennlp still calls.
            from transformers import PreTrainedTokenizerBase as _PTTB
            if not hasattr(_PTTB, "encode_plus"):
                _PTTB.encode_plus = lambda self, text, text_pair=None, **kw: self(
                    text, text_pair=text_pair, **kw
                )

            from radgraph import F1RadGraph
            f1fn = F1RadGraph(reward_level="all", model_type=rg_model)
            hyps_clean = [h if h.strip() else "no findings." for h in hypotheses]
            refs_clean = [r if r.strip() else "no findings." for r in references]
            _, reward_list, *_ = f1fn(hyps=hyps_clean, refs=refs_clean)
            quality     = np.array([t[1] for t in reward_list])   # RG_ER
            metric_type = "RadGraph"
            logger.info("Calibration using RadGraph F1 (%s)", rg_model)
            break
        except Exception as exc:
            logger.warning("RadGraph (%s) unavailable: %s", rg_model, exc)

    # BLEU-2 fallback
    if quality is None:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
            sf = SmoothingFunction().method7
            quality = np.array([
                sentence_bleu(
                    [word_tokenize(r.lower())],
                    word_tokenize(h.lower()),
                    weights=(0.5, 0.5),
                    smoothing_function=sf,
                )
                for h, r in zip(hypotheses, references)
            ])
            metric_type    = "BLEU-2"
            quality_threshold = 0.50   # sensible default for BLEU
            logger.info("Calibration using BLEU-2 fallback")
        except Exception as exc:
            logger.error("BLEU-2 also failed: %s — returning empty analysis", exc)
            return {"ece": float("nan"), "metric_type": "none"}

    confidence  = uncertainty_to_confidence(entropy)
    correctness = (quality >= quality_threshold).astype(float)
    pct_correct = correctness.mean() * 100

    ece, bin_stats = compute_ece(confidence, correctness, n_bins=n_bins)

    logger.info(
        "Calibration analysis — metric=%s threshold=%.2f "
        "pct_correct=%.1f%% ECE=%.4f",
        metric_type, quality_threshold, pct_correct, ece,
    )

    return {
        "ece":               ece,
        "bin_stats":         bin_stats,
        "confidence":        confidence,
        "quality":           quality,
        "correctness":       correctness,
        "metric_type":       metric_type,
        "quality_threshold": quality_threshold,
        "mean_quality":      float(quality.mean()),
        "pct_correct":       pct_correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def calibration_summary(results: dict) -> str:
    """Format calibration analysis results as a readable string."""
    if results.get("metric_type") == "none":
        return "Calibration analysis failed — no quality metric available."

    lines = [
        "=== Calibration Analysis ===",
        f"Quality metric       : {results['metric_type']}",
        f"Quality threshold    : {results['quality_threshold']:.2f}",
        f"Mean quality score   : {results['mean_quality']:.4f}",
        f"Fraction correct     : {results['pct_correct']:.1f}%",
        f"ECE                  : {results['ece']:.4f}",
    ]
    return "\n".join(lines)


def plot_calibration(
    results: dict,
    save_path: Optional[str] = None,
):
    """Plot calibration curve and uncertainty-vs-quality scatter.

    Args:
        results:   Output of :func:`calibration_analysis`.
        save_path: If provided, save figure to this path.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    bs = results.get("bin_stats", {})
    bin_conf  = bs.get("bin_conf",  np.array([]))
    bin_acc   = bs.get("bin_acc",   np.array([]))
    quality   = results.get("quality",   np.array([]))
    conf      = results.get("confidence", np.array([]))
    ece       = results.get("ece", float("nan"))
    metric    = results.get("metric_type", "quality")
    threshold = results.get("quality_threshold", 0.3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: calibration curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    if len(bin_conf):
        ax.bar(bin_conf, bin_acc, width=0.07, alpha=0.7,
               color="steelblue", label="model")
    ax.set_xlabel("Confidence (1 − norm. entropy)")
    ax.set_ylabel(f"Accuracy ({metric} ≥ {threshold:.2f})")
    ax.set_title(f"Calibration Curve  |  ECE = {ece:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()

    # Right: entropy vs quality scatter
    ax = axes[1]
    if len(quality) and len(conf):
        sc = ax.scatter(1 - conf, quality, c="steelblue", alpha=0.4, s=12)
        ax.axhline(threshold, color="red", linestyle="--",
                   label=f"correct threshold={threshold:.2f}")
        ax.set_xlabel("Token entropy (↑ = more uncertain)")
        ax.set_ylabel(f"{metric} score (↑ = better)")
        ax.set_title(f"Uncertainty vs. {metric}")
        ax.legend()

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Calibration plot saved → %s", save_path)

    return fig
