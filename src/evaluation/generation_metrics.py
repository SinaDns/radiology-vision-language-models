"""Generation evaluation metrics: BLEU-1/2/3/4, ROUGE-L, METEOR.

All functions accept lists of hypothesis and reference strings.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK downloads (safe to call multiple times)
# ---------------------------------------------------------------------------

def _ensure_nltk():
    import nltk
    for resource in ("punkt", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info("Downloading NLTK resource: %s", resource)
                nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 4,
) -> dict[str, float]:
    """Compute corpus-level BLEU-1 through BLEU-4.

    Args:
        hypotheses: List of generated reports (one per sample).
        references: List of ground-truth reports (one per sample).
        max_n:      Highest n-gram order (default 4).

    Returns:
        Dict ``{"bleu_1": ..., "bleu_2": ..., "bleu_3": ..., "bleu_4": ...}``.
    """
    _ensure_nltk()
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    sf = SmoothingFunction().method1

    tokenised_hyps = [h.lower().split() for h in hypotheses]
    tokenised_refs = [[r.lower().split()] for r in references]   # each ref is a list of refs

    results = {}
    for n in range(1, max_n + 1):
        weights = tuple(1.0 / n for _ in range(n))
        try:
            score = corpus_bleu(tokenised_refs, tokenised_hyps, weights=weights, smoothing_function=sf)
        except Exception as exc:
            logger.warning("BLEU-%d computation failed: %s", n, exc)
            score = 0.0
        results[f"bleu_{n}"] = round(score, 4)

    logger.info("BLEU scores: %s", results)
    return results


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute average ROUGE-1, ROUGE-2, and ROUGE-L F1.

    Args:
        hypotheses: Generated report strings.
        references: Ground-truth report strings.

    Returns:
        Dict with keys ``rouge_1``, ``rouge_2``, ``rouge_l``.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("rouge-score is not installed. Run: pip install rouge-score") from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_total = r2_total = rl_total = 0.0
    n = len(hypotheses)

    for hyp, ref in zip(hypotheses, references):
        try:
            scores = scorer.score(ref, hyp)
            r1_total += scores["rouge1"].fmeasure
            r2_total += scores["rouge2"].fmeasure
            rl_total += scores["rougeL"].fmeasure
        except Exception as exc:
            logger.warning("ROUGE scoring failed for one sample: %s", exc)
            n -= 1

    if n == 0:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    results = {
        "rouge_1": round(r1_total / n, 4),
        "rouge_2": round(r2_total / n, 4),
        "rouge_l": round(rl_total / n, 4),
    }
    logger.info("ROUGE scores: %s", results)
    return results


# ---------------------------------------------------------------------------
# METEOR
# ---------------------------------------------------------------------------

def compute_meteor(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute average METEOR score.

    Args:
        hypotheses: Generated report strings.
        references: Ground-truth report strings.

    Returns:
        Dict with key ``meteor``.
    """
    _ensure_nltk()
    from nltk.translate.meteor_score import meteor_score

    total = 0.0
    n = len(hypotheses)

    for hyp, ref in zip(hypotheses, references):
        try:
            score = meteor_score([ref.lower().split()], hyp.lower().split())
            total += score
        except Exception as exc:
            logger.warning("METEOR scoring failed for one sample: %s", exc)
            n -= 1

    result = round(total / n, 4) if n > 0 else 0.0
    logger.info("METEOR score: %.4f", result)
    return {"meteor": result}


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute BLEU-1/2/3/4, ROUGE-1/2/L, and METEOR.

    Args:
        hypotheses: N generated reports.
        references: N ground-truth reports.

    Returns:
        Flat dict with all metric scores.
    """
    logger.info(
        "Computing generation metrics on %d samples …", len(hypotheses)
    )
    metrics: dict[str, float] = {}
    metrics.update(compute_bleu(hypotheses, references))
    metrics.update(compute_rouge(hypotheses, references))
    metrics.update(compute_meteor(hypotheses, references))
    logger.info("All generation metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Report table
# ---------------------------------------------------------------------------

def generation_report(metrics: dict[str, float]) -> str:
    """Return a formatted string table of generation metrics."""
    order = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_1", "rouge_2", "rouge_l"]
    lines = [f"{'Metric':<15} {'Score':>8}", "-" * 24]
    for key in order:
        if key in metrics:
            lines.append(f"{key:<15} {metrics[key]:>8.4f}")
    return "\n".join(lines)
