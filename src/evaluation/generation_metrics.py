"""Generation evaluation metrics: BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr, CheXbert F1.

All functions accept lists of hypothesis and reference strings.
"""

import logging

import nltk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK resource downloader — handles both pre-3.9 (punkt) and 3.9+ (punkt_tab)
# ---------------------------------------------------------------------------

def _ensure_nltk():
    """Download required NLTK resources if not already present.

    NLTK 3.9+ renamed ``punkt`` → ``punkt_tab``.  We try both so the
    code works with any version installed in a Colab runtime.
    """
    # punkt / punkt_tab (sentence tokenizer)
    for resource, category in [
        ("punkt_tab", "tokenizers"),
        ("punkt",     "tokenizers"),
    ]:
        try:
            nltk.data.find(f"{category}/{resource}")
            break   # found one, stop trying
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                break
            except Exception:
                continue   # try next variant

    # wordnet + omw (for METEOR)
    for resource, category in [
        ("wordnet",  "corpora"),
        ("omw-1.4",  "corpora"),
    ]:
        try:
            nltk.data.find(f"{category}/{resource}")
        except LookupError:
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
    tokenised_refs = [[r.lower().split()] for r in references]

    results: dict[str, float] = {}
    for n in range(1, max_n + 1):
        weights = tuple(1.0 / n for _ in range(n))
        try:
            score = corpus_bleu(
                tokenised_refs, tokenised_hyps,
                weights=weights, smoothing_function=sf,
            )
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
        from rouge_score import rouge_scorer as rs_lib
    except ImportError as exc:
        raise ImportError(
            "rouge-score is not installed. Run: pip install rouge-score"
        ) from exc

    scorer = rs_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_total = r2_total = rl_total = 0.0
    n = len(hypotheses)

    for hyp, ref in zip(hypotheses, references):
        try:
            scores     = scorer.score(ref, hyp)
            r1_total  += scores["rouge1"].fmeasure
            r2_total  += scores["rouge2"].fmeasure
            rl_total  += scores["rougeL"].fmeasure
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
    n     = len(hypotheses)

    for hyp, ref in zip(hypotheses, references):
        try:
            score  = meteor_score([ref.lower().split()], hyp.lower().split())
            total += score
        except Exception as exc:
            logger.warning("METEOR scoring failed for one sample: %s", exc)
            n -= 1

    result = round(total / n, 4) if n > 0 else 0.0
    logger.info("METEOR score: %.4f", result)
    return {"meteor": result}


# ---------------------------------------------------------------------------
# CIDEr
# ---------------------------------------------------------------------------

def compute_cider(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute CIDEr score using pycocoevalcap.

    CIDEr measures consensus between a generated text and a set of reference
    texts using TF-IDF weighted n-gram matching (n=1..4).  It is the standard
    automatic metric for image-captioning and report-generation benchmarks.

    Args:
        hypotheses: Generated report strings (one per sample).
        references: Ground-truth report strings (one per sample).

    Returns:
        Dict ``{"cider": <float>}``.  Returns 0.0 if pycocoevalcap is not
        installed; a warning is logged in that case.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        logger.warning(
            "pycocoevalcap is not installed — CIDEr skipped. "
            "Run: pip install pycocoevalcap"
        )
        return {"cider": 0.0}

    # pycocoevalcap expects {image_id: [list_of_reference_strings]}
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [hyp] for i, hyp in enumerate(hypotheses)}

    try:
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        result = round(float(score), 4)
    except Exception as exc:
        logger.warning("CIDEr computation failed: %s", exc)
        result = 0.0

    logger.info("CIDEr score: %.4f", result)
    return {"cider": result}


# ---------------------------------------------------------------------------
# CheXbert
# ---------------------------------------------------------------------------

#: 14 CheXpert conditions predicted by CheXbert (in order of model heads).
CHEXBERT_LABELS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def _load_chexbert_labeler(device: str = "cpu"):
    """Build the CheXbert labeler and load pretrained weights.

    Architecture matches ``stanfordmlgroup/CheXbert`` exactly:
      - backbone: ``bert-base-uncased``
      - 14 independent 4-class linear heads (blank/positive/negative/uncertain)

    The pretrained checkpoint is downloaded from the HuggingFace Hub on first
    call (~440 MB) and cached automatically by ``huggingface_hub``.

    Args:
        device: Torch device string.

    Returns:
        Tuple ``(labeler, tokenizer)`` — both moved to *device* and set to eval.
    """
    import torch
    import torch.nn as nn
    from transformers import BertModel, BertTokenizer

    class _CheXbertLabeler(nn.Module):
        """BERT + 14 independent 4-class classification heads."""

        def __init__(self) -> None:
            super().__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.dropout = nn.Dropout(p=0.1)
            # 14 heads; 4 classes each: blank(0), positive(1), negative(2), uncertain(3)
            self.linear_heads = nn.ModuleList(
                [nn.Linear(768, 4) for _ in range(len(CHEXBERT_LABELS))]
            )

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = self.dropout(out.pooler_output)  # (B, 768)
            return [head(cls) for head in self.linear_heads]  # list of 14 × (B, 4)

    labeler = _CheXbertLabeler()

    try:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="stanfordmlgroup/CheXbert",
            filename="pytorch_model.bin",
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # strict=False: the saved checkpoint also contains linear_heads_uncertain.*
        # keys (from the original training code) which we intentionally skip.
        missing, unexpected = labeler.load_state_dict(state_dict, strict=False)
        loaded_heads = sum(
            1 for k in state_dict if k.startswith("linear_heads.")
            and not k.startswith("linear_heads_uncertain.")
        )
        logger.info(
            "CheXbert loaded — %d/14 head weight tensors present, "
            "%d unexpected keys ignored.",
            loaded_heads // 2,   # weight + bias per head → divide by 2
            len(unexpected),
        )
        if missing:
            logger.warning("CheXbert: %d missing keys (first 5: %s)", len(missing), missing[:5])
    except Exception as exc:
        logger.warning(
            "CheXbert pretrained weights could not be loaded (%s). "
            "Classification heads are randomly initialised — metric values "
            "will be meaningless.", exc,
        )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    labeler.to(device).eval()
    return labeler, tokenizer


def compute_chexbert_metrics(
    hypotheses: list[str],
    references: list[str],
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute CheXbert label-level precision, recall, and F1.

    Runs both generated and reference reports through the CheXbert BERT
    classifier to extract binary pathology labels (positive vs. not-positive),
    then computes per-label and macro-average F1.

    This metric captures *clinical label accuracy* and is complementary to
    RadGraph F1 (entity/relation accuracy) and BLEU/ROUGE (surface overlap).

    Args:
        hypotheses:  N generated report strings.
        references:  N ground-truth report strings.
        device:      Torch device string (``"cuda"`` or ``"cpu"``).
        batch_size:  Inference batch size for CheXbert.

    Returns:
        Flat dict containing:
        - ``chexbert_<label>_f1`` for each of the 14 CheXbert conditions
        - ``chexbert_macro_f1``, ``chexbert_macro_precision``, ``chexbert_macro_recall``

        All values are ``0.0`` if CheXbert cannot be loaded.
    """
    import numpy as np
    import torch

    try:
        labeler, tokenizer = _load_chexbert_labeler(device)
    except Exception as exc:
        logger.error("CheXbert could not be initialised: %s — returning zeros.", exc)
        return {
            "chexbert_macro_f1":        0.0,
            "chexbert_macro_precision": 0.0,
            "chexbert_macro_recall":    0.0,
        }

    def _label_texts(texts: list[str]) -> np.ndarray:
        """Return (N, 14) binary array: 1 = positive prediction, 0 otherwise."""
        all_preds: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits_list = labeler(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                )
            # logits_list: 14 tensors each (B, 4) → argmax → (B, 14)
            preds = torch.stack(
                [lg.argmax(-1) for lg in logits_list], dim=1
            ).cpu().numpy()
            all_preds.append(preds)
        return np.concatenate(all_preds, axis=0)  # (N, 14)

    logger.info("CheXbert: labelling %d hypotheses …", len(hypotheses))
    hyp_labels = (_label_texts(hypotheses) == 1).astype(int)   # positive class = 1

    logger.info("CheXbert: labelling %d references …", len(references))
    ref_labels = (_label_texts(references) == 1).astype(int)

    from sklearn.metrics import f1_score, precision_score, recall_score

    results: dict[str, float] = {}
    per_f1: list[float] = []
    per_p:  list[float] = []
    per_r:  list[float] = []

    for i, label in enumerate(CHEXBERT_LABELS):
        p = precision_score(ref_labels[:, i], hyp_labels[:, i], zero_division=0)
        r = recall_score(   ref_labels[:, i], hyp_labels[:, i], zero_division=0)
        f = f1_score(       ref_labels[:, i], hyp_labels[:, i], zero_division=0)
        key = "chexbert_" + label.lower().replace(" ", "_") + "_f1"
        results[key] = round(float(f), 4)
        per_f1.append(f)
        per_p.append(p)
        per_r.append(r)

    results["chexbert_macro_f1"]        = round(float(np.mean(per_f1)), 4)
    results["chexbert_macro_precision"] = round(float(np.mean(per_p)),  4)
    results["chexbert_macro_recall"]    = round(float(np.mean(per_r)),  4)

    logger.info(
        "CheXbert macro — P=%.4f  R=%.4f  F1=%.4f",
        results["chexbert_macro_precision"],
        results["chexbert_macro_recall"],
        results["chexbert_macro_f1"],
    )
    return results


# ---------------------------------------------------------------------------
# Convenience: compute all n-gram metrics at once (BLEU + ROUGE + METEOR + CIDEr)
# ---------------------------------------------------------------------------

def compute_all_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, and CIDEr in one call.

    CheXbert is **not** included here because it requires a ``device`` argument
    and downloads a ~440 MB model; call :func:`compute_chexbert_metrics`
    separately after the standard metrics.

    Args:
        hypotheses: N generated reports.
        references: N ground-truth reports.

    Returns:
        Flat dict with all metric scores.
    """
    logger.info("Computing generation metrics on %d samples …", len(hypotheses))
    metrics: dict[str, float] = {}
    metrics.update(compute_bleu(hypotheses, references))
    metrics.update(compute_rouge(hypotheses, references))
    metrics.update(compute_meteor(hypotheses, references))
    metrics.update(compute_cider(hypotheses, references))
    logger.info("All generation metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Report tables
# ---------------------------------------------------------------------------

def generation_report(metrics: dict[str, float]) -> str:
    """Return a formatted string table of n-gram generation metrics."""
    order = [
        "bleu_1", "bleu_2", "bleu_3", "bleu_4",
        "meteor",
        "rouge_1", "rouge_2", "rouge_l",
        "cider",
    ]
    lines = [f"{'Metric':<15} {'Score':>8}", "-" * 24]
    for key in order:
        if key in metrics:
            lines.append(f"{key:<15} {metrics[key]:>8.4f}")
    return "\n".join(lines)


def chexbert_report(metrics: dict[str, float]) -> str:
    """Return a formatted string table of CheXbert label-level metrics."""
    lines = [
        f"{'Condition':<30} {'F1':>8}",
        "-" * 40,
    ]
    for label in CHEXBERT_LABELS:
        key = "chexbert_" + label.lower().replace(" ", "_") + "_f1"
        if key in metrics:
            lines.append(f"{label:<30} {metrics[key]:>8.4f}")
    lines.append("-" * 40)
    for agg in ("chexbert_macro_f1", "chexbert_macro_precision", "chexbert_macro_recall"):
        if agg in metrics:
            display = agg.replace("chexbert_", "").replace("_", " ").title()
            lines.append(f"{'Macro ' + display.split()[-1]:<30} {metrics[agg]:>8.4f}")
    return "\n".join(lines)
