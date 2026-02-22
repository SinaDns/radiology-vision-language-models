"""IU X-Ray dataset wrapper for seq2seq (R2Gen) training.

Wraps :class:`IUXrayDataset` and applies :class:`ReportTokenizer`
to return batches ready for the R2Gen teacher-forced training loop.
"""

import logging
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset

from src.data_loaders.iu_xray import IUXrayDataset
from src.data_loaders.report_tokenizer import ReportTokenizer

logger = logging.getLogger(__name__)


class IUXraySeq2SeqDataset(Dataset):
    """IU X-Ray dataset returning tokenised (image, input_ids, target_ids) triples.

    Each item:
    - ``image``      – ``(3, H, W)`` tensor
    - ``input_ids``  – ``(max_length,)`` BOS + tokens  (decoder input)
    - ``target_ids`` – ``(max_length,)`` tokens + EOS  (loss target)
    - ``report``     – raw report string (for metric evaluation)
    - ``study_id``   – study identifier

    Args:
        data_dir:    Root IU X-Ray data directory.
        tokenizer:   Pre-built :class:`ReportTokenizer`.
        split:       ``"train"`` or ``"val"``.
        val_fraction: Fraction of studies for validation.
        transform:   Image transform.
        max_length:  Maximum tokenised sequence length (default 100).
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: ReportTokenizer,
        split: str = "train",
        val_fraction: float = 0.1,
        transform: Optional[Callable] = None,
        max_length: int = 100,
    ):
        self.base = IUXrayDataset(
            data_dir=data_dir,
            split=split,
            val_fraction=val_fraction,
            transform=transform,
        )
        self.tokenizer  = tokenizer
        self.max_length = max_length

        logger.info(
            "IUXraySeq2SeqDataset — split=%s samples=%d max_length=%d",
            split, len(self.base), max_length,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        report = item["report"]

        tokens = self.tokenizer(
            [report],
            max_length=self.max_length,
            add_bos=True,
            add_eos=True,
        )

        return {
            "image":      item["image"],
            "input_ids":  tokens["input_ids"][0],    # (max_length,)
            "target_ids": tokens["target_ids"][0],   # (max_length,)
            "lengths":    tokens["lengths"][0],
            "report":     report,
            "study_id":   item["study_id"],
        }


def build_tokenizer(
    data_dir: str,
    save_path: Optional[str] = None,
    min_freq: int = 3,
    val_fraction: float = 0.1,
) -> ReportTokenizer:
    """Build (or load cached) a ReportTokenizer from all training reports.

    Args:
        data_dir:     IU X-Ray root directory.
        save_path:    If provided, save/load vocabulary from this JSON path.
        min_freq:     Minimum word frequency to include in vocab.
        val_fraction: Val fraction used when splitting studies.

    Returns:
        Fitted :class:`ReportTokenizer`.
    """
    if save_path and Path(save_path).exists():
        logger.info("Loading cached vocabulary from %s", save_path)
        return ReportTokenizer.load(save_path)

    logger.info("Building tokenizer vocabulary from training reports …")
    # Load train split without transforms to collect raw reports
    train_ds = IUXrayDataset(
        data_dir=data_dir,
        split="train",
        val_fraction=val_fraction,
        transform=None,
    )
    reports = [item["report"] for item in train_ds]
    logger.info("Collected %d training reports for vocabulary", len(reports))

    tokenizer = ReportTokenizer.build(reports, min_freq=min_freq)

    if save_path:
        tokenizer.save(save_path)

    return tokenizer
