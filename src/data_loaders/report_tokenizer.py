"""Word-level tokenizer for radiology report generation (R2Gen).

Builds a vocabulary from training reports and converts between token
strings and integer IDs.  Special tokens follow the convention used in
the original R2Gen implementation.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

# Special tokens
PAD_TOKEN = "<PAD>"   # 0
UNK_TOKEN = "<UNK>"   # 1
BOS_TOKEN = "<BOS>"   # 2  (beginning of sequence / start decoding)
EOS_TOKEN = "<EOS>"   # 3  (end of sequence / stop decoding)

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def _tokenise_text(text: str) -> list[str]:
    """Lower-case word tokeniser that strips punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class ReportTokenizer:
    """Word-level vocabulary + tokeniser for radiology reports.

    Usage::

        tok = ReportTokenizer.build(list_of_report_strings, min_freq=3)
        tok.save("vocab.json")

        tok = ReportTokenizer.load("vocab.json")
        ids   = tok.encode("no acute cardiopulmonary findings")
        text  = tok.decode(ids)
        batch = tok(["report one", "report two"], max_length=64)

    Args:
        word2idx: Pre-built word→id mapping (including special tokens).
    """

    def __init__(self, word2idx: dict[str, int]):
        self.word2idx: dict[str, int] = word2idx
        self.idx2word: dict[int, str] = {v: k for k, v in word2idx.items()}
        self.pad_id = word2idx[PAD_TOKEN]
        self.unk_id = word2idx[UNK_TOKEN]
        self.bos_id = word2idx[BOS_TOKEN]
        self.eos_id = word2idx[EOS_TOKEN]

        logger.info(
            "ReportTokenizer — vocab_size=%d pad=%d unk=%d bos=%d eos=%d",
            len(self.word2idx), self.pad_id, self.unk_id, self.bos_id, self.eos_id,
        )

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        reports: list[str],
        min_freq: int = 3,
    ) -> "ReportTokenizer":
        """Build vocabulary from a list of report strings.

        Args:
            reports:  All training report texts.
            min_freq: Minimum word frequency to include (default 3).

        Returns:
            Fitted :class:`ReportTokenizer`.
        """
        logger.info(
            "Building vocabulary from %d reports (min_freq=%d) …",
            len(reports), min_freq,
        )
        counter: Counter = Counter()
        for report in reports:
            counter.update(_tokenise_text(report))

        word2idx: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for word, freq in sorted(counter.items()):
            if freq >= min_freq:
                word2idx[word] = len(word2idx)

        n_rare = sum(1 for f in counter.values() if f < min_freq)
        logger.info(
            "Vocabulary built — total_words=%d vocab_size=%d "
            "rare_words_dropped=%d",
            len(counter), len(word2idx), n_rare,
        )
        return cls(word2idx)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReportTokenizer":
        """Load vocabulary from a JSON file saved by :meth:`save`."""
        path = Path(path)
        logger.info("Loading vocabulary from %s", path)
        with open(path) as f:
            word2idx = json.load(f)
        return cls(word2idx)

    def save(self, path: Union[str, Path]):
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.word2idx, f, indent=2)
        logger.info("Vocabulary saved → %s (%d tokens)", path, len(self.word2idx))

    # ── Encoding / decoding ───────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Convert a report string to a list of token IDs (no BOS/EOS)."""
        return [
            self.word2idx.get(w, self.unk_id)
            for w in _tokenise_text(text)
        ]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert token IDs back to a string."""
        skip = {self.pad_id, self.bos_id, self.eos_id} if skip_special else set()
        words = [
            self.idx2word.get(i, UNK_TOKEN)
            for i in ids
            if i not in skip
        ]
        return " ".join(words)

    # ── Batch tokenisation (callable) ─────────────────────────────────────────

    def __call__(
        self,
        texts: list[str],
        max_length: int = 100,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenise a batch of report strings and return padded tensors.

        Args:
            texts:      List of B report strings.
            max_length: Maximum sequence length (sequences are truncated /
                        padded to this length).
            add_bos:    Prepend ``<BOS>`` token.
            add_eos:    Append ``<EOS>`` token (before truncation check).

        Returns:
            Dict with keys:
            - ``input_ids``    (B, max_length) – BOS + tokens (for decoder input)
            - ``target_ids``   (B, max_length) – tokens + EOS (for loss)
            - ``lengths``      (B,)            – actual (unpadded) lengths
        """
        all_ids: list[list[int]] = []
        all_targets: list[list[int]] = []
        lengths: list[int] = []

        for text in texts:
            ids = self.encode(text)
            # Truncate leaving room for special tokens
            max_content = max_length - int(add_bos) - int(add_eos)
            ids = ids[:max_content]

            inp = ([self.bos_id] if add_bos else []) + ids
            tgt = ids + ([self.eos_id] if add_eos else [])

            # Pad to max_length
            inp_len = len(inp)
            tgt_len = len(tgt)
            inp = inp + [self.pad_id] * (max_length - inp_len)
            tgt = tgt + [self.pad_id] * (max_length - tgt_len)

            all_ids.append(inp[:max_length])
            all_targets.append(tgt[:max_length])
            lengths.append(inp_len)

        return {
            "input_ids":  torch.tensor(all_ids, dtype=torch.long),
            "target_ids": torch.tensor(all_targets, dtype=torch.long),
            "lengths":    torch.tensor(lengths, dtype=torch.long),
        }

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
