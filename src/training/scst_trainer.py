"""SCST (Self-Critical Sequence Training) fine-tuner for R2Gen.

Implements the policy-gradient objective from Rennie et al. (2017):
  loss = CE_loss + λ * RL_loss
where
  RL_loss = -E[(R(sample) - R(greedy)) * sum_t log p(a_t)]

Reward R is the RG_ER component of F1-RadGraph (entity+relation F1).
If the RadGraph package is unavailable, BLEU-2 is used as a fallback.

Key design choices (matching the notebook):
  • The visual encoder (ResNet-101) is frozen — only decoder + memory
    parameters receive gradient updates.
  • One epoch alternates between teacher-forced CE loss and RL loss
    for every training batch.
  • Batches are expected in the same dict format as R2GenTrainer:
    {image, input_ids, target_ids, report, study_id}.

Reference: Self-Critical Sequence Training for Image Captioning
           (Rennie et al., CVPR 2017)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.amp
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gather_sequence_logp(
    seq: torch.Tensor,
    log_probs: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Sum log p(token_t) over non-padding positions.

    Args:
        seq:       ``(B, L)`` sampled token IDs.
        log_probs: ``(B, L, V)`` full log-softmax distributions.
        pad_id:    Token ID to treat as padding (excluded from sum).

    Returns:
        ``(B,)`` sum of chosen log-probs per sequence.
    """
    # log prob of the chosen action at each step
    logp_tok = log_probs.gather(
        2, seq.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)                                         # (B, L)

    mask = (seq != pad_id).float()
    return (logp_tok * mask).sum(dim=1)                  # (B,)


def _decode_tokens(tokenizer, seqs: torch.Tensor) -> list[str]:
    """Decode a batch of token-ID tensors to strings."""
    results = []
    for row in seqs.cpu().tolist():
        results.append(tokenizer.decode(row))
    return results


def _bleu_rewards(hyps: list[str], refs: list[str]) -> list[float]:
    """Sentence-level BLEU-2 as fallback reward when RadGraph unavailable."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
    except ImportError:
        return [0.0] * len(hyps)

    sf = SmoothingFunction().method7
    rewards = []
    for h, r in zip(hyps, refs):
        ht = word_tokenize(h.lower())
        rt = word_tokenize(r.lower())
        rewards.append(sentence_bleu([rt], ht, weights=(0.5, 0.5), smoothing_function=sf))
    return rewards


def _radgraph_rewards(f1_fn, hyps: list[str], refs: list[str]) -> list[float]:
    """Extract per-sample RG_ER (entity+relation F1) from F1RadGraph."""
    try:
        _, reward_list, *_ = f1_fn(hyps=hyps, refs=refs)
        # reward_list: list of (RG_E, RG_ER, RG_BAR_ER) tuples
        return [t[1] for t in reward_list]
    except Exception as exc:
        logger.warning("RadGraph reward failed (%s) — falling back to BLEU", exc)
        return _bleu_rewards(hyps, refs)


# ─────────────────────────────────────────────────────────────────────────────
# SCST Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SCSTTrainer:
    """Fine-tunes R2Gen with Self-Critical Sequence Training.

    Args:
        model:        Trained :class:`~src.models.r2gen.R2GenModel`.
        tokenizer:    :class:`~src.data_loaders.report_tokenizer.ReportTokenizer`.
        config:       Config dict (``experiments/configs/scst.yaml``).
        train_loader: DataLoader yielding ``{image, input_ids, target_ids, report}``.
        val_loader:   DataLoader for validation loss tracking.
        device:       Compute device.
        logger_:      Optional external logger.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger_: Optional[logging.Logger] = None,
    ):
        self.model        = model
        self.tokenizer    = tokenizer
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.log          = logger_ or logger

        tcfg = config["training"]
        self.epochs       = tcfg["epochs"]
        self.lambda_fact  = tcfg.get("lambda_fact", 0.1)
        self.temperature  = tcfg.get("temperature", 1.0)
        self.grad_clip    = tcfg.get("grad_clip", 5.0)
        self.max_length   = config["model"].get("max_seq_len", 60)
        self.pad_id       = tokenizer.pad_id

        freeze = tcfg.get("freeze_encoder", True)
        if freeze:
            for p in model.visual_extractor.parameters():
                p.requires_grad_(False)
            self.log.info("Visual encoder frozen — only decoder parameters updated")

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        # Mixed precision
        _use_amp = device.type == "cuda"
        self.scaler   = torch.amp.GradScaler(device=device.type, enabled=_use_amp)
        self._use_amp = _use_amp
        self._device_type = device.type

        # Try to load RadGraph
        rg_model = tcfg.get("radgraph_model_type", "radgraph-xl")
        self.f1radgraph = None
        for attempt in [rg_model, "radgraph"]:
            try:
                # Compatibility shim: transformers>=4.46 removed encode_plus,
                # which radgraph's vendored allennlp still calls.
                from transformers import PreTrainedTokenizerBase as _PTTB
                if not hasattr(_PTTB, "encode_plus"):
                    _PTTB.encode_plus = lambda self, text, text_pair=None, **kw: self(
                        text, text_pair=text_pair, **kw
                    )

                from radgraph import F1RadGraph
                self.f1radgraph = F1RadGraph(reward_level="all", model_type=attempt)
                self.log.info("RadGraph loaded (%s) — using RG_ER as SCST reward", attempt)
                break
            except Exception as exc:
                self.log.warning("RadGraph (%s) unavailable: %s", attempt, exc)

        if self.f1radgraph is None:
            self.log.warning("Falling back to BLEU-2 as SCST reward")

        ckpt_dir = Path(config["paths"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log.info(
            "SCSTTrainer — trainable_params=%.2fM epochs=%d "
            "lambda=%.3f temperature=%.2f",
            total / 1e6, self.epochs, self.lambda_fact, self.temperature,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_rewards(self, hyps: list[str], refs: list[str]) -> torch.Tensor:
        """Compute per-sample rewards as a CPU float tensor."""
        # Sanitise empty outputs
        hyps = [h if h.strip() else "no findings." for h in hyps]
        refs = [r if r.strip() else "no findings." for r in refs]

        if self.f1radgraph is not None:
            r = _radgraph_rewards(self.f1radgraph, hyps, refs)
        else:
            r = _bleu_rewards(hyps, refs)

        return torch.tensor(r, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one SCST fine-tuning epoch.

        Returns:
            Dict with ``train_loss``, ``ce_loss``, ``rl_loss``.
        """
        self.model.train()
        if hasattr(self.model, "visual_extractor"):
            self.model.visual_extractor.eval()   # keep BN in eval during freeze

        total_loss = total_ce = total_rl = 0.0
        n_batches  = 0
        t0         = time.time()

        for i, batch in enumerate(self.train_loader):
            images     = batch["image"].to(self.device)          # (B, 3, H, W)
            input_ids  = batch["input_ids"].to(self.device)      # (B, T)
            target_ids = batch["target_ids"].to(self.device)     # (B, T)
            refs       = list(batch["report"])                   # list[str]
            B          = images.size(0)

            # ── 1. Teacher-forced CE loss ─────────────────────────────────
            with torch.amp.autocast(device_type=self._device_type, enabled=self._use_amp):
                logits  = self.model(images, input_ids)           # (B, T, V)
                V       = logits.size(-1)
                loss_ce = self.criterion(
                    logits.reshape(-1, V),
                    target_ids.reshape(-1),
                )

            # ── 2. SCST RL loss ───────────────────────────────────────────
            # Stochastic sample — keep computation graph through log_probs
            sampled_seq, log_probs = self.model.sample(
                images,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_length=self.max_length,
                temperature=self.temperature,
            )                             # (B, L), (B, L, V)

            # Greedy baseline — no gradient needed
            greedy_seq = self.model.greedy_decode(
                images,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_length=self.max_length,
            )                             # (B, L)

            hyps_s = _decode_tokens(self.tokenizer, sampled_seq)
            hyps_b = _decode_tokens(self.tokenizer, greedy_seq)

            with torch.no_grad():
                r_s = self._compute_rewards(hyps_s, refs).to(self.device)
                r_b = self._compute_rewards(hyps_b, refs).to(self.device)
                adv = r_s - r_b                                    # (B,)

            # Policy gradient: -E[advantage * log_p(sequence)]
            logp   = _gather_sequence_logp(sampled_seq, log_probs, self.pad_id)
            loss_rl = -(adv * logp).mean()

            # ── 3. Combined loss + backward ───────────────────────────────
            loss = loss_ce + self.lambda_fact * loss_rl

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.grad_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_ce   += loss_ce.item()
            total_rl   += loss_rl.item()
            n_batches  += 1

            if (i + 1) % 20 == 0:
                self.log.info(
                    "SCST epoch %d [%d/%d] loss=%.4f ce=%.4f rl=%.4f "
                    "adv_mean=%.4f r_s=%.4f r_b=%.4f",
                    epoch, i + 1, len(self.train_loader),
                    loss.item(), loss_ce.item(), loss_rl.item(),
                    adv.mean().item(), r_s.mean().item(), r_b.mean().item(),
                )

        elapsed = time.time() - t0
        stats = {
            "train_loss": total_loss / max(n_batches, 1),
            "ce_loss":    total_ce   / max(n_batches, 1),
            "rl_loss":    total_rl   / max(n_batches, 1),
        }
        self.log.info(
            "SCST epoch %d done — loss=%.4f ce=%.4f rl=%.4f time=%.1fs",
            epoch, stats["train_loss"], stats["ce_loss"], stats["rl_loss"], elapsed,
        )
        return stats

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self) -> float:
        """Compute teacher-forced CE loss on the validation set."""
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            images     = batch["image"].to(self.device)
            input_ids  = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            with torch.amp.autocast(device_type=self._device_type, enabled=self._use_amp):
                logits  = self.model(images, input_ids)
                V       = logits.size(-1)
                loss    = self.criterion(logits.reshape(-1, V), target_ids.reshape(-1))
            total += loss.item()
            n     += 1
        val_loss = total / max(n, 1)
        self.log.info("SCST val CE loss = %.4f", val_loss)
        return val_loss

    # ──────────────────────────────────────────────────────────────────────────
    def train(self, resume_path: Optional[str] = None) -> None:
        """Run full SCST fine-tuning for ``self.epochs`` epochs."""
        start_epoch = 0
        best_val    = float("inf")

        if resume_path and Path(resume_path).exists():
            ck = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(ck["model_state_dict"])
            start_epoch = ck.get("epoch", 0) + 1
            best_val    = ck.get("val_loss", float("inf"))
            self.log.info("Resumed from %s (epoch %d)", resume_path, start_epoch)

        self.log.info(
            "Starting SCST fine-tuning — epochs=%d start=%d device=%s",
            self.epochs, start_epoch, self.device,
        )

        for epoch in range(start_epoch, self.epochs):
            train_stats = self.train_epoch(epoch)
            val_loss    = self.validate()

            # Save latest checkpoint
            latest_path = self.ckpt_dir / "scst_latest.pt"
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_loss":         val_loss,
                    "train_stats":      train_stats,
                },
                latest_path,
            )

            # Save best checkpoint
            if val_loss < best_val:
                best_val = val_loss
                best_path = self.ckpt_dir / "scst_best.pt"
                torch.save(
                    {
                        "epoch":            epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss":         val_loss,
                    },
                    best_path,
                )
                self.log.info(
                    "New best SCST checkpoint — val_loss=%.4f → %s",
                    val_loss, best_path,
                )

            print(
                f"Epoch {epoch}/{self.epochs - 1} — "
                f"train={train_stats['train_loss']:.4f} "
                f"(ce={train_stats['ce_loss']:.4f} rl={train_stats['rl_loss']:.4f}) "
                f"val={val_loss:.4f}"
            )

        self.log.info("SCST fine-tuning complete — best val_loss=%.4f", best_val)
