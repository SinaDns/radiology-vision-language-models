"""Contrastive trainer for CheXzero (CLIP objective on IU X-Ray)."""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.training.losses import clip_loss

logger = logging.getLogger(__name__)


def _gpu_memory_str(device: torch.device) -> str:
    """Return a short string with current GPU memory usage."""
    if device.type != "cuda":
        return "N/A (CPU)"
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved  = torch.cuda.memory_reserved(device) / 1e9
    return f"{allocated:.2f}/{reserved:.2f} GB (alloc/reserved)"


class ContrastiveTrainer:
    """Trains a CheXzero model with the CLIP contrastive objective.

    Features:
    - Mixed-precision training (``torch.cuda.amp``)
    - Gradient clipping
    - Best-checkpoint saving (by validation loss)
    - Optional WandB logging
    - Comprehensive per-batch and per-epoch logging for remote debugging

    Args:
        model:        CheXzero model instance.
        config:       Nested config dict (from ``load_config``).
        train_loader: DataLoader yielding ``{image, report, study_id}``.
        val_loader:   DataLoader yielding ``{image, report, study_id}``.
        device:       torch.device to train on.
        logger:       Optional external logger; defaults to module logger.
    """

    BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self._log = logger or logging.getLogger(__name__)

        train_cfg = config["training"]
        self.epochs         = train_cfg["epochs"]
        self.grad_clip      = train_cfg.get("grad_clip", 1.0)
        self.mixed_precision = train_cfg.get("mixed_precision", True)
        self.use_wandb      = config.get("logging", {}).get("use_wandb", False)
        self.log_every      = train_cfg.get("log_every_n_batches", 50)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        self.scaler = GradScaler(enabled=self.mixed_precision)

        self._log.info(
            "ContrastiveTrainer — device=%s epochs=%d lr=%s bs=%d "
            "mixed_precision=%s grad_clip=%s",
            device, self.epochs, train_cfg["lr"],
            train_cfg["batch_size"], self.mixed_precision, self.grad_clip,
        )

        self._log.info("Loading tokenizer %s …", self.BERT_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.BERT_MODEL)
        self.max_text_length = 256

        checkpoint_dir = config["paths"]["checkpoint_dir"]
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self._log.info(
            "Tokenizer loaded — vocab_size=%d max_text_length=%d",
            self.tokenizer.vocab_size, self.max_text_length,
        )

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            max_length=self.max_text_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        epoch_start = time.time()

        self._log.info(
            "=== Epoch %d/%d START — batches=%d gpu_mem=%s ===",
            epoch, self.epochs - 1, len(self.train_loader),
            _gpu_memory_str(self.device),
        )

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            images  = batch["image"].to(self.device, non_blocking=True)
            reports = batch["report"]

            try:
                tokens = self._tokenize(reports)
                input_ids     = tokens["input_ids"].to(self.device, non_blocking=True)
                attention_mask = tokens["attention_mask"].to(self.device, non_blocking=True)
            except Exception as exc:
                self._log.error(
                    "Tokenisation failed at epoch=%d batch=%d: %s",
                    epoch, batch_idx, exc, exc_info=True,
                )
                raise

            self.optimizer.zero_grad()

            try:
                with autocast(enabled=self.mixed_precision):
                    logits_i, logits_t = self.model(images, input_ids, attention_mask)
                    loss = clip_loss(logits_i, logits_t)
            except Exception as exc:
                self._log.error(
                    "Forward pass failed at epoch=%d batch=%d "
                    "images.shape=%s input_ids.shape=%s: %s",
                    epoch, batch_idx, images.shape, input_ids.shape, exc,
                    exc_info=True,
                )
                raise

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            batch_ms = (time.time() - batch_start) * 1000

            if batch_idx % self.log_every == 0:
                logit_scale = self.model.logit_scale.exp().item()
                self._log.info(
                    "Epoch %d [%d/%d] loss=%.4f grad_norm=%.3f "
                    "logit_scale=%.3f batch_ms=%.0f gpu_mem=%s",
                    epoch, batch_idx, len(self.train_loader),
                    loss.item(), grad_norm.item(),
                    logit_scale, batch_ms,
                    _gpu_memory_str(self.device),
                )

                if self.use_wandb:
                    self._wandb_log({
                        "train/batch_loss": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/logit_scale": logit_scale,
                    })

        avg_loss = total_loss / len(self.train_loader)
        epoch_sec = time.time() - epoch_start
        self._log.info(
            "=== Epoch %d DONE — avg_train_loss=%.4f epoch_time=%.1fs ===",
            epoch, avg_loss, epoch_sec,
        )
        return avg_loss

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        start = time.time()

        self._log.info("Validation — %d batches", len(self.val_loader))

        for batch_idx, batch in enumerate(self.val_loader):
            images  = batch["image"].to(self.device, non_blocking=True)
            reports = batch["report"]

            try:
                tokens = self._tokenize(reports)
                input_ids      = tokens["input_ids"].to(self.device, non_blocking=True)
                attention_mask = tokens["attention_mask"].to(self.device, non_blocking=True)
            except Exception as exc:
                self._log.error(
                    "Val tokenisation failed at batch=%d: %s",
                    batch_idx, exc, exc_info=True,
                )
                raise

            with autocast(enabled=self.mixed_precision):
                logits_i, logits_t = self.model(images, input_ids, attention_mask)
                loss = clip_loss(logits_i, logits_t)

            total_loss += loss.item()

        avg = total_loss / len(self.val_loader)
        self._log.info(
            "Validation done — avg_val_loss=%.4f time=%.1fs",
            avg, time.time() - start,
        )
        return avg

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }, path)
        self._log.info("Checkpoint saved → %s  (epoch=%d val_loss=%.4f)", path, epoch, val_loss)

    def load_checkpoint(self, path: str) -> int:
        self._log.info("Loading checkpoint from %s …", path)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        epoch = checkpoint["epoch"]
        val_loss = checkpoint.get("val_loss", float("nan"))
        self._log.info(
            "Resumed from epoch=%d val_loss=%.4f", epoch, val_loss
        )
        return epoch

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, resume_path: Optional[str] = None):
        start_epoch = 0
        if resume_path is not None:
            start_epoch = self.load_checkpoint(resume_path) + 1

        best_val_loss = float("inf")
        self._log.info(
            "Training starts — start_epoch=%d target_epochs=%d",
            start_epoch, self.epochs,
        )

        for epoch in range(start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate()

            self._log.info(
                "Epoch %d summary | train_loss=%.4f val_loss=%.4f best_val=%.4f",
                epoch, train_loss, val_loss, best_val_loss,
            )

            if self.use_wandb:
                self._wandb_log({
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss,
                    "epoch": epoch,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")
                self._log.info("New best checkpoint at epoch %d", epoch)

            self.save_checkpoint(epoch, val_loss, tag="latest")

        self._log.info(
            "Training complete — best_val_loss=%.4f checkpoint_dir=%s",
            best_val_loss, self.checkpoint_dir,
        )

    # ── WandB helper ──────────────────────────────────────────────────────────

    def _wandb_log(self, data: dict):
        try:
            import wandb
            wandb.log(data)
        except ImportError:
            self._log.warning("wandb not installed; skipping wandb.log()")
        except Exception as exc:
            self._log.warning("wandb.log() failed: %s", exc)
