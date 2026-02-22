"""Sequence-to-sequence trainer for R2Gen report generation.

Uses torch.amp.autocast (PyTorch ≥ 2.0) to avoid the deprecation of
torch.cuda.amp.autocast introduced in PyTorch 2.4.
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


def _gpu_memory_str(device: torch.device) -> str:
    if device.type != "cuda":
        return "N/A"
    alloc    = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    return f"{alloc:.2f}/{reserved:.2f} GB (alloc/reserved)"


class R2GenTrainer:
    """Trains an R2Gen model with teacher-forced cross-entropy loss.

    Expected batch keys from the DataLoader:
    - ``image``      ``(B, 3, H, W)``
    - ``input_ids``  ``(B, T)``  — BOS + tokens (decoder input)
    - ``target_ids`` ``(B, T)``  — tokens + EOS (loss targets)

    Args:
        model:        R2GenModel instance.
        config:       Config dict from ``load_config``.
        train_loader: DataLoader for training split.
        val_loader:   DataLoader for validation split.
        device:       Compute device.
        pad_id:       Padding token ID (ignored in CE loss).
        logger:       Optional external logger.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        pad_id: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        self.model        = model.to(device)
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.pad_id       = pad_id
        self._log         = logger or logging.getLogger(__name__)
        self._device_type = device.type

        train_cfg = config["training"]
        self.epochs          = train_cfg["epochs"]
        self.grad_clip       = train_cfg.get("grad_clip", 1.0)
        self.mixed_precision = train_cfg.get("mixed_precision", True)
        self.use_wandb       = config.get("logging", {}).get("use_wandb", False)
        self.log_every       = train_cfg.get("log_every_n_batches", 50)

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg["epochs"],
        )
        self.scaler = torch.amp.GradScaler(
            device=self._device_type,
            enabled=(self.mixed_precision and device.type == "cuda"),
        )

        checkpoint_dir = config["paths"]["checkpoint_dir"]
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self._log.info(
            "R2GenTrainer — device=%s epochs=%d lr=%s bs=%d mixed_precision=%s",
            device, self.epochs, train_cfg["lr"],
            train_cfg["batch_size"], self.mixed_precision,
        )

    # ── AMP context helper ────────────────────────────────────────────────────

    def _amp_ctx(self):
        """Return a torch.amp.autocast context compatible with PyTorch 2.x."""
        return torch.amp.autocast(
            device_type=self._device_type,
            enabled=(self.mixed_precision and self._device_type == "cuda"),
        )

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_tokens   = 0
        epoch_start = time.time()

        self._log.info(
            "=== R2Gen Epoch %d/%d START — batches=%d gpu_mem=%s ===",
            epoch, self.epochs - 1, len(self.train_loader),
            _gpu_memory_str(self.device),
        )

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            images     = batch["image"].to(self.device, non_blocking=True)
            input_ids  = batch["input_ids"].to(self.device, non_blocking=True)
            target_ids = batch["target_ids"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            try:
                with self._amp_ctx():
                    logits = self.model(images, input_ids)       # (B, T, V)
                    B, T, V = logits.shape
                    loss = self.criterion(
                        logits.reshape(B * T, V),
                        target_ids.reshape(B * T),
                    )
            except Exception as exc:
                self._log.error(
                    "Forward pass failed — epoch=%d batch=%d "
                    "images=%s input_ids=%s: %s",
                    epoch, batch_idx, list(images.shape),
                    list(input_ids.shape), exc, exc_info=True,
                )
                raise

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            non_pad     = (target_ids != self.pad_id).sum().item()
            total_loss += loss.item() * non_pad
            n_tokens   += non_pad
            batch_ms    = (time.time() - batch_start) * 1000

            if batch_idx % self.log_every == 0:
                self._log.info(
                    "Epoch %d [%d/%d] loss=%.4f ppl=%.2f grad_norm=%.3f "
                    "batch_ms=%.0f gpu_mem=%s",
                    epoch, batch_idx, len(self.train_loader),
                    loss.item(), 2 ** loss.item(),
                    grad_norm.item(), batch_ms,
                    _gpu_memory_str(self.device),
                )
                if self.use_wandb:
                    self._wandb_log({
                        "r2gen/train_batch_loss": loss.item(),
                        "r2gen/train_ppl":         2 ** loss.item(),
                        "r2gen/grad_norm":         grad_norm.item(),
                    })

        self.scheduler.step()
        avg_loss = total_loss / max(n_tokens, 1)
        avg_ppl  = 2 ** avg_loss
        elapsed  = time.time() - epoch_start

        self._log.info(
            "=== R2Gen Epoch %d DONE — avg_loss=%.4f avg_ppl=%.2f time=%.1fs ===",
            epoch, avg_loss, avg_ppl, elapsed,
        )
        return avg_loss

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_tokens   = 0
        start = time.time()

        self._log.info("R2Gen validation — %d batches", len(self.val_loader))

        for batch_idx, batch in enumerate(self.val_loader):
            images     = batch["image"].to(self.device, non_blocking=True)
            input_ids  = batch["input_ids"].to(self.device, non_blocking=True)
            target_ids = batch["target_ids"].to(self.device, non_blocking=True)

            try:
                with self._amp_ctx():
                    logits = self.model(images, input_ids)
                    B, T, V = logits.shape
                    loss = self.criterion(
                        logits.reshape(B * T, V),
                        target_ids.reshape(B * T),
                    )
            except Exception as exc:
                self._log.error(
                    "Val forward failed — batch=%d: %s",
                    batch_idx, exc, exc_info=True,
                )
                raise

            non_pad     = (target_ids != self.pad_id).sum().item()
            total_loss += loss.item() * non_pad
            n_tokens   += non_pad

        avg_loss = total_loss / max(n_tokens, 1)
        self._log.info(
            "R2Gen validation done — avg_loss=%.4f ppl=%.2f time=%.1fs",
            avg_loss, 2 ** avg_loss, time.time() - start,
        )
        return avg_loss

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        torch.save({
            "epoch":                  epoch,
            "model_state_dict":       self.model.state_dict(),
            "optimizer_state_dict":   self.optimizer.state_dict(),
            "scheduler_state_dict":   self.scheduler.state_dict(),
            "scaler_state_dict":      self.scaler.state_dict(),
            "val_loss":               val_loss,
            "config":                 self.config,
        }, path)
        self._log.info(
            "R2Gen checkpoint → %s (epoch=%d val_loss=%.4f)",
            path, epoch, val_loss,
        )

    def load_checkpoint(self, path: str) -> int:
        self._log.info("Loading R2Gen checkpoint from %s …", path)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self._log.info(
            "Resumed from epoch=%d val_loss=%.4f",
            ckpt["epoch"], ckpt.get("val_loss", float("nan")),
        )
        return ckpt["epoch"]

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, resume_path: Optional[str] = None):
        start_epoch = 0
        if resume_path:
            start_epoch = self.load_checkpoint(resume_path) + 1

        best_val_loss = float("inf")
        self._log.info(
            "R2Gen training starts — start_epoch=%d target_epochs=%d",
            start_epoch, self.epochs,
        )

        for epoch in range(start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate()

            self._log.info(
                "R2Gen Epoch %d | train_loss=%.4f val_loss=%.4f "
                "train_ppl=%.2f val_ppl=%.2f best_val=%.4f",
                epoch, train_loss, val_loss,
                2 ** train_loss, 2 ** val_loss, best_val_loss,
            )

            if self.use_wandb:
                self._wandb_log({
                    "r2gen/epoch_train_loss": train_loss,
                    "r2gen/epoch_val_loss":   val_loss,
                    "r2gen/val_ppl":          2 ** val_loss,
                    "epoch":                  epoch,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")
                self._log.info("New R2Gen best at epoch %d", epoch)

            self.save_checkpoint(epoch, val_loss, tag="latest")

        self._log.info(
            "R2Gen training complete — best_val_loss=%.4f checkpoint=%s",
            best_val_loss, self.checkpoint_dir,
        )

    def _wandb_log(self, data: dict):
        try:
            import wandb
            wandb.log(data)
        except Exception as exc:
            self._log.warning("wandb.log() failed: %s", exc)
