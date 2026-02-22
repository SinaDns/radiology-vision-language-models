import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.training.losses import clip_loss


class ContrastiveTrainer:
    """Trains a CheXzero model with the CLIP contrastive objective.

    Supports mixed-precision training, gradient clipping, best-checkpoint
    saving, and optional WandB logging.
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
        self.logger = logger or logging.getLogger(__name__)

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.mixed_precision = train_cfg.get("mixed_precision", True)
        self.use_wandb = config.get("logging", {}).get("use_wandb", False)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        self.scaler = GradScaler(enabled=self.mixed_precision)

        self.tokenizer = AutoTokenizer.from_pretrained(self.BERT_MODEL)
        self.max_text_length = 256

        checkpoint_dir = config["paths"]["checkpoint_dir"]
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

    def _tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            max_length=self.max_text_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            reports = batch["report"]

            tokens = self._tokenize(reports)
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=self.mixed_precision):
                logits_i, logits_t = self.model(images, input_ids, attention_mask)
                loss = clip_loss(logits_i, logits_t)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                logit_scale = self.model.logit_scale.exp().item()
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} logit_scale={logit_scale:.3f}"
                )
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            "train/batch_loss": loss.item(),
                            "train/logit_scale": logit_scale,
                        })
                    except ImportError:
                        pass

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            reports = batch["report"]

            tokens = self._tokenize(reports)
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)

            with autocast(enabled=self.mixed_precision):
                logits_i, logits_t = self.model(images, input_ids, attention_mask)
                loss = clip_loss(logits_i, logits_t)

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.logger.info(f"Resumed from checkpoint: {path} (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]

    def train(self, resume_path: Optional[str] = None):
        start_epoch = 0
        if resume_path is not None:
            start_epoch = self.load_checkpoint(resume_path) + 1

        best_val_loss = float("inf")

        for epoch in range(start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.logger.info(
                f"Epoch {epoch} complete | train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "train/epoch_loss": train_loss,
                        "val/loss": val_loss,
                        "epoch": epoch,
                    })
                except ImportError:
                    pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")

            self.save_checkpoint(epoch, val_loss, tag="latest")

        self.logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
