#!/usr/bin/env python3
"""Fine-tune R2Gen with factuality-constrained loss (Extension 2).

Loads a pre-trained R2Gen checkpoint and continues training with the
ProxyFactualityLoss (clinical keyword coverage penalty).

Usage:
    python experiments/scripts/train_factuality.py \
        --config experiments/configs/factuality.yaml \
        [--device cuda]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.amp
from torch.utils.data import DataLoader

from src.data_loaders.iu_xray_seq2seq import IUXraySeq2SeqDataset, build_tokenizer
from src.data_loaders.transforms import get_train_transforms, get_val_transforms
from src.evaluation.generation_metrics import compute_all_metrics, generation_report
from src.models.r2gen import R2GenModel
from src.training.factuality_loss import ProxyFactualityLoss
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger, init_wandb


def _gpu_memory_str(device):
    if device.type != "cuda":
        return "N/A"
    return f"{torch.cuda.memory_allocated(device)/1e9:.2f} GB"


def parse_args():
    parser = argparse.ArgumentParser(description="Factuality fine-tuning of R2Gen")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config["paths"]["log_dir"], config["logging"]["run_name"])

    logger.info("=== Factuality Fine-Tuning ===")
    logger.info("Config: %s", args.config)

    if config["logging"].get("use_wandb", False):
        init_wandb(config)

    device = torch.device(args.device)
    logger.info("Device: %s", device)
    _use_amp = config["training"].get("mixed_precision", True) and device.type == "cuda"

    # ── Tokenizer & datasets ─────────────────────────────────────────────────
    tok_cfg  = config["tokenizer"]
    data_cfg = config["data"]
    iu_dir   = config["paths"]["iu_xray_dir"]

    tokenizer = build_tokenizer(
        data_dir=iu_dir,
        save_path=tok_cfg.get("vocab_save_path"),
        min_freq=tok_cfg.get("min_freq", 3),
        val_fraction=data_cfg["val_split"],
    )
    logger.info("Vocabulary size: %d", tokenizer.vocab_size)

    image_size = data_cfg["image_size"]
    max_length = tok_cfg.get("max_length", 100)

    train_ds = IUXraySeq2SeqDataset(
        data_dir=iu_dir, tokenizer=tokenizer, split="train",
        val_fraction=data_cfg["val_split"],
        transform=get_train_transforms(image_size), max_length=max_length,
    )
    val_ds = IUXraySeq2SeqDataset(
        data_dir=iu_dir, tokenizer=tokenizer, split="val",
        val_fraction=data_cfg["val_split"],
        transform=get_val_transforms(image_size), max_length=max_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=True,
    )
    logger.info("Train=%d Val=%d", len(train_ds), len(val_ds))

    # ── Model: load from R2Gen checkpoint ────────────────────────────────────
    model_cfg = config["model"]
    model = R2GenModel(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_enc_layers=model_cfg["num_enc_layers"],
        num_dec_layers=model_cfg["num_dec_layers"],
        dim_ff=model_cfg["dim_ff"],
        dropout=model_cfg["dropout"],
        num_mem_slots=model_cfg["num_mem_slots"],
        max_seq_len=model_cfg["max_seq_len"],
        pretrained_image=False,
        pad_id=tokenizer.pad_id,
    )

    r2gen_ckpt = config["model"]["r2gen_checkpoint"]
    logger.info("Loading R2Gen weights from %s …", r2gen_ckpt)
    ckpt = torch.load(r2gen_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    logger.info("R2Gen weights loaded (base epoch=%s)", ckpt.get("epoch", "?"))

    # ── Factuality loss ───────────────────────────────────────────────────────
    criterion = ProxyFactualityLoss(
        vocab=tokenizer.word2idx,
        coverage_weight=config["training"].get("coverage_weight", 0.3),
        pad_id=tokenizer.pad_id,
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=_use_amp)

    ckpt_dir = config["paths"]["checkpoint_dir"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs = config["training"]["epochs"]
    log_every = config["training"].get("log_every_n_batches", 50)

    logger.info("Starting factuality fine-tuning — epochs=%d", epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce   = 0.0
        total_fact = 0.0
        n_batches  = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images     = batch["image"].to(device, non_blocking=True)
            input_ids  = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)

            optimizer.zero_grad()

            try:
                with torch.amp.autocast(device_type=device.type, enabled=_use_amp):
                    logits = model(images, input_ids)
                    loss, ce_loss, fact_loss = criterion(logits, target_ids)
            except Exception as exc:
                logger.error(
                    "Factuality forward failed epoch=%d batch=%d: %s",
                    epoch, batch_idx, exc, exc_info=True,
                )
                raise

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce   += ce_loss.item()
            total_fact += fact_loss.item()
            n_batches  += 1

            if batch_idx % log_every == 0:
                logger.info(
                    "Epoch %d [%d/%d] total=%.4f ce=%.4f fact=%.4f "
                    "grad_norm=%.3f gpu_mem=%s",
                    epoch, batch_idx, len(train_loader),
                    loss.item(), ce_loss.item(), fact_loss.item(),
                    grad_norm.item(), _gpu_memory_str(device),
                )

        avg_loss = total_loss / max(n_batches, 1)
        avg_ce   = total_ce   / max(n_batches, 1)
        avg_fact = total_fact / max(n_batches, 1)
        logger.info(
            "Epoch %d done — avg_loss=%.4f avg_ce=%.4f avg_fact=%.4f time=%.1fs",
            epoch, avg_loss, avg_ce, avg_fact, time.time() - epoch_start,
        )

        # Validation (CE-only)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images     = batch["image"].to(device, non_blocking=True)
                input_ids  = batch["input_ids"].to(device, non_blocking=True)
                target_ids = batch["target_ids"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=_use_amp):
                    logits = model(images, input_ids)
                    loss, _, _ = criterion(logits, target_ids)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)
        logger.info("Epoch %d val_loss=%.4f", epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, os.path.join(ckpt_dir, "best.pt"))
            logger.info("Saved best factuality checkpoint at epoch %d", epoch)

    logger.info("Factuality fine-tuning complete — best_val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
