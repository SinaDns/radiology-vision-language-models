#!/usr/bin/env python3
"""Entry point for R2Gen report generation training.

Usage:
    python experiments/scripts/train_r2gen.py \
        --config experiments/configs/r2gen.yaml \
        [--device cuda] \
        [--resume path/to/checkpoint.pt]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from src.data_loaders.iu_xray_seq2seq import IUXraySeq2SeqDataset, build_tokenizer
from src.data_loaders.transforms import get_train_transforms, get_val_transforms
from src.models.r2gen import R2GenModel
from src.training.r2gen_trainer import R2GenTrainer
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger, init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train R2Gen baseline")
    parser.add_argument("--config",  required=True, help="Path to YAML config")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",  default=None,  help="Path to checkpoint to resume")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    logger = setup_logger(config["paths"]["log_dir"], config["logging"]["run_name"])
    logger.info("=== R2Gen Training Script ===")
    logger.info("Config: %s", args.config)
    logger.info("Device: %s", args.device)

    if config["logging"].get("use_wandb", False):
        init_wandb(config)

    device = torch.device(args.device)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok_cfg    = config["tokenizer"]
    data_cfg   = config["data"]
    iu_dir     = config["paths"]["iu_xray_dir"]

    logger.info("Building/loading tokenizer …")
    tokenizer = build_tokenizer(
        data_dir=iu_dir,
        save_path=tok_cfg.get("vocab_save_path"),
        min_freq=tok_cfg.get("min_freq", 3),
        val_fraction=data_cfg["val_split"],
    )
    logger.info("Vocabulary size: %d", tokenizer.vocab_size)

    # ── Datasets ─────────────────────────────────────────────────────────────
    image_size = data_cfg["image_size"]
    max_length = tok_cfg.get("max_length", 100)

    train_dataset = IUXraySeq2SeqDataset(
        data_dir=iu_dir,
        tokenizer=tokenizer,
        split="train",
        val_fraction=data_cfg["val_split"],
        transform=get_train_transforms(image_size),
        max_length=max_length,
    )
    val_dataset = IUXraySeq2SeqDataset(
        data_dir=iu_dir,
        tokenizer=tokenizer,
        split="val",
        val_fraction=data_cfg["val_split"],
        transform=get_val_transforms(image_size),
        max_length=max_length,
    )

    logger.info("Train samples: %d  Val samples: %d", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
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
        pretrained_image=model_cfg.get("pretrained_image", True),
        pad_id=tokenizer.pad_id,
    )
    logger.info("R2GenModel initialised")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = R2GenTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pad_id=tokenizer.pad_id,
        logger=logger,
    )

    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()
