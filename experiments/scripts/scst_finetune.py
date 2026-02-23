"""SCST fine-tuning entry point for R2Gen.

Loads a pre-trained R2Gen checkpoint and fine-tunes it with
Self-Critical Sequence Training (SCST) using RadGraph F1 rewards.

Usage::

    python experiments/scripts/scst_finetune.py \\
        --config experiments/configs/scst.yaml \\
        --device cuda

Requires a trained R2Gen checkpoint at the path specified in the config
(default: experiments/results/checkpoints/r2gen/best.pt).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Make src/ importable when called from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from src.data_loaders.iu_xray_seq2seq import IUXraySeq2SeqDataset, build_tokenizer
from src.data_loaders.transforms import get_train_transforms, get_val_transforms
from src.models.r2gen import R2GenModel
from src.training.scst_trainer import SCSTTrainer
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCST fine-tuning for R2Gen")
    p.add_argument("--config", default="experiments/configs/scst.yaml",
                   help="Path to SCST config YAML")
    p.add_argument("--device", default="cuda",
                   help="Compute device (cuda / cpu)")
    p.add_argument("--resume", default=None,
                   help="Resume from checkpoint path")
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger = setup_logger(config["paths"]["log_dir"], config["logging"]["run_name"])
    logger.info("=== SCST Fine-tuning ===")
    logger.info("Config: %s", args.config)
    logger.info("Device: %s", device)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok_cfg = config.get("tokenizer", {})
    tokenizer = build_tokenizer(
        data_dir=config["paths"]["iu_xray_dir"],
        save_path=tok_cfg.get("vocab_save_path"),
        min_freq=tok_cfg.get("min_freq", 3),
        val_fraction=config["data"]["val_split"],
    )
    logger.info("Vocabulary: %d tokens", tokenizer.vocab_size)

    # ── Datasets ─────────────────────────────────────────────────────────────
    image_size   = config["data"]["image_size"]
    val_fraction = config["data"]["val_split"]
    num_workers  = config["data"]["num_workers"]
    max_seq_len  = config["model"].get("max_seq_len", 100)

    train_ds = IUXraySeq2SeqDataset(
        data_dir=config["paths"]["iu_xray_dir"],
        tokenizer=tokenizer,
        split="train",
        val_fraction=val_fraction,
        transform=get_train_transforms(image_size),
        max_length=max_seq_len,
    )
    val_ds = IUXraySeq2SeqDataset(
        data_dir=config["paths"]["iu_xray_dir"],
        tokenizer=tokenizer,
        split="val",
        val_fraction=val_fraction,
        transform=get_val_transforms(image_size),
        max_length=max_seq_len,
    )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg = config["model"]
    model = R2GenModel(
        vocab_size=tokenizer.vocab_size,
        d_model=mcfg["d_model"],
        num_heads=mcfg["num_heads"],
        num_enc_layers=mcfg["num_enc_layers"],
        num_dec_layers=mcfg["num_dec_layers"],
        dim_ff=mcfg["dim_ff"],
        dropout=mcfg["dropout"],
        num_mem_slots=mcfg["num_mem_slots"],
        max_seq_len=mcfg["max_seq_len"],
        pretrained_image=False,   # weights loaded from checkpoint below
        pad_id=tokenizer.pad_id,
    )

    ckpt_path = mcfg.get("r2gen_checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        logger.info("R2Gen checkpoint loaded from %s (epoch %s)",
                    ckpt_path, ck.get("epoch", "?"))
    else:
        logger.warning(
            "R2Gen checkpoint not found at '%s' — starting SCST from random weights!",
            ckpt_path,
        )

    model.to(device)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SCSTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger_=logger,
    )

    trainer.train(resume_path=args.resume)
    logger.info("SCST fine-tuning complete.")


if __name__ == "__main__":
    main()
