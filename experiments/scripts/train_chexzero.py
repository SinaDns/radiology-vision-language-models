#!/usr/bin/env python3
"""Entry point for CheXzero contrastive training.

Usage:
    python experiments/scripts/train_chexzero.py \
        --config experiments/configs/chexzero.yaml \
        [--device cuda] \
        [--resume path/to/checkpoint.pt]
"""

import argparse
import sys
from pathlib import Path

# Allow imports from repo root regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from src.data_loaders.iu_xray import IUXrayDataset
from src.data_loaders.transforms import get_train_transforms, get_val_transforms
from src.models.chexzero import CheXzero
from src.training.contrastive import ContrastiveTrainer
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger, init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train CheXzero baseline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    log_dir = config["paths"]["log_dir"]
    run_name = config["logging"]["run_name"]
    logger = setup_logger(log_dir, run_name)

    if config["logging"].get("use_wandb", False):
        init_wandb(config)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    iu_xray_dir = config["paths"]["iu_xray_dir"]

    train_dataset = IUXrayDataset(
        data_dir=iu_xray_dir,
        split="train",
        val_fraction=data_cfg["val_split"],
        transform=get_train_transforms(image_size),
    )
    val_dataset = IUXrayDataset(
        data_dir=iu_xray_dir,
        split="val",
        val_fraction=data_cfg["val_split"],
        transform=get_val_transforms(image_size),
    )

    logger.info(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

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

    model = CheXzero(embed_dim=config["model"]["embed_dim"])
    logger.info("Model initialised")

    trainer = ContrastiveTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
    )

    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()
