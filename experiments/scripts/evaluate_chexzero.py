#!/usr/bin/env python3
"""Zero-shot evaluation of CheXzero on NIH ChestX-ray14.

Usage:
    python experiments/scripts/evaluate_chexzero.py \
        --config experiments/configs/chexzero.yaml \
        --checkpoint experiments/results/checkpoints/chexzero/best.pt \
        [--device cuda] \
        [--split-csv data/nih_chestxray14/test_list.txt]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loaders.nih_chestxray14 import NIHChestXray14Dataset, NIH14_LABELS
from src.data_loaders.transforms import get_val_transforms
from src.evaluation.metrics import classification_report, compute_auroc, compute_auprc
from src.evaluation.zero_shot import compute_zero_shot_scores
from src.models.chexzero import CheXzero
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of CheXzero on NIH-14")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--split-csv",
        default=None,
        help="Optional official test split file (one image filename per line)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    # Load model
    model = CheXzero(embed_dim=config["model"]["embed_dim"])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Build NIH-14 dataset
    nih14_dir = config["paths"]["nih14_dir"]
    image_size = config["data"]["image_size"]

    dataset = NIHChestXray14Dataset(
        data_dir=nih14_dir,
        split="test" if args.split_csv else None,
        split_csv=args.split_csv,
        transform=get_val_transforms(image_size),
    )
    print(f"NIH-14 evaluation samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # Compute zero-shot scores
    print("Computing zero-shot scores...")
    scores_dict = compute_zero_shot_scores(
        model=model,
        image_loader=loader,
        labels=NIH14_LABELS,
        device=device,
    )

    # Collect ground-truth labels
    all_labels = {label: [] for label in NIH14_LABELS}
    for batch in loader:
        gt = batch["labels"].numpy()   # (B, 14)
        for i, label in enumerate(NIH14_LABELS):
            all_labels[label].append(gt[:, i])
    labels_dict = {label: np.concatenate(all_labels[label]) for label in NIH14_LABELS}

    # Metrics
    auroc_results = compute_auroc(scores_dict, labels_dict)
    auprc_results = compute_auprc(scores_dict, labels_dict)

    # Build score/label matrices for the report
    score_matrix = np.stack([scores_dict[l] for l in NIH14_LABELS], axis=1)
    label_matrix = np.stack([labels_dict[l] for l in NIH14_LABELS], axis=1)

    report = classification_report(score_matrix, label_matrix, label_names=NIH14_LABELS)
    print("\n=== Zero-Shot Classification Results (NIH ChestX-ray14) ===")
    print(report)
    print(f"\nMean AUROC: {auroc_results['mean_auroc']:.4f}")
    print(f"Mean AUPRC: {auprc_results['mean_auprc']:.4f}")


if __name__ == "__main__":
    main()
