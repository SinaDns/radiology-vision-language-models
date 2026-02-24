#!/usr/bin/env python3
"""Evaluate Extension 3: Cross-Dataset Robustness (IU X-Ray → NIH-14).

Steps:
1. Extract IU X-Ray and NIH-14 embeddings using CheXzero
2. Compute domain gap metrics
3. Run zero-shot evaluation on NIH-14
4. Run few-shot linear probe for k ∈ {1, 5, 10, 25}
5. Print recovery table

Usage:
    python experiments/scripts/evaluate_cross_dataset.py \
        --config experiments/configs/cross_dataset.yaml \
        [--device cuda]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loaders.iu_xray import IUXrayDataset
from src.data_loaders.nih_chestxray14 import NIHChestXray14Dataset, NIH14_LABELS
from src.data_loaders.transforms import get_val_transforms
from src.evaluation.cross_dataset import (
    extract_embeddings, compute_domain_gap,
    few_shot_adapt, evaluate_probe, cross_dataset_report,
)
from src.evaluation.zero_shot import compute_zero_shot_scores
from src.evaluation.metrics import compute_auroc
from src.models.chexzero import CheXzero
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset robustness evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config["paths"]["log_dir"], config["logging"]["run_name"])

    logger.info("=== Cross-Dataset Robustness Evaluation ===")

    Path(config["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ── Load CheXzero ─────────────────────────────────────────────────────────
    ckpt_path = config["model"]["chexzero_checkpoint"]
    logger.info("Loading CheXzero from %s …", ckpt_path)
    model = CheXzero(embed_dim=config["model"]["embed_dim"])
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info("Loaded CheXzero checkpoint (epoch=%s)", ckpt.get("epoch", "?"))

    data_cfg   = config["data"]
    image_size = data_cfg["image_size"]
    n_workers  = data_cfg["num_workers"]

    # ── IU X-Ray embeddings (source domain) ──────────────────────────────────
    logger.info("Building IU X-Ray dataset for embedding extraction …")
    iu_ds = IUXrayDataset(
        data_dir=config["paths"]["iu_xray_dir"],
        split="val",
        transform=get_val_transforms(image_size),
    )
    iu_loader = DataLoader(iu_ds, batch_size=64, shuffle=False,
                           num_workers=n_workers, pin_memory=True)
    logger.info("Extracting IU X-Ray embeddings (%d samples) …", len(iu_ds))
    iu_embs = extract_embeddings(model, iu_loader, device)

    # ── NIH-14 embeddings (target domain) ────────────────────────────────────
    logger.info("Building NIH-14 test split …")
    test_csv = config["paths"].get("nih14_test_csv")
    nih_test_ds = NIHChestXray14Dataset(
        data_dir=config["paths"]["nih14_dir"],
        split="test",
        split_csv=test_csv,
        transform=get_val_transforms(image_size),
    )
    nih_loader = DataLoader(nih_test_ds, batch_size=64, shuffle=False,
                            num_workers=n_workers, pin_memory=True)
    logger.info("Extracting NIH-14 embeddings (%d samples) …", len(nih_test_ds))
    nih_embs = extract_embeddings(model, nih_loader, device)

    # Collect NIH-14 labels
    all_labels = []
    for batch in nih_loader:
        all_labels.append(batch["labels"].numpy())
    nih_labels_np = np.concatenate(all_labels, axis=0)   # (N, 14)
    logger.info("NIH-14 labels shape: %s", list(nih_labels_np.shape))

    # ── Domain gap ────────────────────────────────────────────────────────────
    print("\n=== Domain Gap Analysis ===")
    gap = compute_domain_gap(iu_embs, nih_embs)
    for k, v in gap.items():
        print(f"  {k:<25}: {v:.4f}")
    logger.info("Domain gap: %s", gap)

    # ── Zero-shot AUROC on NIH-14 ─────────────────────────────────────────────
    logger.info("Computing zero-shot scores …")
    zs_scores = compute_zero_shot_scores(
        model=model,
        image_loader=nih_loader,
        labels=NIH14_LABELS,
        device=device,
    )
    labels_dict = {l: nih_labels_np[:, i] for i, l in enumerate(NIH14_LABELS)}
    zs_auroc = compute_auroc(zs_scores, labels_dict)
    logger.info("Zero-shot mean AUROC: %.4f", zs_auroc["mean_auroc"])

    # ── Few-shot linear probe for each k ─────────────────────────────────────
    fs_cfg    = config["few_shot"]
    k_shots   = fs_cfg["k_shots"]
    nih_labels_t = torch.tensor(nih_labels_np)

    results_by_k = {}

    for k in k_shots:
        logger.info("Few-shot adaptation — k=%d …", k)
        try:
            probe = few_shot_adapt(
                embeddings=nih_embs,
                labels=nih_labels_t,
                k_shot=k,
                n_classes=len(NIH14_LABELS),
                n_epochs=fs_cfg["n_epochs"],
                lr=fs_cfg["lr"],
                device=device,
            )
            probe_auroc = evaluate_probe(
                probe=probe,
                embeddings=nih_embs,
                labels=nih_labels_np,
                device=device,
                label_names=NIH14_LABELS,
            )
            results_by_k[k] = probe_auroc
            logger.info("k=%d mean AUROC=%.4f", k, probe_auroc["mean_auroc"])
        except Exception as exc:
            logger.error("Few-shot adaptation failed for k=%d: %s", k, exc, exc_info=True)
            results_by_k[k] = {"mean_auroc": float("nan")}

    # ── Print final comparison ────────────────────────────────────────────────
    print("\n=== Cross-Dataset Results (IU X-Ray → NIH ChestX-ray14) ===")
    print(f"\nZero-shot mean AUROC: {zs_auroc['mean_auroc']:.4f}")

    for k in k_shots:
        fs_auroc = results_by_k[k]
        report   = cross_dataset_report(zs_auroc, fs_auroc, NIH14_LABELS)
        print(f"\n--- k={k} few-shot probe ---")
        print(report)

    logger.info("Cross-dataset evaluation complete")


if __name__ == "__main__":
    main()
