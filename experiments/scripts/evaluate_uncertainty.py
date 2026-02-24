#!/usr/bin/env python3
"""Evaluate Extension 1: Uncertainty-Aware Classification on NIH-14.

Usage:
    python experiments/scripts/evaluate_uncertainty.py \
        --config experiments/configs/uncertainty.yaml \
        [--device cuda]
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
from src.evaluation.metrics import compute_auroc, classification_report
from src.evaluation.zero_shot import PATHOLOGY_PROMPTS, _encode_prompts
from src.models.chexzero import CheXzero
from src.models.uncertainty import MCDropoutCheXzero, ConformalPredictor
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Uncertainty-aware classification evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config["paths"]["log_dir"], config["logging"]["run_name"])

    logger.info("=== Uncertainty-Aware Classification Evaluation ===")
    logger.info("Config: %s", args.config)

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ── Load CheXzero + MC Dropout wrapper ──────────────────────────────────
    ckpt_path = config["model"]["chexzero_checkpoint"]
    logger.info("Loading CheXzero checkpoint from %s …", ckpt_path)
    base_model = CheXzero(embed_dim=config["model"]["embed_dim"])
    ckpt = torch.load(ckpt_path, map_location=device)
    base_model.load_state_dict(ckpt["model_state_dict"])

    mc_model = MCDropoutCheXzero(
        chexzero=base_model,
        dropout_rate=config["model"]["dropout_rate"],
        n_samples=config["model"]["n_mc_samples"],
    )
    mc_model.to(device)
    logger.info("MC Dropout model ready — n_mc_samples=%d", config["model"]["n_mc_samples"])

    # ── Tokenizer + text embeddings ──────────────────────────────────────────
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    logger.info("Pre-computing text embeddings for all labels …")
    # Stack positive text embeddings (one per label) for zero-shot scoring
    text_embs = []
    for label in NIH14_LABELS:
        prompts  = PATHOLOGY_PROMPTS.get(label, {})
        pos_list = prompts.get("positive", [f"findings consistent with {label.lower()}"])
        embs = _encode_prompts(base_model, pos_list, tokenizer, device)
        text_embs.append(embs.mean(dim=0, keepdim=True))   # (1, D)
    text_embs_t = torch.cat(text_embs, dim=0).to(device)   # (14, D)
    logger.info("Text embeddings shape: %s", list(text_embs_t.shape))

    # ── Load NIH-14 datasets (calibration + test) ────────────────────────────
    data_cfg  = config["data"]
    nih14_dir = config["paths"]["nih14_dir"]
    image_size = data_cfg["image_size"]

    calib_csv = config["paths"].get("nih14_calib_csv")
    test_csv  = config["paths"].get("nih14_split_csv")

    logger.info("Loading NIH-14 calibration split …")
    calib_ds = NIHChestXray14Dataset(
        data_dir=nih14_dir,
        split="train",
        split_csv=calib_csv,
        transform=get_val_transforms(image_size),
    )
    logger.info("NIH-14 test split …")
    test_ds = NIHChestXray14Dataset(
        data_dir=nih14_dir,
        split="test",
        split_csv=test_csv,
        transform=get_val_transforms(image_size),
    )

    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False,
                              num_workers=data_cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              num_workers=data_cfg["num_workers"], pin_memory=True)

    # ── MC Dropout scoring on test set ───────────────────────────────────────
    logger.info("Computing MC Dropout predictions on test set …")
    all_mean_scores, all_uncertainty, all_labels = [], [], []

    for batch_idx, batch in enumerate(test_loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].numpy()   # (B, 14)

        try:
            result = mc_model.predict_with_uncertainty(
                images, text_embs_t, n_samples=config["model"]["n_mc_samples"]
            )
        except Exception as exc:
            logger.error("MC prediction failed at batch=%d: %s", batch_idx, exc, exc_info=True)
            raise

        all_mean_scores.append(result["mean_scores"].cpu().numpy())
        all_uncertainty.append(result["uncertainty"].cpu().numpy())
        all_labels.append(labels)

        if (batch_idx + 1) % 50 == 0:
            logger.info("MC scoring: %d/%d batches done", batch_idx + 1, len(test_loader))

    mean_scores_np  = np.concatenate(all_mean_scores,  axis=0)  # (N, 14)
    uncertainty_np  = np.concatenate(all_uncertainty,  axis=0)
    labels_np       = np.concatenate(all_labels,       axis=0)

    logger.info(
        "MC scoring complete — N=%d mean_uncertainty=%.4f",
        len(labels_np), uncertainty_np.mean(),
    )

    # ── AUROC with mean scores ────────────────────────────────────────────────
    scores_dict = {l: mean_scores_np[:, i] for i, l in enumerate(NIH14_LABELS)}
    labels_dict = {l: labels_np[:, i] for i, l in enumerate(NIH14_LABELS)}
    auroc = compute_auroc(scores_dict, labels_dict)

    print("\n=== Uncertainty-Aware Zero-Shot Results (NIH-14) ===")
    print(classification_report(mean_scores_np, labels_np, label_names=NIH14_LABELS))
    print(f"\nMean AUROC: {auroc['mean_auroc']:.4f}")

    # ── Conformal prediction calibration on calibration split ────────────────
    logger.info("Computing calibration scores on calibration split …")
    calib_scores, calib_labels = [], []

    for batch in calib_loader:
        images = batch["image"].to(device, non_blocking=True)
        gt     = batch["labels"].numpy()
        with torch.no_grad():
            emb = base_model.encode_image(images).cpu()
            sc  = (emb @ text_embs_t.cpu().T).numpy()
        calib_scores.append(sc)
        calib_labels.append(gt)

    calib_scores_np = np.concatenate(calib_scores, axis=0)
    calib_labels_np = np.concatenate(calib_labels, axis=0)
    logger.info("Calibration set size: %d", len(calib_labels_np))

    predictor = ConformalPredictor(alpha=config["conformal"]["alpha"])
    predictor.calibrate(calib_scores_np, calib_labels_np)

    coverage_stats = predictor.compute_coverage(mean_scores_np, labels_np)
    print(f"\n=== Conformal Prediction (alpha={config['conformal']['alpha']}) ===")
    print(f"Empirical coverage: {coverage_stats['coverage']:.4f}  "
          f"(target: {1 - config['conformal']['alpha']:.2f})")
    print(f"Average prediction set size: {coverage_stats['avg_set_size']:.2f} / 14")

    ece_stats = predictor.compute_ece_reduction(
        calib_scores_np, mean_scores_np, labels_np
    )
    print(f"\nECE before calibration: {ece_stats['ece_before']:.4f}")
    print(f"ECE after calibration:  {ece_stats['ece_after']:.4f}")
    print(f"ECE reduction: {ece_stats['ece_reduction']:.1%}  (target: 20-30%)")

    logger.info("Uncertainty evaluation complete — %s", {**auroc, **coverage_stats, **ece_stats})


if __name__ == "__main__":
    main()
