#!/usr/bin/env python3
"""Evaluate R2Gen on IU X-Ray val split: BLEU, ROUGE-L, METEOR.

Usage:
    python experiments/scripts/evaluate_r2gen.py \
        --config experiments/configs/r2gen.yaml \
        --checkpoint experiments/results/checkpoints/r2gen/best.pt \
        [--device cuda] \
        [--beam-size 3]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from src.data_loaders.iu_xray_seq2seq import IUXraySeq2SeqDataset, build_tokenizer
from src.data_loaders.transforms import get_val_transforms
from src.evaluation.generation_metrics import compute_all_metrics, generation_report
from src.models.r2gen import R2GenModel
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate R2Gen report generation")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam-size",  type=int, default=3)
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config["paths"]["log_dir"], "r2gen-eval")

    logger.info("=== R2Gen Evaluation Script ===")
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Beam size: %d", args.beam_size)

    device = torch.device(args.device)
    logger.info("Device: %s", device)

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

    val_dataset = IUXraySeq2SeqDataset(
        data_dir=iu_dir,
        tokenizer=tokenizer,
        split="val",
        val_fraction=data_cfg["val_split"],
        transform=get_val_transforms(config["data"]["image_size"]),
        max_length=tok_cfg.get("max_length", 100),
    )
    logger.info("Val samples: %d", len(val_dataset))

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
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
        pretrained_image=False,   # weights already in checkpoint
        pad_id=tokenizer.pad_id,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info("Loaded checkpoint from epoch %s", ckpt.get("epoch", "?"))

    # ── Generate ──────────────────────────────────────────────────────────────
    hypotheses: list[str] = []
    references: list[str] = []

    gen_cfg = config.get("generation", {})
    beam_size  = args.beam_size or gen_cfg.get("beam_size", 3)
    max_length = gen_cfg.get("max_length", 100)

    logger.info("Generating reports (beam_size=%d) …", beam_size)

    for batch_idx, batch in enumerate(val_loader):
        images = batch["image"].to(device, non_blocking=True)
        refs   = batch["report"]

        try:
            token_ids_list = model.generate(
                images,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                beam_size=beam_size,
                max_length=max_length,
            )
        except Exception as exc:
            logger.error(
                "Generation failed at batch=%d: %s", batch_idx, exc, exc_info=True
            )
            raise

        for ids in token_ids_list:
            hypotheses.append(tokenizer.decode(ids, skip_special=True))
        references.extend(refs)

        if (batch_idx + 1) % 20 == 0:
            logger.info("Generated %d/%d batches", batch_idx + 1, len(val_loader))

    logger.info("Generation complete — %d samples", len(hypotheses))

    # Sample output
    logger.info("Sample generated report: %s", hypotheses[0] if hypotheses else "(empty)")
    logger.info("Sample reference report: %s", references[0] if references else "(empty)")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_all_metrics(hypotheses, references)

    print("\n=== R2Gen Generation Results (IU X-Ray val split) ===")
    print(generation_report(metrics))
    print("\nTarget: BLEU-4 > 0.10, ROUGE-L > 0.30")

    logger.info("Evaluation complete — metrics=%s", metrics)


if __name__ == "__main__":
    main()
