# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graduate research project for "Biomedical Image Analysis and Processing" at Sharif University of Technology (Jan 2026). Research topic: **Vision-Language Models in Radiology: Zero-Shot Classification, Cross-Modal Retrieval, and Factual Radiology Report Generation**.

The project builds on two base architectures:
- **CheXzero** (Nature BME 2022): contrastive VLM for zero-shot chest X-ray classification
- **R2Gen** (EMNLP 2020): memory-driven transformer for radiology report generation

Primary datasets: **IU X-Ray** (3,955 studies with reports, public) for training; **NIH ChestX-ray14** (112K images, public) for cross-dataset evaluation.

## Implementation Status

**Phase 2 is fully implemented.** All baselines, extensions, training pipelines, and Colab notebooks are complete.

## Repository Structure

```
proposal/        # LaTeX research proposal (main.tex → main.pdf)
paper/           # Research paper (main.tex → main.pdf)
literature/      # Comprehensive paper reviews (.md files)
  vlm-foundations/     # CheXzero, GLORIA, ConVIRT reviews
  report-generation/   # R2Gen review
  evaluation-metrics/  # RadGraph review
  papers_summary_index.md  # Master index of 45+ papers with relevance ratings
  base_paper_analysis.md   # Comparative analysis: CheXzero vs GLORIA vs R2Gen
datasets/        # Dataset documentation (.md files, no actual data)
references/      # bibliography.bib (40+ BibTeX entries)
src/
  models/
    chexzero.py          # ImageEncoder (ResNet-50) + TextEncoder (BioClinicalBERT) + CheXzero
    r2gen.py             # VisualExtractor + RelationalMemory + MeshedDecoder + R2GenModel
    uncertainty.py       # MCDropoutCheXzero, MCDropoutR2Gen, ConformalPredictor
  data_loaders/
    iu_xray.py           # IUXrayDataset — XML parsing, image–report pairing
    iu_xray_seq2seq.py   # IUXraySeq2SeqDataset + build_tokenizer for R2Gen
    nih_chestxray14.py   # NIHChestXray14Dataset — 14-class multi-label loader
    report_tokenizer.py  # ReportTokenizer (word-level, BOS/EOS/PAD/UNK)
    transforms.py        # get_train_transforms / get_val_transforms
  training/
    contrastive.py       # ContrastiveTrainer (CheXzero, CLIP loss, mixed precision)
    r2gen_trainer.py     # R2GenTrainer (CE loss, CosineAnnealingLR)
    scst_trainer.py      # SCSTTrainer (REINFORCE + CE, RadGraph F1 reward)
    factuality_loss.py   # ProxyFactualityLoss (clinical keyword coverage)
    losses.py            # clip_loss (symmetric cross-entropy)
  evaluation/
    zero_shot.py         # PATHOLOGY_PROMPTS + compute_zero_shot_scores (14 NIH labels)
    generation_metrics.py# compute_bleu/rouge/meteor + compute_all_metrics
    metrics.py           # compute_auroc, compute_auprc, compute_ece, classification_report
    calibration.py       # compute_ece, calibration_analysis, plot_calibration
    cross_dataset.py     # extract_embeddings, compute_domain_gap, FewShotLinearProbe
  utils/
    config.py            # load_config (YAML → dict)
    logging_utils.py     # setup_logger, init_wandb
experiments/
  configs/
    chexzero.yaml        # CheXzero training config (embed_dim, lr, epochs, …)
    r2gen.yaml           # R2Gen baseline config (d_model, num_heads, epochs, …)
    scst.yaml            # SCST fine-tuning config (lambda_fact, freeze_encoder, …)
    factuality.yaml      # Factuality fine-tuning config (coverage_weight, …)
    uncertainty.yaml     # MC Dropout / conformal prediction config
    cross_dataset.yaml   # Cross-dataset evaluation config
  scripts/
    train_chexzero.py    # CLI: train CheXzero contrastively on IU X-Ray
    train_r2gen.py       # CLI: train R2Gen with CE loss
    scst_finetune.py     # CLI: SCST fine-tuning from R2Gen checkpoint
    train_factuality.py  # CLI: factuality-constrained R2Gen fine-tuning
    evaluate_chexzero.py # CLI: zero-shot AUROC on NIH-14
    evaluate_r2gen.py    # CLI: BLEU/ROUGE/METEOR/RadGraph F1 on IU X-Ray val
    evaluate_uncertainty.py   # CLI: ECE + conformal prediction
    evaluate_cross_dataset.py # CLI: domain gap + few-shot adaptation
    download_iu_xray.sh  # Download IU X-Ray from NIH OpenI
    download_nih14.sh    # Instructions for NIH ChestX-ray14
  chexzero_colab.ipynb   # Colab: CheXzero train + IU X-Ray retrieval + NIH-14 zero-shot
  r2gen_colab.ipynb      # Colab: R2Gen train + BLEU/ROUGE/METEOR/RadGraph eval
  r2gen_test_colab.ipynb # Colab: SCST fine-tuning + MC Dropout + calibration analysis
  extensions_colab.ipynb # Colab: Ext 1 uncertainty, Ext 2 factuality, Ext 3 cross-dataset
  r2gen_test.ipynb       # Original Kaggle experiment notebook (reference)
  results/               # Saved checkpoints, logs, plots (gitignored)
```

## Google Colab Notebooks

All four notebooks are designed for **T4 GPU (15 GB VRAM)**. Run cells top-to-bottom within each section; sections are otherwise independent.

| Notebook | Purpose | Estimated T4 runtime |
|----------|---------|---------------------|
| `chexzero_colab.ipynb` | Train CheXzero + IU X-Ray retrieval eval + NIH-14 zero-shot | ~30 min training + eval |
| `r2gen_colab.ipynb` | Train R2Gen + generation metric evaluation | ~35 min training + eval |
| `r2gen_test_colab.ipynb` | SCST fine-tuning + MC Dropout uncertainty + calibration | ~20 min fine-tuning + analysis |
| `extensions_colab.ipynb` | All three research extensions | Depends on NIH-14 availability |

**T4 batch size defaults** (all notebooks override configs for T4 safety):
- R2Gen training/inference: `batch_size=8`
- CheXzero training: `batch_size=32`
- NIH-14 inference: `batch_size=32`
- MC Dropout: `n_samples=10`

Increase these for A100 (64/128 batch sizes, 20 MC samples).

**Checkpoint persistence:** Each notebook has commented cells to save/load checkpoints via Google Drive. Run `chexzero_colab` and `r2gen_colab` first to produce checkpoints needed by `extensions_colab`.

## CLI Entry Points

```bash
# Train baselines
python experiments/scripts/train_chexzero.py --config experiments/configs/chexzero.yaml
python experiments/scripts/train_r2gen.py    --config experiments/configs/r2gen.yaml

# Fine-tune
python experiments/scripts/scst_finetune.py     --config experiments/configs/scst.yaml
python experiments/scripts/train_factuality.py  --config experiments/configs/factuality.yaml

# Evaluate
python experiments/scripts/evaluate_chexzero.py \
    --config experiments/configs/chexzero.yaml \
    --checkpoint experiments/results/checkpoints/chexzero/best.pt

python experiments/scripts/evaluate_r2gen.py \
    --config experiments/configs/r2gen.yaml \
    --checkpoint experiments/results/checkpoints/r2gen/best.pt
```

All scripts accept `--device cuda` (default) and `--resume <path>` for checkpoint resumption.

## Python Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `torchvision`, `transformers` (BioClinicalBERT), `timm`, `scikit-learn`, `nltk`, `radgraph` (RadGraph F1 reward), `pyyaml`, `matplotlib`.

PyTorch API note: all mixed-precision uses `torch.amp.autocast` / `torch.amp.GradScaler` (PyTorch 2.0+ API, not the deprecated `torch.cuda.amp`).

## LaTeX Commands

```bash
cd proposal && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd paper    && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# or:
latexmk -pdf proposal/main.tex
latexmk -pdf paper/main.tex
```

LaTeX build artifacts are gitignored for `proposal/` only.

## Research Architecture

**Two baselines + three implemented extensions:**

1. **CheXzero baseline** (`src/models/chexzero.py`, `src/training/contrastive.py`): ResNet-50 + BioClinicalBERT contrastive training on IU X-Ray. Zero-shot classification on NIH-14 via PATHOLOGY_PROMPTS. Target: mean AUROC ≥ 0.75.

2. **R2Gen baseline** (`src/models/r2gen.py`, `src/training/r2gen_trainer.py`): ResNet-101 visual extractor → relational memory → meshed decoder. Teacher-forced CE training. Target: BLEU-4 > 0.10, ROUGE-L > 0.30.

3. **Extension 1 — Uncertainty-Aware Classification** (`src/models/uncertainty.py`): MC Dropout over CheXzero embeddings (n_samples=10) + conformal prediction calibration. Metrics: ECE reduction (target 20–30%), coverage ≥ 90% at α=0.10.

4. **Extension 2 — Factuality-Constrained Generation** (`src/training/factuality_loss.py`, `src/training/scst_trainer.py`): Two-stage approach: (a) proxy factuality loss fine-tuning (clinical keyword coverage), (b) SCST fine-tuning with RadGraph F1 reward. Target: RadGraph F1 improvement from baseline.

5. **Extension 3 — Cross-Dataset Robustness** (`src/evaluation/cross_dataset.py`): Zero-shot transfer from IU X-Ray → NIH-14 + few-shot linear probe adaptation (k=1,5,10,25). Target: few-shot probe recovers 50–70% of cross-dataset performance drop.

## Key Design Decisions

- **MIMIC-CXR avoided**: requires PhysioNet credentialing. IU X-Ray + NIH-14 are fully public.
- **CheXzero chosen over GLORIA**: simpler codebase, better extensibility, equivalent performance.
- **RadGraph F1 is the primary generation metric**: correlates with clinical accuracy (r=0.7) vs BLEU (r=0.3). BLEU/ROUGE reported for baseline comparison only.
- **IU X-Ray image pairing**: images are resolved via exact filenames from XML `<parentImage URI>` elements (not glob matching) to avoid wrong-image pairings.
- **SCST visual encoder frozen**: only decoder + memory parameters updated during RL fine-tuning to prevent feature collapse.
- Compute budget: ~100–150 GPU hours total; single T4 (Colab) sufficient for demo runs.

## Literature Reference

When working on any component, consult:
- `literature/base_paper_analysis.md` — architectural details and extension paths for CheXzero and R2Gen
- `literature/papers_summary_index.md` — relevance-rated index of all referenced papers
- `datasets/iu-xray.md` and `datasets/nih-chestxray14.md` — dataset statistics, preprocessing, and access instructions

External code repos:
- CheXzero: https://github.com/rajpurkarlab/CheXzero
- R2Gen: https://github.com/cuhksz-nlp/R2Gen
- RadGraph: https://github.com/stanfordmlgroup/RadGraph
