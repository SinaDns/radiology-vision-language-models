# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graduate research project for "Biomedical Image Analysis and Processing" at Sharif University of Technology (Jan 2026). Research topic: **Vision-Language Models in Radiology: Zero-Shot Classification, Cross-Modal Retrieval, and Factual Radiology Report Generation**.

The project builds on two base architectures:
- **CheXzero** (Nature BME 2022): contrastive VLM for zero-shot chest X-ray classification
- **R2Gen** (EMNLP 2020): memory-driven transformer for radiology report generation

Primary datasets: **IU X-Ray** (3,955 studies with reports, public) for training; **NIH ChestX-ray14** (112K images, public) for cross-dataset evaluation.

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
src/             # Source code (directory skeleton, not yet implemented)
  models/        # Model architectures
  data_loaders/  # Dataset loading utilities
  training/      # Training pipelines
  evaluation/    # Evaluation metrics
  utils/         # Utilities
experiments/     # Scripts, configs, notebooks, results (not yet implemented)
```

## LaTeX Commands

Compile the proposal or paper (from repo root):
```bash
cd proposal && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Or use latexmk for automatic dependency handling:
```bash
latexmk -pdf proposal/main.tex
latexmk -pdf paper/main.tex
```

LaTeX build artifacts are gitignored for `proposal/` only (`.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.toc`, `.fdb_latexmk`, `.fls`, `.synctex.gz`, `.xdb`).

## Python Environment (When Implemented)

Expected dependencies:
```bash
pip install torch torchvision transformers
pip install scikit-learn scipy pandas matplotlib seaborn
pip install pydicom nibabel SimpleITK   # Medical imaging
pip install radgraph                     # Report evaluation (RadGraph F1)
pip install wandb tensorboard            # Experiment tracking
```

## Research Architecture

**Three planned extensions over the baselines:**

1. **Uncertainty-Aware Classification**: Monte Carlo Dropout + Conformal Prediction on top of CheXzero. Targets 20-30% ECE reduction and 10-15% accuracy gain via selective prediction.

2. **Factuality-Constrained Generation**: RadGraph-based training loss for R2Gen to reduce hallucinations. Primary metric is RadGraph F1 (target: 0.42 → 0.48). BLEU/ROUGE are used only for baseline comparison — RadGraph F1 is the ground truth metric for clinical accuracy.

3. **Cross-Dataset Robustness**: Systematic IU X-Ray → NIH-14 evaluation with domain adaptation analysis. Few-shot adaptation expected to recover 50-70% of cross-dataset performance drop.

## Key Design Decisions

- **MIMIC-CXR avoided**: requires PhysioNet credentialing. IU X-Ray + NIH-14 are fully public and sufficient for methodological contributions.
- **CheXzero chosen over GLORIA**: simpler codebase, better extensibility, equivalent performance.
- **RadGraph F1 is the primary generation metric**: correlates with clinical accuracy (r=0.7) vs BLEU (r=0.3).
- Compute budget: ~100-150 GPU hours total; single GPU (A100 or V100) sufficient.

## Literature Reference

When working on any component, consult:
- `literature/base_paper_analysis.md` — architectural details and extension paths for CheXzero and R2Gen
- `literature/papers_summary_index.md` — relevance-rated index of all referenced papers
- `datasets/iu-xray.md` and `datasets/nih-chestxray14.md` — dataset statistics, preprocessing, and access instructions

External code repos:
- CheXzero: https://github.com/rajpurkarlab/CheXzero
- R2Gen: https://github.com/cuhksz-nlp/R2Gen
- RadGraph: https://github.com/stanfordmlgroup/RadGraph
