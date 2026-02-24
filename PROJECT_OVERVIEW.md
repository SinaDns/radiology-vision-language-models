# Radiology Vision-Language Models Research Project

**Graduate Course:** Biomedical Image Analysis and Processing
**Institution:** Sharif University of Technology
**Date:** January 2026

---

## Project Overview

### Research Topic
**"Vision-Language Models in Radiology: Zero-Shot Classification, Cross-Modal Retrieval, and Factual Radiology Report Generation"**

### Research Goals
1. Develop uncertainty-aware zero-shot chest X-ray classification
2. Create factuality-constrained radiology report generation
3. Establish cross-dataset robustness evaluation benchmarks
4. Target publication at MICCAI, MIDL, or TMI

---

## Repository Structure

```
radiology-vision-language-models/
├── proposal/                    # LaTeX research proposal
│   └── main.tex                 # Complete academic proposal
│
├── paper/                       # Research paper
│   └── main.tex
│
├── literature/                  # Paper reviews & analysis
│   ├── vlm-foundations/
│   │   ├── chexzero.md          ✅ Comprehensive review
│   │   ├── gloria.md            ✅ Comprehensive review
│   │   └── convirt.md           ✅ Comprehensive review
│   ├── report-generation/
│   │   └── r2gen.md             ✅ Comprehensive review
│   ├── evaluation-metrics/
│   │   └── radgraph.md          ✅ Comprehensive review
│   ├── base_paper_analysis.md   ✅ CheXzero vs GLORIA vs R2Gen
│   └── papers_summary_index.md  ✅ Master index of all papers
│
├── datasets/                    # Dataset documentation
│   ├── iu-xray.md               ✅ IU X-Ray: 3,955 studies, PUBLIC
│   └── nih-chestxray14.md       ✅ NIH-14: 112K images, PUBLIC
│
├── references/
│   └── bibliography.bib         ✅ 40+ BibTeX entries
│
├── src/                         # ✅ Fully implemented
│   ├── models/
│   │   ├── chexzero.py          # ImageEncoder + TextEncoder + CheXzero
│   │   ├── r2gen.py             # VisualExtractor + RelationalMemory + R2GenModel
│   │   └── uncertainty.py       # MCDropoutCheXzero, MCDropoutR2Gen, ConformalPredictor
│   ├── data_loaders/
│   │   ├── iu_xray.py           # IUXrayDataset (XML parsing, URI-based image pairing)
│   │   ├── iu_xray_seq2seq.py   # IUXraySeq2SeqDataset + build_tokenizer
│   │   ├── nih_chestxray14.py   # NIHChestXray14Dataset (14-label multi-hot)
│   │   ├── report_tokenizer.py  # ReportTokenizer (word-level, BOS/EOS/PAD/UNK)
│   │   └── transforms.py        # get_train_transforms / get_val_transforms
│   ├── training/
│   │   ├── contrastive.py       # ContrastiveTrainer (CheXzero)
│   │   ├── r2gen_trainer.py     # R2GenTrainer (CE + CosineAnnealingLR)
│   │   ├── scst_trainer.py      # SCSTTrainer (REINFORCE + RadGraph F1 reward)
│   │   ├── factuality_loss.py   # ProxyFactualityLoss (clinical keyword coverage)
│   │   └── losses.py            # clip_loss
│   ├── evaluation/
│   │   ├── zero_shot.py         # PATHOLOGY_PROMPTS + compute_zero_shot_scores
│   │   ├── generation_metrics.py# BLEU/ROUGE/METEOR + compute_all_metrics
│   │   ├── metrics.py           # compute_auroc, compute_auprc, compute_ece
│   │   ├── calibration.py       # calibration_analysis, plot_calibration, ECE
│   │   └── cross_dataset.py     # extract_embeddings, domain gap, FewShotLinearProbe
│   └── utils/
│       ├── config.py            # load_config
│       └── logging_utils.py     # setup_logger, init_wandb
│
├── experiments/                 # ✅ Fully implemented
│   ├── configs/
│   │   ├── chexzero.yaml
│   │   ├── r2gen.yaml
│   │   ├── scst.yaml
│   │   ├── factuality.yaml
│   │   ├── uncertainty.yaml
│   │   └── cross_dataset.yaml
│   ├── scripts/
│   │   ├── train_chexzero.py
│   │   ├── train_r2gen.py
│   │   ├── scst_finetune.py
│   │   ├── train_factuality.py
│   │   ├── evaluate_chexzero.py
│   │   ├── evaluate_r2gen.py
│   │   ├── evaluate_uncertainty.py
│   │   ├── evaluate_cross_dataset.py
│   │   ├── download_iu_xray.sh
│   │   └── download_nih14.sh
│   ├── chexzero_colab.ipynb     # Train CheXzero + retrieval + NIH-14 zero-shot
│   ├── r2gen_colab.ipynb        # Train R2Gen + generation metrics
│   ├── r2gen_test_colab.ipynb   # SCST + MC Dropout + calibration analysis
│   └── extensions_colab.ipynb  # All three extensions
│
├── requirements.txt             # ✅ All dependencies pinned
├── CLAUDE.md                    # ✅ AI assistant guidance (kept up to date)
└── .gitignore
```

---

## Key Decisions Made

### Base Implementation: CheXzero + R2Gen

**Rationale:**
- ✅ Excellent code quality (both have 300-400 GitHub stars)
- ✅ Compatible with PUBLIC datasets (IU X-Ray, NIH-14)
- ✅ Moderate compute (single T4 GPU via Google Colab)
- ✅ Multiple extension paths for novel contributions
- ✅ Strong publication potential

**Alternative Considered:** GLORIA (chose CheXzero for simplicity and extensibility)

---

### Datasets: IU X-Ray + NIH ChestX-ray14

**Primary Training:** IU X-Ray (3,955 studies → ~3,400 train, ~550 val after 90/10 split)
- Fully public, no credentialing
- XML reports parsed using exact `<parentImage URI>` filenames (avoids wrong-image pairing)
- Image-report pairs: one report per study, multiple images per study supported

**Primary Evaluation:** NIH ChestX-ray14 (112,120 images)
- Large-scale for robust evaluation
- Tests cross-dataset generalization
- Public, no barriers

**Rationale for Avoiding MIMIC-CXR:**
- Requires PhysioNet credentialing
- IU X-Ray + NIH-14 sufficient for methodological contributions

---

### Three Implemented Extensions

**Extension 1: Uncertainty-Aware Classification** ✅ Implemented
- MC Dropout over CheXzero embeddings (n_samples=10 on T4)
- Conformal prediction calibration (α=0.10, target coverage ≥ 90%)
- Code: `src/models/uncertainty.py`, `src/evaluation/calibration.py`
- Notebook: `extensions_colab.ipynb` (Section 1)

**Extension 2: Factuality-Constrained Generation** ✅ Implemented
- Two-stage: (a) proxy clinical keyword loss, (b) SCST fine-tuning with RadGraph F1 reward
- Visual encoder frozen during SCST; combined CE + λ·RL loss (λ=0.1)
- Code: `src/training/factuality_loss.py`, `src/training/scst_trainer.py`
- Notebook: `extensions_colab.ipynb` (Section 2), `r2gen_test_colab.ipynb` (Section 4)

**Extension 3: Cross-Dataset Robustness** ✅ Implemented
- Zero-shot IU X-Ray → NIH-14, then few-shot linear probe (k=1,5,10,25)
- Domain gap analysis (cosine similarity between source and target embeddings)
- Code: `src/evaluation/cross_dataset.py`
- Notebook: `extensions_colab.ipynb` (Section 3)

---

## Current Progress

### ✅ Phase 1: Foundation (Complete)

1. Literature review (5 comprehensive reviews + master index)
2. Dataset documentation (IU X-Ray, NIH-14)
3. Bibliography (40+ BibTeX entries)
4. Research proposal (LaTeX)
5. Base paper analysis and architecture selection

### ✅ Phase 2: Implementation (Complete)

1. **CheXzero baseline** — ResNet-50 + BioClinicalBERT contrastive training
2. **R2Gen baseline** — ResNet-101 + relational memory + meshed decoder
3. **CIDEr metric** — via `pycocoevalcap`, integrated into `compute_all_metrics()`
4. **CheXbert evaluation** — `stanfordmlgroup/CheXbert` label-level F1 via `compute_chexbert_metrics()`
5. **Extension 1** — MC Dropout uncertainty + conformal prediction
6. **Extension 2** — Proxy factuality loss + SCST fine-tuning
5. **Extension 3** — Domain gap analysis + few-shot linear probe
6. **Google Colab notebooks** — 4 notebooks, all T4-compatible
7. **CLI scripts** — 8 training/evaluation entry points
8. **Config files** — 6 YAML configs

### ⏳ Phase 3: Evaluation & Analysis (Next)

- Run full training on Colab with saved checkpoints
- Collect quantitative results (AUROC, BLEU-4, RadGraph F1, ECE)
- Ablation studies for each extension
- Write paper draft

---

## Next Steps

### Immediate (Run Experiments)

```bash
# Order of operations:
# 1. chexzero_colab.ipynb   → produces chexzero/best.pt
# 2. r2gen_colab.ipynb      → produces r2gen/best.pt
# 3. r2gen_test_colab.ipynb → produces scst/best.pt + calibration plots
# 4. extensions_colab.ipynb → all three extensions (needs NIH-14 for Ext 1 & 3)
```

Save all checkpoints to Google Drive between sessions.

### Then: Write Paper

- Target: **MICCAI Workshop** (CLIP-Med, May 2026) with Extensions 1 & 2
- Extend to MICCAI main or MIDL with all 3 extensions

---

## Research Timeline

### Phase 1: Foundation (Months 1-2) ✅
- ✅ Literature review
- ✅ Proposal writing
- ✅ Dataset documentation
- ✅ Architecture selection

### Phase 2: Implementation (Months 3-5) ✅
- ✅ CheXzero baseline
- ✅ R2Gen baseline
- ✅ Extension 1: Uncertainty quantification
- ✅ Extension 2: Factuality-constrained generation
- ✅ Extension 3: Cross-dataset robustness
- ✅ Colab notebooks + CLI scripts

### Phase 3: Evaluation & Analysis (Month 6) ⏳
- Run experiments, collect metrics
- Ablation studies
- Error analysis

### Phase 4: Writing & Submission (Months 7-8)
- Paper draft (4 weeks)
- Internal review (1 week)
- Revision + submission to MICCAI/MIDL

---

## Key Research Questions

1. **Can we quantify uncertainty in zero-shot medical VLMs to enable safe clinical deployment?**
   - Method: MC Dropout + conformal prediction on CheXzero
   - Target: 20-30% ECE reduction, coverage ≥ 90% at α=0.10

2. **Does RadGraph-based factuality loss reduce hallucination in report generation?**
   - Method: Proxy clinical keyword loss + SCST with RadGraph F1 reward
   - Target: RadGraph F1 improvement from baseline; CheXbert macro F1 improvement

3. **How robust are medical VLMs to cross-institutional distribution shift?**
   - Method: Zero-shot IU X-Ray → NIH-14 + few-shot linear probe
   - Target: k=5 few-shot probe recovers 50-70% of performance drop

---

## Publication Strategy

### Target Venues (in priority order)

**Option A: MICCAI Workshop** (Recommended for initial submission)
- CLIP-Med Workshop
- Submission: May 2026
- Format: 4 pages

**Option B: MIDL (Medical Imaging with Deep Learning)**
- Submission: next cycle
- Format: 8 pages

**Option C: MICCAI Main Conference**
- Submission: March 2026
- Format: 8 pages + 1 supplement

### Recommended Path:
1. Target **MICCAI workshop** (May 2026) with Extensions 1 & 2
2. Extend to **MICCAI main** or **TMI** with all 3 extensions + clinical validation

---

## Success Criteria

### Minimum Viable Project (Pass Graduate Course)
- ✅ Reproduce CheXzero baseline on IU X-Ray
- ✅ Reproduce R2Gen baseline on IU X-Ray
- ✅ Add RadGraph F1 evaluation
- ✅ Cross-dataset evaluation (IU → NIH-14)
- ⏳ Written report documenting findings (in progress)

### Target Outcome (Workshop Paper)
- ✅ Above + Extensions 1-3 implemented
- ⏳ Ablation studies (needs experimental results)
- ⏳ 4-page workshop paper drafted

### Stretch Goal (Conference Paper)
- ✅ All 3 extensions implemented
- ⏳ Comprehensive evaluation on multiple datasets
- ⏳ Clinical error analysis
- ⏳ 8-page conference paper

---

## Resources & Tools

### Code
- **This repo:** All baselines and extensions implemented in `src/`
- **Reference:** CheXzero — https://github.com/rajpurkarlab/CheXzero
- **Reference:** R2Gen — https://github.com/cuhksz-nlp/R2Gen
- **Reference:** RadGraph — https://github.com/stanfordmlgroup/RadGraph

### Datasets
- **IU X-Ray:** https://openi.nlm.nih.gov/
- **NIH-14:** https://nihcc.app.box.com/v/ChestXray-NIHCC

### Environment
```bash
pip install -r requirements.txt
```

### Compute
- **Google Colab T4:** sufficient for all experiments (see notebooks)
- **Estimated:** 100-150 GPU hours total

---

## Version History

- **v0.1 (2026-01-01):** Initial repository setup, literature review started
- **v0.2 (2026-01-07):** Base paper selected, proposal drafted
- **v0.3 (2026-01-14):** Dataset documented, environment configured
- **v1.0 (2026-02-24):** Full implementation complete — baselines + 3 extensions + Colab notebooks

---

**Last Updated:** February 2026
**Status:** ✅ Phase 2 Implementation Complete → Ready to Run Experiments
**Current Branch:** `feature/phase2-implementation`
