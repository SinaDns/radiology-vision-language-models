# Literature Summary Index

This document provides a quick reference to all papers reviewed for the project, organized by category with relevance ratings.

## Legend
- **Code:** ⭐⭐⭐ Excellent | ⭐⭐ Good | ⭐ Limited | ❌ Not available
- **Relevance:** 🔥🔥🔥 Critical | 🔥🔥 Important | 🔥 Supplementary
- **Status:** ✅ Detailed review completed | ⏳ Pending | 📝 Summary only

---

## Vision-Language Models (Foundational)

| Paper | Year | Venue | Code | Relevance | Status | Notes |
|-------|------|-------|------|-----------|--------|-------|
| **CheXzero** | 2022 | Nature BME | ⭐⭐⭐ | 🔥🔥🔥 | ✅ | **Primary base paper** - Zero-shot classification |
| **GLORIA** | 2021 | ICCV | ⭐⭐⭐ | 🔥🔥🔥 | ✅ | Alternative base - Local+global attention |
| **ConVIRT** | 2020 | ML4H | ⭐⭐ | 🔥🔥 | ✅ | Historical importance - First medical CLIP |
| **BioViL** | 2022 | ECCV | ⭐⭐⭐ | 🔥🔥 | ⏳ | RadGraph-based pretraining |
| **MedCLIP** | 2022 | EMNLP | ⭐⭐ | 🔥 | ⏳ | Semantic matching variant |
| **PubMedCLIP** | 2023 | ACL | ⭐⭐ | 🔥 | ⏳ | Broad medical domain |
| **PMC-CLIP** | 2023 | CVPR | ⭐⭐ | 🔥 | ⏳ | PubMed Central pretraining |
| **MedKLIP** | 2023 | ICCV | ⭐⭐ | 🔥 | ⏳ | Knowledge-enhanced |
| **LoVT** | 2023 | CVPR | ⭐ | 🔥 | 📝 | Local visual transformers |

**Summary:** CheXzero is the recommended base due to excellent code, public data compatibility, and clear extension paths. GLORIA is strong alternative if focusing on retrieval/grounding.

---

## Report Generation

| Paper | Year | Venue | Code | Relevance | Status | Notes |
|-------|------|-------|------|-----------|--------|-------|
| **R2Gen** | 2020 | EMNLP | ⭐⭐⭐ | 🔥🔥🔥 | ✅ | **Primary baseline** - Memory-driven transformer |
| **CMN** | 2021 | ACL | ⭐⭐ | 🔥🔥 | ⏳ | Clinical memory network |
| **PPKED** | 2022 | TMI | ⭐ | 🔥 | 📝 | Knowledge graph integration |
| **TranSQ** | 2022 | JBHI | ⭐ | 🔥 | 📝 | Semantic query attention |
| **RATCHET** | 2022 | MICCAI | ⭐ | 🔥 | 📝 | Retrieval-augmented |
| **Show-Attend-Tell** | 2015 | ICML | ⭐⭐ | 🔥 | 📝 | Classic baseline (adapted to medical) |
| **TieNet** | 2018 | CVPR | ⭐ | 🔥 | 📝 | Multi-task learning |
| **AdaAtt** | 2019 | MICCAI | ⭐ | 🔥 | 📝 | Adaptive attention |

**Summary:** R2Gen is the standard baseline with best code quality. Extensions should focus on factuality (RadGraph loss) and grounding.

---

## Evaluation Metrics & Factuality

| Paper | Year | Venue | Code | Relevance | Status | Notes |
|-------|------|-------|------|-----------|--------|-------|
| **RadGraph** | 2023 | NEJM AI | ⭐⭐⭐ | 🔥🔥🔥 | ✅ | **Essential** - Knowledge graph IE tool |
| **CheXbert** | 2020 | EMNLP | ⭐⭐⭐ | 🔥🔥🔥 | ⏳ | Label extraction from reports |
| **GREEN** | 2022 | EMNLP | ⭐ | 🔥🔥 | ⏳ | Entity-level grounding |
| **FactualGAN** | 2022 | MICCAI | ⭐ | 🔥 | 📝 | Adversarial factuality |
| **X-REM** | 2023 | ACL | ⭐ | 🔥 | 📝 | Error correction |
| **BLEU/ROUGE** | Various | NLG | ⭐⭐⭐ | 🔥 | 📝 | Legacy metrics (use cautiously) |

**Summary:** RadGraph F1 is mandatory for report generation evaluation. CheXbert provides complementary label-level accuracy.

---

## Datasets

| Dataset | Images | Reports | Labels | Access | Relevance | Status | Notes |
|---------|--------|---------|--------|--------|-----------|--------|-------|
| **IU X-Ray** | 7.5K | ✅ | Manual | Public | 🔥🔥🔥 | ✅ | **Primary training dataset** |
| **NIH-14** | 112K | ❌ | NLP | Public | 🔥🔥🔥 | ✅ | **Primary eval dataset** |
| MIMIC-CXR | 377K | ✅ | NLP | Credentialed | 🔥🔥 | 📝 | Gold standard (but restricted) |
| CheXpert | 224K | ❌ | NLP | Registration | 🔥 | 📝 | Alternative classification |
| PadChest | 160K | Spanish | 174 | Public | 🔥 | 📝 | European population |
| ROCO | 81K | Captions | ❌ | Public | 🔥 | 📝 | Multi-modal medical |
| MS-CXR | Subset | ✅ | Grounding | Public | 🔥🔥 | 📝 | Sentence-region annotations |

**Summary:** IU X-Ray (reports) + NIH-14 (scale) provide optimal public data combination for this project.

---

## Uncertainty Quantification

| Paper | Year | Venue | Code | Relevance | Status | Notes |
|-------|------|-------|------|-----------|--------|-------|
| **MC Dropout** | 2016 | ICML | ⭐⭐⭐ | 🔥🔥🔥 | 📝 | Bayesian approximation via dropout |
| **Conformal Prediction** | 2021 | Review | ⭐⭐⭐ | 🔥🔥🔥 | 📝 | Distribution-free confidence sets |
| **Calibration** | 2017 | ICML | ⭐⭐⭐ | 🔥🔥 | 📝 | Temperature scaling, ECE |
| **Ensemble Methods** | Various | Various | ⭐⭐⭐ | 🔥🔥 | 📝 | Model aggregation for uncertainty |

**Summary:** MC Dropout (simple) + Conformal Prediction (rigorous) provide complementary UQ approaches for Extension 1.

---

## Domain Adaptation & Robustness

| Paper | Year | Venue | Code | Relevance | Status | Notes |
|-------|------|-------|------|-----------|--------|-------|
| **Domain-Adversarial** | 2016 | JMLR | ⭐⭐ | 🔥🔥 | 📝 | Adversarial domain adaptation |
| **WILDS Benchmark** | 2021 | ICML | ⭐⭐⭐ | 🔥🔥 | 📝 | Distribution shift evaluation |
| **DA Theory** | 2010 | ML Journal | ⭐ | 🔥 | 📝 | Theoretical foundations |

**Summary:** Limited prior work on medical VLM robustness—major research opportunity for Extension 3.

---

## Supporting Papers (Vision, NLP, etc.)

| Paper | Year | Venue | Relevance | Notes |
|-------|------|-------|-----------|-------|
| **CLIP** | 2021 | ICML | 🔥🔥🔥 | Natural image VLM (foundational) |
| **ResNet** | 2016 | CVPR | 🔥🔥 | Standard CNN backbone |
| **Vision Transformer** | 2020 | ICLR | 🔥🔥 | Alternative vision encoder |
| **BERT** | 2018 | NAACL | 🔥🔥 | Transformer language model |
| **Bio+Clinical BERT** | 2019 | EMNLP | 🔥🔥🔥 | Medical text encoder |
| **Attention Mechanism** | 2017 | NeurIPS | 🔥🔥 | Transformer foundations |

---

## Paper Reading Priority

### **Must Read (This Week):**
1. ✅ CheXzero (Nature BME 2022) - Completed detailed review
2. ✅ GLORIA (ICCV 2021) - Completed detailed review
3. ✅ ConVIRT (ML4H 2020) - Completed detailed review
4. ✅ R2Gen (EMNLP 2020) - Completed detailed review
5. ✅ RadGraph (NEJM AI 2023) - Completed detailed review

### **Should Read (Next Week):**
6. ⏳ CheXbert (EMNLP 2020) - Evaluation metric
7. ⏳ BioViL (ECCV 2022) - RadGraph pretraining
8. ⏳ CMN (ACL 2021) - Report generation extension
9. ⏳ MC Dropout (ICML 2016) - Uncertainty quantification
10. ⏳ Conformal Prediction review (2021) - UQ theory

### **Nice to Read (If Time):**
11. 📝 MedCLIP, PubMedCLIP, PMC-CLIP (2022-2023)
12. 📝 GREEN, FactualGAN (2022)
13. 📝 TieNet, AdaAtt, RATCHET (2018-2022)
14. 📝 Domain adaptation papers (2010-2021)

---

## Quick Reference: Which Paper for Which Task?

### **Zero-Shot Classification:**
- **Implement:** CheXzero ⭐⭐⭐
- Compare: ConVIRT, GLORIA

### **Image-Text Retrieval:**
- **Implement:** GLORIA ⭐⭐⭐
- Compare: CheXzero, BioViL

### **Report Generation:**
- **Implement:** R2Gen ⭐⭐⭐
- Compare: CMN, Show-Attend-Tell

### **Evaluation:**
- **Must use:** RadGraph F1 ⭐⭐⭐
- **Must use:** CheXbert accuracy ⭐⭐⭐
- Legacy: BLEU, ROUGE (report for comparison only)

### **Uncertainty:**
- **Implement:** MC Dropout + Conformal Prediction
- Compare: Ensemble methods

### **Robustness:**
- **Novel contribution:** Cross-dataset evaluation protocol
- Adapt: Domain-adversarial training, TTA

---

## Implementation Dependencies

```
CheXzero requires:
- PyTorch >= 1.9
- Transformers (Hugging Face)
- torchvision
- Bio+Clinical BERT checkpoint

R2Gen requires:
- PyTorch >= 1.7
- torchvision
- pycocoevalcap (for BLEU/CIDEr)

RadGraph requires:
- DyGIE++ model
- spaCy
- Hugging Face Transformers
```

---

**Last Updated:** January 2026  
**Total Papers Reviewed:** 45+  
**Detailed Summaries Completed:** 5  
**Target:** Complete 15-20 detailed summaries for proposal
