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
│   ├── main.tex                # Complete academic proposal
│   ├── figures/                # Architecture diagrams (to be added)
│   └── tables/                 # Results tables (to be added)
│
├── literature/                  # Paper reviews & analysis
│   ├── vlm-foundations/        # CheXzero, GLORIA, ConVIRT, BioViL
│   │   ├── chexzero.md        ✅ Comprehensive review
│   │   ├── gloria.md          ✅ Comprehensive review
│   │   └── convirt.md         ✅ Comprehensive review
│   ├── report-generation/      # R2Gen, CMN, PPKED
│   │   └── r2gen.md           ✅ Comprehensive review
│   ├── evaluation-metrics/     # RadGraph, CheXbert
│   │   └── radgraph.md        ✅ Comprehensive review
│   ├── clinical-applications/  # Clinical use cases
│   ├── base_paper_analysis.md ✅ CheXzero vs GLORIA vs R2Gen
│   └── papers_summary_index.md ✅ Master index of all papers
│
├── datasets/                    # Dataset documentation
│   ├── iu-xray.md             ✅ IU X-Ray: 3,955 studies, PUBLIC
│   ├── nih-chestxray14.md     ✅ NIH-14: 112K images, PUBLIC
│   ├── mimic-cxr.md           ⏳ Reference only (credentialed)
│   ├── chexpert.md            ⏳ Registration required
│   ├── padchest.md            ⏳ Spanish reports, public
│   └── roco.md                ⏳ PubMed captions
│
├── references/                  # Bibliography & citations
│   ├── bibliography.bib        ✅ 40+ BibTeX entries
│   └── papers/                 # PDF storage (gitignored)
│
├── experiments/                 # Code & results (future)
│   ├── configs/                # YAML config files
│   ├── scripts/                # Training & evaluation
│   ├── notebooks/              # Jupyter analysis
│   └── results/                # Logs, checkpoints, figures
│
├── notes/                       # Research journal
│   ├── meeting_notes.md        # Advisor meetings
│   ├── ideas.md                # Research ideas
│   └── timeline.md             # Weekly progress
│
├── .gitignore                   # Ignore data, PDFs, checkpoints
└── README.md                    # (Will be created at project end)
```

---

## Key Decisions Made

### Base Implementation: CheXzero + R2Gen

**Rationale:**
- ✅ Excellent code quality (both have 300-400 GitHub stars)
- ✅ Compatible with PUBLIC datasets (IU X-Ray, NIH-14)
- ✅ Moderate compute (1× GPU, 24-48 hours training)
- ✅ Multiple extension paths for novel contributions
- ✅ Strong publication potential

**Alternative Considered:** GLORIA (chose CheXzero for simplicity and extensibility)

---

### Datasets: IU X-Ray + NIH ChestX-ray14

**Primary Training:** IU X-Ray (3,955 studies with reports)
- Fully public, no credentialing
- Well-structured radiology reports
- Standard benchmark for report generation

**Primary Evaluation:** NIH ChestX-ray14 (112,120 images)
- Large-scale for robust evaluation
- Tests cross-dataset generalization
- Public, no barriers

**Rationale for Avoiding MIMIC-CXR:**
- Requires PhysioNet credentialing (1-2 week delay)
- Unnecessary for proof-of-concept graduate project
- Public datasets sufficient for methodological contributions

---

### Three Research Extensions

**Extension 1: Uncertainty-Aware Classification** (Priority: HIGH)
- Method: Monte Carlo Dropout + Conformal Prediction
- Gap: Clinical deployment requires knowing when model is uncertain
- Publication venue: Radiology AI, MIDL
- Feasibility: High (add dropout layers, calibration set)

**Extension 2: Factuality-Constrained Generation** (Priority: HIGH)
- Method: RadGraph-based training loss
- Gap: Models hallucinate findings not present in images
- Publication venue: MICCAI, EMNLP, TMI
- Feasibility: Moderate (requires RadGraph integration)

**Extension 3: Cross-Dataset Robustness** (Priority: MEDIUM)
- Method: Systematic evaluation IU X-Ray → NIH-14
- Gap: Most papers ignore external validation
- Publication venue: MICCAI, MIDL
- Feasibility: High (evaluation-focused, no new training)

---

## Current Progress

### ✅ Completed (Week 1)

1. **Repository Structure**
   - Created formal directory hierarchy
   - Organized by research function (proposal, literature, datasets)

2. **Literature Review**
   - **5 comprehensive paper reviews:**
     - CheXzero (Nature BME 2022) - 15 pages
     - GLORIA (ICCV 2021) - 13 pages
     - ConVIRT (ML4H 2020) - 9 pages
     - R2Gen (EMNLP 2020) - 12 pages
     - RadGraph (NEJM AI 2023) - 14 pages
   - Master index of 45+ papers with relevance ratings
   - Base paper comparison and recommendation

3. **Dataset Documentation**
   - IU X-Ray: Complete documentation with access instructions
   - NIH-14: Complete documentation with preprocessing guide
   - Comparison table and selection rationale

4. **Bibliography**
   - 40+ BibTeX entries properly formatted
   - Covers: VLMs, generation, evaluation, UQ, datasets

5. **Research Proposal (In Progress)**
   - LaTeX document started
   - Sections completed:
     - Abstract ✅
     - Introduction ✅ (clinical motivation, gaps, objectives)
     - Related Work ✅ (VLMs, generation, evaluation, robustness)
     - Datasets ✅ (detailed descriptions, comparison)
     - Methodology ✅ (baseline + 3 extensions with math)
   - Sections remaining:
     - Experimental Plan
     - Timeline
     - References integration

---

## Next Steps (Week 2)

### Immediate Priorities

1. **Complete LaTeX Proposal** (2-3 days)
   - Section 5: Experimental Plan and Evaluation
   - Section 6: Expected Contributions
   - Section 7: Ethical Considerations
   - Section 8: Timeline (Gantt chart)
   - Section 9: Conclusion
   - Compile and generate PDF

2. **Additional Literature Reviews** (2-3 days)
   - CheXbert (evaluation metric) - HIGH priority
   - BioViL (RadGraph pretraining) - MEDIUM priority
   - CMN (report generation variant) - MEDIUM priority
   - MC Dropout paper (UQ method) - HIGH priority

3. **Dataset Access** (1 day)
   - Download IU X-Ray (7 GB)
   - Download NIH-14 (45 GB)
   - Verify data integrity, explore statistics

4. **Environment Setup** (1 day)
   - Install PyTorch, Transformers, medical imaging libraries
   - Clone CheXzero and R2Gen repositories
   - Test on sample data

---

## Research Timeline (8 Months)

### Phase 1: Foundation (Months 1-2)
- ✅ Literature review
- ✅ Proposal writing
- ⏳ Dataset preparation
- ⏳ Baseline reproduction (CheXzero, R2Gen)

### Phase 2: Extension Development (Months 3-5)
- Extension 1: Uncertainty quantification (4 weeks)
- Extension 2: Factuality loss (6 weeks)
- Extension 3: Cross-dataset eval (2 weeks)

### Phase 3: Evaluation & Analysis (Month 6)
- Comprehensive experiments
- Ablation studies
- Error analysis

### Phase 4: Writing & Submission (Months 7-8)
- Paper draft (4 weeks)
- Internal review (1 week)
- Revision (2 weeks)
- Submission to MICCAI/MIDL

---

## Key Research Questions

1. **Can we quantify uncertainty in zero-shot medical VLMs to enable safe clinical deployment?**
   - Hypothesis: MC Dropout + conformal prediction provide calibrated confidence
   - Expected: 20-30% ECE reduction, 10-15% accuracy gain by deferring 15% uncertain cases

2. **Does RadGraph-based factuality loss reduce hallucination in report generation?**
   - Hypothesis: Knowledge graph supervision improves clinical correctness
   - Expected: RadGraph F1 improvement from 0.42 → 0.48 (+14% relative)

3. **How robust are medical VLMs to cross-institutional distribution shift?**
   - Hypothesis: Performance drops 5-10% AUROC on external datasets
   - Expected: Few-shot adaptation (5% labeled data) recovers 50-70% of drop

---

## Publication Strategy

### Target Venues (in priority order)

**Option A: MICCAI Workshop** (Recommended for initial submission)
- CLIP-Med Workshop
- DGM4MICCAI (Deep Generative Models)
- Submission: May 2026
- Format: 4 pages
- Pros: Lower bar, fast feedback, community engagement

**Option B: MIDL (Medical Imaging with Deep Learning)**
- Submission: January 2026 (MISSED) or next cycle
- Format: 8 pages
- Pros: Strong methodological focus, fair reviews

**Option C: MICCAI Main Conference**
- Submission: March 2026
- Format: 8 pages + 1 supplement
- Pros: Prestigious, high impact
- Cons: Competitive, requires strong validation

**Option D: TMI Journal** (If results very strong)
- Submission: Rolling
- Format: Full paper (15-20 pages)
- Pros: High impact factor, comprehensive presentation
- Cons: Longer review cycle (4-6 months)

### Recommended Path:
1. Target **MICCAI workshop** (May 2026) with 1-2 extensions
2. If accepted: Present at MICCAI (October 2026)
3. Extend to **MICCAI main** or **TMI** with all 3 extensions + clinical validation

---

## Risk Assessment & Mitigation

### Risk 1: Baseline Reproduction Fails
- **Probability:** Low (both have excellent code)
- **Mitigation:** Start with pretrained weights, use exact hyperparameters
- **Contingency:** Reach out to authors (both are responsive)

### Risk 2: Public Data Insufficient for Strong Results
- **Probability:** Medium (IU X-Ray only 3,955 studies)
- **Mitigation:** Frame as data efficiency / few-shot learning
- **Contingency:** Combine IU + NIH (with pseudo-reports)

### Risk 3: Extension 2 (Factuality Loss) Too Complex
- **Probability:** Medium (RadGraph integration non-trivial)
- **Mitigation:** Start with simpler CheXbert auxiliary loss
- **Contingency:** Focus on Extensions 1 & 3 only

### Risk 4: Compute Limitations
- **Probability:** Low-Medium
- **Mitigation:** Apply for Sharif HPC cluster, use cloud credits
- **Contingency:** Reduce model size, use mixed precision

---

## Success Criteria

### Minimum Viable Project (Pass Graduate Course)
- ✅ Reproduce CheXzero baseline on IU X-Ray
- ✅ Reproduce R2Gen baseline on IU X-Ray  
- ✅ Add RadGraph F1 evaluation
- ✅ Cross-dataset evaluation (IU → NIH-14)
- ✅ Written report documenting findings

### Target Outcome (Workshop Paper)
- ✅ Above + 1-2 extensions implemented
- ✅ Ablation studies
- ✅ 4-page workshop paper drafted
- ✅ Submission to MICCAI workshop

### Stretch Goal (Conference Paper)
- ✅ All 3 extensions implemented
- ✅ Comprehensive evaluation on multiple datasets
- ✅ Clinical error analysis
- ✅ 8-page conference paper
- ✅ Submission to MICCAI main or MIDL

---

## Resources & Tools

### Code Repositories
- **CheXzero:** https://github.com/rajpurkarlab/CheXzero
- **R2Gen:** https://github.com/cuhksz-nlp/R2Gen
- **RadGraph:** https://github.com/stanfordmlgroup/RadGraph

### Datasets
- **IU X-Ray:** https://openi.nlm.nih.gov/
- **NIH-14:** https://nihcc.app.box.com/v/ChestXray-NIHCC

### Libraries
```bash
pip install torch torchvision transformers
pip install scikit-learn scipy pandas matplotlib seaborn
pip install pydicom nibabel SimpleITK  # Medical imaging
pip install radgraph  # Report evaluation
pip install wandb tensorboard  # Experiment tracking
```

### Compute
- **Local:** Sharif HPC cluster (申请中)
- **Cloud:** Google Cloud credits ($300), AWS Educate
- **Estimated:** 100-150 GPU hours total

---

## Contact & Collaboration

### Advisor
- **Name:** [To be filled]
- **Meetings:** Weekly (tentative)
- **Focus:** Research direction, paper writing

### Potential Collaborators
- **Clinical validation:** Radiology department at Sharif Hospital
- **Technical discussion:** Medical AI lab members

---

## Version History

- **v0.1 (2026-01-01):** Initial repository setup, literature review started
- **v0.2 (2026-01-07):** Base paper selected, proposal drafted
- **v0.3 (Future):** Dataset downloaded, baseline reproduced
- **v1.0 (Future):** Extensions implemented, paper submitted

---

**Last Updated:** January 1, 2026  
**Status:** 📊 Planning Phase Complete → Ready for Implementation  
**Next Milestone:** Complete LaTeX proposal + Download datasets (Week 2)
