# Base Paper Analysis & Recommendation

## Executive Summary

After comprehensive review of medical vision-language models, we recommend **CheXzero** as the primary base implementation for this graduate research project, with **R2Gen** as a complementary component for the report generation track.

**Justification:** CheXzero offers the optimal balance of:
- ✅ Excellent code quality and documentation
- ✅ Compatibility with public datasets (IU X-Ray, NIH-14)
- ✅ Moderate computational requirements (feasible for graduate project)
- ✅ Multiple clear extension paths toward publication
- ✅ Strong foundation for classification, retrieval, AND generation tasks

---

## Top 3 Candidates Comparison

### 1. CheXzero (RECOMMENDED)

**Paper:** Tiu et al., "Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning," Nature BME 2022

**GitHub:** https://github.com/rajpurkarlab/CheXzero  
**Stars:** ~300 | **Code Quality:** ⭐⭐⭐⭐⭐ Excellent

#### Technical Specifications
- **Architecture:** ResNet-50 (image) + Bio+Clinical BERT (text)
- **Training:** Contrastive learning (modified CLIP)
- **Parameters:** ~50M total
- **GPU Requirements:** 1× A100 (24 hours) or 1× V100 (2-3 days)
- **Inference:** Real-time on single GPU

#### Public Data Compatibility
| Dataset | Compatible | Use Case |
|---------|------------|----------|
| IU X-Ray | ✅ Perfect fit | Training with reports |
| NIH-14 | ✅ Excellent | Zero-shot evaluation |
| PadChest | ⚠️ Spanish reports | Cross-dataset validation |
| ROCO | ⚠️ Captions only | Multi-modal extension |

**Training Data Required:** 3,955+ image-report pairs (IU X-Ray sufficient)

#### Extension Paths (Publication Potential)

**Extension 1: Cross-Dataset Robustness Study** ⭐⭐⭐⭐⭐  
**Gap:** Most papers train and test on single dataset  
**Approach:** Systematic evaluation on IU → NIH-14, IU → PadChest  
**Novelty:** Analyze failure modes, propose domain adaptation  
**Venue:** MICCAI, MIDL (clinical robustness focus)  
**Feasibility:** High (no new model architecture needed)

**Extension 2: Uncertainty-Aware Zero-Shot Classification** ⭐⭐⭐⭐⭐  
**Gap:** CheXzero produces point estimates without confidence  
**Approach:** 
- Monte Carlo Dropout for Bayesian uncertainty
- Conformal prediction for calibrated confidence intervals
- Ensemble methods (train 5 models with different seeds)
**Novelty:** Uncertainty quantification critical for clinical deployment  
**Venue:** Radiology AI, TMI (clinical safety focus)  
**Feasibility:** High (add dropout layers, modify inference)

**Extension 3: Automated Prompt Learning** ⭐⭐⭐⭐  
**Gap:** CheXzero requires manual prompt engineering  
**Approach:**
- Continuous prompt tuning (learnable soft prompts)
- Reinforcement learning to optimize prompts
- Meta-learning for few-shot prompt adaptation
**Novelty:** Remove human bottleneck in prompt design  
**Venue:** CVPR, ICCV (vision-language methods)  
**Feasibility:** Moderate (requires prompt tuning implementation)

**Extension 4: Retrieval + Classification** ⭐⭐⭐⭐  
**Gap:** CheXzero only does classification  
**Approach:** Use learned embeddings for image-text retrieval  
**Novelty:** Demonstrate multi-task capability  
**Venue:** MICCAI workshop, MIDL  
**Feasibility:** High (reuse embeddings, add retrieval eval)

**Extension 5: Report Generation Decoder** ⭐⭐⭐⭐⭐  
**Gap:** CheXzero lacks generation capability  
**Approach:** Add GPT-2 decoder on top of encoder  
**Novelty:** Unified model for classification + generation  
**Venue:** MICCAI, TMI  
**Feasibility:** Moderate-High (combine with R2Gen decoder)

#### Pros
- ✅ Best-in-class code and documentation
- ✅ Published in prestigious medical journal (Nature BME)
- ✅ Contrastive learning framework supports multiple tasks
- ✅ Zero-shot capability reduces labeling burden
- ✅ Moderate compute (trainable on single GPU)
- ✅ Clear gaps for novel contributions
- ✅ Active research area (700+ citations expected by 2026)

#### Cons
- ❌ Requires manual prompt engineering (can be extension)
- ❌ Binary classification per disease (not multi-label joint)
- ❌ No built-in report generation (can add decoder)
- ❌ Single modality (chest X-ray only)

---

### 2. GLORIA (Alternative if focusing on retrieval)

**Paper:** Huang et al., "GLoRIA: A Multimodal Global-Local Representation Learning Framework," ICCV 2021

**GitHub:** https://github.com/marshuang80/gloria  
**Stars:** ~200 | **Code Quality:** ⭐⭐⭐⭐ Very Good

#### Technical Specifications
- **Architecture:** ResNet-50 + BERT + Attention
- **Training:** Global + local contrastive learning
- **Parameters:** ~60M total
- **GPU Requirements:** 4× V100 (2-3 days) or 1× V100 (1-2 weeks)

#### Public Data Compatibility
- IU X-Ray: ✅ Excellent (sentence-level reports)
- NIH-14: ❌ No reports (cannot train)
- MS-CXR: ✅ Public grounding annotations

#### Extension Paths

**Extension 1: Grounded Report Generation** ⭐⭐⭐⭐⭐  
**Approach:** Use local attention as cross-attention for decoder  
**Novelty:** Link each sentence to image regions (reduce hallucination)

**Extension 2: Hierarchical Attention** ⭐⭐⭐⭐  
**Approach:** Multi-scale attention (pixel → region → image)  
**Novelty:** Fine-grained localization

#### Pros
- ✅ State-of-the-art retrieval performance
- ✅ Built-in grounding via attention
- ✅ Interpretable (attention visualizations)
- ✅ Strong for factuality research (attention = evidence)

#### Cons
- ❌ Higher compute requirements (4× GPUs preferred)
- ❌ More complex architecture (harder to debug)
- ❌ Requires sentence-level tokenization
- ❌ Slower inference (attention computation overhead)

**Verdict:** Choose GLORIA if project focuses on **retrieval or grounding**; otherwise CheXzero more versatile.

---

### 3. R2Gen (Best for Report Generation)

**Paper:** Chen et al., "Generating Radiology Reports via Memory-driven Transformer," EMNLP 2020

**GitHub:** https://github.com/cuhksz-nlp/R2Gen  
**Stars:** ~400 | **Code Quality:** ⭐⭐⭐⭐⭐ Excellent

#### Technical Specifications
- **Architecture:** ResNet-101 + ViT + Memory + Transformer Decoder
- **Training:** Cross-entropy (teacher forcing)
- **Parameters:** ~80M total
- **GPU Requirements:** 1× V100 (12 hours on IU X-Ray)

#### Public Data Compatibility
- IU X-Ray: ✅ Perfect (standard benchmark)
- NIH-14: ❌ No reports

#### Extension Paths

**Extension 1: RadGraph-Based Factuality Loss** ⭐⭐⭐⭐⭐  
**Gap:** R2Gen optimizes BLEU, not clinical correctness  
**Approach:** Add auxiliary loss penalizing entity hallucination  
**Novelty:** First to integrate RadGraph during training (not just eval)  
**Venue:** MICCAI, EMNLP

**Extension 2: Retrieval-Augmented Generation** ⭐⭐⭐⭐⭐  
**Approach:** Retrieve similar reports before generation  
**Novelty:** Hybrid retrieval-generation for factuality

#### Pros
- ✅ Excellent code and reproducibility
- ✅ Standard baseline for report generation
- ✅ Moderate compute (single GPU)
- ✅ Memory mechanism interpretable
- ✅ Natural extension to factuality

#### Cons
- ❌ Generation-only (no classification/retrieval built-in)
- ❌ Evaluated with BLEU/ROUGE (need to add RadGraph)
- ❌ Teacher forcing (exposure bias)

**Verdict:** **Combine with CheXzero** — use CheXzero encoder + R2Gen decoder for unified model.

---

## Final Recommendation: Hybrid Approach

### Phase 1: Baseline Implementation (Months 1-2)

**Primary:** CheXzero on IU X-Ray
- Reproduce zero-shot classification results
- Extend to image-text retrieval evaluation
- Test cross-dataset on NIH-14

**Secondary:** R2Gen on IU X-Ray
- Reproduce report generation results
- Add RadGraph F1 evaluation
- Baseline for generation experiments

### Phase 2: Novel Contribution (Months 3-5)

**Option A (Classification Focus):**
1. **Uncertainty-Aware CheXzero**
   - MC Dropout + conformal prediction
   - Calibration evaluation
   - Test on rare diseases

**Option B (Generation Focus):**
2. **Factuality-Aware Report Generation**
   - CheXzero encoder + R2Gen decoder
   - RadGraph-based training loss
   - Attention grounding evaluation

**Option C (Robustness Focus):**
3. **Cross-Dataset Robustness**
   - Train on IU X-Ray
   - Test on NIH-14, PadChest
   - Domain adaptation techniques

**Recommended:** **Option B** — Best novelty + publication potential

### Phase 3: Publication Preparation (Months 6-8)

- Comprehensive experiments (ablations, comparisons)
- Clinical evaluation metrics (RadGraph, CheXbert)
- Error analysis and case studies
- Write paper for MICCAI or MIDL

---

## Implementation Roadmap

### Week 1-2: Setup & Reproduction
- [ ] Set up environment (PyTorch, Transformers, medical imaging libs)
- [ ] Download IU X-Ray (7 GB) and NIH-14 (45 GB)
- [ ] Reproduce CheXzero baseline (AUROC validation)
- [ ] Reproduce R2Gen baseline (BLEU validation)

### Week 3-4: Baseline Extension
- [ ] Add RadGraph F1 evaluation to R2Gen
- [ ] Add retrieval evaluation to CheXzero
- [ ] Test cross-dataset (IU X-Ray → NIH-14)
- [ ] Document baseline performance

### Week 5-8: Novel Contribution
- [ ] Implement chosen extension (Option A, B, or C)
- [ ] Iterative development and debugging
- [ ] Ablation studies

### Week 9-12: Evaluation & Analysis
- [ ] Comprehensive evaluation on all datasets
- [ ] Statistical significance testing
- [ ] Error analysis (qualitative + quantitative)
- [ ] Attention visualizations

### Week 13-16: Writing
- [ ] Draft paper (8 pages MICCAI format)
- [ ] Create figures and tables
- [ ] Literature review refinement
- [ ] Internal review and revision

---

## Compute Budget Estimate

### Training (One-Time)
- CheXzero on IU X-Ray: 24 hours × 1× A100 = $10-20 (cloud)
- R2Gen on IU X-Ray: 12 hours × 1× V100 = $5-10
- Extension experiments: 50-100 GPU hours = $50-100

**Total Training:** ~$100-150 (or free if university GPU access)

### Inference & Evaluation
- Negligible (can run on single GPU or CPU)

### Storage
- Datasets: 60 GB (IU X-Ray + NIH-14)
- Checkpoints: 5 GB
- Results: 1 GB

**Total Storage:** ~70 GB

---

## Expected Outcomes by Track

### Track A: Classification + Uncertainty

**Expected Results:**
- CheXzero baseline: 0.81 AUROC (5 diseases, IU X-Ray)
- With uncertainty: 0.82 AUROC + calibrated confidence
- Cross-dataset (NIH-14): 0.76 AUROC (modest drop)

**Paper Title:** "Uncertainty-Aware Zero-Shot Classification for Chest X-Ray Diagnosis"  
**Venue:** Radiology AI, MIDL, ML4H Workshop

### Track B: Factual Report Generation

**Expected Results:**
- R2Gen baseline: BLEU-4 = 0.20, RadGraph F1 = 0.42
- Factual R2Gen: BLEU-4 = 0.19, RadGraph F1 = 0.48 (+6%)
- Reduced hallucination: 25% fewer false entities

**Paper Title:** "Factuality-Guided Radiology Report Generation with Knowledge Graph Supervision"  
**Venue:** MICCAI, EMNLP, TMI

### Track C: Cross-Dataset Robustness

**Expected Results:**
- Train IU X-Ray: 0.81 AUROC
- Test NIH-14: 0.74 AUROC (baseline)
- With adaptation: 0.77 AUROC (+3%)

**Paper Title:** "Evaluating and Improving Cross-Institutional Robustness of Medical Vision-Language Models"  
**Venue:** MICCAI, MIDL

---

## Risk Mitigation

### Risk 1: Baseline Reproduction Fails

**Probability:** Low (both have excellent code)  
**Mitigation:**
- Start with official repos and pretrained weights
- Use exact hyperparameters from papers
- Reach out to authors if stuck (both are responsive)

### Risk 2: Public Data Insufficient

**Probability:** Medium (IU X-Ray small)  
**Mitigation:**
- Focus on data efficiency narrative
- Frame as few-shot or transfer learning
- Combine IU X-Ray + NIH-14 (pseudo-reports)

### Risk 3: Novel Contribution Not Strong Enough

**Probability:** Medium  
**Mitigation:**
- Choose well-defined gap (uncertainty, factuality, robustness)
- Thorough evaluation beats modest improvement
- Target workshops first (lower bar), then extend to conference

### Risk 4: Compute Limitations

**Probability:** Low-Medium  
**Mitigation:**
- Use university GPU cluster (申请 Sharif HPC)
- Cloud credits (Google Cloud, AWS Educate)
- Optimize code (mixed precision, gradient accumulation)

---

## Alternative: Pivot to GLORIA

**If CheXzero proves too challenging**, pivot to GLORIA:

**Pros of GLORIA:**
- Retrieval focus (different from classification)
- Built-in grounding (good for factuality)
- Less competition (fewer papers extend GLORIA vs. CheXzero)

**Cons:**
- Higher compute (4× GPUs preferred)
- More complex debugging

**Decision Point:** End of Week 2 (after baseline reproduction attempt)

---

## Summary Table

| Criterion | CheXzero | GLORIA | R2Gen |
|-----------|----------|--------|-------|
| **Code Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Public Data Fit** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Compute** | ⭐⭐⭐⭐⭐ Low | ⭐⭐⭐ High | ⭐⭐⭐⭐ Low |
| **Extensibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Publication Gaps** | ⭐⭐⭐⭐⭐ Many | ⭐⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Many |
| **Novelty Potential** | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ High |

**Final Verdict:** **CheXzero (primary) + R2Gen (complementary) = Optimal for publication-oriented graduate project**

---

**Last Updated:** January 2026  
**Decision:** ✅ Proceed with CheXzero + R2Gen hybrid approach
