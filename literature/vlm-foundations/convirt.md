# ConVIRT: Contrastive Learning of Medical Visual Representations from Paired Images and Text

## Citation
**Title:** Contrastive Learning of Medical Visual Representations from Paired Images and Text  
**Authors:** Zhang, Y., Jiang, H., Miura, Y., Manning, C. D., Langlotz, C. P.  
**Venue:** Machine Learning for Healthcare (ML4H), 2020  
**arXiv:** [2010.00747](https://arxiv.org/abs/2010.00747)  
**Code:** [https://github.com/edreisMD/ConVIRT-pytorch](https://github.com/edreisMD/ConVIRT-pytorch) ⭐ (Unofficial but widely used)  

---

## Core Contribution

ConVIRT is one of the **pioneering works** applying contrastive learning to medical imaging using image-text pairs. It demonstrates that self-supervised pretraining on radiology images and reports significantly improves downstream task performance and data efficiency compared to ImageNet pretraining.

**Historical Importance:** ConVIRT predates CLIP (published around the same time) and established that contrastive vision-language learning works effectively in medical domains with limited annotated data.

---

## Methodology

### Architecture

**Image Encoder:**
- ResNet-50 (randomly initialized or ImageNet pretrained)
- Input: 224×224 chest X-rays
- Output: 2048-dim feature vector $\mathbf{v}$
- Global average pooling applied

**Text Encoder:**
- BERT-base (randomly initialized or Bio+Clinical BERT)
- Input: Radiology report text (findings + impressions)
- Output: 768-dim feature from [CLS] token $\mathbf{u}$

**Projection Heads:**
- MLP (2-layer) projects both to shared 512-dim embedding space
- Applied to both encoders

---

### Contrastive Learning Objective

**Bidirectional Loss:**

$$
\mathcal{L}_{\text{ConVIRT}} = \mathcal{L}_{v \rightarrow u} + \mathcal{L}_{u \rightarrow v}
$$

**Image-to-Text Loss:**

$$
\mathcal{L}_{v \rightarrow u} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{u}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{v}_i, \mathbf{u}_j) / \tau)}
$$

**Text-to-Image Loss:**

$$
\mathcal{L}_{u \rightarrow v} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{u}_i, \mathbf{v}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{u}_i, \mathbf{v}_j) / \tau)}
$$

- $\text{sim}(\mathbf{v}, \mathbf{u}) = \mathbf{v}^T \mathbf{u} / (\|\mathbf{v}\| \|\mathbf{u}\|)$ (cosine similarity)
- $\tau = 0.1$ (temperature parameter)
- $N$: batch size (typically 64-128)

---

### Training Details

- **Optimizer:** Adam (lr=1e-4)
- **Batch Size:** 64-128
- **Epochs:** 50-100
- **Weight Decay:** 1e-6
- **Warmup:** Linear warmup for first 10% of training
- **Hardware:** 4× NVIDIA V100 GPUs
- **Training Time:** ~48 hours on MIMIC-CXR

---

## Datasets Used

### Pretraining
- **MIMIC-CXR** (377,110 images, 227,835 studies)
  - Used findings and impressions sections
  - No disease labels needed

### Evaluation

**1. Transfer Learning (Linear Probing):**
- CheXpert (224,316 images, 14 labels)
- Shenzhen TB (662 images, tuberculosis detection)
- RSNA Pneumonia (26,684 images)

**2. Few-Shot Learning:**
- CheXpert: 1%, 10%, 50%, 100% of labeled data

**3. Image-Text Retrieval:**
- MIMIC-CXR test set (5,000 studies)

---

## Results

### Transfer Learning (AUROC on CheXpert, Full Training Set)

| Method | Atelectasis | Cardiomegaly | Edema | Pleural Effusion | Mean (14 diseases) |
|--------|-------------|--------------|-------|------------------|---------------------|
| ImageNet Pretrained | 0.803 | 0.828 | 0.895 | 0.921 | 0.825 |
| ConVIRT (from scratch) | 0.811 | 0.841 | 0.902 | 0.929 | 0.841 |
| **ConVIRT (best)** | **0.823** | **0.854** | **0.913** | **0.937** | **0.857** |

**Key Finding:** ConVIRT pretraining outperforms ImageNet initialization by **+3.2% average AUROC**

---

### Few-Shot Learning (AUROC, CheXpert)

| Training Data | ImageNet Init | ConVIRT Init | Improvement |
|---------------|---------------|--------------|-------------|
| 1% (2,240 images) | 0.712 | **0.768** | +5.6% |
| 10% (22,400 images) | 0.781 | **0.823** | +4.2% |
| 50% (112,000 images) | 0.819 | **0.847** | +2.8% |
| 100% (224,316 images) | 0.825 | **0.857** | +3.2% |

**Key Finding:** Largest gains in low-data regimes (1-10% labeled data) — critical for rare diseases

---

### Cross-Dataset Generalization

| Target Dataset | Task | ImageNet Init | ConVIRT Init |
|----------------|------|---------------|--------------|
| Shenzhen TB | TB detection | 0.832 | **0.881** (+4.9%) |
| RSNA Pneumonia | Pneumonia detection | 0.778 | **0.819** (+4.1%) |

---

### Image-Text Retrieval (MIMIC-CXR Test Set)

| Direction | Metric | ConVIRT |
|-----------|--------|---------|
| Image → Text | Recall@1 | 25.1% |
| Image → Text | Recall@5 | 50.2% |
| Text → Image | Recall@1 | 24.7% |
| Text → Image | Recall@5 | 49.8% |

**Note:** These are baseline results; later methods (GLORIA, BioViL) improve significantly.

---

## Strengths

1. **Pioneering Work:** First large-scale application of contrastive vision-language learning to medical imaging
2. **Data Efficiency:** Significant gains in few-shot scenarios (5-6% AUROC improvement with 1% data)
3. **Better Than ImageNet:** Demonstrates domain-specific pretraining superior to natural image pretraining
4. **Simple Architecture:** Easy to implement and reproduce
5. **No Manual Annotations:** Self-supervised on unlabeled image-text pairs
6. **Strong Transfer:** Generalizes well to diverse downstream tasks (classification, detection, retrieval)
7. **Publicly Reproducible:** Can train on public IU X-Ray or NIH-14 datasets

---

## Limitations

1. **Global Matching Only:** No fine-grained region-sentence alignment (addressed by GLORIA)
2. **Slower Than Supervised:** Pretraining + fine-tuning takes longer than end-to-end supervised training
3. **Hyperparameter Sensitivity:** Temperature and learning rate require careful tuning
4. **Text Encoder Initialization:** Performance depends on text encoder quality (Bio+Clinical BERT vs. standard BERT)
5. **Report Quality Dependency:** Noisy or incomplete reports reduce learned representation quality
6. **No Interpretability:** Global embeddings don't provide localization or attention
7. **Outdated Baselines:** 2020 paper; newer methods (CheXzero, GLORIA, BioViL) outperform
8. **Limited Modality Coverage:** Only chest X-ray evaluated; CT/MRI not tested

---

## Comparison with Later Methods

| Method | Year | Global/Local | AUROC (CheXpert) | Retrieval R@1 | Grounding |
|--------|------|--------------|------------------|---------------|-----------|
| ConVIRT | 2020 | Global | 0.857 | 24.7% | ❌ |
| GLORIA | 2021 | Global + Local | 0.878 | 31.5% | ✅ |
| CheXzero | 2022 | Global | 0.873 | Not reported | ❌ |
| BioViL | 2022 | Global + Local | 0.881 | 33.8% | ✅ |

**Takeaway:** ConVIRT established the foundation; later works add local alignment, better architectures, and grounding.

---

## Relevance to Our Project

### Why ConVIRT Matters:

1. **Historical Context:** Understanding ConVIRT helps position our work in the evolution of medical VLMs
2. **Simple Baseline:** Easy to implement as initial baseline before adding complexity
3. **Public Data Compatible:** Designed for scenarios with limited annotated data (perfect for IU X-Ray)
4. **Extension Path:** Natural starting point to add local attention (→ GLORIA) or zero-shot classification (→ CheXzero)

### Should We Use ConVIRT as Base?

**No, prefer CheXzero or GLORIA:**
- ConVIRT is foundational but older (2020)
- CheXzero has better code, more recent architecture, zero-shot capabilities
- GLORIA adds grounding and local alignment
- ConVIRT best as **baseline comparison** rather than main implementation

### Role in Our Project:

1. **Literature Review:** Important to cite as pioneering work
2. **Baseline Comparison:** Implement as simple baseline to show improvement
3. **Ablation Study:** Compare global-only (ConVIRT) vs. global+local (GLORIA)

---

## Implementation Considerations

### Compute Requirements
- **Training:** 4× V100 (48 hours) or 1× V100 (1 week)
- **Inference:** 1× GPU (real-time capable)
- **Memory:** 16-24 GB GPU RAM

### Preprocessing
1. Resize to 224×224
2. Normalize with ImageNet mean/std (if using pretrained encoder)
3. Tokenize reports (max 512 tokens)
4. Random augmentation: horizontal flip, rotation (±10°)

### Key Hyperparameters
- Temperature: 0.1 (higher than CLIP's 0.07)
- Learning rate: 1e-4 (higher than CheXzero's 5e-5)
- Batch size: 64-128 (smaller than later methods)
- Embedding dim: 512

---

## Reproducibility on Public Data

### Suggested Protocol:

**Pretraining:**
- Use IU X-Ray (3,955 studies) for initial experiments
- Combine IU + NIH-14 (no reports, use disease labels as text) for larger scale

**Fine-Tuning:**
- CheXpert (if registration is simple)
- NIH ChestX-ray14 (fully public)
- PadChest (public, large-scale)

**Expected Results on IU X-Ray:**
- Smaller dataset → lower absolute performance
- But relative improvement over ImageNet init should hold (+3-5% AUROC)

---

## Related Papers & Code

- **CLIP:** Radford et al., ICML 2021
- **SimCLR:** Chen et al., ICML 2020 (contrastive learning for vision)
- **Bio+Clinical BERT:** Alsentzer et al., EMNLP 2019
- **Unofficial PyTorch Implementation:** [github.com/edreisMD/ConVIRT-pytorch](https://github.com/edreisMD/ConVIRT-pytorch)

---

## BibTeX Entry

```bibtex
@article{zhang2020convirt,
  title={Contrastive learning of medical visual representations from paired images and text},
  author={Zhang, Yuhao and Jiang, Hang and Miura, Yasuhide and Manning, Christopher D and Langlotz, Curtis P},
  journal={Machine Learning for Healthcare (ML4H)},
  year={2020},
  archivePrefix={arXiv},
  eprint={2010.00747}
}
```

---

**Last Updated:** January 2026  
**Review Status:** ✅ Comprehensive - Important for historical context and baseline comparison
