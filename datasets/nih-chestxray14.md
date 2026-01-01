# NIH ChestX-ray14 Dataset

## Dataset Overview

**Full Name:** NIH Clinical Center Chest X-ray Dataset (ChestX-ray14)  
**Institution:** National Institutes of Health (NIH) Clinical Center  
**Year Released:** 2017  
**Access:** **Fully Public** (No registration required) ⭐⭐⭐  
**Predecessor:** ChestX-ray8 (2017, 8 diseases)

---

## Key Statistics

| Attribute | Value |
|-----------|-------|
| **Total Images** | 112,120 |
| **Total Studies** | ~30,805 patients |
| **Patients** | 30,805 unique |
| **Modality** | Chest X-ray (frontal view only: PA/AP) |
| **Image Format** | PNG (grayscale, 8-bit) |
| **Resolution** | 1024×1024 pixels (standardized) |
| **Reports** | ❌ No free-text reports available |
| **Disease Labels** | 14 labels (text-mined from reports) |
| **Bounding Boxes** | ✅ 880 images (8 diseases, subset) |
| **Temporal Data** | ❌ No longitudinal studies |

---

## Disease Labels

**14 Pathologies (Multi-Label Classification):**

| Disease | Positive Cases | Prevalence | Notes |
|---------|----------------|------------|-------|
| Atelectasis | 11,559 | 10.3% | Lung collapse |
| Cardiomegaly | 2,776 | 2.5% | Enlarged heart |
| Effusion | 13,317 | 11.9% | Pleural fluid |
| Infiltration | 19,894 | 17.7% | Lung infiltrates (most common) |
| Mass | 5,782 | 5.2% | Lung mass/tumor |
| Nodule | 6,331 | 5.6% | Small lung nodule |
| Pneumonia | 1,431 | 1.3% | Infection |
| Pneumothorax | 5,302 | 4.7% | Collapsed lung |
| Consolidation | 4,667 | 4.2% | Lung consolidation |
| Edema | 2,303 | 2.1% | Pulmonary edema |
| Emphysema | 2,516 | 2.2% | Chronic lung disease |
| Fibrosis | 1,686 | 1.5% | Lung scarring |
| Pleural Thickening | 3,385 | 3.0% | Thickened pleura |
| Hernia | 227 | 0.2% | Diaphragmatic hernia (very rare) |
| **No Finding** | 60,361 | 53.8% | Normal chest X-ray |

**Multi-Label:** Images can have 0-5 labels (avg: 1.2 labels per image)

---

## Label Extraction Method

**Important:** Labels were **automatically extracted** from radiology reports using Natural Language Processing (NLP), not manually annotated.

**Pipeline:**
1. Parse radiology reports (not included in public release)
2. Apply rule-based NLP to extract disease mentions
3. Assign labels based on keyword matching and negation detection

**Implications:**
- **Label Noise:** Estimated 10-20% error rate
- **False Positives:** "rule out pneumonia" might be labeled as pneumonia
- **False Negatives:** Mentions in impression but not findings might be missed
- **Uncertainty:** "possible mass" treated same as "definite mass"

**Validation Studies:**
- Expert radiologist annotations on subset: Cohen's Kappa = 0.65-0.75 (moderate agreement)
- Highest quality: Atelectasis, Cardiomegaly, Effusion
- Lowest quality: Infiltration, Pneumonia (high noise)

---

## Bounding Box Annotations (Subset)

**Size:** 880 images with radiologist-drawn bounding boxes

**Covered Diseases (8 out of 14):**
1. Atelectasis (117 boxes)
2. Cardiomegaly (90 boxes)
3. Effusion (171 boxes)
4. Infiltration (185 boxes)
5. Mass (126 boxes)
6. Nodule (107 boxes)
7. Pneumonia (20 boxes)
8. Pneumothorax (64 boxes)

**File Format:** `BBox_List_2017.csv`
- Image filename
- Disease label
- Bounding box coordinates (x, y, width, height)

**Use Case:**
- Weakly supervised localization
- Training object detection models
- Evaluating attention maps

---

## Data Splits

### Official Split (Most Common)

| Split | Images | Percentage |
|-------|--------|------------|
| **Train** | 70,000 | ~62% |
| **Validation** | 17,120 | ~15% |
| **Test** | 25,000 | ~23% |

**Split Type:** Random at image level (not patient level)

**Note:** Some papers use different splits (80/10/10) or patient-level splits.

---

## Download Instructions

### Method 1: Direct Download from NIH (Recommended)

**Website:**
```
https://nihcc.app.box.com/v/ChestXray-NIHCC
```

**Files:**
- `images_001.tar.gz` to `images_012.tar.gz` (12 parts, ~5 GB each)
- `Data_Entry_2017.csv` (image labels)
- `BBox_List_2017.csv` (bounding boxes)
- `README_ChestXray.pdf` (documentation)

**Total Size:** ~60 GB (compressed), ~45 GB (uncompressed PNG)

**Download Time:** 2-6 hours depending on connection

### Method 2: AWS S3 (Faster)

Some users mirror on AWS:
```bash
aws s3 sync s3://nih-chest-xrays/images/ ./nih_data/
```

### Method 3: Kaggle

**Kaggle Dataset:**
```
https://www.kaggle.com/datasets/nih-chest-xrays/data
```

**Advantages:**
- Easier download via Kaggle CLI
- Preprocessed versions available
- Community notebooks for exploration

---

## Preprocessing Requirements

### Images

**Standard Pipeline:**
1. **Load PNG files** (1024×1024, grayscale)
2. **Resize:**
   - 224×224 (ResNet, standard)
   - 320×320 (CheXzero)
   - 512×512 (higher resolution)
3. **Normalize:**
   - ImageNet mean/std (if using pretrained models)
   - Or custom computed mean/std from training set
4. **Data Augmentation (training only):**
   - Random horizontal flip (anatomically valid)
   - Random rotation (±10°)
   - Random crop with padding
   - Color jitter (limited for grayscale)

### Labels

**Multi-Label Encoding:**
```python
# Example label encoding
diseases = ['Atelectasis', 'Cardiomegaly', ..., 'Hernia', 'No Finding']
label_vector = [1, 0, 1, 0, ..., 0]  # 14-dim binary vector
```

**Handling "No Finding":**
- Option A: Exclusive (if "No Finding", all others are 0)
- Option B: Non-exclusive (some papers allow overlap)

**Most Common:** Treat "No Finding" as inverse of any disease presence.

---

## Strengths

1. **Large Scale:** 112K images — good for deep learning
2. **Fully Public:** No registration or credentialing ⭐⭐⭐
3. **Standard Benchmark:** Widely used (500+ papers cite it)
4. **Multiple Diseases:** 14 pathologies cover common findings
5. **Localization Subset:** 880 bounding boxes enable weakly supervised learning
6. **Diverse Population:** NIH Clinical Center (diverse patients)
7. **Preprocessed:** Standardized 1024×1024 resolution
8. **Fast Download:** Well-hosted on NIH Box

---

## Limitations

1. **No Reports:** Only labels, no free-text reports
   - **Cannot use for report generation**
   - **Cannot use for image-text retrieval (VLM pretraining)**
2. **Label Noise:** 10-20% estimated error rate due to NLP extraction
   - Some papers manually re-annotate subset for validation
3. **Class Imbalance:** "Hernia" only 227 cases (0.2%)
4. **Frontal View Only:** No lateral views
5. **Single Institution:** All from NIH Clinical Center
6. **No Demographics (Public):** Age, gender redacted for privacy
7. **Temporal Gaps:** Not designed for longitudinal studies
8. **Bounding Box Subset Small:** Only 880 images (<1%) have boxes

---

## Use Cases

### 1. Multi-Label Classification (Primary Use)

**Task:** Predict 14 diseases from chest X-ray

**Typical Baselines:**
- DenseNet-121: 0.77-0.80 mean AUROC
- ResNet-50: 0.74-0.76 mean AUROC
- Vision Transformer: 0.78-0.82 mean AUROC

**State-of-the-Art (Supervised):**
- CheXNet (DenseNet-121): 0.841 AUROC (Rajpurkar et al., 2017)
- CheXNeXt: 0.850+ AUROC

**State-of-the-Art (Zero-Shot):**
- CheXzero: 0.810 AUROC (5 diseases subset)
- GLORIA: 0.825 AUROC

### 2. Zero-Shot Classification Evaluation

**Use Case:** Test VLMs pretrained on MIMIC-CXR or IU X-Ray

**Protocol:**
- Pretrain on dataset with reports (e.g., IU X-Ray)
- Test zero-shot on NIH-14 using clinical prompts
- Report AUROC on 14 diseases

**Advantage:** Large test set (25K images) provides robust evaluation

### 3. Weakly Supervised Localization

**Use Case:** Train class activation maps (CAM), attention, or object detection

**Method:**
- Train classification model on 112K images (image-level labels)
- Evaluate localization on 880 bounding box subset
- Metrics: IoU, Pointing Accuracy

**Papers:**
- Rajpurkar et al., "CheXNet" (2017)
- Wang et al., "TieNet" (2018)

### 4. Transfer Learning Source

**Use Case:** Pretrain encoder, then transfer to other medical tasks

**Target Tasks:**
- Pneumonia detection (RSNA)
- Tuberculosis detection (Shenzhen, Montgomery)
- Pediatric CXR (external datasets)

### 5. Cross-Dataset Validation

**Use Case:** Evaluate model generalization

**Example:**
- Train on IU X-Ray or MIMIC-CXR
- Test on NIH-14 (distribution shift)
- Measure AUROC drop

---

## Comparison with Other Datasets

| Dataset | Images | Reports | Labels | Bbox | Access | Best For |
|---------|--------|---------|--------|------|--------|----------|
| **NIH-14** | 112K | ❌ | ✅ 14 (NLP) | ⚠️ 880 | ⭐⭐⭐ Public | Classification, Zero-shot eval |
| IU X-Ray | 7.5K | ✅ | ✅ 14 (Manual) | ❌ | ⭐⭐⭐ Public | Report generation |
| MIMIC-CXR | 377K | ✅ | ✅ 14 (NLP) | ❌ | 🔒 Credentialed | Pretraining |
| CheXpert | 224K | ❌ | ✅ 14 (NLP) | ❌ | ⚠️ Registration | Multi-label classification |
| VinDr-CXR | 18K | ❌ | ✅ 22 (Manual) | ✅ All | 🔒 Credentialed | High-quality localization |

**NIH-14 Niche:** Large-scale public classification benchmark without reports.

---

## Recommended Usage for Our Project

### Scenario 1: Zero-Shot Classification Evaluation

**Setup:**
1. Pretrain VLM (CheXzero) on IU X-Ray (with reports)
2. Test zero-shot on NIH-14 (without fine-tuning)
3. Report AUROC on 14 diseases

**Advantage:** Demonstrates generalization to large external dataset

### Scenario 2: Cross-Dataset Robustness

**Setup:**
1. Train classifier on IU X-Ray (3,955 studies)
2. Test on NIH-14 (25K test images)
3. Analyze performance drop, error patterns

**Research Question:** How robust are models to distribution shift?

### Scenario 3: Weakly Supervised Localization

**Setup:**
1. Train attention-based VLM
2. Evaluate attention maps on NIH-14 bounding box subset (880 images)
3. Compute IoU, Pointing Accuracy

**Extension:** Compare global-only (CheXzero) vs. local attention (GLORIA)

### Scenario 4: Pseudo-Report Generation for Pretraining

**Creative Approach:**
1. Convert 14 labels to pseudo-reports using templates:
   - "Findings: There is evidence of atelectasis and pleural effusion."
2. Pretrain VLM on NIH-14 + templates
3. Evaluate on IU X-Ray generation

**Caveat:** Synthetic reports lack linguistic diversity

---

## Potential Issues & Solutions

### Issue 1: Label Noise

**Problem:** NLP-extracted labels have errors

**Solutions:**
- Use only high-confidence diseases (Atelectasis, Cardiomegaly, Effusion)
- Manually re-annotate validation/test sets (labor-intensive)
- Use label smoothing or noise-robust losses

### Issue 2: Class Imbalance

**Problem:** "Hernia" only 227 cases

**Solutions:**
- Weighted loss (higher weight for rare classes)
- Focal loss (focus on hard examples)
- Report macro-averaged metrics (not just micro-average)

### Issue 3: No Reports

**Problem:** Can't train report generation or VLMs

**Solutions:**
- Use IU X-Ray for report tasks, NIH-14 for classification eval
- Combine with other datasets (PadChest has Spanish reports)
- Focus on classification and zero-shot evaluation

---

## Citation & Attribution

**Primary Paper:**
```
Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and 
Benchmarks on Weakly-Supervised Classification and Localization of 
Common Thorax Diseases," CVPR 2017.
```

**Updated Paper (ChestX-ray14):**
```
Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with 
Uncertainty Labels and Expert Comparison," AAAI 2019.
```

**BibTeX (ChestX-ray14):**
```bibtex
@inproceedings{wang2017chestxray,
  title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2097--2106},
  year={2017}
}
```

---

## Related Resources

- **Official NIH Page:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **CheXNet Paper (Benchmark):** Rajpurkar et al., arXiv 2017
- **Papers with Code:** https://paperswithcode.com/dataset/chestx-ray14
- **Kaggle Notebooks:** Community explorations and baselines

---

**Last Updated:** January 2026  
**Status:** ✅ Fully accessible and widely used  
**Recommendation:** **Primary dataset for classification and zero-shot evaluation** in our project
