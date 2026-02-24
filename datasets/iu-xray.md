# IU X-Ray (Indiana University Chest X-Ray Collection)

## Dataset Overview

**Full Name:** Indiana University Chest X-ray Collection  
**Alternative Names:** IU X-Ray, OpenI Indiana, Indiana CXR  
**Institution:** Indiana University School of Medicine  
**Year Released:** 2015  
**Access:** **Fully Public** (No registration required) ⭐⭐⭐  

---

## Key Statistics

| Attribute | Value |
|-----------|-------|
| **Total Images** | 7,470 |
| **Total Studies** | 3,955 |
| **Patients** | 3,955 (one study per patient) |
| **Modality** | Chest X-ray (frontal and lateral views) |
| **Image Format** | PNG (grayscale) |
| **Resolution** | Variable (typically 2048×2048 to 3000×2500) |
| **Reports** | 3,955 free-text radiology reports |
| **Manual Annotations** | 3,955 studies (14 disease labels + severity) |
| **MeSH Tags** | Available for all studies |

---

## Data Composition

### Images per Study
- **Frontal view only:** ~40% of studies
- **Frontal + Lateral:** ~60% of studies
- Total: 1.89 images per study (average)

### Patient Demographics
- **Age:** Not provided
- **Gender:** Not provided
- **Disease Prevalence:** See table below

---

## Disease Labels (Manual Annotations)

**14 Observations with Binary Labels:**

| Disease/Finding | Positive Cases | Prevalence | Severity Levels |
|-----------------|----------------|------------|-----------------|
| Normal | 1,110 | 28.1% | N/A |
| Cardiomegaly | 376 | 9.5% | Mild, Moderate, Severe |
| Lung Opacity | 658 | 16.6% | Mild, Moderate, Severe |
| Lung Lesion | 54 | 1.4% | Small, Large |
| Edema | 89 | 2.2% | Mild, Moderate, Severe |
| Consolidation | 173 | 4.4% | Mild, Moderate, Severe |
| Pneumonia | 157 | 4.0% | Mild, Moderate, Severe |
| Atelectasis | 200 | 5.1% | Mild, Moderate, Severe |
| Pneumothorax | 42 | 1.1% | Small, Large |
| Pleural Effusion | 303 | 7.7% | Small, Moderate, Large |
| Pleural Thickening | 54 | 1.4% | Mild, Moderate, Severe |
| Fracture | 48 | 1.2% | N/A |
| Support Devices | 175 | 4.4% | N/A |
| Calcinosis | 65 | 1.6% | N/A |

**Note:** Studies can have multiple labels (multi-label classification).

---

## Radiology Reports

### Report Structure

Each report contains **4 sections:**

1. **Comparison:** Reference to prior studies (if any)
2. **Indication:** Clinical reason for exam
3. **Findings:** Detailed observations
4. **Impression:** Summary and diagnosis

**Example Report:**
```
COMPARISON: None

INDICATION: ___-year-old with cough

FINDINGS: The cardiac silhouette and mediastinum size are within normal limits. 
There is no pulmonary edema. There is no focal consolidation. There are no XXXX 
of a pleural effusion. There is no evidence of pneumothorax.

IMPRESSION: No acute disease.
```

### Report Statistics

| Metric | Value |
|--------|-------|
| **Avg Words per Report** | 35-40 words |
| **Avg Sentences** | 4-6 sentences |
| **Min Length** | 10 words |
| **Max Length** | 150 words |
| **Most Common Findings** | "no acute", "heart size normal", "lungs clear" |

**Characteristics:**
- Relatively **short** compared to MIMIC-CXR (which averages 53 words)
- Structured format (same 4 sections for all)
- Many negative findings ("no evidence of...")
- Some PHI redacted with "XXXX" or "___"

---

## MeSH (Medical Subject Headings)

Each study tagged with **controlled vocabulary** from MeSH:

**Example Tags:**
- Pulmonary Disease, Chronic Obstructive
- Pneumonia, Bacterial
- Cardiomegaly
- Pleural Effusion
- Lung Neoplasms

**Use Case:** Can be used as additional supervision signal or for retrieval.

---

## Data Splits (Standard in Literature)

### Official Split (Most Commonly Used)

| Split | Studies | Images | Percentage |
|-------|---------|--------|------------|
| **Train** | 2,770 | ~5,200 | 70% |
| **Validation** | 585 | ~1,100 | 15% |
| **Test** | 600 | ~1,170 | 15% |

**Split Type:** Random at study level (not patient level, but 1 study per patient anyway)

**Papers Using This Split:**
- R2Gen (EMNLP 2020)
- Show-Attend-Tell for Medical (various)
- CMN (CVPR 2021)
- Most report generation papers

---

### Alternative Splits

Some papers use different splits:
- 80/10/10 split
- 5-fold cross-validation (rare)

**Recommendation:** Use **70/15/15 split** for comparability with prior work.

### This Project's Split

We use a **90/10 train/val split** (no held-out test set) because:
- IU X-Ray has no standard test-set lock-out; test performance is reported on NIH-14 (zero-shot)
- 90/10 maximises training data for the small dataset
- Implemented in `src/data_loaders/iu_xray.py` and `src/data_loaders/iu_xray_seq2seq.py`

---

## Download Instructions

### Method 1: Direct Download (Recommended)

**OpenI Website:**
```
https://openi.nlm.nih.gov/faq#collection
```

**Steps:**
1. Visit OpenI website
2. Navigate to "Collections" → "Indiana University Chest X-Ray Collection"
3. Download image set + reports XML file
4. No login required

**File Size:** ~7 GB (images) + 5 MB (reports XML)

### Method 2: Kaggle (Preprocessed)

Some users have uploaded preprocessed versions to Kaggle:
```
https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
```

**Includes:**
- Resized images (224×224 or 512×512)
- CSV file with reports (easier than XML)
- Train/val/test split annotations

---

## Preprocessing Requirements

### Images

1. **Load PNG files** (grayscale, 8-bit or 16-bit)
2. **Resize to target resolution:**
   - 224×224 (standard for ResNet-50, ViT)
   - 320×320 (CheXzero)
   - 512×512 (higher resolution for localization)
3. **Normalize pixel values:**
   - Option A: [0, 1] range
   - Option B: ImageNet mean/std (if using pretrained encoder)
4. **Handle multi-view:**
   - Single view: use frontal only
   - Multi-view: concatenate or encode separately

### Reports

1. **Parse XML format:**
   - Extract findings and impression sections
   - Optionally include indication
2. **Tokenization:**
   - Use medical-specific tokenizer (e.g., Bio+Clinical BERT tokenizer)
   - Max length: 128 tokens (most reports fit)
3. **Sentence Segmentation:**
   - For GLORIA-style models: split into sentences
   - For generation: keep as full text

**Implementation note:** Our loader (`src/data_loaders/iu_xray.py`) extracts image filenames from the `<parentImage URI="...">` element in each XML rather than using glob matching. This avoids pairing an image with the wrong study when multiple studies share a similar naming prefix. FINDINGS and IMPRESSION sections are concatenated as the report text.

---

## Strengths

1. **Fully Public:** No registration, credentialing, or usage restrictions ⭐
2. **Well-Structured Reports:** Consistent 4-section format makes parsing easy
3. **Manual Annotations:** Gold-standard disease labels for 14 findings
4. **Multi-View:** Frontal + lateral provides richer information
5. **Standard Benchmark:** Widely used in report generation literature (easy comparison)
6. **Moderate Size:** Manageable for single-GPU training (~12 hours for R2Gen)
7. **Clean Data:** High-quality clinical reports from academic medical center
8. **MeSH Tags:** Additional structured metadata

---

## Limitations

1. **Small Scale:** Only 3,955 studies (vs. MIMIC-CXR's 227K)
   - Limits generalization and model capacity
   - Zero-shot learning challenging
2. **Single Institution:** All from Indiana University → potential bias
   - Equipment: Same scanners
   - Population: Midwestern US demographics
   - Reporting style: Single institution's conventions
3. **No Demographics:** Age, gender, race not provided → can't study fairness
4. **Short Reports:** Avg 35 words → less rich than MIMIC-CXR (53 words)
5. **PHI Redaction:** Some information masked (dates, ages)
6. **Imbalanced Labels:** Rare diseases have <50 examples (pneumothorax: 42)
7. **No Temporal Data:** Can't study longitudinal comparisons
8. **Limited Modalities:** Chest X-ray only (no CT, MRI)

---

## Use Cases in Literature

### 1. Report Generation (Primary Use)

**Standard Papers:**
- R2Gen (EMNLP 2020): Achieved BLEU-4 = 0.203
- CMN (CVPR 2021): Achieved CIDEr = 0.512
- AdaAtt, CoAtt, TieNet, etc.

**Typical Setup:**
- Input: Frontal + lateral images (or frontal only)
- Output: Findings + impression sections
- Metrics: BLEU, ROUGE, CIDEr (legacy), RadGraph F1 (modern)

### 2. Multi-Label Classification

**Task:** Predict 14 disease labels from images

**Typical Performance:**
- DenseNet-121: ~0.75 mean AUROC
- Vision Transformer: ~0.78 mean AUROC

### 3. Image-Text Retrieval

**Less Common:** Too small for large-scale retrieval benchmarks

**Possible:** Use as test set after pretraining on larger dataset

### 4. Few-Shot Learning

**Use Case:** Pretrain on other dataset, few-shot adapt to IU X-Ray

---

## Comparison with Other Datasets

| Dataset | Size (Studies) | Access | Report Length | Multi-View | Institution | Best For |
|---------|----------------|--------|---------------|------------|-------------|----------|
| **IU X-Ray** | 3,955 | ⭐⭐⭐ Public | 35 words | ✅ | Single | Report generation baseline |
| MIMIC-CXR | 227,835 | 🔒 Credentialed | 53 words | ✅ | Single | Pretraining, large-scale |
| NIH-14 | 112,120 | ⭐⭐⭐ Public | ❌ No reports | ❌ | NIH | Classification only |
| CheXpert | 224,316 | ⚠️ Registration | ❌ No reports | ❌ | Single | Multi-label classification |
| PadChest | 160,000 | ⭐⭐ Public | Spanish | ✅ | Single (Spain) | Large-scale, European |

**Sweet Spot:** IU X-Ray is the **only fully public dataset with high-quality paired images and reports**.

---

## Usage in This Project

### What We Actually Use IU X-Ray For

| Task | Split | Loader |
|------|-------|--------|
| CheXzero contrastive training | 90% train | `IUXrayDataset` |
| CheXzero val loss monitoring | 10% val | `IUXrayDataset` |
| IU X-Ray image→text retrieval (val eval) | 10% val | `IUXrayDataset` |
| R2Gen training | 90% train | `IUXraySeq2SeqDataset` |
| R2Gen / SCST / Factuality val loss | 10% val | `IUXraySeq2SeqDataset` |
| Report generation evaluation (BLEU/ROUGE/METEOR/RadGraph F1) | 10% val | `IUXraySeq2SeqDataset` |
| Calibration analysis (MC Dropout reference) | 10% val | `IUXraySeq2SeqDataset` |

NIH-14 is used exclusively for zero-shot classification evaluation and cross-dataset robustness (no IU X-Ray test split needed for that).

---

## Citation & Attribution

**Primary Paper:**
```
Demner-Fushman et al., "Preparing a collection of radiology examinations for 
distribution and retrieval," Journal of the American Medical Informatics 
Association, 2016.
```

**BibTeX:**
```bibtex
@article{demner2016preparing,
  title={Preparing a collection of radiology examinations for distribution and retrieval},
  author={Demner-Fushman, Dina and Kohli, Marc D and Rosenman, Marc B and Shooshan, Sonya E and Rodriguez, Laritza and Antani, Sameer and Thoma, George R and McDonald, Clement J},
  journal={Journal of the American Medical Informatics Association},
  volume={23},
  number={2},
  pages={304--310},
  year={2016},
  publisher={Oxford University Press},
  doi={10.1093/jamia/ocv080}
}
```

**Always Cite When Using This Dataset**

---

## Related Resources

- **OpenI Portal:** https://openi.nlm.nih.gov/
- **MICCAI CXR-Rep-Learn Workshop:** Uses IU X-Ray as benchmark
- **Papers with Code:** https://paperswithcode.com/dataset/iu-x-ray

---

**Last Updated:** February 2026
**Status:** ✅ Integrated — data loaders implemented in `src/data_loaders/iu_xray.py` and `src/data_loaders/iu_xray_seq2seq.py`
**Role in project:** Primary training dataset for both CheXzero and R2Gen
