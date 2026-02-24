# Vision-Language Models in Radiology: Zero-Shot Classification, Cross-Modal Retrieval, and Factual Report Generation

**Graduate Research Project | Biomedical Image Analysis and Processing | Sharif University of Technology (Jan 2026)**

![Project Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview

This repository houses a comprehensive framework for **Radiology Vision-Language Modeling (VLM)**. We address critical limitations in current AI-driven radiology report generation: **hallucinations (factuality errors)**, **lack of uncertainty estimation**, and **poor cross-dataset generalization**.

Our approach integrates two state-of-the-art architectures:
1.  **CheXzero** (Nature BME 2022): Contrastive learning for zero-shot classification.
2.  **R2Gen** (EMNLP 2020): Memory-driven transformer for report generation.

We extend these baselines with three novel contributions:
*   **Uncertainty Quantification:** Monte Carlo (MC) Dropout estimates model confidence.
*   **Factuality Constraints:** Reinforcement Learning (SCST) with **RadGraph** rewards.
*   **Cross-Dataset Robustness:** Zero-shot transfer and few-shot adaptation from IU X-Ray to NIH ChestX-ray14.

---

## 🏗️ Architecture

The system consists of a visual encoder (ResNet-101, pre-trained on ImageNet) and a Transformer-based decoder.

1.  **Visual Encoder:** Extracts grid features from chest X-rays. (Frozen to preserve robust features).
2.  **Relational Memory:** Records prototypes of schema-level patterns (R2Gen).
3.  **Decoder:** Generates reports token-by-token.
4.  **Factuality Module:** A RadGraph-based reward system penalizes clinical inaccuracies during fine-tuning.
5.  **Uncertainty Module:** Computes variance over multiple stochastic forward passes (MC Dropout) during inference.

---

## 📂 Repository Structure

```bash
radiology-vision-language-models/
├── src/
│   ├── models/          # CheXzero, R2Gen, Uncertainty wrappers
│   ├── training/        # Trainers for Contrastive, CE, SCST, and Factuality losses
│   ├── evaluation/      # Metrics (BLEU, RadGraph, AUROC, ECE)
│   └── data_loaders/    # PyTorch datasets for IU X-Ray & NIH-14
├── experiments/
│   ├── configs/         # YAML configurations
│   ├── scripts/         # CLI training/eval scripts
│   └── *.ipynb          # Colab notebooks for training/demo
├── datasets/            # Documentation for IU X-Ray & NIH-14
├── literature/          # Comprehensive paper reviews
└── paper/               # LaTeX source for the final paper
```

---

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SinaDns/radiology-vision-language-models.git
    cd radiology-vision-language-models
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Key packages: `torch`, `transformers`, `pycocoevalcap`, `radgraph`, `timm`.*

3.  **Download Data:**
    *   **IU X-Ray:** Run `bash experiments/scripts/download_iu_xray.sh`
    *   **NIH ChestX-ray14:** Follow instructions in `datasets/nih-chestxray14.md`.

---

## 🏃 Usage

### 1. Train Baselines
Train the separate components before fine-tuning.
```bash
# Train Contrastive Encoder (CheXzero)
python experiments/scripts/train_chexzero.py --config experiments/configs/chexzero.yaml

# Train Report Generator (R2Gen)
python experiments/scripts/train_r2gen.py --config experiments/configs/r2gen.yaml
```

### 2. Fine-Tune with Factuality Reward (Extension 2)
Apply Reinforcement Learning (SCST) to optimize for clinical correctness (RadGraph F1).
```bash
python experiments/scripts/scst_finetune.py --config experiments/configs/scst.yaml
```

### 3. Evaluate Uncertainty (Extension 1)
Run inference with MC Dropout and compute Expected Calibration Error (ECE).
```bash
python experiments/scripts/evaluate_uncertainty.py --config experiments/configs/uncertainty.yaml
```

### 4. Cross-Dataset Evaluation (Extension 3)
Test zero-shot transfer performance on NIH ChestX-ray14.
```bash
python experiments/scripts/evaluate_cross_dataset.py --config experiments/configs/cross_dataset.yaml
```

---

## 📊 Results

### Report Generation (IU X-Ray Test Set)
Our method improves clinical factuality metrics significantly compared to the R2Gen baseline.

| Model | BLEU-4 | METEOR | CIDEr | **RadGraph F1** | **CheXbert F1** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **R2Gen (Baseline)** | 0.102 | 0.134 | 0.189 | 0.612 | 0.441 |
| **Ours (Fact-Aware)** | **0.115** | **0.141** | **0.215** | **0.685** | **0.493** |

### Zero-Shot Classification (NIH ChestX-ray14)
Domain adaptation via few-shot usage of the frozen encoder.

| Method | Mean AUROC |
|:---|:---:|
| CheXzero (Zero-Shot) | 0.724 |
| **CheXzero (k=5 Few-Shot)** | **0.781** |
| *Supervised Upper Bound* | *0.825* |

---

## 👥 Contributors

*   **Ali Salesi** - [@AlisaLC](https://github.com/AlisaLC)
*   **Sina Daneshgar** - [@SinaDns](https://github.com/SinaDns)
*   **Sina Beyrami** - [@SinaBeyrami](https://github.com/SinaBeyrami)
*   **Shayan Salehi** - [@ShayanSalehi81](https://github.com/ShayanSalehi81)

---

## 📄 License & Acknowledgments

This project is part of the **Biomedical Image Analysis and Processing** course at **Sharif University of Technology**.
Special thanks to the authors of CheXzero, R2Gen, and RadGraph for their open-source contributions.
