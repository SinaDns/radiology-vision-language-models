from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Positive/negative prompt templates for each NIH-14 pathology.
# Based on the CheXzero paper's approach: soft prompts that describe
# radiological findings rather than bare label names.
PATHOLOGY_PROMPTS: dict[str, dict[str, list[str]]] = {
    "Atelectasis": {
        "positive": [
            "findings consistent with atelectasis",
            "atelectasis is present",
            "there is evidence of atelectasis",
            "subsegmental atelectasis noted",
            "bibasilar atelectasis",
            "linear atelectasis",
            "discoid atelectasis observed",
        ],
        "negative": [
            "no atelectasis",
            "lungs are clear",
            "no evidence of atelectasis",
            "clear lungs bilaterally",
            "no focal atelectasis",
        ],
    },
    "Cardiomegaly": {
        "positive": [
            "cardiomegaly is present",
            "enlarged cardiac silhouette",
            "the heart is enlarged",
            "findings consistent with cardiomegaly",
            "increased cardiac size",
            "cardiomegaly noted",
        ],
        "negative": [
            "no cardiomegaly",
            "normal heart size",
            "cardiac silhouette is normal",
            "heart size within normal limits",
            "no cardiac enlargement",
        ],
    },
    "Consolidation": {
        "positive": [
            "consolidation is present",
            "airspace consolidation noted",
            "findings consistent with consolidation",
            "there is consolidation",
            "lobar consolidation",
            "patchy consolidation",
        ],
        "negative": [
            "no consolidation",
            "no airspace consolidation",
            "lungs are clear without consolidation",
            "no evidence of consolidation",
            "clear lungs",
        ],
    },
    "Edema": {
        "positive": [
            "pulmonary edema is present",
            "findings consistent with pulmonary edema",
            "there is pulmonary edema",
            "interstitial edema noted",
            "vascular congestion and edema",
            "perihilar edema",
        ],
        "negative": [
            "no pulmonary edema",
            "no evidence of edema",
            "no vascular congestion",
            "no interstitial edema",
            "clear lungs without edema",
        ],
    },
    "Effusion": {
        "positive": [
            "pleural effusion is present",
            "there is a pleural effusion",
            "findings consistent with pleural effusion",
            "small pleural effusion noted",
            "bilateral pleural effusions",
            "costophrenic angle blunting",
        ],
        "negative": [
            "no pleural effusion",
            "no effusion",
            "costophrenic angles are sharp",
            "no evidence of pleural effusion",
            "clear costophrenic angles",
        ],
    },
    "Emphysema": {
        "positive": [
            "emphysema is present",
            "findings consistent with emphysema",
            "hyperinflation noted",
            "there is emphysema",
            "centrilobular emphysema",
            "hyperinflated lungs",
        ],
        "negative": [
            "no emphysema",
            "no hyperinflation",
            "no evidence of emphysema",
            "normal lung volumes",
            "lungs are not hyperinflated",
        ],
    },
    "Fibrosis": {
        "positive": [
            "pulmonary fibrosis is present",
            "findings consistent with fibrosis",
            "interstitial fibrosis noted",
            "there is fibrosis",
            "reticular opacities consistent with fibrosis",
            "fibrotic changes observed",
        ],
        "negative": [
            "no fibrosis",
            "no pulmonary fibrosis",
            "no evidence of fibrosis",
            "no interstitial fibrosis",
            "clear lungs without fibrosis",
        ],
    },
    "Hernia": {
        "positive": [
            "hiatal hernia is present",
            "findings consistent with hernia",
            "there is a hernia",
            "diaphragmatic hernia noted",
            "bowel loops in the chest",
        ],
        "negative": [
            "no hernia",
            "no hiatal hernia",
            "no evidence of hernia",
            "diaphragm is intact",
            "no diaphragmatic hernia",
        ],
    },
    "Infiltration": {
        "positive": [
            "infiltrate is present",
            "pulmonary infiltrate noted",
            "findings consistent with infiltration",
            "there are infiltrates",
            "bilateral infiltrates",
            "patchy infiltrates",
        ],
        "negative": [
            "no infiltrate",
            "no pulmonary infiltrates",
            "lungs are clear",
            "no evidence of infiltration",
            "no focal infiltrates",
        ],
    },
    "Mass": {
        "positive": [
            "a pulmonary mass is present",
            "lung mass noted",
            "findings consistent with a mass",
            "there is a mass lesion",
            "soft tissue mass in the lung",
        ],
        "negative": [
            "no mass",
            "no pulmonary mass",
            "no evidence of a mass",
            "no focal mass lesion",
            "lungs are clear without a mass",
        ],
    },
    "Nodule": {
        "positive": [
            "a pulmonary nodule is present",
            "lung nodule noted",
            "findings consistent with a nodule",
            "there is a nodule",
            "solitary pulmonary nodule",
            "multiple pulmonary nodules",
        ],
        "negative": [
            "no nodule",
            "no pulmonary nodule",
            "no evidence of a nodule",
            "no focal nodular opacity",
            "lungs are clear without nodules",
        ],
    },
    "Pleural_Thickening": {
        "positive": [
            "pleural thickening is present",
            "findings consistent with pleural thickening",
            "there is pleural thickening",
            "apical pleural thickening noted",
            "bilateral pleural thickening",
        ],
        "negative": [
            "no pleural thickening",
            "no evidence of pleural thickening",
            "pleura appears normal",
            "no apical pleural thickening",
            "no pleural abnormality",
        ],
    },
    "Pneumonia": {
        "positive": [
            "pneumonia is present",
            "findings consistent with pneumonia",
            "there is pneumonia",
            "lobar pneumonia noted",
            "bilateral pneumonia",
            "airspace disease consistent with pneumonia",
        ],
        "negative": [
            "no pneumonia",
            "no evidence of pneumonia",
            "lungs are clear",
            "no airspace disease",
            "no focal pneumonia",
        ],
    },
    "Pneumothorax": {
        "positive": [
            "pneumothorax is present",
            "findings consistent with pneumothorax",
            "there is a pneumothorax",
            "small pneumothorax noted",
            "left-sided pneumothorax",
            "right-sided pneumothorax",
        ],
        "negative": [
            "no pneumothorax",
            "no evidence of pneumothorax",
            "lung margins are intact",
            "no free air in the pleural space",
            "no pneumothorax identified",
        ],
    },
}


def _encode_prompts(
    model: torch.nn.Module,
    prompts: list[str],
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode a list of text prompts into L2-normalised embeddings."""
    all_embeddings = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            embeddings = model.encode_text(input_ids, attention_mask)  # (n, D)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)   # (N, D)


@torch.no_grad()
def compute_zero_shot_scores(
    model: torch.nn.Module,
    image_loader: DataLoader,
    labels: list[str],
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
) -> dict[str, np.ndarray]:
    """Compute zero-shot classification scores for each pathology label.

    For each image the score for a label is:
        mean(cosine_sim(image, positive_prompts))
        - mean(cosine_sim(image, negative_prompts))

    Returns:
        Dict mapping label name → 1-D array of scores (one per image).
    """
    if tokenizer is None:
        from transformers import AutoTokenizer as _AT
        tokenizer = _AT.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    model.eval()

    # Pre-compute text embeddings for every prompt (done once).
    pos_embeddings: dict[str, torch.Tensor] = {}
    neg_embeddings: dict[str, torch.Tensor] = {}

    for label in labels:
        prompts = PATHOLOGY_PROMPTS.get(label, {})
        pos_prompts = prompts.get("positive", [f"findings consistent with {label.lower()}"])
        neg_prompts = prompts.get("negative", [f"no {label.lower()}"])

        pos_embeddings[label] = _encode_prompts(model, pos_prompts, tokenizer, device)
        neg_embeddings[label] = _encode_prompts(model, neg_prompts, tokenizer, device)

    scores: dict[str, list] = {label: [] for label in labels}

    for batch in image_loader:
        images = batch["image"].to(device)
        with torch.no_grad():
            img_feats = model.encode_image(images).cpu()   # (B, D)

        for label in labels:
            pos_sim = img_feats @ pos_embeddings[label].T   # (B, n_pos)
            neg_sim = img_feats @ neg_embeddings[label].T   # (B, n_neg)
            score = pos_sim.mean(dim=1) - neg_sim.mean(dim=1)   # (B,)
            scores[label].append(score.numpy())

    return {label: np.concatenate(scores[label]) for label in labels}
