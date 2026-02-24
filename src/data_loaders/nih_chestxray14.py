"""NIH ChestX-ray14 dataset loader for zero-shot / few-shot evaluation."""

import logging
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NIH14_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


class NIHChestXray14Dataset(Dataset):
    """NIH ChestX-ray14 dataset for zero-shot evaluation.

    Reads ``Data_Entry_2017.csv`` and resolves images from ``images/``.
    Supports an optional official patient-level split CSV (one filename
    per line, no header).

    Args:
        data_dir:  Root containing ``images/`` and ``Data_Entry_2017.csv``.
        split:     Informational label (not used for filtering unless
                   ``split_csv`` is also provided).
        split_csv: Path to a file listing image filenames for this split.
        transform: torchvision transform applied to each PIL image.

    Raises:
        FileNotFoundError: If ``Data_Entry_2017.csv`` is not found.
    """

    def __init__(
        self,
        data_dir: str,
        split: Optional[str] = None,
        split_csv: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_names = NIH14_LABELS

        logger.info(
            "NIHChestXray14Dataset init — data_dir=%s split=%s split_csv=%s",
            data_dir, split, split_csv,
        )

        csv_path = self.data_dir / "Data_Entry_2017.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {csv_path}\n"
                "Download NIH ChestX-ray14 from "
                "https://nihcc.app.box.com/v/ChestXray-NIHCC"
            )

        df = pd.read_csv(csv_path)
        logger.info("Loaded metadata CSV — total rows: %d", len(df))

        df = df.rename(columns={
            "Image Index": "image_file",
            "Finding Labels": "labels",
        })

        if split_csv is not None and split is not None:
            split_path = Path(split_csv)
            if not split_path.exists():
                raise FileNotFoundError(f"Split CSV not found: {split_path}")
            split_df = pd.read_csv(split_path, header=None, names=["image_file"])
            split_files = set(split_df["image_file"].tolist())
            before = len(df)
            df = df[df["image_file"].isin(split_files)].reset_index(drop=True)
            logger.info(
                "Applied split filter '%s': %d → %d rows",
                split, before, len(df),
            )

        self.samples = self._build_samples(df)
        logger.info(
            "Dataset ready — %d samples, labels: %s",
            len(self.samples), self.label_names,
        )

        # Log label prevalence summary
        if self.samples:
            label_matrix = torch.stack([s["labels"] for s in self.samples])
            prevalences = label_matrix.mean(dim=0)
            for name, prev in zip(self.label_names, prevalences):
                logger.debug("  %-22s prevalence=%.3f", name, prev.item())

    def _build_samples(self, df: pd.DataFrame) -> list:
        samples = []
        images_dir = self.data_dir / "images"
        n_missing = 0

        if not images_dir.exists():
            logger.warning("Images directory not found: %s", images_dir)

        for _, row in df.iterrows():
            img_path = images_dir / row["image_file"]
            if not img_path.exists():
                n_missing += 1
                if n_missing <= 5:
                    logger.debug("Image file missing: %s", img_path)
                continue
            label_vec = self._parse_labels(row["labels"])
            samples.append({
                "image_path": str(img_path),
                "labels": label_vec,
            })

        if n_missing:
            logger.warning(
                "%d image files listed in CSV but not found on disk "
                "(check extraction completeness)",
                n_missing,
            )
        logger.info("Built %d valid samples (%d missing images skipped)", len(samples), n_missing)
        return samples

    def _parse_labels(self, label_string: str) -> torch.Tensor:
        active = set(label_string.split("|"))
        vec = torch.zeros(len(self.label_names), dtype=torch.float32)
        for i, name in enumerate(self.label_names):
            if name in active:
                vec[i] = 1.0
        return vec

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        try:
            image = Image.open(sample["image_path"]).convert("L")
        except Exception as exc:
            logger.error(
                "Failed to open image %s (idx=%d): %s",
                sample["image_path"], idx, exc,
            )
            raise

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "labels": sample["labels"],
            "image_path": sample["image_path"],
        }
