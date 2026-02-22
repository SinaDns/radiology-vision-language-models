from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

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

    Reads Data_Entry_2017.csv and resolves images from the images/ subdirectory.
    Supports an optional official patient-level split CSV.
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

        csv_path = self.data_dir / "Data_Entry_2017.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {csv_path}\n"
                "Download NIH ChestX-ray14 from https://nihcc.app.box.com/v/ChestXray-NIHCC"
            )

        df = pd.read_csv(csv_path)
        df = df.rename(columns={"Image Index": "image_file", "Finding Labels": "labels"})

        if split_csv is not None and split is not None:
            split_df = pd.read_csv(split_csv, header=None, names=["image_file"])
            split_files = set(split_df["image_file"].tolist())
            df = df[df["image_file"].isin(split_files)].reset_index(drop=True)

        self.samples = self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> list:
        samples = []
        images_dir = self.data_dir / "images"
        for _, row in df.iterrows():
            img_path = images_dir / row["image_file"]
            if not img_path.exists():
                continue
            label_vec = self._parse_labels(row["labels"])
            samples.append({
                "image_path": str(img_path),
                "labels": label_vec,
            })
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
        image = Image.open(sample["image_path"]).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "labels": sample["labels"],
            "image_path": sample["image_path"],
        }
