import os
from pathlib import Path
from typing import Optional, Callable
from xml.etree import ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset


class IUXrayDataset(Dataset):
    """IU X-Ray dataset returning (image, report_text) pairs.

    Each study may have up to two views (frontal + lateral); both images
    are paired with the same report text.  The dataset is split 90/10
    by study_id so no study leaks across train and val.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_fraction: float = 0.1,
        transform: Optional[Callable] = None,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split}"
        self.transform = transform

        reports_dir = Path(data_dir) / "reports"
        images_dir = Path(data_dir) / "images"

        if not reports_dir.exists():
            raise FileNotFoundError(
                f"Reports directory not found: {reports_dir}\n"
                "Run experiments/scripts/download_iu_xray.sh first."
            )

        samples = self._build_samples(reports_dir, images_dir)

        # Deterministic study-level split
        study_ids = sorted({s["study_id"] for s in samples})
        n_val = max(1, int(len(study_ids) * val_fraction))
        val_ids = set(study_ids[-n_val:])

        if split == "train":
            self.samples = [s for s in samples if s["study_id"] not in val_ids]
        else:
            self.samples = [s for s in samples if s["study_id"] in val_ids]

    def _build_samples(self, reports_dir: Path, images_dir: Path) -> list:
        samples = []
        for xml_path in sorted(reports_dir.glob("*.xml")):
            study_id = xml_path.stem
            report_text = self._parse_report(xml_path)
            if not report_text.strip():
                continue

            # Images follow naming convention: study_id-*.png or
            # CXR<study_id>_IM-<id>-<view>.png; search by prefix.
            image_paths = sorted(images_dir.glob(f"*{study_id}*"))
            if not image_paths:
                continue

            for img_path in image_paths:
                samples.append({
                    "study_id": study_id,
                    "image_path": str(img_path),
                    "report": report_text,
                })

        return samples

    def _parse_report(self, xml_path: Path) -> str:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            return ""

        sections = []
        for section in root.iter("AbstractText"):
            label = section.get("Label", "")
            text = (section.text or "").strip()
            if label.upper() in ("FINDINGS", "IMPRESSION") and text:
                sections.append(text)

        return " ".join(sections)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("L")  # grayscale
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "report": sample["report"],
            "study_id": sample["study_id"],
        }
