"""IU X-Ray dataset loader.

Parses OpenI XML reports (FINDINGS + IMPRESSION) and resolves paired
PNG images.  Returns (image, report_text) samples for contrastive
training (CheXzero) or seq2seq training (R2Gen).
"""

import logging
import os
from pathlib import Path
from typing import Optional, Callable
from xml.etree import ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class IUXrayDataset(Dataset):
    """IU X-Ray dataset returning (image, report_text) pairs.

    Each study may have up to two views (frontal + lateral); both images
    are paired with the same report text.  The dataset is split 90/10
    by study_id so no study leaks across train and val.

    Args:
        data_dir:     Root directory containing ``images/`` and ``reports/``.
        split:        ``"train"`` or ``"val"``.
        val_fraction: Fraction of studies withheld for validation (default 0.1).
        transform:    torchvision transform applied to each PIL image.
        return_tokens: If True, also return ``report`` as-is for external
                       tokenisation (used by R2Gen dataloader).

    Raises:
        FileNotFoundError: If ``data_dir/reports/`` does not exist.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_fraction: float = 0.1,
        transform: Optional[Callable] = None,
        return_tokens: bool = False,
    ):
        assert split in ("train", "val"), (
            f"split must be 'train' or 'val', got '{split}'"
        )
        self.transform = transform
        self.return_tokens = return_tokens

        reports_dir = Path(data_dir) / "reports"
        images_dir = Path(data_dir) / "images"

        logger.info("IUXrayDataset init — data_dir=%s split=%s", data_dir, split)

        if not reports_dir.exists():
            raise FileNotFoundError(
                f"Reports directory not found: {reports_dir}\n"
                "Run experiments/scripts/download_iu_xray.sh first."
            )
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}\n"
                "Run experiments/scripts/download_iu_xray.sh first."
            )

        logger.info("Scanning reports from %s …", reports_dir)
        samples = self._build_samples(reports_dir, images_dir)
        logger.info("Raw samples (before split): %d", len(samples))

        # Deterministic study-level split
        study_ids = sorted({s["study_id"] for s in samples})
        n_val = max(1, int(len(study_ids) * val_fraction))
        val_ids = set(study_ids[-n_val:])
        logger.info(
            "Studies total=%d  val=%d  train=%d",
            len(study_ids), len(val_ids), len(study_ids) - len(val_ids),
        )

        if split == "train":
            self.samples = [s for s in samples if s["study_id"] not in val_ids]
        else:
            self.samples = [s for s in samples if s["study_id"] in val_ids]

        logger.info(
            "[%s split] %d image-report pairs from %d studies",
            split, len(self.samples), len({s["study_id"] for s in self.samples}),
        )

        if len(self.samples) == 0:
            logger.warning(
                "Dataset is EMPTY for split='%s'. "
                "Check that images/ and reports/ are populated.",
                split,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_samples(self, reports_dir: Path, images_dir: Path) -> list:
        samples = []
        xml_files = sorted(reports_dir.glob("*.xml"))
        n_xml = len(xml_files)
        n_empty_report = 0
        n_no_image = 0

        logger.info("Found %d XML report files", n_xml)

        for xml_path in xml_files:
            study_id = xml_path.stem
            report_text = self._parse_report(xml_path)

            if not report_text.strip():
                n_empty_report += 1
                logger.debug("Skipping study %s — empty report", study_id)
                continue

            # Match images by study_id substring in filename
            image_paths = sorted(images_dir.glob(f"*{study_id}*"))
            if not image_paths:
                n_no_image += 1
                logger.debug("Skipping study %s — no matching images", study_id)
                continue

            for img_path in image_paths:
                samples.append({
                    "study_id": study_id,
                    "image_path": str(img_path),
                    "report": report_text,
                })

        logger.info(
            "Build complete — kept=%d  skipped(empty report)=%d  skipped(no image)=%d",
            len(samples), n_empty_report, n_no_image,
        )
        return samples

    def _parse_report(self, xml_path: Path) -> str:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as exc:
            logger.warning("XML parse error in %s: %s", xml_path.name, exc)
            return ""

        sections = []
        for section in root.iter("AbstractText"):
            label = section.get("Label", "")
            text = (section.text or "").strip()
            if label.upper() in ("FINDINGS", "IMPRESSION") and text:
                sections.append(text)

        return " ".join(sections)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        try:
            image = Image.open(sample["image_path"]).convert("L")
        except Exception as exc:
            logger.error(
                "Failed to open image %s (idx=%d, study=%s): %s",
                sample["image_path"], idx, sample["study_id"], exc,
            )
            raise

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "report": sample["report"],
            "study_id": sample["study_id"],
            "image_path": sample["image_path"],
        }
