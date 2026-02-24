"""IU X-Ray dataset loader.

Parses OpenI XML reports (FINDINGS + IMPRESSION sections) and resolves
the paired PNG images using the exact filenames stored in each XML's
``<parentImage URI="...">`` elements.  This avoids the substring-glob
pitfall (e.g. study "1" matching CXR*1*.png incorrectly).

Dataset structure after extraction (see download_iu_xray.sh):
    data/iu_xray/
        images/   ← CXR<id>_IM-<view>-<slice>.png  (flat or nested)
        reports/  ← <uid>.xml
"""

import logging
from pathlib import Path
from typing import Optional, Callable
from xml.etree import ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class IUXrayDataset(Dataset):
    """IU X-Ray dataset returning (image, report_text) pairs.

    Image-report pairing uses the exact filenames declared in each XML's
    ``parentImage URI`` attribute, so studies with 2 views (frontal +
    lateral) both correctly link to the same report text.

    The dataset is split 90/10 by ``study_id`` — no study leaks across
    train and val splits.

    Args:
        data_dir:     Root directory containing ``images/`` and ``reports/``.
        split:        ``"train"`` or ``"val"``.
        val_fraction: Fraction of studies withheld for validation (0.1).
        transform:    torchvision transform applied to each PIL image.

    Raises:
        FileNotFoundError: If ``data_dir/reports/`` or ``data_dir/images/``
                           do not exist.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_fraction: float = 0.1,
        transform: Optional[Callable] = None,
    ):
        assert split in ("train", "val"), (
            f"split must be 'train' or 'val', got '{split}'"
        )
        self.transform = transform

        reports_dir = Path(data_dir) / "reports"
        images_dir  = Path(data_dir) / "images"

        logger.info("IUXrayDataset init — data_dir=%s split=%s", data_dir, split)

        if not reports_dir.exists():
            raise FileNotFoundError(
                f"Reports directory not found: {reports_dir}\n"
                "Run: bash experiments/scripts/download_iu_xray.sh"
            )
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}\n"
                "Run: bash experiments/scripts/download_iu_xray.sh"
            )

        samples = self._build_samples(reports_dir, images_dir)
        logger.info("Raw samples (before split): %d", len(samples))

        # Deterministic study-level split (sort to ensure reproducibility)
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

        n_studies = len({s["study_id"] for s in self.samples})
        logger.info(
            "[%s split] %d image-report pairs from %d studies",
            split, len(self.samples), n_studies,
        )

        if not self.samples:
            logger.warning(
                "Dataset is EMPTY for split='%s'. "
                "Verify that images/ and reports/ are populated.",
                split,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_samples(self, reports_dir: Path, images_dir: Path) -> list:
        """Build (image_path, report_text, study_id) triples.

        Uses the exact PNG filenames from ``parentImage URI`` attributes
        inside each XML, then looks them up in an index of all available
        images (handles nested subdirectories transparently).
        """
        # Build a flat index: filename → absolute path
        # rglob handles both flat and nested extraction layouts.
        image_index: dict[str, Path] = {}
        for p in images_dir.rglob("*.png"):
            image_index[p.name] = p

        xml_files = sorted(reports_dir.glob("*.xml"))
        logger.info(
            "Found %d XML report files, %d PNG images in index",
            len(xml_files), len(image_index),
        )

        samples = []
        n_empty_report = 0
        n_no_uri       = 0
        n_uri_missing  = 0

        for xml_path in xml_files:
            study_id = xml_path.stem

            report_text = self._parse_report(xml_path)
            if not report_text.strip():
                n_empty_report += 1
                logger.debug("Skipping study %s — empty FINDINGS/IMPRESSION", study_id)
                continue

            # Extract exact image URIs declared in this study's XML.
            uris = self._parse_image_uris(xml_path)

            if not uris:
                # Rare: XML has no parentImage elements (malformed).
                n_no_uri += 1
                logger.debug("Skipping study %s — no parentImage URIs", study_id)
                continue

            found_any = False
            for uri in uris:
                if uri in image_index:
                    samples.append({
                        "study_id":   study_id,
                        "image_path": str(image_index[uri]),
                        "report":     report_text,
                    })
                    found_any = True
                else:
                    n_uri_missing += 1
                    logger.debug(
                        "Study %s — URI '%s' not found in image index", study_id, uri
                    )

            if not found_any:
                logger.debug("Study %s — none of its URIs resolved to a file", study_id)

        logger.info(
            "Build complete — pairs=%d  skipped(empty_report)=%d  "
            "skipped(no_uri)=%d  unresolved_uris=%d",
            len(samples), n_empty_report, n_no_uri, n_uri_missing,
        )
        return samples

    def _parse_image_uris(self, xml_path: Path) -> list[str]:
        """Return image filenames declared in a study XML.

        Handles two ``<parentImage>`` formats found in the IU X-Ray release:

        * ``<parentImage URI="CXR1_IM-0001-1001.png">``  (URI attribute)
        * ``<parentImage id="CXR1_IM-0001-1001">``       (id attribute, no .png)

        In both cases only the bare filename (no path prefix) is returned so
        it can be looked up in the flat image index built from ``images/``.
        Returns an empty list on parse error.
        """
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            logger.warning("XML parse error in %s: %s", xml_path.name, exc)
            return []

        filenames = []
        for img in root.iter("parentImage"):
            # Format 1: explicit URI attribute (may be a full URL or bare filename)
            uri = img.get("URI", "").strip()
            if uri:
                name = Path(uri.replace("\\", "/")).name
                if name:
                    filenames.append(name)
                continue

            # Format 2: id attribute holds the filename without the .png extension
            img_id = img.get("id", "").strip()
            if img_id:
                filenames.append(img_id + ".png")

        return filenames

    def _parse_report(self, xml_path: Path) -> str:
        """Concatenate FINDINGS and IMPRESSION sections from a study XML."""
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            logger.warning("XML parse error in %s: %s", xml_path.name, exc)
            return ""

        sections = []
        for section in root.iter("AbstractText"):
            label = section.get("Label", "")
            text  = (section.text or "").strip()
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
                "Failed to open image %s (idx=%d study=%s): %s",
                sample["image_path"], idx, sample["study_id"], exc,
            )
            raise

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image":      image,
            "report":     sample["report"],
            "study_id":   sample["study_id"],
            "image_path": sample["image_path"],
        }
