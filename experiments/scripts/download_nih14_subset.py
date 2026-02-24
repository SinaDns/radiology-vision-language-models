"""
Download a 5% subset of NIH ChestX-ray14 using Kaggle API credentials.

Usage:
    python experiments/scripts/download_nih14_subset.py \
        --data-dir data/nih_chestxray14 \
        --fraction 0.05 \
        --kaggle-username YOUR_USERNAME \
        --kaggle-key YOUR_API_KEY
"""
import argparse
import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

from tqdm import tqdm


def kaggle_download(dataset: str, file: str, dest_dir: Path) -> Path:
    """Download a single file from a Kaggle dataset using the Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

    api = KaggleApi()
    api.authenticate()
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Kaggle → {dataset}/{file}")
    api.dataset_download_file(dataset, file_name=file, path=str(dest_dir), force=True)

    # Kaggle wraps single files in a .zip
    zipped = dest_dir / (file + ".zip")
    if zipped.exists():
        with zipfile.ZipFile(zipped) as z:
            z.extractall(dest_dir)
        zipped.unlink()

    return dest_dir / file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",        default="data/nih_chestxray14")
    parser.add_argument("--fraction",        type=float, default=0.05)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--kaggle-username", required=True, help="Kaggle account username")
    parser.add_argument("--kaggle-key",      required=True, help="Kaggle API key token")
    parser.add_argument("--keep-archive",    action="store_true",
                        help="Do not delete the tar.gz after extraction")
    args = parser.parse_args()

    # ── Inject credentials into environment (no kaggle.json needed) ──────────
    os.environ["KAGGLE_USERNAME"] = args.kaggle_username
    os.environ["KAGGLE_KEY"]      = args.kaggle_key

    data_dir   = Path(args.data_dir)
    images_dir = data_dir / "images"
    staging    = data_dir / "staging"

    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    staging.mkdir(exist_ok=True)

    DATASET = "nih-chest-xrays/data"

    # ── 1. Metadata ──────────────────────────────────────────────────────────
    print("=== Step 1/3 — Downloading metadata ===")
    for fname in ["Data_Entry_2017.csv", "test_list.txt", "train_val_list.txt"]:
        dest = data_dir / fname
        if dest.exists():
            print(f"  {fname} already present, skipping.")
        else:
            kaggle_download(DATASET, fname, data_dir)
            print(f"  {fname} ✓")

    # ── 2. First image archive (~9 GB, ~9,300 images) ────────────────────────
    print("\n=== Step 2/3 — Downloading images_001.tar.gz ===")
    archive = staging / "images_001.tar.gz"
    if archive.exists():
        print(f"  Archive already present ({archive.stat().st_size / 1e9:.1f} GB), skipping.")
    else:
        kaggle_download(DATASET, "images_001.tar.gz", staging)

    # ── 3. Sample fraction and extract ───────────────────────────────────────
    print(f"\n=== Step 3/3 — Extracting {args.fraction * 100:.0f}% sample ===")
    random.seed(args.seed)

    with tarfile.open(archive, "r:gz") as tar:
        all_members = [m for m in tar.getmembers() if m.name.lower().endswith(".png")]
        sample_size = max(100, int(len(all_members) * args.fraction))
        sampled     = random.sample(all_members, sample_size)
        print(f"  Archive: {len(all_members)} images → extracting {sample_size}")
        for member in tqdm(sampled, desc="Extracting"):
            member.name = Path(member.name).name  # flatten directory
            tar.extract(member, images_dir)

    if not args.keep_archive:
        archive.unlink()
        print("  Archive deleted (use --keep-archive to retain it).")

    shutil.rmtree(staging, ignore_errors=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(list(images_dir.glob("*.png")))
    print(f"\n✓ Images on disk : {n}")
    print(f"✓ Metadata       : {data_dir}")
    print(f"✓ Ready for      : NIHChestXray14Dataset(data_dir='{data_dir}')")


if __name__ == "__main__":
    main()