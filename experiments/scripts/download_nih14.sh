#!/usr/bin/env bash
# Instructions for downloading NIH ChestX-ray14.
#
# NIH-14 cannot be downloaded automatically because Box.com requires a
# browser-based session.  Follow the manual steps below.
#
# After downloading you should have:
#   data/nih_chestxray14/images/          (112,120 PNG images across 12 batch zips)
#   data/nih_chestxray14/Data_Entry_2017.csv
#   data/nih_chestxray14/train_val_list.txt   (optional, for official split)
#   data/nih_chestxray14/test_list.txt        (optional)
#
# Reference: https://nihcc.app.box.com/v/ChestXray-NIHCC
#            Wang et al. (2017) "ChestX-ray8: Hospital-scale Chest X-ray Database"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/nih_chestxray14"

echo "======================================================================"
echo "  NIH ChestX-ray14 — Manual Download Instructions"
echo "======================================================================"
echo ""
echo "1. Open the dataset page in a browser:"
echo "   https://nihcc.app.box.com/v/ChestXray-NIHCC"
echo ""
echo "2. Download ALL of the following files into ${DATA_DIR}/zips/:"
echo "   images_001.tar.gz  ...  images_012.tar.gz   (12 batch archives)"
echo "   Data_Entry_2017.csv"
echo "   train_val_list.txt"
echo "   test_list.txt"
echo ""
echo "3. After downloading, run this script again with --extract to unpack."
echo ""

if [[ "${1:-}" != "--extract" ]]; then
    echo "Re-run with --extract once all files are in ${DATA_DIR}/zips/:"
    echo "   bash experiments/scripts/download_nih14.sh --extract"
    exit 0
fi

# ── Extraction ───────────────────────────────────────────────────────────────
ZIPS_DIR="${DATA_DIR}/zips"
IMAGES_DIR="${DATA_DIR}/images"

if [ ! -d "${ZIPS_DIR}" ]; then
    echo "ERROR: ${ZIPS_DIR} does not exist. Download the files first."
    exit 1
fi

mkdir -p "${IMAGES_DIR}"

echo "Extracting image archives..."
for i in $(seq -w 1 12); do
    ARCHIVE="${ZIPS_DIR}/images_0${i}.tar.gz"
    if [ -f "${ARCHIVE}" ]; then
        echo "  Extracting ${ARCHIVE}..."
        tar -xzf "${ARCHIVE}" -C "${IMAGES_DIR}" --strip-components=1
    else
        echo "  WARNING: ${ARCHIVE} not found, skipping."
    fi
done

# Copy metadata files
for META in Data_Entry_2017.csv train_val_list.txt test_list.txt; do
    SRC="${ZIPS_DIR}/${META}"
    if [ -f "${SRC}" ]; then
        cp "${SRC}" "${DATA_DIR}/${META}"
        echo "Copied ${META}"
    fi
done

# ── Verification ─────────────────────────────────────────────────────────────
IMAGE_COUNT=$(find "${IMAGES_DIR}" -name "*.png" | wc -l)
echo ""
echo "Verification:"
echo "  PNG images: ${IMAGE_COUNT}  (expected ~112,120)"

if [ "${IMAGE_COUNT}" -lt 100000 ]; then
    echo "WARNING: fewer images than expected — some batch archives may be missing."
fi

echo ""
echo "NIH ChestX-ray14 extraction complete."
echo "Data directory: ${DATA_DIR}"
