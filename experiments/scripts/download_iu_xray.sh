#!/usr/bin/env bash
# Download IU X-Ray (Indiana University Chest X-ray) dataset from OpenI.
#
# After running this script you should have:
#   data/iu_xray/images/   (~7,470 PNG images)
#   data/iu_xray/reports/  (~3,955 XML report files)
#
# Usage:
#   bash experiments/scripts/download_iu_xray.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/iu_xray"

echo "Downloading IU X-Ray dataset to: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

# ── Images ──────────────────────────────────────────────────────────────────
IMAGE_TGZ="${DATA_DIR}/NLMCXR_png.tgz"
if [ ! -f "${IMAGE_TGZ}" ]; then
    echo "Downloading images archive..."
    wget -q --show-progress \
        "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz" \
        -O "${IMAGE_TGZ}"
else
    echo "Images archive already downloaded, skipping."
fi

echo "Extracting images..."
mkdir -p "${DATA_DIR}/images"
tar -xzf "${IMAGE_TGZ}" -C "${DATA_DIR}/images" --strip-components=1
echo "Images extracted."

# ── Reports ─────────────────────────────────────────────────────────────────
REPORT_TGZ="${DATA_DIR}/NLMCXR_reports.tgz"
if [ ! -f "${REPORT_TGZ}" ]; then
    echo "Downloading reports archive..."
    wget -q --show-progress \
        "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz" \
        -O "${REPORT_TGZ}"
else
    echo "Reports archive already downloaded, skipping."
fi

echo "Extracting reports..."
mkdir -p "${DATA_DIR}/reports"
tar -xzf "${REPORT_TGZ}" -C "${DATA_DIR}/reports" --strip-components=1
echo "Reports extracted."

# ── Verification ────────────────────────────────────────────────────────────
IMAGE_COUNT=$(find "${DATA_DIR}/images" -name "*.png" | wc -l)
REPORT_COUNT=$(find "${DATA_DIR}/reports" -name "*.xml" | wc -l)

echo ""
echo "Verification:"
echo "  PNG images : ${IMAGE_COUNT}  (expected ~7,470)"
echo "  XML reports: ${REPORT_COUNT}  (expected ~3,955)"

if [ "${IMAGE_COUNT}" -lt 7000 ]; then
    echo "WARNING: fewer images than expected — download may be incomplete."
fi
if [ "${REPORT_COUNT}" -lt 3000 ]; then
    echo "WARNING: fewer reports than expected — download may be incomplete."
fi

echo ""
echo "IU X-Ray download complete."
