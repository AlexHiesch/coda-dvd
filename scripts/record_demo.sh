#!/usr/bin/env bash
# record_demo.sh — Record CODA-DVD pipeline demo with asciinema
# Usage: ./scripts/record_demo.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p media/recordings

CAST_FILE="media/recordings/coda_dvd_demo.cast"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  CODA-DVD Demo Recording                         ║"
echo "║  Press Ctrl+D when finished to stop recording    ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "Recording to: $CAST_FILE"
echo ""

# Record with 80 cols x 24 rows for clean rendering
asciinema rec \
  --cols 100 \
  --rows 30 \
  --command "source .venv/bin/activate && python coda_dvd_pipeline.py" \
  --title "CODA-DVD: Dark Vessel Detection Pipeline" \
  --overwrite \
  "$CAST_FILE"

echo ""
echo "Recording saved: $CAST_FILE"
echo "Next: run ./scripts/render_demo.sh to convert to MP4"
