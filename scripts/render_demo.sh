#!/usr/bin/env bash
# render_demo.sh — Convert asciinema recording to MP4 via agg + ffmpeg
# Usage: ./scripts/render_demo.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p media/renders media/final

CAST="media/recordings/coda_dvd_demo.cast"
GIF="media/renders/coda_dvd_demo.gif"
MP4_RAW="media/renders/coda_dvd_demo_raw.mp4"

if [[ ! -f "$CAST" ]]; then
  echo "ERROR: No recording found at $CAST"
  echo "Run ./scripts/record_demo.sh first"
  exit 1
fi

echo "── Step 1: asciinema cast → GIF (agg) ──"
agg \
  --font-size 18 \
  --theme monokai \
  --cols 100 \
  --rows 30 \
  --speed 1.0 \
  "$CAST" "$GIF"

echo "── Step 2: GIF → MP4 (ffmpeg) ──"
# Convert to proper H.264 MP4 with good quality
ffmpeg -y \
  -i "$GIF" \
  -movflags faststart \
  -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 \
  -crf 18 \
  -preset slow \
  "$MP4_RAW"

echo ""
echo "✓ Terminal MP4 ready: $MP4_RAW"
echo "  Duration: $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$MP4_RAW")s"
echo ""
echo "Next: run python scripts/generate_voiceover.py to add narration"
