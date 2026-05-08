#!/usr/bin/env bash
# render_excalidraw.sh — Export Excalidraw to high-res PNG
# Requires: npx @excalidraw/cli  (auto-installed via npx)
# Usage: ./scripts/render_excalidraw.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p media/renders

INPUT="coda_dvd_architecture.excalidraw"
OUTPUT="media/renders/coda_dvd_architecture.png"

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: $INPUT not found"
  exit 1
fi

echo "── Rendering Excalidraw → high-res PNG ──"

# Try excalidraw CLI first
if npx --yes @excalidraw/excalidraw-export \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --scale 3 \
  --theme light 2>/dev/null; then
  echo "✓ Exported via excalidraw CLI: $OUTPUT"
else
  # Fallback: use kroki.io API
  echo "  excalidraw CLI not available, trying kroki.io API..."
  CONTENT=$(cat "$INPUT" | python3 -c "import sys,base64,zlib; d=sys.stdin.buffer.read(); print(base64.urlsafe_b64encode(zlib.compress(d,9)).decode())")
  curl -s "https://kroki.io/excalidraw/png/${CONTENT}" -o "$OUTPUT" 2>/dev/null

  if [[ -f "$OUTPUT" ]] && [[ $(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null) -gt 1000 ]]; then
    echo "✓ Exported via kroki.io: $OUTPUT"
  else
    echo "⚠ Both methods failed. Open coda_dvd_architecture.excalidraw in Excalidraw"
    echo "  and export manually: Menu → Export → PNG (3x scale)"
    echo "  Save to: $OUTPUT"
    exit 1
  fi
fi

echo "  Size: $(identify "$OUTPUT" 2>/dev/null | awk '{print $3}' || echo 'unknown')"
