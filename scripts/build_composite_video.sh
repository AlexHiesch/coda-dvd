#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Building composite demo video ==="

# Step 1: Convert slides to video clips
for slide in media/slides/*.png; do
  name=$(basename "$slide" .png)
  dur=5
  if [[ "$name" == "01_architecture" ]]; then dur=8; fi
  if [[ "$name" == "04_downlink" ]]; then dur=7; fi
  ffmpeg -y -loop 1 -i "$slide" -c:v libx264 -t $dur -pix_fmt yuv420p \
    -vf "scale=1920:1080" -r 30 -crf 18 -preset fast \
    "media/slides/${name}.mp4" 2>/dev/null
  echo "  Created ${name}.mp4 (${dur}s)"
done

# Step 2: Scale the terminal recording to 1920x1080
ffmpeg -y -i media/renders/coda_dvd_demo_raw.mp4 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x12121800" \
  -c:v libx264 -pix_fmt yuv420p -r 30 -crf 18 -preset fast \
  media/slides/06_terminal.mp4 2>/dev/null
echo "  Created 06_terminal.mp4 (terminal recording)"

# Step 3: Create concat list
cat > media/slides/concat.txt << EOF
file '01_architecture.mp4'
file '02_sentinel.mp4'
file '03_mapbox.mp4'
file '04_downlink.mp4'
file '05_singapore.mp4'
file '06_terminal.mp4'
EOF

# Step 4: Concatenate all clips
ffmpeg -y -f concat -safe 0 -i media/slides/concat.txt \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast \
  media/renders/coda_dvd_demo_composite.mp4 2>/dev/null

# Report
duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 media/renders/coda_dvd_demo_composite.mp4)
size=$(du -h media/renders/coda_dvd_demo_composite.mp4 | cut -f1)
echo ""
echo "=== Composite video ready ==="
echo "  File: media/renders/coda_dvd_demo_composite.mp4"
echo "  Duration: ${duration}s"
echo "  Size: ${size}"
