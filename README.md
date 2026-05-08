# CODA-DVD: Cascaded On-Board Attention for Dark Vessel Detection

> **AI in Space Hackathon** — Liquid Track | Team: Alex Hiesch
> A bandwidth-efficient on-board satellite AI pipeline using **LFM2.5-VL** for autonomous dark vessel detection.

---

## The Problem

Earth Observation satellites generate **terabytes of data daily**, but downlink bandwidth is severely limited (often just minutes of ground station contact per orbit). Traditional approaches beam down full images for ground-based analysis — wasting 95%+ bandwidth on empty ocean, clouds, and irrelevant data.

**Dark vessels** (ships operating without AIS transponders) are a critical maritime security challenge: illegal fishing, sanctions evasion, and trafficking. Detecting them requires processing vast ocean areas, but transmitting full imagery is infeasible.

## Our Solution

**CODA-DVD** implements a **3-stage cascaded filter** running entirely on-board the satellite. Each stage progressively discards irrelevant data, so only tiny, confirmed anomaly packets reach the ground station.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CODA-DVD Pipeline                                │
├──────────────────┬─────────────────────┬────────────────────────────────┤
│    Stage 1       │      Stage 2        │         Stage 3                │
│   CloudFilter    │  AnomalyDetector    │   SuperResolutionTrigger       │
│                  │                     │                                │
│  SCL band +      │  NIR threshold +    │   Mapbox high-res zoom +       │
│  metadata check  │  LFM2.5-VL confirm  │   LFM2.5-VL classification     │
│                  │                     │                                │
│  ☁ >70% cloud?  │  No bright spots    │   128×128 crop + JSON metadata │
│  → DISCARD       │  on dark ocean?     │   → TRANSMIT (15KB vs 2MB)     │
│  (100% saved)    │  → DISCARD          │                                │
│                  │  (100% saved)       │   98-99% bandwidth saved       │
└──────────────────┴─────────────────────┴────────────────────────────────┘
```

### Bandwidth Impact

| Scenario | Traditional | CODA-DVD | Savings |
|----------|-------------|----------|---------|
| Cloudy image | 820 KB transmitted | 0 bytes | **100%** |
| Clear, no vessel | 820 KB transmitted | 0 bytes | **100%** |
| Vessel detected | 820 KB transmitted | ~15 KB (JSON + crop) | **98.1%** |
| High-res confirmation | 2 MB transmitted | ~22 KB | **99.1%** |

## Architecture Deep-Dive

### Stage 1: Cloud Filter
- Uses **Scene Classification Layer (SCL)** from Sentinel-2 multispectral data
- Counts cloud pixels (classes 8, 9) for sub-scene cloud detection
- Cross-validates with API metadata cloud percentage
- **Multispectral band usage:** `scl` band for pixel-level classification

### Stage 2: Anomaly Detection (NIR + VLM)
- **Phase 1 — Fast CV:** Near-Infrared (NIR) band thresholding. Ocean is dark (<30/255), vessels reflect brightly (>80/255). Connected component analysis finds candidate blobs.
- **Phase 2 — VLM Confirmation:** Each candidate crop is fed to **LFM2.5-VL** with prompt: *"Is there a vessel in this satellite image crop?"*
- Pixel-to-GPS coordinate conversion using Sentinel footprint metadata
- **Multispectral band usage:** `nir` for contrast-based pre-filtering

### Stage 3: Super Resolution + Classification
- Fetches **Mapbox high-resolution imagery** (1280×1280 @ 0.5m/px) at anomaly GPS coordinates
- **LFM2.5-VL** classifies: vessel type, estimated length, heading direction
- Generates minimal **downlink packet**: 500B JSON metadata + 128×128 PNG crop
- Graceful fallback to 4× bicubic upscale if high-res unavailable

## LFM2.5-VL Usage (Liquid Track)

The model is used at **three critical decision points** — it's not just a captioner, it's the autonomous decision engine:

1. **Anomaly confirmation** (Stage 2): Reduces false positives from CV pre-filter
2. **Vessel classification** (Stage 3): Adds intelligence to the downlink packet
3. **Grounding format** output: `[{"label": "ship", "bbox": [x1,y1,x2,y2]}]` (0-1 normalized)

**Runtime characteristics** (Apple M3 Max simulating NVIDIA Orin 16GB):
- Model: `mlx-community/LFM2.5-VL-1.6B-8bit`
- Inference: ~220 tokens/sec
- Memory: 2.6 GB
- Suitable for edge deployment on space-grade hardware

## Fine-Tuning (Vessel Grounding)

We prepared a complete fine-tuning pipeline for vessel grounding using the official Liquid AI framework:

**Dataset:** [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench) (NeurIPS 2024)
- 36K visual grounding samples from satellite imagery
- Format: referring expression → bounding box `[x1, y1, x2, y2]` normalized 0-1
- Includes ships, vehicles, and other objects in remote sensing imagery

**Framework:** `leap-finetune` + Modal (H100 GPU)

**Configuration:** `fine_tuning/vessel_grounding_modal.yaml`
- LoRA (r=16, α=32) for efficient adaptation
- 3 epochs, cosine LR schedule (3e-5)
- Evaluation: IoU@0.5 and IoU@0.25 on held-out grounding set

```bash
# Prepare data (runs on Modal, downloads VRSBench)
python liquid-cookbook/examples/satellite-vlm/prepare_vrsbench.py --task grounding --modal

# Launch fine-tuning
uv run leap-finetune fine_tuning/vessel_grounding_modal.yaml
```

## Multispectral Data Usage

| Band | Purpose | Stage |
|------|---------|-------|
| Red, Green, Blue | Visual input for VLM | 2, 3 |
| NIR (Near-Infrared) | Water/vessel contrast for CV pre-filter | 2 |
| SCL (Scene Classification) | Cloud pixel classification | 1 |

## Running the Demo

### Prerequisites
```bash
# 1. Start SimSat (provides Sentinel-2 + Mapbox API)
cd SimSat && docker compose up -d

# 2. Set Mapbox token (optional, enables Stage 3 high-res)
export MAPBOX_ACCESS_TOKEN="your_token_here"

# 3. Python environment
python -m venv .venv && source .venv/bin/activate
pip install mlx-vlm numpy Pillow requests
```

### Run
```bash
python coda_dvd_pipeline.py
```

The pipeline runs 3 test scenarios end-to-end:
1. **North Atlantic** — Cloudy → discarded at Stage 1
2. **Open Pacific** — Clear but no vessels → discarded at Stage 2
3. **English Channel** — Busy shipping lane → vessel detected, classified, minimal downlink

### Expected Output
```
╔═══════════════════════════════════════════════════════════╗
║   CODA-DVD: Cascaded On-Board Attention                  ║
║   for Dark Vessel Detection                              ║
╚═══════════════════════════════════════════════════════════╝

[Stage 1] Cloud Filter
  ☀ Clear sky: 0.0% cloud cover

[Stage 2] Anomaly Detection (NIR + LFM2.5-VL)
  ● 1 anomaly detected
  🚢 Vessel at 51.0°N, 1.3°E [vlm]

[Stage 3] High-Res Zoom + Classification
  ✓ High-res acquired (mapbox)
  🔍 Classification: cargo vessel, ~150m, heading W
  📦 Downlink packet: 22,104 bytes (vs 820,000 original)
  💾 Bandwidth saved: 97.3%

MISSION SUMMARY
  Total bandwidth saved: 99.1%
```

## Project Structure

```
├── coda_dvd_pipeline.py              # Complete pipeline (single runnable file)
├── fine_tuning/
│   └── vessel_grounding_modal.yaml   # Fine-tuning config (LoRA, Modal H100)
├── liquid-cookbook/                   # Liquid AI official cookbook (reference)
│   └── examples/satellite-vlm/
│       ├── prepare_vrsbench.py       # Data preparation script
│       └── configs/                  # Reference configs
├── leap-finetune/                    # Liquid AI fine-tuning framework
├── SimSat/                           # DPhi Space satellite simulator (Docker)
└── README.md
```

## Hardware Target

CODA-DVD is designed for the **NVIDIA Jetson Orin NX 16GB** (hackathon prize hardware):
- LFM2.5-VL-1.6B-8bit: 2.6 GB model weight → fits comfortably
- Inference: expected 100-200 tok/s on Orin (vs 220 tok/s on M3 Max)
- Full pipeline: ~30s per observation pass including Sentinel fetch
- Power budget: 15-25W — within CubeSat solar panel capacity

## Technologies Used

- **LFM2.5-VL** (Liquid AI) — Vision-Language Model for on-board decisions
- **SimSat** (DPhi Space) — Satellite simulator providing real Sentinel-2 + Mapbox data
- **Sentinel-2** (Copernicus) — 10m multispectral satellite imagery
- **MLX** (Apple) — Efficient inference on Apple Silicon
- **Modal** — Serverless GPU cloud for fine-tuning
- **VRSBench** — Remote sensing visual grounding benchmark (NeurIPS 2024)

## License

MIT

---

*Built for [AI in Space Hackathon #05](https://lu.ma/AI_in_Space_5) by Liquid AI x DPhi Space*
