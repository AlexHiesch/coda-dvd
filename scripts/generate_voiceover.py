#!/usr/bin/env python3
"""
generate_voiceover.py — Add Azure Neural TTS voiceover to CODA-DVD demo video.

Pattern adapted from keynote_rdus. Generates per-segment WAV files,
mixes with precise timing via FFmpeg filter_complex, muxes onto video.

Usage:
  python scripts/generate_voiceover.py              # full pipeline
  python scripts/generate_voiceover.py --dry-run    # print timeline only
  python scripts/generate_voiceover.py --audio-only # TTS only, skip mux

Voice: Azure Cognitive Services  voice=en-US-OnyxTurboMultilingualNeural
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ── Voice config ─────────────────────────────────────────────────────────────
VOICE          = "en-US-OnyxTurboMultilingualNeural"
SPEECH_KEY     = os.environ.get("AZURE_SPEECH_KEY", "")
SERVICE_REGION = os.environ.get("AZURE_SPEECH_REGION", "westeurope")


def _plain(text: str) -> str:
    if text.strip().startswith("<"):
        return re.sub(r"<[^>]+>", " ", text).strip()
    return text


# ── Scene definitions ────────────────────────────────────────────────────────
# Format: (start_ms, "narration text")
# Timings are approximate — adjust after first recording.
# Run --dry-run to preview timeline before generating.

# Video timeline (composite):
#   0.0s –  8.0s  01_architecture  (8s still)
#   8.0s – 13.0s  02_sentinel      (5s still)
#  13.0s – 18.0s  03_mapbox        (5s still)
#  18.0s – 25.0s  04_downlink      (7s still)
#  25.0s – 30.0s  05_singapore     (5s still)
#  30.0s – 44.2s  06_terminal      (14.2s recording)
# Total: ~44.2s

# Each segment: (slide_start_ms, slide_end_ms, "narration text")
# The script will generate TTS, measure actual duration,
# and center/anchor each segment within its slide window.

SCENES = {
    "01_CompositeDemo": {
        "video": "media/renders/coda_dvd_demo_composite.mp4",
        "segments": [
            # ── Architecture slide (0-8s) ──
            (500,   7500,  "CODA-DVD. Dark Vessel Detection from orbit. "
                           "Three-stage on-board AI pipeline."),

            # ── Sentinel slide (8-13s) ──
            (8500,  12500, "Sentinel-2 at ten meters. "
                           "Regions of interest in Hamburg."),

            # ── Mapbox slide (13-18s) ──
            (13500, 17500, "Mapbox high-res. "
                           "Vessels classified by the VLM."),

            # ── Downlink slide (18-25s) ──
            (18500, 24500, "Downlink: just fifteen kilobytes. "
                           "Ninety-eight percent bandwidth saved."),

            # ── Singapore slide (25-30s) ──
            (25500, 29500, "Singapore anchorage. "
                           "Real vessel targets at scale."),

            # ── Terminal demo (30-44s) ──
            (31000, 36000, "Live demo. Three passes processed on-board."),
            (37500, 43500, "Vessel confirmed. Mission complete."),
        ],
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"\nERROR: {' '.join(cmd[:3])} ...\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result


def media_duration_ms(path):
    r = run(["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)])
    return int(float(r.stdout.strip()) * 1000)


def generate_tts(text, out_path):
    if out_path.exists():
        print(f"    [cache]  {out_path.name}")
        return out_path

    plain = _plain(text)
    preview = plain[:70] + ("…" if len(plain) > 70 else "")
    print(f"    [tts]    {out_path.name}")
    print(f"             \"{preview}\"")

    import azure.cognitiveservices.speech as speechsdk
    config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
    config.speech_synthesis_voice_name = VOICE
    synth = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)

    is_ssml = text.strip().startswith("<speak")
    result = (synth.speak_ssml_async(text) if is_ssml
              else synth.speak_text_async(text)).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        out_path.write_bytes(result.audio_data)
    else:
        details = ""
        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details.error_details
        print(f"\nERROR: TTS failed: {preview}\n{details}", file=sys.stderr)
        sys.exit(1)
    return out_path


def build_audio_track(segments_with_paths, video_ms, out_path):
    """Build a single audio track with segments placed at their start times."""
    n = len(segments_with_paths)
    dur_s = video_ms / 1000.0

    filter_parts = [
        f"anullsrc=r=16000:cl=mono[base_raw];"
        f"[base_raw]atrim=0:{dur_s}[base]"
    ]
    for i, (start_ms, audio_path) in enumerate(segments_with_paths):
        # Convert to 16kHz mono to match TTS output, then delay
        filter_parts.append(
            f"[{i}:a]aresample=16000,aformat=channel_layouts=mono,"
            f"adelay={start_ms}|{start_ms}[a{i}]"
        )

    mix_inputs = "[base]" + "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={n + 1}:normalize=0:dropout_transition=0[aout]"
    )

    inputs = []
    for _, audio_path in segments_with_paths:
        inputs += ["-i", str(audio_path)]

    cmd = (["ffmpeg", "-y"] + inputs + [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[aout]",
        "-t", str(dur_s),
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ])
    print(f"  Building audio track → {out_path.name}")
    run(cmd)


def mux(video_path, audio_path, srt_path, output_path):
    """Mux video + audio, burning in subtitles if SRT exists."""
    if srt_path and srt_path.exists():
        # Re-encode video to burn in subtitles
        # Use absolute path and escape for FFmpeg subtitle filter
        abs_srt = str(srt_path.resolve()).replace(":", r"\:").replace("'", r"\'")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
            "-vf", f"subtitles={abs_srt}:force_style="
                   f"'FontName=Helvetica,FontSize=22,PrimaryColour=&Hffffff&,"
                   f"OutlineColour=&H000000&,Outline=2,MarginV=40'",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]
    print(f"  Muxing → {output_path.name}")
    run(cmd)


def generate_srt(segments_info, out_path):
    """Generate SRT subtitle file from segment timing info.
    
    segments_info: list of (start_ms, duration_ms, text)
    """
    lines = []
    for i, (start_ms, dur_ms, text) in enumerate(segments_info, 1):
        end_ms = start_ms + dur_ms
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(start_ms)} --> {_fmt_srt_time(end_ms)}")
        lines.append(_plain(text))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Captions → {out_path.name} ({len(segments_info)} entries)")


def _fmt_srt_time(ms):
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1000
    ms_r = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_r:03d}"


def concat_scenes(scene_paths, output_path):
    """Concatenate multiple scene MP4s into one final video."""
    list_file = Path("media/final/_concat_list.txt")
    with open(list_file, "w") as f:
        for p in scene_paths:
            f.write(f"file '{p.resolve()}'\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    print(f"\n  Concatenating {len(scene_paths)} scenes → {output_path.name}")
    run(cmd)
    list_file.unlink()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scene", metavar="PREFIX")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--no-captions", action="store_true",
                        help="Skip subtitle generation")
    args = parser.parse_args()

    audio_dir = Path("media/audio")
    final_dir = Path("media/final")
    if not args.dry_run:
        audio_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

    scenes = SCENES
    if args.scene:
        scenes = {k: v for k, v in SCENES.items() if k.startswith(args.scene)}
        if not scenes:
            print(f"No scene matching '{args.scene}'. Keys: {list(SCENES.keys())}")
            sys.exit(1)

    final_scene_paths = []

    for scene_key, scene in scenes.items():
        video_path = Path(scene["video"])
        segments = scene["segments"]

        print(f"\n{'═' * 62}")
        print(f"  {scene_key}")
        print(f"  video : {video_path}")

        if not segments:
            print(f"  segs  : 0  →  skipped")
            continue

        print(f"  segs  : {len(segments)}")
        print(f"{'═' * 62}")

        if not video_path.exists():
            print(f"  ⚠  Video not found: {video_path}")
            print(f"     Run ./scripts/record_demo.sh and ./scripts/render_demo.sh first")
            continue

        video_ms = media_duration_ms(video_path)
        print(f"  video duration: {video_ms / 1000:.1f}s\n")

        if args.dry_run:
            prev_end = 0
            for i, (start_ms, end_ms, text) in enumerate(segments):
                plain = _plain(text)
                words = len(plain.split())
                est_s = words / 2.5
                est_end = start_ms + int(est_s * 1000)
                gap = start_ms - prev_end
                window = end_ms - start_ms
                flag = ""
                if gap < 0:
                    flag = " ⚠ OVERLAP"
                if est_end > end_ms:
                    flag += " ⚠ OVERRUN"
                print(f"  [{i+1}] {start_ms/1000:6.2f}s → ~{est_end/1000:.2f}s  "
                      f"(window {window/1000:.1f}s, {words}w ~{est_s:.1f}s, "
                      f"gap {gap/1000:+.2f}s){flag}")
                print(f"       \"{plain[:80]}{'…' if len(plain)>80 else ''}\"")
                prev_end = est_end
            if prev_end > video_ms:
                print(f"\n  ⚠ Audio overruns video by {(prev_end - video_ms)/1000:.1f}s!")
            continue

        # Generate TTS segments
        segments_with_paths = []
        segments_info = []  # for SRT generation
        for i, (start_ms, end_ms, text) in enumerate(segments):
            wav = audio_dir / f"{scene_key}_{i:02d}.wav"
            generate_tts(text, wav)
            # Measure actual duration
            dur_ms = media_duration_ms(wav)
            if dur_ms > (end_ms - start_ms):
                print(f"    ⚠ Seg {i}: {dur_ms/1000:.1f}s > window "
                      f"{(end_ms-start_ms)/1000:.1f}s — will be clipped")
            segments_with_paths.append((start_ms, wav))
            segments_info.append((start_ms, min(dur_ms, end_ms - start_ms), text))

        # Build combined audio
        combined = audio_dir / f"{scene_key}_combined.m4a"
        build_audio_track(segments_with_paths, video_ms, combined)

        # Generate SRT captions
        srt_path = None
        if not args.no_captions:
            srt_path = final_dir / f"{scene_key}.srt"
            generate_srt(segments_info, srt_path)

        if args.audio_only:
            continue

        # Mux with captions
        output = final_dir / f"{scene_key}.mp4"
        mux(video_path, combined, srt_path, output)
        final_scene_paths.append(output)
        print(f"  ✓ {output}")

    # Concatenate all scenes
    if len(final_scene_paths) > 1 and not args.dry_run and not args.audio_only:
        final_output = final_dir / "coda_dvd_demo_final.mp4"
        concat_scenes(final_scene_paths, final_output)
        dur = media_duration_ms(final_output) / 1000
        print(f"\n  ✓ Final video: {final_output} ({dur:.1f}s)")
    elif final_scene_paths:
        print(f"\n  ✓ Final video: {final_scene_paths[0]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
