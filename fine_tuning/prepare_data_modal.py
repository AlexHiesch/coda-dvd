"""
Prepare VRSBench grounding data on Modal.
Workaround for Python 3.14 local → 3.12 remote serialization mismatch.
Runs the preparation as a subprocess in the container.
"""
import modal

VOLUME_NAME = "satellite-vlm"
MOUNT_POINT = "/satellite-vlm"

app = modal.App("coda-dvd-data-prep")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "tqdm")
    .add_local_file(
        "liquid-cookbook/examples/satellite-vlm/prepare_vrsbench.py",
        "/app/prepare_vrsbench.py",
        copy=True,
    )
)


@app.function(
    image=image,
    volumes={MOUNT_POINT: volume},
    timeout=3600,
)
def prepare_grounding(limit: int = 0):
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "/app/prepare_vrsbench.py",
        "--task", "grounding",
        "--data-dir", f"{MOUNT_POINT}/data/vrsbench",
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main(limit: int = 0):
    prepare_grounding.remote(limit)
    print(f"\nData ready in Modal volume '{VOLUME_NAME}'.")
    print("Next: modal run fine_tuning/run_finetune_modal.py")
