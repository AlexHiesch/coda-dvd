"""
Run vessel grounding fine-tuning on Modal H100.
Standalone script that avoids Python 3.14 → 3.12 serialization issues
by running leap-finetune as a subprocess inside the container.
"""
import pathlib

import modal
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LEAP_ROOT = _REPO_ROOT / "leap-finetune"

VOLUME_NAME = "satellite-vlm"
MOUNT_POINT = "/satellite-vlm"

# Load config to embed in the image
CONFIG_PATH = _REPO_ROOT / "fine_tuning" / "vessel_grounding_modal.yaml"
with open(CONFIG_PATH) as f:
    config_dict = yaml.safe_load(f)

# Strip modal section for in-container use
train_config = {k: v for k, v in config_dict.items() if k != "modal"}
train_config.setdefault("training_config", {})["output_dir"] = MOUNT_POINT

app = modal.App("coda-dvd-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Build image with leap-finetune installed
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("uv")
    .add_local_file(str(_LEAP_ROOT / "pyproject.toml"), remote_path="/app/pyproject.toml", copy=True)
    .add_local_file(str(_LEAP_ROOT / "uv.lock"), remote_path="/app/uv.lock", copy=True)
    .run_commands(
        "cd /app && uv export --frozen --no-dev --no-emit-project --no-hashes"
        " > requirements.txt"
        " && grep -v flash-attn requirements.txt > requirements-modal.txt"
        " && uv pip install --system -r requirements-modal.txt",
    )
    .add_local_dir(str(_LEAP_ROOT / "src" / "leap_finetune"), remote_path="/app/src/leap_finetune", copy=True)
    .env({
        "PYTHONPATH": "/app/src",
        "LEAP_FINETUNE_DIR": "/app",
        "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "1",
    })
)


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={MOUNT_POINT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(config_yaml: str):
    import os
    import subprocess
    import sys
    import tempfile

    os.environ["OUTPUT_DIR"] = MOUNT_POINT
    os.environ.pop("HF_HUB_OFFLINE", None)

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        tmp_path = f.name

    # Run leap-finetune directly
    sys.argv = ["leap-finetune", tmp_path]
    from leap_finetune import main as leap_main
    leap_main()


@app.local_entrypoint()
def main():
    config_str = yaml.dump(train_config)
    print(f"Launching fine-tuning on Modal H100...")
    print(f"  Model: {train_config.get('model_name')}")
    print(f"  Dataset: {train_config.get('dataset', {}).get('path')}")
    print(f"  Epochs: {train_config.get('training_config', {}).get('num_train_epochs')}")
    print(f"  LoRA: r={config_dict.get('peft_config', {}).get('lora_r')}")
    train.remote(config_str)
    print("\nFine-tuning complete! Check Modal volume for outputs.")
