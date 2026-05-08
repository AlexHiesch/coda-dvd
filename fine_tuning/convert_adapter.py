"""
Convert PyTorch PEFT LoRA adapter to MLX format for mlx-vlm.

Downloads the adapter from Modal volume, converts weights to MLX safetensors,
and creates the adapter_config.json expected by mlx-vlm.
"""
import json
import shutil
from pathlib import Path

import numpy as np


def convert_peft_to_mlx(peft_dir: Path, output_dir: Path):
    """Convert a HuggingFace PEFT adapter directory to mlx-vlm format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PEFT adapter_config.json
    with open(peft_dir / "adapter_config.json") as f:
        peft_config = json.load(f)

    r = peft_config.get("r", peft_config.get("rank", 16))
    lora_alpha = peft_config.get("lora_alpha", 32)
    target_modules = peft_config.get("target_modules", [])

    # Create mlx-vlm adapter_config.json
    mlx_config = {
        "rank": r,
        "alpha": lora_alpha,
        "dropout": peft_config.get("lora_dropout", 0.0),
        "target_modules": target_modules,
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)

    # Convert weights from PyTorch safetensors to MLX safetensors
    # Both use the same safetensors format, so we can often just copy
    import safetensors.torch
    import mlx.core as mx
    from mlx.utils import save_safetensors

    # Find adapter weight files
    weight_files = list(peft_dir.glob("adapter_model*.safetensors"))
    if not weight_files:
        # Try .bin format
        weight_files = list(peft_dir.glob("adapter_model*.bin"))

    if not weight_files:
        raise FileNotFoundError(f"No adapter weights found in {peft_dir}")

    all_weights = {}
    for wf in weight_files:
        if wf.suffix == ".safetensors":
            tensors = safetensors.torch.load_file(str(wf))
        else:
            import torch
            tensors = torch.load(str(wf), map_location="cpu")

        for key, tensor in tensors.items():
            # Convert key format: base_model.model.X.lora_A.weight -> X.lora_A
            clean_key = key.replace("base_model.model.", "")
            clean_key = clean_key.replace(".lora_A.weight", ".lora_A")
            clean_key = clean_key.replace(".lora_B.weight", ".lora_B")
            clean_key = clean_key.replace(".lora_A.default.weight", ".lora_A")
            clean_key = clean_key.replace(".lora_B.default.weight", ".lora_B")

            # Convert to numpy then to mlx
            if hasattr(tensor, "numpy"):
                np_array = tensor.float().numpy()
            else:
                np_array = np.array(tensor)
            all_weights[clean_key] = mx.array(np_array)

    # Save as MLX safetensors
    save_safetensors(str(output_dir / "adapters.safetensors"), all_weights)
    print(f"Converted {len(all_weights)} tensors to {output_dir / 'adapters.safetensors'}")
    print(f"Config: rank={r}, alpha={lora_alpha}, targets={target_modules}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--peft-dir", required=True, help="Path to PEFT adapter directory")
    parser.add_argument("--output-dir", default="./fine_tuning/mlx_adapter", help="Output directory")
    args = parser.parse_args()

    convert_peft_to_mlx(Path(args.peft_dir), Path(args.output_dir))
