#!/usr/bin/env python3
"""Prepare the Qwen-Image-2.1-Turbo transformer for Maestro's Draft model.

Downloads (or reuses) the official diffusers checkpoint and merges its sharded
transformer into the single safetensors file named by
app/defaults/qwen_image_21_7B_turbo.json. The text encoder, VAE and processor are
shared with the existing Qwen Image 2.1 model and are not duplicated.

Usage:
    python scripts/prepare_qwen21_turbo.py [--ckpts app/ckpts] [--revision <sha>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = "Qwen/Qwen-Image-2.1-Turbo"
TARGET = "qwen_image_21_7B_turbo_bf16.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpts", type=Path, default=Path(__file__).resolve().parents[1] / "app" / "ckpts")
    parser.add_argument("--revision", default=None, help="Pin a Hugging Face revision of the Turbo repo.")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    snapshot = Path(snapshot_download(REPO, revision=args.revision,
                                      allow_patterns=["transformer/*", "model_index.json"]))
    shards = sorted((snapshot / "transformer").glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No transformer shards found in {snapshot}")

    tensors = {}
    for shard in shards:
        with safe_open(str(shard), "pt") as handle:
            for key in handle.keys():
                if key in tensors:
                    raise SystemExit(f"Duplicate tensor {key!r} across shards")
                tensors[key] = handle.get_tensor(key)

    sigmas = json.loads((snapshot / "model_index.json").read_text()).get("sample_sigmas")
    args.ckpts.mkdir(parents=True, exist_ok=True)
    target = args.ckpts / TARGET
    save_file(tensors, str(target), metadata={"source": REPO, "snapshot": snapshot.name,
                                              "sample_sigmas": json.dumps(sigmas)})

    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    print(json.dumps({"target": str(target), "tensors": len(tensors), "snapshot": snapshot.name,
                      "sample_sigmas": sigmas, "sha256": digest.hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
