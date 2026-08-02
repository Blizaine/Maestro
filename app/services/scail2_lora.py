"""Utilities for converting official SCAIL-2 SAT LoRAs to Wan format.

The key mapping follows zai-org/SCAIL-2's Apache-2.0 licensed
``convert_lora.py`` from the ``wan-scail2`` branch. Maestro verifies the
official checkpoint's pinned SHA-256 before this module is called, and this
module verifies it again before allowing PyTorch to deserialize the SAT file.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class ModuleParser:
    def __init__(self, key: str):
        self.key = key
        self.modules = key.split(".")
        self.idx = 0

    def match(self, pattern: str) -> bool:
        parts = pattern.split(".")
        if self.modules[self.idx:self.idx + len(parts)] != parts:
            return False
        self.idx += len(parts)
        return True

    def step(self) -> str:
        if self.idx >= len(self.modules):
            raise ValueError(f"Unexpected end of LoRA key: {self.key}")
        value = self.modules[self.idx]
        self.idx += 1
        return value

    def eof(self) -> bool:
        return self.idx == len(self.modules)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sat_module_to_wan_prefix(key: str) -> str:
    parser = ModuleParser(key)
    if not parser.match("model.diffusion_model.transformer.layers"):
        raise ValueError(f"Unsupported SCAIL-2 LoRA key: {key}")

    block_idx = parser.step()
    modules = ["diffusion_model", "blocks", block_idx]

    if parser.match("attention"):
        modules.append("self_attn")
        if parser.match("query_key_value.lora_layer"):
            qkv_idx = parser.step()
            qkv_map = {"0": "q", "1": "k", "2": "v"}
            if qkv_idx not in qkv_map:
                raise ValueError(f"Unsupported QKV LoRA index in {key}")
            modules.append(qkv_map[qkv_idx])
        elif parser.match("dense.lora_layer"):
            modules.append("o")
        else:
            raise ValueError(f"Unsupported self-attention LoRA key: {key}")
    elif parser.match("cross_attention"):
        modules.append("cross_attn")
        if parser.match("query.lora_layer"):
            modules.append("q")
        elif parser.match("key_value.lora_layer"):
            kv_idx = parser.step()
            kv_map = {"0": "k", "1": "v"}
            if kv_idx not in kv_map:
                raise ValueError(f"Unsupported cross-attention KV index in {key}")
            modules.append(kv_map[kv_idx])
        elif parser.match("dense.lora_layer"):
            modules.append("o")
        else:
            raise ValueError(f"Unsupported cross-attention LoRA key: {key}")
    elif parser.match("mlp"):
        modules.append("ffn")
        if parser.match("dense_h_to_4h.lora_layer"):
            modules.append("0")
        elif parser.match("dense_4h_to_h.lora_layer"):
            modules.append("2")
        else:
            raise ValueError(f"Unsupported MLP LoRA key: {key}")
    else:
        raise ValueError(f"Unsupported transformer LoRA key: {key}")

    if not parser.eof():
        raise ValueError(f"Unparsed LoRA key suffix in {key}")
    return ".".join(modules)


def convert_key(key: str) -> str:
    if key.endswith(".down.weight"):
        sat_module = key[:-len(".down.weight")]
        suffix = ".lora_down.weight"
    elif key.endswith(".up.weight"):
        sat_module = key[:-len(".up.weight")]
        suffix = ".lora_up.weight"
    elif key.endswith(".diff_b"):
        sat_module = key[:-len(".diff_b")]
        suffix = ".diff_b"
    else:
        raise ValueError(f"Unsupported SCAIL-2 LoRA tensor key: {key}")
    return _sat_module_to_wan_prefix(sat_module) + suffix


def convert_state_dict(state_dict, dtype=None):
    converted = {}
    for key, value in state_dict.items():
        if ".lora_layer" not in key:
            continue
        new_key = convert_key(key)
        if new_key in converted:
            raise ValueError(f"Duplicate converted key {new_key} from {key}")
        tensor = value.detach().cpu().contiguous()
        if dtype is not None:
            tensor = tensor.to(dtype)
        converted[new_key] = tensor
    if not converted:
        raise ValueError("No SCAIL-2 LoRA tensors were found in the checkpoint")
    return converted


def _load_module_state(path: Path):
    import torch

    # This file is a pickle-based SAT checkpoint. Callers must verify its
    # pinned official hash before deserialization; convert_scail2_sat_lora
    # enforces that contract immediately below.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for name in ("module", "state_dict", "model"):
            state_dict = checkpoint.get(name)
            if isinstance(state_dict, dict):
                return state_dict
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Could not find a tensor state dict in {path}")


def convert_scail2_sat_lora(
    input_path: str | Path,
    output_path: str | Path,
    *,
    expected_sha256: str,
) -> int:
    """Convert one hash-pinned official SAT LoRA to Wan safetensors.

    Returns the number of converted tensors. The output may use any temporary
    extension; safetensors identifies the format by contents rather than name.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    actual_sha256 = sha256_file(input_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "Official SCAIL-2 Relighting LoRA failed SHA-256 verification "
            f"(expected {expected_sha256}, got {actual_sha256})"
        )

    # Importing safetensors pulls in Torch. Keep both out of the process until
    # the pickle-based source has passed the immutable upstream hash check.
    from safetensors.torch import save_file

    state_dict = _load_module_state(input_path)
    converted = convert_state_dict(state_dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    return len(converted)
