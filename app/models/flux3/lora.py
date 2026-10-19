"""LoRA key conversion for FLUX 3's native DiT architecture."""


_ROOT_PREFIXES = {"transformer", "diffusion_model"}
_ATTENTION_PROJECTIONS = {
    "to_q": "q_proj",
    "to_k": "k_proj",
    "to_v": "v_proj",
}
_FEED_FORWARD_PROJECTIONS = {
    "linear_in": "mlp_in",
    "linear_out": "mlp_out",
}
_LORA_SUFFIXES = (
    ".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight",
    ".alpha", ".dora_scale", ".lokr_w1", ".lokr_w2", ".diff", ".diff_b",
)


def _normalize_module(module, module_names):
    """Resolve public Diffusers or wrapped names to an actual native module."""
    resolved = module
    if resolved in module_names:
        return resolved
    while True:
        for prefix in ("base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer."):
            if resolved.startswith(prefix):
                resolved = resolved[len(prefix):]
                if resolved in module_names:
                    return resolved
                break
        else:
            break

    parts = resolved.split(".")
    if parts and parts[0] in _ROOT_PREFIXES:
        parts = parts[1:]

    if parts[:2] == ["mode_blocks", "image"] or parts[:2] == ["mode_blocks", "image_cond"]:
        parts[0] = "content_mode_blocks"
    elif parts and parts[0] == "single_transformer_blocks":
        parts[0] = "single_blocks"
    elif parts[:2] == ["x_embedders", "image"] or parts[:2] == ["x_embedders", "image_cond"]:
        parts[0] = "emb_in"
    elif parts and parts[0] == "context_embedder":
        parts[0] = "txt_in"
    elif parts and parts[0] == "proj_out":
        parts[:1] = ["final_layer", "image", "linear"]

    index = 0
    while index < len(parts):
        if parts[index] == "attn" and index + 1 < len(parts):
            projection = parts[index + 1]
            if projection in _ATTENTION_PROJECTIONS:
                parts[index:index + 2] = [_ATTENTION_PROJECTIONS[projection]]
            elif projection == "to_out" and index + 2 < len(parts) and parts[index + 2] == "0":
                parts[index:index + 3] = ["attn_out"]
        elif parts[index] == "ff" and index + 1 < len(parts):
            projection = _FEED_FORWARD_PROJECTIONS.get(parts[index + 1])
            if projection:
                parts[index:index + 2] = [projection]
        index += 1

    resolved = ".".join(parts)
    return resolved if resolved in module_names else None


def preprocess_lora_state_dict(model_type, state_dict, module_names, split_linear_modules_map=None):
    """Map public Diffusers FLUX 3 adapters onto Maestro's native DiT.

    Unrecognized target names stay unchanged so MMGP's compatibility check can
    reject an adapter that belongs to another architecture or unsupported format.
    """
    module_names = set(module_names)
    for fused, spec in (split_linear_modules_map or {}).items():
        for name in tuple(module_names):
            if any(name == split or name.endswith("." + split) for split in spec["mapped_modules"]):
                prefix = name[:-len(name.rsplit(".", 1)[-1])]
                module_names.add(prefix + fused)

    encoded_names = {}
    for name in module_names:
        encoded = name.replace(".", "_")
        if encoded in encoded_names and encoded_names[encoded] != name:
            raise ValueError(f"Ambiguous FLUX 3 underscore LoRA module name: {encoded!r}.")
        encoded_names[encoded] = name

    converted = {}
    for key, value in state_dict.items():
        normalized_key = key.replace(".default.weight", ".weight")
        suffix = next((candidate for candidate in _LORA_SUFFIXES if normalized_key.endswith(candidate)), None)
        if suffix is None:
            converted[key] = value
            continue

        source_module = normalized_key[:-len(suffix)]
        if source_module.startswith("lora_unet_"):
            encoded_name = source_module[len("lora_unet_"):]
            module = encoded_names.get(encoded_name)
            if module is None:
                for prefix in ("base_model_model_", "model_diffusion_model_", "diffusion_model_", "transformer_"):
                    if encoded_name.startswith(prefix):
                        module = encoded_names.get(encoded_name[len(prefix):])
                        if module is not None:
                            break
        else:
            module = _normalize_module(source_module, module_names)
        mapped_key = f"{module}{suffix}" if module is not None else normalized_key
        if mapped_key in converted:
            raise ValueError(f"Flux 3 LoRA key conversion collision for {mapped_key!r}.")
        converted[mapped_key] = value
    return converted
