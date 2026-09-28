"""Normalize flattened Kohya/Musubi LoRA module names for MiniMax H3."""


_MMGP_PREFIXES = ("diffusion_model.", "transformer.")

# MMGP accepts these spellings for the same low-rank factors. They are used
# only to detect collisions; source keys and suffixes are never rewritten.
_FACTOR_SUFFIX_ALIASES = (
    (".lora_down.weight", ".lora_A.weight"),
    (".lora_up.weight", ".lora_B.weight"),
    (".lora_down.default.weight", ".lora_A.weight"),
    (".lora_up.default.weight", ".lora_B.weight"),
    (".lora_A.default.weight", ".lora_A.weight"),
    (".lora_B.default.weight", ".lora_B.weight"),
    (".lora.down.weight", ".lora_A.weight"),
    (".lora.up.weight", ".lora_B.weight"),
    (".lora.down.default.weight", ".lora_A.weight"),
    (".lora.up.default.weight", ".lora_B.weight"),
    (".lora.A.weight", ".lora_A.weight"),
    (".lora.B.weight", ".lora_B.weight"),
    (".lora.A.default.weight", ".lora_A.weight"),
    (".lora.B.default.weight", ".lora_B.weight"),
)


def _flattened_module_token(key: str) -> tuple[str, str] | None:
    """Return a flattened module token and untouched adapter suffix, if any."""
    if key.startswith("lora_unet_"):
        remainder = key[len("lora_unet_"):]
    elif key.startswith("token_refiner_blocks_"):
        remainder = key
    else:
        return None

    module_token, separator, suffix = remainder.partition(".")
    if not separator:
        return None
    return module_token, suffix


def _flattened_module_paths(transformer) -> dict[str, set[str]]:
    """Index only module paths present in this H3 model, plus virtual QKV."""
    named_modules = getattr(transformer, "named_modules", None)
    if not callable(named_modules):
        return {}

    logical_paths = set()
    attentions = []
    for path, module in named_modules():
        if not path:
            continue
        logical_paths.add(path)
        if path.endswith(".attn"):
            attentions.append((path, module))

    for path, attention in attentions:
        has_fused_qkv = hasattr(attention, "qkv_proj")
        has_split_qkv = all(
            hasattr(attention, projection)
            for projection in ("q_proj", "k_proj", "v_proj")
        )
        if has_fused_qkv or has_split_qkv:
            # MMGP can split the backbone's logical qkv_proj into Q/K/V
            # modules. Keep the source-facing target name available either way.
            logical_paths.add(f"{path}.qkv_proj")

    flattened = {}
    for path in logical_paths:
        token = path.replace(".", "_")
        flattened.setdefault(token, set()).add(path)
    return flattened


def _mmgp_collision_key(key: str) -> str:
    """Canonicalize only MMGP's known module/factor aliases for comparison."""
    for prefix in _MMGP_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    for source, target in _FACTOR_SUFFIX_ALIASES:
        if key.endswith(source):
            return key[:-len(source)] + target
    return key


def normalize_flattened_lora_names(state_dict, transformer) -> dict:
    """Resolve flat H3 module tokens without guessing at underscore boundaries.

    A token is converted only when it uniquely matches a module name from the
    live transformer. Unknown tokens remain untouched so MMGP can report them.
    Tensor values and every adapter suffix remain unchanged.
    """
    source = dict(state_dict)
    candidates = {}
    for key in source:
        flattened = _flattened_module_token(key)
        if flattened is not None:
            candidates[key] = flattened
    if not candidates:
        return source

    module_paths = _flattened_module_paths(transformer)
    normalized = {}
    owners = {}
    for key, value in source.items():
        target = key
        flattened = candidates.get(key)
        was_mapped = False
        if flattened is not None:
            token, suffix = flattened
            matches = module_paths.get(token, set())
            if len(matches) > 1:
                paths = ", ".join(sorted(matches))
                raise ValueError(
                    f"Ambiguous flattened H3 LoRA module {token!r}: {paths}"
                )
            if len(matches) == 1:
                target = f"{next(iter(matches))}.{suffix}"
                was_mapped = True

        collision_key = _mmgp_collision_key(target)
        previous = owners.get(collision_key)
        if previous is not None and (was_mapped or previous[1]):
            raise ValueError(
                "Flattened H3 LoRA key collision: "
                f"{previous[0]!r} and {key!r} resolve to the same MMGP target"
            )
        owners[collision_key] = (key, was_mapped)
        normalized[target] = value
    return normalized
