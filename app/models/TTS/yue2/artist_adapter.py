"""Use validated YuE2 AR/NAR LoRAs without modifying quantized base weights."""
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def target_shapes(branch):
    attention = "self_attn" if branch == "ar" else "nar_self_attn"
    mlp = "mlp" if branch == "ar" else "nar_mlp"
    for layer in range(28):
        for group, names in ((attention, (("q_proj", 2048, 2048), ("k_proj", 2048, 1024),
                                          ("v_proj", 2048, 1024), ("o_proj", 2048, 2048))),
                             (mlp, (("gate_proj", 2048, 6144), ("up_proj", 2048, 6144),
                                    ("down_proj", 6144, 2048)))):
            for name, input_size, output_size in names:
                yield f"model.layers.{layer}.{group}.{name}", input_size, output_size


def validate_tensors(tensors, branch, *, require_io=True):
    expected = set()
    ranks = set()
    for name, input_size, output_size in target_shapes(branch):
        a, b = tensors.get(name + ".A"), tensors.get(name + ".B")
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError(f"The {branch.upper()} adapter is missing {name}")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != input_size or b.shape != (output_size, a.shape[0]):
            raise ValueError(f"Unsupported YuE2 adapter dimensions at {name}")
        if not 1 <= a.shape[0] <= 128 or not a.is_floating_point() or not b.is_floating_point():
            raise ValueError("YuE2 music adapters require floating-point matrices of rank 1–128")
        ranks.add(a.shape[0])
        expected.update((name + ".A", name + ".B"))
    if len(ranks) != 1:
        raise ValueError("The YuE2 artist adapter must use one rank across its layers")
    if branch == "nar" and require_io:
        for name, shape in (("vae2llm.weight", (2048, 64)), ("vae2llm.bias", (2048,)),
                            ("llm2vae.weight", (64, 2048)), ("llm2vae.bias", (64,))):
            if not isinstance(tensors.get(name), torch.Tensor) or tuple(tensors[name].shape) != shape or not tensors[name].is_floating_point():
                raise ValueError(f"The NAR adapter is missing a compatible {name}")
            expected.add(name)
    if set(tensors) != expected:
        raise ValueError("The music adapter contains unsupported or unexpected tensor targets")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        raise ValueError("Music adapter weights contain non-finite values")
    return next(iter(ranks))


def read_upstream_adapter(path, branch, *, require_io=True):
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        tensors = load_file(str(path), device="cpu")
        # Mothersuperior's named safetensors use the same unfused matrices as
        # the older positional .pt files, with a different key spelling.
        if any(key.startswith("layers.") for key in tensors):
            tensors = {("model." + key.replace(".lora_A", ".A").replace(".lora_B", ".B")
                        if key.startswith("layers.") else key): value for key, value in tensors.items()}
    elif path.suffix.lower() == ".pt":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        weights = checkpoint.get("lora") if isinstance(checkpoint, dict) else None
        if not isinstance(weights, (list, tuple)) or len(weights) != 392:
            raise ValueError("Expected an upstream YuE2 artist checkpoint with 196 A/B adapter pairs")
        tensors = {}
        for index, (name, _, _) in enumerate(target_shapes(branch)):
            tensors[name + ".A"], tensors[name + ".B"] = weights[index * 2:index * 2 + 2]
        if branch == "nar":
            for name in ("vae2llm", "llm2vae"):
                for key, value in (checkpoint.get("io", {}).get(name, {})).items():
                    tensors[f"{name}.{key}"] = value
    else:
        raise ValueError("Import a YuE2 .pt or .safetensors artist adapter")
    validate_tensors(tensors, branch, require_io=require_io)
    return tensors


@contextmanager
def active_artist(pipeline, custom_settings):
    """Single-bundle compatibility for reconstruction and older callers."""
    with active_artists(pipeline, custom_settings) as artists:
        if len(artists) > 1:
            raise ValueError("This comparison requires one music LoRA")
        yield artists[0] if artists else None


@contextmanager
def active_artists(pipeline, custom_settings):
    from services.music_styles import generation_styles, style_directory
    from services.music_contracts import adapter_contract, tokenizer_pair
    from .music_assets import ASSETS, ensure_asset

    settings = custom_settings or {}
    selected = generation_styles(settings, verify=True)
    if not selected:
        yield []
        return
    adapters, metadata = [], []
    for item in selected:
        style_id, strength, manifest = item['id'], item['strength'], item['manifest']
        sources = {branch: style_directory(style_id) / manifest[branch]['file'] for branch in ('ar', 'nar')}
        mode = adapter_contract(manifest)
        nar_asset = tokenizer_pair(manifest)['nar']
        if settings.get('base_acoustic') is True:
            if manifest.get('adapted_pair'):
                raise ValueError('Use the original / before / after sound-pair comparison for an adapted tokenizer; a generic decoder would not match its tokens')
            if mode == 'joint':
                sources.pop('nar')  # Joint LoRAs were trained against the base decoder.
            else:
                sources['nar'] = ensure_asset(nar_asset)
        adapters.append({'sources': sources, 'strength': strength, 'mode': mode})
        metadata.append({"id": style_id, "name": manifest["name"], "trigger": manifest["trigger"], "strength": strength,
                         "ar_sha256": manifest["ar"]["sha256"], "nar_sha256": manifest["nar"]["sha256"],
                         "active_nar_sha256": (None if mode == 'joint' else ASSETS[nar_asset]['sha256']) if settings.get('base_acoustic') is True else manifest['nar']['sha256'],
                         "adapter_mode": mode, "tokenizer_revision": manifest["tokenizer_revision"]})
    with active_adapter_stack(pipeline, adapters):
        yield metadata


def artist_style_prompt(style, artists):
    """Keep delivery instructions intact and add each missing training trigger once."""
    style = str(style or '')
    missing = []
    seen = set()
    for artist in artists:
        trigger = artist['trigger'].strip()
        folded = trigger.casefold()
        if trigger and folded not in style.casefold() and folded not in seen:
            missing.append(trigger)
            seen.add(folded)
    return ', '.join([*missing, style] if style else missing)


@contextmanager
def active_adapters(pipeline, sources, strength, *, mode="separate"):
    """Shared inference hooks for validated library bundles and private checkpoints.

    Sources are resolved by the caller, never taken from generation settings.
    Omitting AR uses base conditioning (for an audio-only experiment).
    """
    with active_adapter_stack(pipeline, [{'sources': sources, 'strength': strength, 'mode': mode}]):
        yield


@contextmanager
def active_adapter_stack(pipeline, adapters):
    """Add LoRA deltas; blend full decoder companions once, including their I/O.

    Separate NAR files are complete companions, not independent style deltas.
    Their relative strengths form a convex blend; summing them would double
    the shared decoder adaptation. Joint NAR files are ordinary additive LoRAs.
    A single separate bundle always keeps its decoder, including at AR strength
    zero (used by existing reconstruction comparisons). All-zero selections
    retain an equal decoder blend, with no AR or joint LoRA contribution.
    """
    from services.music_styles import validate_strength
    adapters = [{**item, 'strength': validate_strength(item['strength'])} for item in adapters]
    if any(item['mode'] not in {'separate', 'joint'} for item in adapters):
        raise ValueError("Unsupported music adapter mode")
    companions = [item for item in adapters if item['mode'] == 'separate' and 'nar' in item['sources']]
    companion_total = sum(item['strength'] for item in companions)
    handles, buffers = [], []
    projections = {}
    try:
        # Remove any graphs from a previous song before changing adapter hooks.
        pipeline.engine.release_runtime_allocations()
        # Python hooks must be captured only for this song; generation's finally
        # clears CUDA graphs before these buffers are released.
        for item in adapters:
            sources, strength, mode = item['sources'], item['strength'], item['mode']
            for branch, model in (("ar", pipeline.text_encoder), ("nar", pipeline.transformer)):
                if branch not in sources:
                    continue
                scale = strength if branch == 'ar' or mode == 'joint' else (
                    strength / companion_total if companion_total else 1 / len(companions))
                if scale == 0:
                    continue
                tensors = read_upstream_adapter(sources[branch], branch, require_io=mode != "joint")
                modules = dict(model.named_modules())
                for name, _, _ in target_shapes(branch):
                    a, b = (tensors[name + suffix].to(device="cuda", dtype=torch.bfloat16) for suffix in (".A", ".B"))
                    buffers.extend((a, b))
                    def add_adapter(module, inputs, output, a=a, b=b, scale=scale):
                        return output + F.linear(F.linear(inputs[0].to(a.dtype), a), b).to(output.dtype) * scale
                    handles.append(modules[name].register_forward_hook(add_adapter))
                if branch == "nar" and mode != "joint":
                    for name in ("vae2llm", "llm2vae"):
                        for key in ('weight', 'bias'):
                            target = f'{name}.{key}'
                            value = tensors[target].float() * scale
                            projections[target] = projections.get(target, 0) + value
                del tensors
        if projections:
            modules = dict(pipeline.transformer.named_modules())
            for name in ('vae2llm', 'llm2vae'):
                weight, bias = (projections[f'{name}.{key}'].to(device='cuda', dtype=torch.bfloat16) for key in ('weight', 'bias'))
                buffers.extend((weight, bias))
                def replace_projection(module, inputs, output, weight=weight, bias=bias):
                    return F.linear(inputs[0].to(weight.dtype), weight, bias).to(output.dtype)
                handles.append(modules[name].register_forward_hook(replace_projection))
        projections.clear()
        yield
    finally:
        try:
            pipeline.engine.release_runtime_allocations()
        finally:
            for handle in handles:
                handle.remove()
            buffers.clear()
            projections.clear()
