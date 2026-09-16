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


def validate_tensors(tensors, branch):
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
    if branch == "nar":
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


def read_upstream_adapter(path, branch):
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        tensors = load_file(str(path), device="cpu")
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
    validate_tensors(tensors, branch)
    return tensors


@contextmanager
def active_artist(pipeline, custom_settings):
    from services.music_styles import load_style, style_directory, validate_strength
    settings = custom_settings or {}
    style_id = settings.get("artist_id")
    if not style_id:
        yield None
        return
    strength = validate_strength(settings.get("artist_strength", 1.0))
    manifest = load_style(style_id, verify=True)
    sources = {branch: style_directory(style_id) / manifest[branch]['file'] for branch in ('ar', 'nar')}
    from .music_assets import ASSETS, ensure_asset
    if settings.get('base_acoustic') is True:
        sources['nar'] = ensure_asset('nar')
    with active_adapters(pipeline, sources, strength):
        yield {"id": style_id, "name": manifest["name"], "trigger": manifest["trigger"], "strength": strength,
               "ar_sha256": manifest["ar"]["sha256"], "nar_sha256": manifest["nar"]["sha256"],
               "active_nar_sha256": ASSETS['nar']['sha256'] if settings.get('base_acoustic') is True else manifest['nar']['sha256'],
               "tokenizer_revision": manifest["tokenizer_revision"]}


@contextmanager
def active_adapters(pipeline, sources, strength):
    """Shared inference hooks for validated library bundles and private checkpoints.

    Sources are resolved by the caller, never taken from generation settings.
    Omitting AR uses base conditioning (for an audio-only experiment).
    """
    handles, buffers = [], []
    try:
        # Python hooks must be captured only for this song; generation's finally
        # clears CUDA graphs before these buffers are released.
        for branch, model in (("ar", pipeline.text_encoder), ("nar", pipeline.transformer)):
            if branch not in sources or branch == "ar" and strength == 0:
                continue
            tensors = read_upstream_adapter(sources[branch], branch)
            modules = dict(model.named_modules())
            scale = strength if branch == "ar" else 1.0
            for name, _, _ in target_shapes(branch):
                a, b = (tensors[name + suffix].to(device="cuda", dtype=torch.bfloat16) for suffix in (".A", ".B"))
                buffers.extend((a, b))
                def add_adapter(module, inputs, output, a=a, b=b, scale=scale):
                    return output + F.linear(F.linear(inputs[0].to(a.dtype), a), b).to(output.dtype) * scale
                handles.append(modules[name].register_forward_hook(add_adapter))
            if branch == "nar":
                for name in ("vae2llm", "llm2vae"):
                    weight, bias = (tensors[f"{name}.{key}"].to(device="cuda", dtype=torch.bfloat16) for key in ("weight", "bias"))
                    buffers.extend((weight, bias))
                    def replace_projection(module, inputs, output, weight=weight, bias=bias):
                        return F.linear(inputs[0].to(weight.dtype), weight, bias).to(output.dtype)
                    handles.append(modules[name].register_forward_hook(replace_projection))
        yield
    finally:
        pipeline.engine.release_runtime_allocations()
        for handle in handles:
            handle.remove()
        buffers.clear()
