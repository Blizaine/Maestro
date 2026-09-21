"""Bounded, resumable AR-LoRA training with a held-out and minted control set.

Adapts Mothersuperior's score-free ar_lora.py objective to Maestro's split AR
model. It does not train the tokenizer or NAR branch. Those use the paired v4
release, preserving one token dialect across preparation and generation.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config

from .artist_adapter import target_shapes
from .music_assets import ensure_asset, TOKENIZER_REVISION
from .protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, SongRequest, token_prefixes
from .tokenization_yue2 import YuE2TextTokenizer
from .transformer import YuE2AR
from services.music_training import project_directory, update_project
from services.music_contracts import tokenizer_pair


class TrainableAdapter(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(rank, base.weight.shape[1], device=base.weight.device, dtype=torch.float32) / math.sqrt(base.weight.shape[1]))
        self.B = nn.Parameter(torch.zeros(base.weight.shape[0], rank, device=base.weight.device, dtype=torch.float32))

    def forward(self, values):
        return self.base(values) + F.linear(F.linear(values.float(), self.A), self.B).to(values.dtype)


def _norm(values, module):
    return F.rms_norm(values, (values.shape[-1],), module.weight, module.eps)


def _layer(layer, values):
    attention = layer.self_attn
    normalized = _norm(values, layer.input_layernorm)
    length = len(values)
    query = attention.q_proj(normalized).view(length, attention.num_heads, attention.head_dim)
    key = attention.k_proj(normalized).view(length, attention.num_kv_heads, attention.head_dim)
    value = attention.v_proj(normalized).view(length, attention.num_kv_heads, attention.head_dim)
    query, key = _norm(query, attention.q_norm), _norm(key, attention.k_norm)
    query, key = attention.rotary_emb(torch.arange(length, device=values.device), query, key)
    groups = attention.num_heads // attention.num_kv_heads
    key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    # The CUDA flash/efficient SDPA backends require batch/head/sequence/width.
    # A three-dimensional call silently selects the quadratic math backend.
    hidden = F.scaled_dot_product_attention(query.transpose(0, 1)[None], key.transpose(0, 1)[None],
                                            value.transpose(0, 1)[None], is_causal=True)[0]
    values = values + attention.o_proj(hidden.transpose(0, 1).reshape(length, -1))
    normalized = _norm(values, layer.post_attention_layernorm)
    return values + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(normalized)) * layer.mlp.up_proj(normalized))


def _aligned_layer(layer, values):
    from .training_math import ar_layer
    return ar_layer(layer, values)[0]


def sequence_loss(model, item, maximum, *, gradients, cursor_head=None, metrics=None, production_norm=False):
    prefix = item["prefix"]
    if len(prefix) >= maximum - 1:
        raise ValueError("Song lyrics leave no room for training audio tokens; shorten the lyrics")
    codec = item["codec"]
    room = maximum - len(prefix) - 1
    body = [int(code) + CODEC_OFFSET for code in codec[:room]]
    if len(codec) <= room:
        body.append(MUSIC_END)
    ids = torch.tensor(prefix + body, device="cuda", dtype=torch.long)
    values = model.model.embed_tokens(ids)
    layer_fn = _aligned_layer if cursor_head is not None or production_norm else _layer
    for layer in model.model.layers:
        values = checkpoint(layer_fn, layer, values, use_reentrant=False) if gradients else layer_fn(layer, values)
    if cursor_head is not None or production_norm:
        from .training_math import ar_norm
        normalized = ar_norm(values, model.model.norm)
        hidden = normalized[len(prefix) - 1:-1]
    else:
        hidden = _norm(values[len(prefix) - 1:-1], model.model.norm)
    targets = ids[len(prefix):]
    loss = hidden.new_zeros((), dtype=torch.float32)
    def head_loss(part, target):
        # Training must not inherit the inference engine's last-token context.
        return F.cross_entropy(F.linear(part, model.lm_head.weight).float(), target, reduction="sum")
    # Recompute each vocabulary projection during backward instead of retaining
    # an entire song's 184k-class logits in VRAM.
    for start in range(0, len(hidden), 256):
        args = hidden[start:start + 256], targets[start:start + 256]
        loss = loss + (checkpoint(head_loss, *args, use_reentrant=False) if gradients else head_loss(*args))
    lm = loss / len(targets)
    if metrics is not None:
        metrics['lm'] = float(lm.detach())
    if cursor_head is not None and item.get('cursor'):
        from .lyric_alignment import cursor_loss
        cursor = cursor_loss(normalized, len(prefix), item['cursor'], cursor_head)
        if metrics is not None:
            metrics['cursor'] = float(cursor.detach())
        return lm + .08 * cursor
    return lm


def _regularizer(path):
    # The pinned upstream pack contains NumPy arrays. Keep torch's restricted
    # loader enabled and allow only their reconstruction types.
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype,
                                           np.dtypes.Int32DType, np.dtypes.Int64DType]):
        data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        data = data.get("records", data.get("items", data.get("data")))
    if not isinstance(data, list) or not data:
        raise ValueError("The YuE2 regularizer pack is empty or incompatible")
    return data


def _dataset(project, regularizer_path, tokenizer, aligned=False):
    directory = project_directory(project["id"]) / "prepared"
    groups = {"artist": [], "heldout": [], "minted": [], "minted_val": []}
    for track in project["tracks"]:
        codes_path = directory / f"{track['id']}.npy"
        if not codes_path.is_file():
            raise ValueError("Prepare all recordings before training")
        request = SongRequest(style=f"{project['trigger']}, {track['style']}", lyrics=track["lyrics"], cot="off")
        item = {"prefix": token_prefixes(request, tokenizer), "codec": np.load(codes_path, allow_pickle=False).tolist()}
        if aligned:
            from .lyric_alignment import cursor_targets
            item['cursor'] = cursor_targets(project, track, item['prefix'], tokenizer, len(item['codec']))
            if not track['holdout'] and item['cursor'] is not None and not item['cursor']['rows']:
                raise ValueError('This recording has no confident lyric timing targets; review the alignment')
        groups["heldout" if track["holdout"] else "artist"].append(item)
    for index, record in enumerate(_regularizer(regularizer_path)):
        source = record.get("src")
        group = "minted_val" if source == "minted_val" else "minted"
        if source not in {"minted", "minted_val"}:
            # Stable control split for packs without source labels.
            group = "minted_val" if index % 20 == 0 else "minted"
        codes = np.asarray(record["codec"], dtype=np.int64).reshape(-1)
        if not len(codes) or (codes < 0).any() or (codes >= CODEC_SIZE).any():
            raise ValueError("Regularizer contains invalid YuE2 semantic codes")
        request = SongRequest(style=str(record["style"]), lyrics=str(record["lyrics"]), cot="off")
        groups[group].append({"prefix": token_prefixes(request, tokenizer), "codec": codes.tolist()})
    if any(not values for values in groups.values()):
        raise ValueError("Training requires artist, held-out, minted and minted validation examples")
    return groups


def train_project(project, options, *, report, cancelled, pause_at_checkpoint=False):
    directory = project_directory(project["id"])
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    pair = tokenizer_pair(project)
    if not project.get("prepared") or project["prepared"].get("tokenizer_revision") != pair['revision']:
        raise ValueError("Prepare this dataset with the current music tokenizer before training")
    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise ValueError("YuE2 style training currently requires a GPU with at least 20 GB VRAM")
    regularizer_path = ensure_asset("regularizer", cancelled=cancelled, report=report)
    model_path = ensure_asset("ar_training", cancelled=cancelled, report=report)
    tokenizer_path = ensure_asset("text_tokenizer", cancelled=cancelled, report=report)
    aligned = options.get('lyric_alignment', False)
    if aligned and not project.get('alignment', {}).get('ready'):
        raise ValueError('Align and review lyrics before enabling lyric timing training')
    groups = _dataset(project, regularizer_path, YuE2TextTokenizer(str(tokenizer_path)), aligned)
    seed = options["seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = optimizer = cursor_head = None
    adapters, completed = {}, 0
    latest = checkpoints / "resume.pt"
    contract = {key: options[key] for key in ("rank", "seed", "learning_rate", "artist_fraction", "max_tokens", "accumulation_steps")}
    contract["dataset_digest"] = project["dataset_digest"]
    # Preserve resume compatibility for the original v4 projects.
    if pair['revision'] != TOKENIZER_REVISION:
        contract['tokenizer_revision'] = pair['revision']
    if aligned:
        from .lyric_alignment import ALIGNMENT_VERSION
        from services.music_styles import file_digest
        contract.update(lyric_alignment=True, cursor_weight=.08, objective=ALIGNMENT_VERSION,
                        alignment_hashes={track['id']: file_digest(directory / 'alignment' / (track['id'] + '.json')) for track in project['tracks']})
    history = list(project.get("checkpoints") or [])
    started = time.monotonic()
    try:
        report("Loading the optional BF16 training model", 0)
        config = Qwen3Config.from_json_file(str(Path(__file__).with_name("yue2_ar.json")))
        with init_empty_weights():
            model = YuE2AR(config)
        model.load_state_dict(load_file(str(model_path)), strict=True, assign=True)
        model = model.to("cuda").eval().requires_grad_(False)
        for name, _, _ in target_shapes("ar"):
            parent_name, attribute = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            adapter = TrainableAdapter(getattr(parent, attribute), options["rank"])
            setattr(parent, attribute, adapter)
            adapters[name] = adapter
        parameters = [parameter for adapter in adapters.values() for parameter in (adapter.A, adapter.B)]
        if aligned:
            cursor_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False, device='cuda', dtype=torch.float32)
            with torch.no_grad():
                cursor_head.weight.copy_(torch.eye(config.hidden_size, device='cuda', dtype=torch.float32))
            parameters += list(cursor_head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=options["learning_rate"], weight_decay=0.0, betas=(0.9, 0.95))
        if options["resume"]:
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            if saved.get("contract") != contract:
                raise ValueError("Resume requires the same dataset, rank, seed and learning rate")
            with torch.no_grad():
                for name, adapter in adapters.items():
                    adapter.A.copy_(saved["weights"][name + ".A"])
                    adapter.B.copy_(saved["weights"][name + ".B"])
            optimizer.load_state_dict(saved["optimizer"])
            if aligned:
                cursor_head.load_state_dict(saved['cursor_head'])
            completed = int(saved["step"])
            rng.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        elif latest.exists():
            raise ValueError("This project already has a checkpoint. Resume it or create a new training project")

        def save_resume():
            weights = {name + suffix: getattr(adapter, suffix[1:]).detach().cpu()
                       for name, adapter in adapters.items() for suffix in (".A", ".B")}
            temporary = latest.with_suffix(".tmp")
            torch.save({"contract": contract, "step": completed, "weights": weights,
                        "cursor_head": {k: v.detach().cpu() for k, v in cursor_head.state_dict().items()} if aligned else None,
                        "optimizer": optimizer.state_dict(), "python_rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
            temporary.replace(latest)
            update_project(project["id"], resume_available=True, completed_steps=completed, training_options=contract)
            return weights

        @torch.no_grad()
        def evaluate():
            result = {}
            for group in ("heldout", "minted_val"):
                scores = []
                for item in groups[group][:3]:
                    if cancelled():
                        raise InterruptedError("Music training cancelled")
                    metrics = {}
                    sequence_loss(model, item, options["max_tokens"], gradients=False, cursor_head=cursor_head, metrics=metrics)
                    scores.append(metrics['lm'])
                result[group] = sum(scores) / len(scores)
            return result

        if not options["resume"]:
            baseline = evaluate()
            update_project(project["id"], baseline=baseline)
        # The cosine horizon stays fixed across resumes, matching the upstream
        # recommendation; extending a run does not silently change its schedule.
        for step in range(completed + 1, options["steps"] + 1):
            if cancelled():
                save_resume()
                raise InterruptedError("Music training cancelled; checkpoint saved")
            factor = min(1, step / 50) * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(step, 3000) / 3000)))
            for group in optimizer.param_groups:
                group["lr"] = options["learning_rate"] * factor
            loss_value = 0.0
            lm_value, cursor_values = 0.0, []
            for _ in range(options["accumulation_steps"]):
                group = "artist" if rng.random() < options['artist_fraction'] else "minted"
                item = rng.choice(groups[group])
                metrics = {}
                loss = sequence_loss(model, item, options["max_tokens"], gradients=True, cursor_head=cursor_head, metrics=metrics) / options["accumulation_steps"]
                lm_value += metrics['lm'] / options['accumulation_steps']
                if 'cursor' in metrics:
                    cursor_values.append(metrics['cursor'])
                loss.backward()
                loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step
            details = f" · token loss {lm_value:.3f}" + (f" · lyric timing {sum(cursor_values)/len(cursor_values):.3f}" if cursor_values else '') if aligned else f" · loss {loss_value:.3f}"
            report(f"Training step {step}/{options['steps']}" + details, step / options["steps"] * 100)
            if step % options["checkpoint_every"] == 0 or step == options["steps"] or cancelled():
                weights = save_resume()
                filename = f"step-{step}.safetensors"
                save_file(weights, str(checkpoints / filename))
                scores = {} if cancelled() else evaluate()
                history = [item for item in history if item["step"] != step]
                history.append({"step": step, "file": filename, "scores": scores,
                                "elapsed_seconds": round(time.monotonic() - started)})
                update_project(project["id"], checkpoints=history, completed_steps=step, training_options=contract)
                if pause_at_checkpoint:
                    return history
        return history
    finally:
        optimizer = model = cursor_head = None
        adapters.clear()
        gc.collect()
        torch.cuda.empty_cache()
