"""Experimental joint AR/NAR LoRA training with detached audio conditioning.

Uses full-song next-token loss plus a cropped latent flow loss. Fused projection
groups share their A matrix, matching AI-Toolkit's YuE2 adapter parameterization.
Saved-style starts retain independent A matrices and the saved decoder's I/O.
Base weights, tokenizer and I/O projections stay frozen. This score-free recipe
does not claim to reproduce an external checkpoint's undocumented dataset/config.
"""
import gc
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config

from .acoustic_training import load_data
from .artist_adapter import read_upstream_adapter, target_shapes
from .artist_training import TrainableAdapter, sequence_loss
from .music_assets import ensure_asset, VAE_REVISION
from .modules import YuE2Config
from .protocol import CODEC_OFFSET, MUSIC_END
from .tokenization_yue2 import YuE2TextTokenizer
from .training_math import ar_condition, acoustic_forward
from .transformer import YuE2AR, YuE2Acoustic
from services.music_contracts import adapter_contract, tokenizer_pair
from services.music_styles import file_digest, load_style, style_directory
from services.music_training import project_directory, update_project


def install_adapters(model, branch, rank, weights=None, *, shared_fused_a=True):
    adapters, shared = {}, {}
    for name, _, _ in target_shapes(branch):
        parent, attribute = name.rsplit('.', 1)
        group = ('qkv' if attribute in ('q_proj', 'k_proj', 'v_proj') else
                 'gate_up' if attribute in ('gate_proj', 'up_proj') else attribute)
        key = parent + '.' + group
        adapter = TrainableAdapter(model.get_submodule(name), rank)
        if shared_fused_a and key in shared:
            if weights is not None and not torch.equal(shared[key].detach().cpu(), weights[name + '.A'].float().cpu()):
                raise ValueError('Cannot share unequal saved adapter matrices')
            adapter.A = shared[key]
        elif shared_fused_a:
            shared[key] = adapter.A
        if weights:
            with torch.no_grad():
                adapter.A.copy_(weights[name + '.A'])
                adapter.B.copy_(weights[name + '.B'])
        setattr(model.get_submodule(parent), attribute, adapter)
        adapters[name] = adapter
    return adapters


def initial_style_weights(project, style_id):
    """Validate a saved pair before changing a new experiment's models."""
    manifest = load_style(style_id, verify=True)
    if manifest.get('tokenizer_revision') != tokenizer_pair(project)['revision']:
        raise ValueError('The starting style must use this project\'s tokenizer pair')
    if manifest['trigger'] != project['trigger']:
        raise ValueError('Use the starting style\'s exact trigger in this project')
    mode = adapter_contract(manifest)
    weights = {branch: read_upstream_adapter(style_directory(style_id) / manifest[branch]['file'], branch,
                                             require_io=mode != 'joint') for branch in ('ar', 'nar')}
    identity = {'style_id': style_id, 'style_name': manifest['name'], 'adapter_mode': mode}
    for branch in ('ar', 'nar'):
        identity[branch + '_sha256'] = manifest[branch]['sha256']
        identity[branch + '_rank'] = next(value.shape[0] for key, value in weights[branch].items() if key.endswith('.A'))
    return identity, weights


def restore_decoder_io(model, weights):
    for name in ('vae2llm', 'llm2vae'):
        getattr(model, name).load_state_dict({key: weights[name + '.' + key] for key in ('weight', 'bias')})


def adapter_snapshot(model, adapters, *, include_io=False):
    weights = {name + suffix: getattr(adapter, suffix[1:]).detach().cpu().clone()
               for name, adapter in adapters.items() for suffix in ('.A', '.B')}
    if include_io:
        for name in ('vae2llm', 'llm2vae'):
            weights.update({name + '.' + key: value.detach().cpu().clone()
                            for key, value in getattr(model, name).state_dict().items()})
    return weights


def flow_loss(ar, nar, item, start, count, t, noise, *, gradients):
    z = torch.tensor(np.array(item['latent'][start:start + count]), device=noise.device)
    ids = item['prefix'] + [int(c) + CODEC_OFFSET for c in item['codec'][start:start + count]] + [MUSIC_END]
    # AR learns from next-token CE only; audio error must not change its cache.
    cache = ar_condition(ar, ids)
    prediction = acoustic_forward(nar, t * noise + (1 - t) * z,
        torch.logit(torch.tensor(t, device=noise.device)), cache, len(ids), gradients=gradients)
    return F.mse_loss(prediction.float(), noise - z)


def train_joint(project, options, *, report, cancelled, pause_at_checkpoint=False):
    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise ValueError('Joint music training requires at least 20 GB VRAM')
    directory = project_directory(project['id']) / 'joint_checkpoints'
    directory.mkdir(exist_ok=True)
    latest = directory / 'resume.pt'
    if latest.exists() != options['resume']:
        raise ValueError('Resume the saved joint run, or create a new experiment before training')
    saved = torch.load(latest, map_location='cpu', weights_only=True) if options['resume'] else None
    initial_id = options.get('initial_style_id', '')
    if saved:
        initial = saved['contract'].get('initialization')
        if initial_id and initial_id != (initial or {}).get('style_id'):
            raise ValueError('Resume must keep the original starting style')
        weights = saved['weights']
    elif initial_id:
        initial, weights = initial_style_weights(project, initial_id)
    else:
        initial, weights = None, {}
    mode = initial['adapter_mode'] if initial else 'joint'
    shared_fused_a = not bool(initial)
    paths = {key: ensure_asset(key, report=report, cancelled=cancelled)
             for key in ('ar_training', 'nar_training', 'text_tokenizer')}
    groups, regularizer_digest = load_data(project, YuE2TextTokenizer(str(paths['text_tokenizer'])), report=report, cancelled=cancelled)
    # Unlike NAR crops, AR token loss always begins at the start of the song.
    # Do not silently label the beginning of a long song with its full lyrics.
    if any(len(item['codec']) + len(item['prefix']) + 1 > options['max_tokens']
           for group in groups.values() for item in group):
        raise ValueError('Joint training needs shorter, lyric-aligned recordings that fit the 12,288-token training context')
    contract = {key: options[key] for key in ('rank', 'seed', 'learning_rate', 'artist_fraction',
                                             'window_frames', 'max_tokens', 'accumulation_steps')}
    contract.update(objective='joint-ar-nar-v1', tokenizer_revision=tokenizer_pair(project)['revision'],
                    dataset_digest=project['dataset_digest'], vae_revision=VAE_REVISION,
                    regularizer_digest=regularizer_digest, decoder_initialization='base',
                    ar_weight=1.0, nar_weight=1.0, shared_fused_a=shared_fused_a)
    if initial:
        contract.update(initialization=initial, decoder_initialization='saved-style',
                        conditioning_trigger=project['trigger'])
    if saved and saved['contract'] != contract:
        raise ValueError('Joint resume requires the same data, tokenizer pair and training settings')
    rng = random.Random(options['seed'])
    torch.manual_seed(options['seed'])
    torch.cuda.manual_seed_all(options['seed'])
    completed = int(saved['step']) if saved else 0
    ar = nar = optimizer = None
    adapters, parameters = {}, []
    history = list(project.get('joint_checkpoints', []))
    started = time.monotonic()
    try:
        report('Loading joint music and audio training models', 0)
        with init_empty_weights():
            ar = YuE2AR(Qwen3Config.from_json_file(str(Path(__file__).with_name('yue2_ar.json'))))
            nar = YuE2Acoustic(YuE2Config())
        for branch, model in (('ar', ar), ('nar', nar)):
            model.load_state_dict(load_file(str(paths[branch + '_training'])), strict=True, assign=True)
            model.cuda().eval().requires_grad_(False)
            rank = initial[branch + '_rank'] if initial else options['rank']
            adapters[branch] = install_adapters(model, branch, rank, weights.get(branch), shared_fused_a=shared_fused_a)
            if branch == 'nar' and mode == 'separate':
                restore_decoder_io(model, weights[branch])
        ar.configure_engine('legacy', cancelled)
        parameters = list(dict.fromkeys(p for group in adapters.values() for adapter in group.values() for p in (adapter.A, adapter.B)))
        optimizer = torch.optim.AdamW(parameters, lr=options['learning_rate'], betas=(.9, .95), weight_decay=0)
        if saved:
            optimizer.load_state_dict(saved['optimizer'])
            rng.setstate(saved['python_rng'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved, weights
        torch.cuda.reset_peak_memory_stats()

        @torch.no_grad()
        def evaluate():
            scores = {}
            generator = torch.Generator(device='cpu').manual_seed(123)
            for group in ('heldout', 'minted_val'):
                ces, flows = [], []
                for item in groups[group][:2]:
                    if cancelled():
                        return {}
                    ces.append(float(sequence_loss(ar, item, options['max_tokens'], gradients=False, production_norm=True)))
                    count = min(options['window_frames'], len(item['codec']))
                    start = (len(item['codec']) - count) // 2
                    noise = torch.randn(count, 64, generator=generator, device='cpu').cuda()
                    flows.append(float(flow_loss(ar, nar, item, start, count, .5, noise, gradients=False)))
                scores[group + '_tokens'] = sum(ces) / len(ces)
                scores[group + '_audio'] = sum(flows) / len(flows)
            return scores

        def snapshot(scores):
            weights = {branch: adapter_snapshot(model, adapters[branch], include_io=branch == 'nar' and mode == 'separate')
                       for branch, model in (('ar', ar), ('nar', nar))}
            files = {}
            for branch in ('ar', 'nar'):
                path = directory / f'{branch}-step-{completed}.safetensors'
                temporary = path.with_suffix('.tmp')
                save_file(weights[branch], str(temporary)); temporary.replace(path)
                files[branch] = path
            temporary = latest.with_suffix('.tmp')
            torch.save({'contract': contract, 'step': completed, 'weights': weights, 'optimizer': optimizer.state_dict(),
                        'python_rng': rng.getstate(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temporary)
            temporary.replace(latest)
            row = {'step': completed, 'file': files['ar'].name, 'nar_file': files['nar'].name,
                   'adapter_mode': mode,
                   'ar_sha256': file_digest(files['ar']), 'nar_sha256': file_digest(files['nar']), 'scores': scores,
                   'elapsed_seconds': round(time.monotonic() - started),
                   'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
            history[:] = [row for row in history if row['step'] != completed] + [row]
            # Project row is the commit marker; no half-pair enters the picker.
            update_project(project['id'], joint_resume_available=True, joint_completed_steps=completed,
                           joint_training_options=contract, joint_checkpoints=history)

        if not options['resume']:
            baseline = evaluate()
            update_project(project['id'], joint_baseline=baseline)
            if initial:
                # Keep an auditable pre-update pair, including the saved decoder.
                snapshot(baseline)
        for step in range(completed + 1, options['steps'] + 1):
            if cancelled():
                snapshot({})
                raise InterruptedError('Joint training stopped; both adapters and optimizer are saved')
            factor = min(1, step / 50) * (.2 + .8 * .5 * (1 + math.cos(math.pi * min(step, 3000) / 3000)))
            for group in optimizer.param_groups:
                group['lr'] = options['learning_rate'] * factor
            item = rng.choice(groups['artist' if rng.random() < options['artist_fraction'] else 'minted'])
            ce = sequence_loss(ar, item, options['max_tokens'], gradients=True, production_norm=True)
            if not torch.isfinite(ce):
                raise ValueError('Joint token loss is non-finite')
            ce.backward()  # Free the AR graph before allocating NAR activations.
            ce_value = float(ce.detach())
            del ce
            count = min(options['window_frames'], len(item['codec']))
            start = rng.randrange(len(item['codec']) - count + 1)
            t = min(.95, max(.05, rng.random()))
            loss = flow_loss(ar, nar, item, start, count, t, torch.randn(count, 64, device='cuda'), gradients=True)
            if not torch.isfinite(loss):
                raise ValueError('Joint audio loss is non-finite')
            loss.backward()
            audio_value = float(loss.detach())
            del loss
            torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            completed = step
            report(f'Joint step {step}/{options["steps"]} · tokens {ce_value:.3f} · audio {audio_value:.4f}', step / options['steps'] * 100)
            if step % options['checkpoint_every'] == 0 or step == options['steps'] or cancelled():
                snapshot({} if cancelled() else evaluate())
                if pause_at_checkpoint:
                    return history
        return history
    finally:
        ar = nar = optimizer = None
        adapters.clear(); parameters.clear()
        gc.collect(); torch.cuda.empty_cache()
