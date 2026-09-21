"""Resumable fixed-tokenizer NAR adaptation to real source audio.

Mothersuperior v4 flow objective: 75% source windows, 25% disjoint minted
regularizer windows. Frozen AR conditioning; train paired NAR LoRA + I/O only.
"""
import gc
import hashlib
import json
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

from .artist_adapter import read_upstream_adapter, target_shapes
from .artist_training import TrainableAdapter
from .audio_training_data import target_identity
from .music_assets import ensure_asset, VAE_REVISION
from services.music_contracts import tokenizer_pair, pair_asset
from .modules import YuE2Config
from .protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, SongRequest, token_prefixes
from .tokenization_yue2 import YuE2TextTokenizer
from .training_math import ar_condition, acoustic_forward
from .transformer import YuE2AR, YuE2Acoustic
from services.music_training import project_directory, update_project
from services.music_styles import file_digest


def load_data(project, tokenizer, *, report, cancelled):
    from .music_assets import ROOT
    directory = project_directory(project['id'])
    if project.get('prepared', {}).get('tokenizer_revision') != tokenizer_pair(project)['revision']:
        raise ValueError('Prepare this project with its selected tokenizer pair before training')
    groups = {'artist': [], 'heldout': [], 'minted': [], 'minted_val': []}
    def item(codes_path, latent_path, request):
        codes = np.load(codes_path, mmap_mode='r', allow_pickle=False)
        latents = np.load(latent_path, mmap_mode='r', allow_pickle=False)
        if (codes.ndim != 1 or not np.issubdtype(codes.dtype, np.integer) or codes.size == 0
                or codes.min() < 0 or codes.max() >= CODEC_SIZE or latents.ndim != 2
                or latents.shape[1] != 64 or not np.isfinite(latents).all()):
            raise ValueError('Invalid acoustic training codes or targets')
        count = min(len(codes), len(latents))
        if count < 128 or abs(len(codes) - len(latents)) > 2:
            raise ValueError('Audio targets and semantic codes must share the 25 Hz timeline')
        prefix = token_prefixes(request, tokenizer)
        if len(prefix) + 514 > 12288:
            raise ValueError('Shorten the lyric text before audio adaptation')
        return {'codec': codes[:count], 'latent': latents[:count], 'prefix': prefix}
    for track in project['tracks']:
        if file_digest(Path(track['audio_path'])) != track['audio_sha256']:
            raise ValueError('A source recording changed; create a new project')
        if json.loads((directory / 'audio_targets' / (track['id'] + '.json')).read_text()) != target_identity(project, track):
            raise ValueError('Prepare current source-audio targets before adapting sound')
        if json.loads((directory / 'prepared' / (track['id'] + '.json')).read_text()) != project['prepared']:
            raise ValueError('Prepared tokens do not match this dataset')
        request = SongRequest(style=f"{project['trigger']}, {track['style']}", lyrics=track['lyrics'], cot='off')
        groups['heldout' if track['holdout'] else 'artist'].append(item(
            directory / 'prepared' / (track['id'] + '.npy'), directory / 'audio_targets' / (track['id'] + '.npy'), request))
    manifest = json.loads(Path(__file__).with_name('acoustic_regularizer.json').read_text())
    for entry in manifest['files']:
        if cancelled():
            raise InterruptedError('Audio adaptation cancelled')
        ensure_asset('acoustic:' + entry['path'], cancelled=cancelled, report=report)
    for track in manifest['tracks']:
        source = ROOT / 'acoustic_regularizer' / track['id']
        request = json.loads((source / 'request.json').read_text())
        groups['minted_val' if track['holdout'] else 'minted'].append(item(source / 'semantic.npy', source / 'latent.npy',
            SongRequest(style=request['style'], lyrics=request['lyrics'], cot='off')))
    if any(not group for group in groups.values()):
        raise ValueError('Audio adaptation needs source and minted training/held-out examples')
    return groups, hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def train_audio(project, options, *, report, cancelled, pause_at_checkpoint=False):
    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise ValueError('Audio adaptation requires at least 20 GB VRAM')
    directory = project_directory(project['id'])
    checkpoint_dir = directory / 'audio_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    latest = checkpoint_dir / 'resume.pt'
    if latest.exists() and not options['resume']:
        raise ValueError('Resume this audio experiment or create a new experiment')
    if options['resume'] and not latest.exists():
        raise ValueError('No saved audio training state is available to resume')
    pair = tokenizer_pair(project)
    paths = {key: ensure_asset(key, report=report, cancelled=cancelled) for key in
             ('ar_training', 'nar_training', 'text_tokenizer')}
    paths['nar'] = pair_asset(project, 'nar', cancelled=cancelled, report=report)
    groups, regularizer_digest = load_data(project, YuE2TextTokenizer(str(paths['text_tokenizer'])), report=report, cancelled=cancelled)
    condition = options['conditioning_checkpoint']
    condition_path = directory / 'checkpoints' / condition if condition else None
    if condition and condition not in {row['file'] for row in project.get('checkpoints', [])}:
        raise ValueError('Choose a saved music checkpoint for audio conditioning')
    contract = {key: options[key] for key in ('rank', 'seed', 'learning_rate', 'io_learning_rate', 'window_frames', 'conditioning_checkpoint')}
    contract.update(objective='fixed-tokenizer-flow-v1', dataset_digest=project['dataset_digest'],
                    tokenizer_revision=pair['revision'], vae_revision=VAE_REVISION,
                    regularizer_digest=regularizer_digest, conditioning_sha256=file_digest(condition_path) if condition else 'base-ar')
    saved = torch.load(latest, map_location='cpu', weights_only=True) if options['resume'] else None
    if saved and saved['contract'] != contract:
        raise ValueError('Resume requires unchanged audio settings, sources, conditioning checkpoint and regularizer')
    rng = random.Random(options['seed'])
    torch.manual_seed(options['seed'])
    torch.cuda.manual_seed_all(options['seed'])
    ar = nar = optimizer = None
    adapters, ar_adapters, parameters = {}, {}, []
    history = list(project.get('audio_checkpoints') or [])
    completed = int(saved['step']) if saved else 0
    started = time.monotonic()
    try:
        report('Loading audio adaptation models', 0)
        with init_empty_weights():
            ar = YuE2AR(Qwen3Config.from_json_file(str(Path(__file__).with_name('yue2_ar.json'))))
            nar = YuE2Acoustic(YuE2Config())
        ar.load_state_dict(load_file(str(paths['ar_training'])), strict=True, assign=True)
        nar.load_state_dict(load_file(str(paths['nar_training'])), strict=True, assign=True)
        ar.cuda().eval().requires_grad_(False)
        ar.configure_engine('legacy', cancelled)
        nar.cuda().eval().requires_grad_(False)
        if condition:
            weights = read_upstream_adapter(condition_path, 'ar')
            for name, _, _ in target_shapes('ar'):
                parent, attribute = name.rsplit('.', 1)
                module = TrainableAdapter(ar.get_submodule(name), weights[name + '.A'].shape[0])
                with torch.no_grad():
                    module.A.copy_(weights[name + '.A']); module.B.copy_(weights[name + '.B'])
                module.requires_grad_(False)
                setattr(ar.get_submodule(parent), attribute, module)
                ar_adapters[name] = module
            del weights
        weights = saved['weights'] if saved else read_upstream_adapter(paths['nar'], 'nar')
        for name, _, _ in target_shapes('nar'):
            parent, attribute = name.rsplit('.', 1)
            module = TrainableAdapter(nar.get_submodule(name), options['rank'])
            with torch.no_grad():
                module.A.copy_(weights[name + '.A']); module.B.copy_(weights[name + '.B'])
            setattr(nar.get_submodule(parent), attribute, module)
            adapters[name] = module
        for name in ('vae2llm', 'llm2vae'):
            module = getattr(nar, name).float().requires_grad_(True)
            module.load_state_dict({key: weights[name + '.' + key] for key in ('weight', 'bias')})
        parameters = [p for adapter in adapters.values() for p in (adapter.A, adapter.B)]
        io = list(nar.vae2llm.parameters()) + list(nar.llm2vae.parameters())
        optimizer = torch.optim.AdamW([{'params': parameters, 'lr': options['learning_rate']},
                                      {'params': io, 'lr': options['io_learning_rate']}], betas=(.9, .95), weight_decay=0)
        parameters += io
        if saved:
            optimizer.load_state_dict(saved['optimizer'])
            rng.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved, weights
        torch.cuda.reset_peak_memory_stats()

        def snapshot():
            weights = {name + suffix: getattr(module, suffix[1:]).detach().cpu()
                       for name, module in adapters.items() for suffix in ('.A', '.B')}
            for name in ('vae2llm', 'llm2vae'):
                weights.update({name + '.' + key: value.detach().cpu() for key, value in getattr(nar, name).state_dict().items()})
            temp = latest.with_suffix('.tmp')
            torch.save({'contract': contract, 'step': completed, 'weights': weights, 'optimizer': optimizer.state_dict(),
                        'python_rng': rng.getstate(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp)
            temp.replace(latest)
            update_project(project['id'], audio_resume_available=True, audio_completed_steps=completed, audio_training_options=contract)
            return weights

        def loss(item, start, t, noise, gradients):
            n = len(noise)
            z = torch.tensor(np.array(item['latent'][start:start+n]), device='cuda')
            ids = item['prefix'] + [int(c) + CODEC_OFFSET for c in item['codec'][start:start+n]] + [MUSIC_END]
            cache = ar_condition(ar, ids)
            x = t * noise + (1 - t) * z
            prediction = acoustic_forward(nar, x, torch.logit(torch.tensor(t, device='cuda')), cache, len(ids), gradients=gradients)
            return F.mse_loss(prediction.float(), noise - z)

        @torch.no_grad()
        def evaluate():
            scores = {}
            for group in ('heldout', 'minted_val'):
                values = []
                generator = torch.Generator(device='cpu').manual_seed(123)
                for item in groups[group][:3]:
                    n = min(options['window_frames'], len(item['codec']))
                    for fraction in (.25, .5, .75):
                        start = int((len(item['codec']) - n) * fraction)
                        noise = torch.randn(n, 64, generator=generator, device='cpu').cuda()
                        for t in (.2, .5, .8):
                            if cancelled():
                                return {}
                            values.append(float(loss(item, start, t, noise, False)))
                scores[group] = sum(values) / len(values)
            return scores

        if not options['resume']:
            update_project(project['id'], audio_baseline=evaluate())
        for step in range(completed + 1, options['steps'] + 1):
            if cancelled():
                snapshot()
                raise InterruptedError('Audio adaptation stopped; checkpoint saved')
            factor = min(1, step / 50) * (.2 + .8 * .5 * (1 + math.cos(math.pi * min(step, 3000) / 3000)))
            for group, lr in zip(optimizer.param_groups, (options['learning_rate'], options['io_learning_rate'])):
                group['lr'] = lr * factor
            item = rng.choice(groups['artist' if rng.random() < .75 else 'minted'])
            n = min(options['window_frames'], len(item['codec']))
            start = rng.randrange(len(item['codec']) - n + 1)
            t = min(.95, max(.05, rng.betavariate(2, 2)))
            value = loss(item, start, t, torch.randn(n, 64, device='cuda'), True)
            if not torch.isfinite(value):
                raise ValueError('Audio training loss became non-finite; resume from a saved checkpoint with reviewed settings')
            value.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            completed = step
            report(f"Adapting source sound {step}/{options['steps']} · flow loss {float(value.detach()):.4f}", step / options['steps'] * 100)
            if step % options['checkpoint_every'] == 0 or step == options['steps'] or cancelled():
                weights = snapshot()
                filename = f'step-{step}.safetensors'
                temporary = checkpoint_dir / (filename + '.tmp')
                save_file(weights, str(temporary)); temporary.replace(checkpoint_dir / filename)
                del weights
                scores = {} if cancelled() else evaluate()
                history = [row for row in history if row['step'] != step] + [{'step': step, 'file': filename, 'scores': scores,
                    'conditioning_checkpoint': condition, 'elapsed_seconds': round(time.monotonic() - started),
                    'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3)}]
                update_project(project['id'], audio_checkpoints=history, audio_completed_steps=step, audio_training_options=contract)
                if pause_at_checkpoint:
                    return history
        return history
    finally:
        ar = nar = optimizer = None
        parameters.clear(); adapters.clear(); ar_adapters.clear()
        gc.collect(); torch.cuda.empty_cache()
