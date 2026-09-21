"""Resumable port of Mothersuperior's head+NAR waveform adaptation objective.

See pair_training_math.py and docs/YuE2-music.md for provenance and explicit
differences from the historical 4,000-anchor research run.
"""
import gc
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config

from .artist_adapter import read_upstream_adapter
from .joint_training import install_adapters, adapter_snapshot, restore_decoder_io
from .music_assets import ensure_asset
from .music_tokenizer import RealAudioHead
from .pair_training_data import load_pair_data
from .pair_training_math import pair_flow, straight_through_embeddings, soft_token_loss, SpectralLoss, waveform_loss
from .audio_training_data import read_stereo
from .modules import YuE2Config
from .protocol import CODEC_OFFSET
from .tokenization_yue2 import YuE2TextTokenizer
from .transformer import YuE2AR, YuE2Acoustic
from .vae import YuE2VAE, YuE2VAEConfig
from services.music_contracts import tokenizer_pair, pair_asset
from services.music_pair_adaptation import OBJECTIVE, AUTHOR_REVISION
from services.music_styles import file_digest
from services.music_training import project_directory, update_project


def load_decoder(path):
    model = YuE2VAE(YuE2VAEConfig()).decoder.float()
    with safe_open(str(path), framework='pt', device='cpu') as source:
        weights = {key.removeprefix('decoder.'): source.get_tensor(key) for key in source.keys() if key.startswith('decoder.')}
    for key in list(weights):
        if key.endswith('.weight_g'):
            base = key.removesuffix('_g')
            weights[base] = torch._weight_norm(weights.pop(base + '_v'), weights.pop(key), 0)
    model.load_state_dict(weights, strict=True)
    return model.cuda().eval().requires_grad_(False)


def head_weights(path):
    return load_file(str(path)) if path.suffix == '.safetensors' else torch.load(path, map_location='cpu', weights_only=True)['model']


def train_pair(project, options, *, report, cancelled):
    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise ValueError('Sound adaptation currently requires at least 20 GB VRAM')
    directory = project_directory(project['id']) / 'pair_checkpoints'
    directory.mkdir(exist_ok=True)
    latest = directory / 'resume.pt'
    if latest.exists() != options['resume']:
        raise ValueError('Resume the saved sound adaptation, or create a new experiment')
    paths = {key: ensure_asset(key, report=report, cancelled=cancelled) for key in
             ('ar_training', 'nar_training', 'vae_training', 'text_tokenizer', 'sem_nbr_idx', 'sem_nbr_cos')}
    groups, skipped = load_pair_data(project, YuE2TextTokenizer(str(paths['text_tokenizer'])))
    initial = {key: pair_asset(project, key, report=report, cancelled=cancelled) for key in ('head', 'nar')}
    contract = {k: v for k, v in options.items() if k not in ('steps', 'resume', 'checkpoint_every')}
    contract.update(objective=OBJECTIVE, author_revision=AUTHOR_REVISION,
        dataset_digest=project['dataset_digest'], tokenizer_revision=tokenizer_pair(project)['revision'],
        initialization={key: file_digest(path) for key, path in initial.items()},
        regularizer_digest=project['pair_prepared']['regularizer_digest'],
        anchors={key: len(groups[key]) for key in ('minted', 'minted_val')},
        soft_label_weight=.25, neighbor_temperature=.05, minted_flow_probability=.25)
    saved = torch.load(latest, map_location='cpu', weights_only=True) if options['resume'] else None
    if saved and saved['contract'] != contract:
        raise ValueError('Resume requires the same dataset, initial pair, anchors and adaptation settings')
    rng = random.Random(options['seed'])
    torch.manual_seed(options['seed']); torch.cuda.manual_seed_all(options['seed'])
    completed = int(saved['step']) if saved else 0
    audio_updates = int(saved.get('audio_updates', 0)) if saved else 0
    ar = nar = head = decoder = optimizer = spectral = None
    adapters, parameters = {}, []
    history = list(project.get('pair_checkpoints', []))
    started = time.monotonic()
    try:
        report(f"Loading author sound adaptation: {len(groups['artist'])} training / {len(groups['heldout'])} check excerpts; {len(skipped)} short excerpts skipped", 0)
        with init_empty_weights():
            ar = YuE2AR(Qwen3Config.from_json_file(str(Path(__file__).with_name('yue2_ar.json'))))
            nar = YuE2Acoustic(YuE2Config())
        for key, model in (('ar', ar), ('nar', nar)):
            model.load_state_dict(load_file(str(paths[key + '_training'])), strict=True, assign=True)
            model.cuda().eval().requires_grad_(False)
        ar.configure_engine('legacy', cancelled)
        nar_weights = saved['nar'] if saved else read_upstream_adapter(initial['nar'], 'nar')
        rank = next(value.shape[0] for key, value in nar_weights.items() if key.endswith('.A'))
        adapters = install_adapters(nar, 'nar', rank, nar_weights, shared_fused_a=False)
        for name in ('vae2llm', 'llm2vae'):
            getattr(nar, name).float().requires_grad_(True)
        restore_decoder_io(nar, nar_weights)
        head = RealAudioHead().cuda()
        head.load_state_dict(saved['head'] if saved else head_weights(initial['head']), strict=True)
        decoder = load_decoder(paths['vae_training'])
        spectral = SpectralLoss().cuda()
        neighbors = torch.as_tensor(np.load(paths['sem_nbr_idx'], allow_pickle=False).astype(np.int64), device='cuda')
        neighbor_weights = torch.as_tensor(np.load(paths['sem_nbr_cos'], allow_pickle=False), device='cuda').float().div(.05).softmax(-1)
        if neighbors.shape[0] != 32768 or neighbors.shape != neighbor_weights.shape or neighbors.min() < 0 or neighbors.max() >= 32768:
            raise ValueError('Invalid semantic-neighbor training tables')
        lora = [p for module in adapters.values() for p in (module.A, module.B)]
        io = list(nar.vae2llm.parameters()) + list(nar.llm2vae.parameters())
        parameters = list(head.parameters()) + lora + io
        optimizer = torch.optim.AdamW([
            {'params': list(head.parameters()), 'lr': options['head_learning_rate'], 'weight_decay': .05},
            {'params': lora, 'lr': options['nar_learning_rate'], 'weight_decay': 0},
            {'params': io, 'lr': options['io_learning_rate'], 'weight_decay': 0}], betas=(.9, .95))
        if saved:
            optimizer.load_state_dict(saved['optimizer']); rng.setstate(saved['python_rng'])
            torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved, nar_weights, lora, io
        embeddings = ar.model.embed_tokens.weight[CODEC_OFFSET:CODEC_OFFSET + 32768]
        torch.cuda.reset_peak_memory_stats()
        update_project(project['id'], pair_skipped_tracks=skipped)

        def encode(item, start, count, gradients):
            features = torch.as_tensor(item['features'][start:start + count][None], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = head(features)[0]
            if gradients:
                return straight_through_embeddings(logits, embeddings)
            idx = logits.float().argmax(-1)
            return embeddings[idx], idx

        def flow(item, start, count, t, noise, gradients, real=True):
            target = torch.tensor(np.array(item['latent'][start:start + count]), device='cuda')
            embed = encode(item, start, count, gradients)[0] if real else embeddings[torch.as_tensor(item['codec'][start:start + count], device='cuda').long()]
            return pair_flow(ar, nar, item['prefix'], embed, target, t, noise, gradients=gradients)

        @torch.no_grad()
        def evaluate():
            head.eval()
            values, waves, accuracy = [], [], []
            generator = torch.Generator(device='cpu').manual_seed(123)
            for item in groups['heldout']:
                if cancelled():
                    return {}
                count = min(512, len(item['latent'])); start = (len(item['latent']) - count) // 2
                noise = torch.randn(count, 64, generator=generator).cuda()
                audio = read_stereo(item['audio_path'])
                for t in (.2, .5, .8):
                    loss, prediction, mixed = flow(item, start, count, t, noise, False)
                    values.append(float(loss))
                    if t == .2:
                        waves.append(float(waveform_loss(decoder, spectral, mixed, prediction, t, audio, start)))
            for item in groups['minted_val']:
                count = min(512, len(item['latent'])); start = (len(item['latent']) - count) // 2
                _, idx = encode(item, start, count, False)
                accuracy.append(float((idx == torch.as_tensor(item['codec'][start:start + count], device='cuda')).float().mean()))
            return {'heldout_flow': sum(values) / len(values), 'heldout_spectral': sum(waves) / len(waves),
                    'anchor_top1': sum(accuracy) / len(accuracy)}

        def snapshot(scores):
            weights = adapter_snapshot(nar, adapters, include_io=True)
            head_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            temporary = latest.with_suffix('.tmp')
            torch.save({'contract': contract, 'step': completed, 'head': head_state, 'nar': weights,
                'optimizer': optimizer.state_dict(), 'python_rng': rng.getstate(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'audio_updates': audio_updates}, temporary)
            temporary.replace(latest)
            row = {'step': completed, 'file': f'head-{completed}.safetensors', 'nar_file': f'nar-{completed}.safetensors',
                   'scores': scores, 'audio_updates': audio_updates, 'elapsed_seconds': round(time.monotonic() - started),
                   'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
            for branch, tensors in (('head', head_state), ('nar', weights)):
                path = directory / row['file' if branch == 'head' else 'nar_file']
                tmp = path.with_suffix('.tmp'); save_file(tensors, str(tmp)); tmp.replace(path)
                row[branch + '_sha256'] = file_digest(path)
            history[:] = [r for r in history if r['step'] != completed] + [row]
            update_project(project['id'], pair_checkpoints=history, pair_completed_steps=completed,
                           pair_resume_available=True, pair_training_options=contract)

        if not options['resume']:
            snapshot(evaluate())
        for step in range(completed + 1, options['steps'] + 1):
            if cancelled():
                snapshot({}); raise InterruptedError('Sound adaptation stopped; checkpoint saved')
            head.train()
            factor = min(1, step / 50) * (.2 + .8 * .5 * (1 + math.cos(math.pi * min(step, 3000) / 3000)))
            for group, key in zip(optimizer.param_groups, ('head_learning_rate', 'nar_learning_rate', 'io_learning_rate')):
                group['lr'] = options[key] * factor
            real = rng.random() >= .25
            item = rng.choice(groups['artist' if real else 'minted'])
            count = 512; start = rng.randrange(len(item['latent']) - count + 1)
            t = min(.95 if real else .98, max(.05 if real else .02, rng.betavariate(2, 2)))
            optimizer.zero_grad(set_to_none=True)
            loss, prediction, mixed = flow(item, start, count, t, torch.randn(count, 64, device='cuda'), True, real)
            flow_value, aux_value, ce_value = float(loss.detach()), 0., 0.
            if real and t <= options['audio_tmax']:
                auxiliary = waveform_loss(decoder, spectral, mixed, prediction, t, read_stereo(item['audio_path']), start,
                                         frames=options['audio_frames'], margin=options['audio_margin'])
                aux_value = float(auxiliary.detach()); loss = loss + options['audio_weight'] * auxiliary
                audio_updates += 1
                del auxiliary
            if not torch.isfinite(loss):
                raise ValueError('Sound adaptation produced a non-finite loss')
            loss.backward()
            del loss, prediction, mixed
            # Accumulate the author's 16-anchor CE batch in small microbatches.
            # This preserves its objective without retaining all 16 head graphs.
            if real:
                for offset in range(0, options['anchor_batch'], options['anchor_microbatch']):
                    size = min(options['anchor_microbatch'], options['anchor_batch'] - offset)
                    xs, ys = [], []
                    for _ in range(size):
                        anchor = rng.choice(groups['minted']); s = rng.randrange(len(anchor['codec']) - 512 + 1)
                        xs.append(anchor['features'][s:s + 512]); ys.append(anchor['codec'][s:s + 512])
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = head(torch.as_tensor(np.stack(xs), device='cuda'))
                    ce = soft_token_loss(logits, torch.as_tensor(np.stack(ys), device='cuda').long(), neighbors, neighbor_weights)
                    ce = ce * size / options['anchor_batch']; ce_value += float(ce.detach())
                    ce.backward(); del ce, logits
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            completed = step
            report(f"Adapting tokenizer + sound {step}/{options['steps']} · flow {flow_value:.3f} · audio {aux_value:.3f} · anchor {ce_value:.3f} · {audio_updates} waveform updates", 100 * step / options['steps'])
            if step % options['checkpoint_every'] == 0 or step == options['steps'] or cancelled():
                snapshot({} if cancelled() else evaluate())
        return history
    finally:
        ar = nar = head = decoder = optimizer = spectral = None
        adapters.clear(); parameters.clear()
        gc.collect(); torch.cuda.empty_cache()
