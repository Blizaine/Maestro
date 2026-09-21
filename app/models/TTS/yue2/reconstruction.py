"""Reconstruct prepared source tokens without sampling a new song or training.

Runs only inside MusicTrainingRunner's shared generation slot. Original audio,
prepared tokens, training state and checkpoint files remain read-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from services.music_styles import file_digest, load_style
from services.music_training import project_directory
from .music_assets import MERT_REVISION
from services.music_contracts import tokenizer_pair
from .protocol import CODEC_OFFSET, CODEC_SIZE, SongRequest, token_prefixes


def prepared_tracks(project, options):
    expected = {"dataset_digest": project["dataset_digest"], "mert_revision": MERT_REVISION,
                "tokenizer_revision": tokenizer_pair(project)['revision'], "feature_layer": 20, "frame_rate": 25}
    if project.get("prepared") != expected:
        raise ValueError("Prepare this project again with the current music tokenizer")
    directory = project_directory(project["id"]) / "prepared"
    by_id = {track["id"]: track for track in project["tracks"]}
    result = []
    for track_id in options["track_ids"]:
        track = by_id[track_id]
        if file_digest(Path(track["audio_path"])) != track["audio_sha256"]:
            raise ValueError("A source recording changed after preparation; create a new training project")
        marker, path = directory / f"{track_id}.json", directory / f"{track_id}.npy"
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Prepared recording tokens do not match this dataset")
        raw = np.load(path, allow_pickle=False)
        if (raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer)
                or raw.min() < 0 or raw.max() >= CODEC_SIZE):
            raise ValueError("Prepared recording contains invalid YuE2 codec IDs")
        excerpt = raw[:options["seconds"] * 25]
        codec = [int(value) + CODEC_OFFSET for value in excerpt]
        digest = hashlib.sha256(excerpt.astype("<i4").tobytes()).hexdigest()
        result.append((track, codec, digest))
    return result


def reconstruct_project(project, options, output_dir, job_id, *, report, cancelled, publish):
    import soundfile as sf
    import torch
    import wgp
    from .artist_adapter import active_artist

    def check_cancel():
        if cancelled():
            raise InterruptedError("Music reconstruction cancelled")

    check_cancel()
    tracks = prepared_tracks(project, options)
    style = load_style(options["style_id"], verify=True)
    if style.get('tokenizer_revision') != tokenizer_pair(project)['revision']:
        raise ValueError('Choose a style trained with this project\'s tokenizer pair for reconstruction')
    compare_audio = options.get('comparison') == 'audio'
    if any(style[branch]["sha256"] != options[f"{branch}_sha256"] for branch in ("ar", "nar")):
        raise ValueError("The selected checkpoint changed while this diagnostic was queued")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"music-reconstruction-{job_id}"
    manifest_path = directory / f"{stem}.json"
    files = []
    manifest = {"version": 1, "project_id": project["id"], "job_id": job_id,
                "source_dataset_digest": project["dataset_digest"], "options": options,
                "checkpoint": style.get("training", {}).get("checkpoint"),
                "tokenizer_revision": tokenizer_pair(project)['revision'], "records": [], "status": "running",
                "comparison": "Fixed source tokens; AR adapter off/on; matching NAR adapter enabled in both"}
    if compare_audio:
        manifest['comparison'] = 'Fixed source tokens and AR conditioning; matched audio adapter versus personal audio adapter'
        manifest['audio_checkpoint'] = style.get('training', {}).get('audio_checkpoint')
    elif options.get('comparison') == 'joint':
        manifest['comparison'] = 'Fixed source tokens; joint AR and NAR adapters off/on, without a generic decoder adapter'

    def record(path, details):
        manifest["records"].append({"file": path.name, **details})
        files.append(path.name)
        label = details["variant"]
        if "artist_strength" in details:
            label += f" · {manifest['checkpoint']} · strength {details['artist_strength']:g}"
        sidecar = {"params": {"model_type": "yue2", "generation_mode": "audio", "seed": options["seed"],
                              "prompt": f"Source reconstruction: {details['track_style']} — {label}",
                              "alt_prompt": details["track_style"]},
                   "generation_mode": "audio", "job_id": job_id, "created_at": time.time(),
                   "output_filename": path.name, "model_details": {"reconstruction": details}}
        path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        publish(files)

    pipeline = None
    try:
        report("Loading YuE2 with the current generation precision/profile", 0)
        # Use the same loader, offloading and precision as normal auditions.
        wgp.wan_model, wgp.offloadobj = wgp.load_models("yue2", output_type="audio")
        pipeline = wgp.wan_model
        pipeline.text_encoder.abort_fn = pipeline.transformer.abort_fn = cancelled
        manifest["engine"] = pipeline.lm_decoder_engine
        total = len(tracks) * 2
        for index, (track, codec, digest) in enumerate(tracks):
            check_cancel()
            seconds = len(codec) / 25
            common = {"track_id": track["id"], "track_style": track["style"], "heldout": track["holdout"],
                      "source_sha256": track["audio_sha256"], "codec_sha256": digest,
                      "codec_frames": len(codec), "start_seconds": 0, "seconds": seconds,
                      "sample_rate": 48000, "channels": 2}
            original = directory / f"{stem}-{index + 1}-original.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-i", track["audio_path"],
                            "-t", str(seconds), "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(original)],
                           capture_output=True, check=True, timeout=120)
            record(original, {**common, "variant": "original"})
            request = SongRequest(style=f"{project['trigger']}, {track['style']}", lyrics=track["lyrics"],
                                  cot="off", seed=options["seed"], cfg_scale=1)
            prefix = token_prefixes(request, pipeline.tokenizer)
            variants = (("adapter-off", 0.0), ("adapter-on", options.get("artist_strength", 1.0)))
            if compare_audio:
                variants = (('audio-before', options['artist_strength']), ('audio-after', options['artist_strength']))
            for variant_index, (variant, strength) in enumerate(variants):
                check_cancel()
                ordinal = index * 2 + variant_index
                report(f"Recording {index + 1}/{len(tracks)}: {variant}, strength {strength:g} ({seconds:g}s)", ordinal / total * 100)

                def callback(**progress):
                    check_cancel()
                    # Decoding follows synthesis; keep displayed progress monotonic.
                    phase = progress.get("denoising_extra", "")
                    if "decoding" in phase:
                        fraction = 0.95
                    else:
                        fraction = 0.9 * (progress.get("step_idx", 0) + 1) / max(1, progress.get("override_num_inference_steps", 1))
                    report(f"Recording {index + 1}/{len(tracks)}: {variant} — {phase}", (ordinal + fraction) / total * 100)

                settings = {"artist_id": options["style_id"], "artist_strength": strength}
                if compare_audio:
                    settings['base_acoustic'] = variant_index == 0
                with active_artist(pipeline, settings):
                    audio = pipeline.decode_codec(prefix, codec, options["seed"], options["steps"], 1024, callback)
                    check_cancel()
                    if not torch.isfinite(audio).all():
                        raise FloatingPointError("Music reconstruction produced non-finite audio")
                    path = directory / f"{stem}-{index + 1}-{variant}.wav"
                    sf.write(str(path), audio[0].float().cpu().clamp(-1, 1).T.numpy(), 48000, subtype="PCM_24")
                pipeline.last_latents = None
                from .music_assets import ASSETS
                record(path, {**common, "variant": variant, "artist_strength": strength,
                              "adapter_mode": style.get('adapter_mode', 'separate'),
                              "nar_strength": strength if options.get('comparison') == 'joint' else 1.0,
                              "style_id": options["style_id"], "seed": options["seed"],
                              "steps": options["steps"], "ar_sha256": options["ar_sha256"],
                              "nar_sha256": ASSETS[tokenizer_pair(project)['nar']]['sha256'] if compare_audio and variant_index == 0 else options["nar_sha256"],
                              "checkpoint": manifest["checkpoint"], "audio_checkpoint": manifest.get('audio_checkpoint')})
        # Confirm the inputs also remained unchanged through the GPU work.
        for track, _, _ in tracks:
            if file_digest(Path(track["audio_path"])) != track["audio_sha256"]:
                raise ValueError("A source recording changed during reconstruction")
        manifest["status"] = "completed"
        return {"files": files, "report": manifest_path.name}
    except BaseException as error:
        manifest["status"] = "cancelled" if isinstance(error, InterruptedError) else "failed"
        manifest["error"] = str(error)
        raise
    finally:
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        finally:
            try:
                if pipeline is not None:
                    pipeline.release()
            finally:
                wgp.release_model()
