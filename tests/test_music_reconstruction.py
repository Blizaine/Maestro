"""Source identity, fixed-token comparisons and queue isolation for diagnostics."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import music_styles, music_training
from services.job_lifecycle import request_cancel
from services.music_reconstruction import reconstruction_options
from services.music_training_runner import MusicTrainingRunner
from models.TTS.yue2.music_assets import MERT_REVISION, TOKENIZER_REVISION
from models.TTS.yue2.protocol import CODEC_OFFSET
from models.TTS.yue2.reconstruction import prepared_tracks


class ReconstructionTests(unittest.TestCase):
    def test_audio_comparison_is_explicit_and_rejects_malformed_modes(self):
        options = reconstruction_options({**self.request, 'comparison': 'audio'}, self.project)
        self.assertEqual(options['comparison'], 'audio')
        for value in ({}, [], None, 'unknown'):
            with self.assertRaises(ValueError):
                reconstruction_options({**self.request, 'comparison': value}, self.project)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for module, name, path in ((music_training, "PROJECT_ROOT", self.root / "projects"),
                                   (music_styles, "STYLE_ROOT", self.root / "styles")):
            context = patch.object(module, name, path)
            context.start()
            self.addCleanup(context.stop)
        tracks = []
        for number in range(2):
            path = self.root / f"song-{number}.wav"
            path.write_bytes(bytes([number]) * 200)
            tracks.append({"audio_path": str(path), "lyrics": "[Verse]\nMorning light",
                           "style": "Acoustic folk", "holdout": number == 1})
        self.project = music_training.create_project("Original songs", "My acoustic sound", tracks, pair='v4')
        self.cache = {"dataset_digest": self.project["dataset_digest"], "mert_revision": MERT_REVISION,
                      "tokenizer_revision": TOKENIZER_REVISION, "feature_layer": 20, "frame_rate": 25}
        self.project = music_training.update_project(self.project["id"], prepared=self.cache,
            status="completed", completed_steps=800, progress=100, job_id="training-finished", resume_available=True)
        directory = music_training.project_directory(self.project["id"]) / "prepared"
        directory.mkdir()
        for track in self.project["tracks"]:
            np.save(directory / f"{track['id']}.npy", np.arange(2000, dtype=np.int32))
            (directory / f"{track['id']}.json").write_text(json.dumps(self.cache))
        style = music_styles.style_directory("checkpoint")
        style.mkdir(parents=True)
        manifest = {"id": "checkpoint", "name": "Step 800", "trigger": "My acoustic sound",
                    "version": 1, "architecture": "yue2", "base_revision": music_styles.BASE_REVISION,
                    "tokenizer_revision": TOKENIZER_REVISION,
                    "training": {"project_id": self.project["id"], "dataset_digest": self.project["dataset_digest"]}}
        for branch in ("ar", "nar"):
            path = style / f"{branch}.safetensors"
            path.write_bytes(branch.encode())
            manifest[branch] = {"file": path.name, "sha256": music_styles.file_digest(path)}
        (style / "style.json").write_text(json.dumps(manifest))
        self.request = {"style_id": "checkpoint", "seconds": 10}
        self.project_file = music_training.project_directory(self.project["id"]) / "project.json"

    def test_options_reject_unbounded_or_foreign_inputs(self):
        for change in ({"seconds": 61}, {"seconds": float("nan")}, {"steps": True}, {"seed": 1.2},
                       {"artist_strength": True}, {"artist_strength": -0.5},
                       {"artist_strength": float("nan")}, {"artist_strength": 1.6},
                       {"track_ids": ["../escape"]}, {"track_ids": []},
                       {"track_ids": [self.project["tracks"][0]["id"]] * 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                reconstruction_options({**self.request, **change}, self.project)
        other = {**self.project, "id": "different-project"}
        with self.assertRaisesRegex(ValueError, "this training project"):
            reconstruction_options(self.request, other)

    def test_reconstruction_strength_defaults_and_fractional_override(self):
        self.assertEqual(reconstruction_options(self.request, self.project)["artist_strength"], 1.0)
        options = reconstruction_options({**self.request, "artist_strength": 0.5}, self.project)
        self.assertEqual(options["artist_strength"], 0.5)
        self.assertEqual(options["ar_sha256"], music_styles.file_digest(
            music_styles.style_directory("checkpoint") / "ar.safetensors"))

    def test_joint_reconstruction_compares_both_adapters(self):
        path = music_styles.style_directory('checkpoint') / 'style.json'
        manifest = json.loads(path.read_text())
        manifest.update(version=2, adapter_mode='joint')
        path.write_text(json.dumps(manifest))
        self.assertEqual(reconstruction_options(self.request, self.project)['comparison'], 'joint')
        for comparison in ('ar', 'audio'):
            with self.assertRaisesRegex(ValueError, 'both adapters'):
                reconstruction_options({**self.request, 'comparison': comparison}, self.project)

    def test_fixed_tokens_keep_order_offset_duration_and_holdout(self):
        options = reconstruction_options(self.request, self.project)
        tracks = prepared_tracks(self.project, options)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0][1], list(range(CODEC_OFFSET, CODEC_OFFSET + 250)))
        self.assertFalse(tracks[0][0]["holdout"])
        self.assertTrue(tracks[1][0]["holdout"])
        self.assertEqual(tracks[0][2], tracks[1][2])

    def test_changed_recordings_and_cache_revision_are_rejected(self):
        options = reconstruction_options(self.request, self.project)
        with self.assertRaisesRegex(ValueError, "current music tokenizer"):
            prepared_tracks({**self.project, "prepared": {}}, options)
        track = self.project["tracks"][0]
        marker = music_training.project_directory(self.project["id"]) / "prepared" / f"{track['id']}.json"
        marker.write_text("{}")
        with self.assertRaisesRegex(ValueError, "do not match"):
            prepared_tracks(self.project, options)
        marker.write_text(json.dumps(self.cache))
        Path(track["audio_path"]).write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "source recording changed"):
            prepared_tracks(self.project, options)

    def test_codec_range_validation_rejects_text_ids_or_float_codes(self):
        options = reconstruction_options(self.request, self.project)
        path = music_training.project_directory(self.project["id"]) / "prepared" / f"{self.project['tracks'][0]['id']}.npy"
        for values in (np.array([CODEC_OFFSET]), np.array([-1]), np.array([0.5]), np.zeros((1, 2), dtype=int)):
            np.save(path, values)
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "invalid YuE2 codec"):
                prepared_tracks(self.project, options)

    def test_cancel_before_gpu_slot_leaves_training_and_loaded_model_alone(self):
        before = self.project_file.read_bytes()
        jobs, states, lock = {}, {}, threading.Lock()
        release = Mock()
        runner = MusicTrainingRunner(jobs, lock, states, release, lambda: "auditions", lambda ws: self.root / ws)
        lock.acquire()
        try:
            with patch("threading.Thread.start"):
                submitted = runner.submit("reconstruct", self.project["id"], self.request)
            job = jobs[submitted["job_id"]]
            request_cancel(job)
            runner.run(job["id"])
        finally:
            lock.release()
        release.assert_not_called()
        self.assertEqual(self.project_file.read_bytes(), before)
        self.assertEqual(job["status"], "cancelled")
        self.assertFalse(states)
        self.assertFalse(runner.project_workers)

    def test_running_diagnostic_owns_slot_and_preserves_project_on_success_or_failure(self):
        before = self.project_file.read_bytes()
        for fails in (False, True):
            jobs, states, lock = {}, {}, threading.Lock()
            release = Mock()
            runner = MusicTrainingRunner(jobs, lock, states, release, lambda: "auditions", lambda ws: self.root / ws)
            def reconstruct(project, options, output_dir, job_id, *, report, cancelled, publish):
                self.assertTrue(lock.locked())
                self.assertEqual(Path(output_dir), self.root / "auditions")
                self.assertFalse(cancelled())
                report("Rendering", 50)
                publish(["comparison.wav"])
                if fails:
                    raise RuntimeError("test decode failure")
                return {"files": ["comparison.wav"], "report": "comparison.json"}
            fake = types.SimpleNamespace(reconstruct_project=reconstruct)
            with patch("threading.Thread.start"):
                submitted = runner.submit("reconstruct", self.project["id"], self.request)
            with patch.dict(sys.modules, {"models.TTS.yue2.reconstruction": fake}), patch("traceback.print_exc"):
                runner.run(submitted["job_id"])
            job = jobs[submitted["job_id"]]
            self.assertEqual(job["status"], "failed" if fails else "completed")
            self.assertEqual(job["output_files"], ["comparison.wav"])
            self.assertEqual(self.project_file.read_bytes(), before)
            self.assertFalse(lock.locked())
            self.assertFalse(states)
            release.assert_called_once()

    def test_acoustic_noise_is_repeatable_cpu_data_with_non_cpu_default(self):
        import torch
        from models.TTS.yue2.protocol import chunk_ranges, MUSIC_END
        # Exercise the production method without importing/loading its models.
        source = Path(__file__).resolve().parents[1] / "app/models/TTS/yue2/pipeline.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "YuE2Pipeline")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "decode_codec")
        method.decorator_list = []
        scope = {"torch": torch, "chunk_ranges": chunk_ranges, "MUSIC_END": MUSIC_END,
                 "tqdm": lambda **kw: unittest.mock.MagicMock()}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
        received, conditioning = [], []
        def synthesize(noise, cache, length, steps, report):
            self.assertEqual(noise.device.type, "cpu")
            received.append(noise.clone())
            return noise
        def condition(ids):
            conditioning.append(ids)
            return []
        pipeline = types.SimpleNamespace(text_encoder=types.SimpleNamespace(condition=condition),
            transformer=types.SimpleNamespace(synthesize=synthesize),
            vae=types.SimpleNamespace(decode_tiled=lambda latent, **kw: latent[:, :2]))
        before = torch.get_default_device()
        try:
            torch.set_default_device("meta")
            for seed in (123, 123, 124):
                scope["decode_codec"](pipeline, [1, 2], [CODEC_OFFSET, CODEC_OFFSET + 1], seed)
        finally:
            torch.set_default_device(before)
        self.assertTrue(torch.equal(received[0], received[1]))
        self.assertFalse(torch.equal(received[0], received[2]))
        self.assertEqual(conditioning, [[1, 2, CODEC_OFFSET, CODEC_OFFSET + 1, MUSIC_END]] * 3)

    def test_generation_fingerprints_actual_acoustic_inputs_independent_of_renderer(self):
        import torch
        from dataclasses import replace
        from models.TTS.yue2.protocol import GenerationConfig, SongRequest
        source = Path(__file__).resolve().parents[1] / "app/models/TTS/yue2/pipeline.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "YuE2Pipeline")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_generate")
        method.decorator_list = []
        scope = {"torch": torch, "replace": replace, "SongRequest": SongRequest,
                 "hashlib": hashlib, "json": json,
                 "token_prefixes": lambda *args: [101, 102]}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
        codec = [CODEC_OFFSET, CODEC_OFFSET + 1]
        pipeline = types.SimpleNamespace(frame_rate=25, sample_rate=48000,
            generation_config=GenerationConfig(), lm_decoder_engine="legacy", tokenizer=None,
            engine=types.SimpleNamespace(release_runtime_allocations=Mock()),
            _tokens=lambda *args: list(codec),
            decode_codec=Mock(return_value=torch.zeros(1, 2, 3)))
        kwargs = dict(input_prompt="[Verse]\nA new song", alt_prompt="Acoustic folk", seed=123,
            duration_seconds=10, sampling_steps=32, guide_scale=1, temperature=1,
            top_k=100, top_p=.95, model_mode=2, VAE_tile_size=256)
        first = scope["_generate"](pipeline, **kwargs)
        pipeline.decode_codec.assert_called_once_with([101, 102], list(codec), 123, 32, 256, None)
        pipeline.decode_codec.return_value = torch.ones(1, 2, 3) * .5
        second = scope["_generate"](pipeline, **kwargs)
        identity = first["artifact_metadata"]["acoustic_inputs"]
        self.assertEqual(identity, second["artifact_metadata"]["acoustic_inputs"])
        self.assertFalse(torch.equal(first["x"], second["x"]))
        codec.reverse()  # Equal duration does not establish an equal performance.
        third = scope["_generate"](pipeline, **kwargs)["artifact_metadata"]["acoustic_inputs"]
        self.assertEqual(identity["codec_tokens"], third["codec_tokens"])
        self.assertNotEqual(identity["codec_sha256"], third["codec_sha256"])
        self.assertEqual(identity["prefix_sha256"], third["prefix_sha256"])
        changed_seed = scope["_generate"](pipeline, **{**kwargs, "seed": 124})["artifact_metadata"]["acoustic_inputs"]
        self.assertNotEqual(third["noise_seed"], changed_seed["noise_seed"])


if __name__ == "__main__":
    unittest.main()
