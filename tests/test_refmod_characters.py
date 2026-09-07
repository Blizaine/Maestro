"""Portable character round trips and the native H3 conditioning contract."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from services import character_library as library, refmod, character_transfer as transfer


class CharacterFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.cwd)
        self.uploads = Path.cwd() / "uploads"
        self.uploads.mkdir()
        self.portrait = self.uploads / "portrait.png"
        Image.new("RGB", (64, 64), (80, 100, 120)).save(self.portrait)
        self.voice = self.uploads / "voice.wav"
        import wave
        with wave.open(str(self.voice), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x00" * 48000)
        self.probe = mock.patch.object(library, "_probe_duration", return_value=3.0)
        self.probe.start()
        self.addCleanup(self.probe.stop)
        self.latent = torch.arange(24 * 4 * 4, dtype=torch.float32).reshape(1, 24, 1, 4, 4) / 500

    def saved(self, voice=True):
        return library.create_character(name="Blaine", visual_path=str(self.portrait),
            visual_type="image", voice_path=str(self.voice) if voice else None)

    def exported(self, voice=True):
        character = self.saved(voice)
        return character, library.export_character(character["id"], encode_visual=lambda *_: self.latent)

    def test_export_keeps_legacy_contract_and_original_voice_bytes(self):
        character, path = self.exported()
        self.assertEqual(path.name, "blaine.maestro.safetensors")
        with safe_open(str(path), framework="pt") as reader:
            metadata = json.loads(reader.metadata()["refmod_meta"])
            self.assertEqual(metadata["_format_version"], 2)
            self.assertEqual(metadata["kind"], "image")
            self.assertTrue(torch.equal(reader.get_tensor("latent"), self.latent))
        info, tensors = refmod.load_refmod(path)
        audio = info["character"]["audio"][0]
        self.assertEqual(audio["role"], "voice")
        self.assertEqual(tensors[audio["tensor"]].numpy().tobytes(), self.voice.read_bytes())
        self.assertNotIn(str(Path.cwd()), json.dumps(info["metadata"]))
        self.assertTrue(Path(library.get_character(character["id"])["refmod"]["path"]).is_file())

    def test_move_to_fresh_library_preserves_voice_appearance_and_extensions(self):
        _, path = self.exported()
        info, tensors = refmod.load_refmod(path)
        tensors["other_app.extra"] = torch.tensor([42])
        info["metadata"]["other_app_meta"] = "opaque future data"
        refmod.write_refmod(path, tensors, info["metadata"])
        fresh = Path.cwd() / "another_machine"
        fresh.mkdir()
        os.chdir(fresh)
        character = library.import_character(path)
        self.assertTrue(Path(character["visual"]["path"]).is_relative_to(fresh))
        self.assertEqual(Path(character["voice"]["path"]).read_bytes(), self.voice.read_bytes())
        again = library.export_character(character["id"], encode_visual=mock.Mock(side_effect=AssertionError("should reuse latent")))
        loaded, data = refmod.load_refmod(again)
        self.assertEqual(loaded["metadata"]["other_app_meta"], "opaque future data")
        self.assertEqual(data["other_app.extra"].item(), 42)
        self.assertTrue(torch.equal(data["latent"], self.latent))
        self.assertEqual(Path(character["visual"]["path"]).read_bytes(), self.portrait.read_bytes())

    def test_plain_refmod_import_preview_then_attach_voice_and_export(self):
        path = Path.cwd() / "external.safetensors"
        save_file({"latent": self.latent}, str(path), metadata=refmod.new_refmod_metadata("Someone", self.latent, source="image"))
        def preview(latent, directory, kind):
            self.assertTrue(torch.equal(latent, self.latent))
            self.assertEqual(kind, "image")
            target = directory / "visual.png"
            target.write_bytes(self.portrait.read_bytes())
            return target
        character = library.import_character(path, decode_preview=preview)
        self.assertIsNone(character["voice"])
        self.assertTrue(character["refmod"]["preview_generated"])
        library.attach_character_voice(character["id"], str(self.voice))
        output = library.export_character(character["id"])
        info, data = refmod.load_refmod(output)
        self.assertEqual(len(info["character"]["audio"]), 1)
        self.assertTrue(torch.equal(data["latent"], self.latent))

    def test_legacy_sidecar_is_promoted_to_embedded_base_metadata(self):
        path = Path.cwd() / "legacy.safetensors"
        save_file({"latent": self.latent}, str(path))
        self.assertFalse(refmod.is_refmod_file(path))
        path.with_suffix(".json").write_text(json.dumps({"name": "Old", "kind": "image"}))
        refmod._is_refmod.cache_clear()
        self.assertTrue(refmod.is_refmod_file(path))
        info = refmod.inspect_refmod(path)
        self.assertEqual(info["refmod"]["name"], "Old")

    def test_non_refmod_weights_are_not_characters(self):
        path = Path.cwd() / "lora.safetensors"
        save_file({"some.lora_A.weight": torch.zeros(2, 2)}, str(path))
        self.assertFalse(refmod.is_refmod_file(path))
        with self.assertRaisesRegex(ValueError, "not an H3 RefMod"):
            library.import_character(path)

    def test_failed_import_leaves_no_character_or_media_directory(self):
        _, path = self.exported()
        info, tensors = refmod.load_refmod(path)
        tensors["maestro.visual.bytes"] = torch.tensor([0, 1, 2], dtype=torch.uint8)
        save_file(tensors, str(path), metadata=info["metadata"])
        before = list(library._root().iterdir())
        with self.assertRaises(Exception):
            library.import_character(path)
        self.assertEqual(list(library._root().iterdir()), before)
        self.assertEqual(len(library.list_characters()), 1)

    def test_invalid_tensor_and_audio_metadata_are_rejected(self):
        _, path = self.exported()
        info, data = refmod.load_refmod(path)
        for mutation in ("nan", "shape", "kind", "audio_type", "extension", "future"):
            with self.subTest(mutation=mutation):
                metadata = dict(info["metadata"])
                tensors = {key: value.clone() for key, value in data.items()}
                extension = json.loads(metadata[refmod.CHARACTER_META])
                if mutation == "nan": tensors["latent"].fill_(float("nan"))
                if mutation == "shape": tensors["latent"] = torch.zeros(1, 16, 1, 4, 4)
                if mutation == "kind": extension["visual"]["type"] = "video"
                if mutation == "audio_type": extension["audio"][0]["type"] = "video"
                if mutation == "extension": extension["audio"][0]["extension"] = "/../file.py"
                if mutation == "future": extension["schema_version"] = 100
                metadata[refmod.CHARACTER_META] = json.dumps(extension)
                bad = Path.cwd() / "bad.safetensors"
                save_file(tensors, str(bad), metadata=metadata)
                self.assertTrue(refmod.is_refmod_file(bad))
                with self.assertRaises(ValueError): library.import_character(bad)

    def test_missing_voice_does_not_silently_export_visual_only(self):
        character, _ = self.exported()
        Path(character["voice"]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "saved voice is missing"):
            library.export_character(character["id"])

    def test_voice_replacement_is_embedded_on_next_export(self):
        character, path = self.exported()
        self.voice.write_bytes(self.voice.read_bytes() + b"replacement")
        library.attach_character_voice(character["id"], str(self.voice))
        library.export_character(character["id"])
        info, data = refmod.load_refmod(path)
        self.assertEqual(data[info["character"]["audio"][0]["tensor"]].numpy().tobytes(), self.voice.read_bytes())

    def test_compressed_voice_is_decoded_for_tts_but_preserved_for_sharing(self):
        _, path = self.exported()
        info, tensors = refmod.load_refmod(path)
        extension = info["character"]
        extension["audio"][0]["extension"] = ".m4a"
        tensors["maestro.audio.0.bytes"] = torch.tensor(list(b"compressed recording"), dtype=torch.uint8)
        info["metadata"][refmod.CHARACTER_META] = json.dumps(extension)
        refmod.write_refmod(path, tensors, info["metadata"])
        with mock.patch.object(library, "_extract_voice", side_effect=lambda source, target: target.write_bytes(self.voice.read_bytes())):
            imported = library.import_character(path)
        self.assertTrue(imported["voice"]["path"].endswith(".wav"))
        exported = library.export_character(imported["id"])
        result, data = refmod.load_refmod(exported)
        self.assertEqual(result["character"]["audio"][0]["extension"], ".m4a")
        self.assertEqual(data["maestro.audio.0.bytes"].numpy().tobytes(), b"compressed recording")

    def test_delete_cannot_race_an_active_export(self):
        character = self.saved()
        started, release = threading.Event(), threading.Event()
        errors = []
        def encoder(*_):
            started.set()
            release.wait(5)
            return self.latent
        def export():
            try: library.export_character(character["id"], encode_visual=encoder)
            except Exception as error: errors.append(error)
        worker = threading.Thread(target=export)
        worker.start()
        try:
            self.assertTrue(started.wait(3))
            with self.assertRaisesRegex(ValueError, "being exported"):
                library.delete_character(character["id"])
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(errors)

    def test_manifest_retains_native_reference_and_rejects_outside_paths(self):
        from models.minimax_h3.reference_manifest import validate_reference_manifest
        from models.minimax_h3.reference_media import normalize_reference_manifest
        character, _ = self.exported()
        character = library.get_character(character["id"])
        item = {"type": "image", "path": character["visual"]["path"], "refmod_path": character["refmod"]["path"],
                "library_character_id": character["id"], "role": "Blaine"}
        result = validate_reference_manifest(normalize_reference_manifest([item]))
        self.assertEqual(result[0]["refmod_path"], item["refmod_path"])
        with self.assertRaisesRegex(ValueError, "Background removal"):
            validate_reference_manifest([{**item, "remove_background": True}])
        with self.assertRaisesRegex(ValueError, "library first"):
            validate_reference_manifest([{**item, "refmod_path": str(Path.cwd() / "outside.safetensors")}])

    def test_native_h3_uses_refmod_without_vae_or_second_normalization(self):
        from models.minimax_h3.packing import patchify_video_latents
        source = ast.parse((ROOT / "app/models/minimax_h3/minimax_h3_main.py").read_text(encoding="utf-8"))
        node = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "_encode_references")
        namespace = {"torch": torch, "_keyframe_latent_stats_cpu": lambda: (0, 1),
            "AUDIO_LATENTS_MEAN": [0] * 32, "AUDIO_LATENTS_STD": [1] * 32,
            "MINIMAX_H3_PIXEL_MEAN": [0] * 3, "MINIMAX_H3_PIXEL_STD": [1] * 3,
            "MINIMAX_H3_KEYFRAME_NOISE_AUG": 0.1, "patchify_video_latents": patchify_video_latents,
            "keyframe_condition_noise": lambda shapes, patch_size, channels, **kwargs: torch.zeros(sum(t * (h//2) * (w//2) for t,h,w in shapes), channels * 4)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "native_reference_encode", "exec"), namespace)
        latent = self.latent.repeat(1, 1, 3, 1, 1)
        reference = SimpleNamespace(kind="video", refmod_latent=latent, has_audio=False)
        scheduler = SimpleNamespace(scale_noise=mock.Mock(side_effect=lambda values, *_: values))
        model = SimpleNamespace(_interrupt=False, device="cpu", patch_size=(1, 2, 2), scheduler=scheduler)
        video, audio = namespace["_encode_references"](model, [reference], torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(video, patchify_video_latents(latent, model.patch_size)))
        self.assertIsNone(audio)
        self.assertEqual(reference.num_latent_frames, 3)
        scheduler.scale_noise.assert_called_once()

    def test_reference_preparation_loads_latents_and_keeps_semantic_image(self):
        from models.minimax_h3.ref2va import prepare_references
        character, _ = self.exported()
        character = library.get_character(character["id"])
        prepared = prepare_references([{"type": "image", "path": character["visual"]["path"],
            "refmod_path": character["refmod"]["path"], "role": "Blaine"}],
            num_frames=49, target_height=256, target_width=256)
        self.assertTrue(torch.equal(prepared[0].refmod_latent, self.latent))
        self.assertIsNotNone(prepared[0].image)
        self.assertEqual(prepared[0].num_video_rows, 4)

    def test_filename_is_portable(self):
        for name, expected in [("Blaine", "blaine"), ("My Character", "my_character"), ("../BLAINE", "blaine"), ("", "character"), ("NUL", "character_nul")]:
            self.assertEqual(refmod.character_filename(name), expected + ".maestro.safetensors")

    def test_lora_guide_scan_skips_refmods_but_keeps_all_real_adapters(self):
        import asyncio
        import uuid
        _, character_file = self.exported()
        loras = Path.cwd() / "loras"
        loras.mkdir()
        (loras / "character.safetensors").write_bytes(character_file.read_bytes())
        for name in ("one", "two"):
            save_file({"lora_A.weight": torch.zeros(2, 2)}, str(loras / f"{name}.safetensors"))
        source = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        node = next(node for node in source.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "scan_and_generate_guides")
        node.decorator_list = []
        namespace = {"os": os, "uuid": uuid, "Request": object, "threading": threading,
            "wgp": SimpleNamespace(server_config={"loras_root": str(loras)}),
            "_get_linked_model_folders": lambda: [], "_register_lora_guide_scan": mock.Mock()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "guide_scan", "exec"), namespace)
        async def body(): return {}
        with mock.patch.object(threading, "Thread"):
            result = asyncio.run(namespace["scan_and_generate_guides"](SimpleNamespace(json=body)))
        self.assertEqual(result["total"], 2)


    def test_upload_export_download_routes(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        _, path = self.exported()
        app = FastAPI()
        codec = mock.Mock(side_effect=AssertionError("Complete characters should not need a GPU"))
        app.include_router(transfer.build_router(codec))
        def finish(client, response):
            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["id"]
            for _ in range(200):
                job = client.get(f"/api/v1/character-transfers/{job_id}").json()
                if job["status"] != "running":
                    self.assertEqual(job["status"], "completed", job)
                    return job["result"]
                time.sleep(0.01)
            self.fail("Transfer did not finish")
        with TestClient(app) as client:
            imported = finish(client, client.post("/api/v1/characters/import", files={"file": (path.name, path.read_bytes())}))["character"]
            self.assertIsNotNone(imported["voice"])
            output = finish(client, client.post(f"/api/v1/characters/{imported['id']}/export"))
            response = client.get(output["url"])
            self.assertEqual(response.status_code, 200, response.text[:100])
            self.assertIn("blaine.maestro.safetensors", response.headers["content-disposition"])
            downloaded = Path.cwd() / "download.safetensors"
            downloaded.write_bytes(response.content)
            self.assertEqual(refmod.inspect_refmod(downloaded)["character"]["name"], "Blaine")
            self.assertEqual(client.post("/api/v1/characters/import", files={"file": ("bad.py", b"bad")}).status_code, 400)
        codec.assert_not_called()


class RemoteCharacterTests(unittest.TestCase):
    @mock.patch("requests.get")
    def test_exact_huggingface_file_is_selected_from_collection(self, get):
        get.return_value.json.return_value = {"sha": "pinned", "siblings": [
            {"rfilename": "Other.safetensors"}, {"rfilename": "people/Blaine refmod.safetensors"}]}
        result = transfer.remote_files("https://huggingface.co/author/collection/blob/main/people/Blaine%20refmod.safetensors?download=true")
        self.assertEqual(result["files"], ["people/Blaine refmod.safetensors"])
        self.assertEqual(result["revision"], "pinned")
        self.assertEqual(result["repo"], "author/collection")

    def test_non_huggingface_sources_use_local_file_import(self):
        for url in ("https://example.com/a.safetensors", "http://127.0.0.1/file", "https://huggingface.co.evil.test/a/b"):
            with self.assertRaisesRegex(ValueError, "Hugging Face"):
                transfer.remote_files(url)


if __name__ == "__main__":
    unittest.main()
