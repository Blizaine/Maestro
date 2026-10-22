"""Offline regressions for the pinned YuE2 source assets."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


APP_ROOT = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_ROOT))

from models.TTS.yue2.asset_sources import DOWNLOAD_REVISION
from models.TTS.yue2 import music_assets, yue2_handler
from services import music_styles


OLD_STYLE_COMPAT_REVISION = "864a479cbf3e810e1b2c1993b438510750e383b2"
CURRENT_DOWNLOAD_REVISION = "6e87e439357e25ebb563628a2ea6e93a46a8c4b1"
REMOTE_TTS_FILES = {
    "YuE2_Acoustic_bf16.safetensors": (
        2929494672,
        "975253f29d0f9d3229fa96e92234ab8c82221c6e511f0c8b68ad3a3f3fee151e",
    ),
    "YuE2_Acoustic_int8_convrot.safetensors": (
        1522568148,
        "6e66f94fa7145fd2e28b491d14182cd51d72653b7965d80ec23ae3ff24a92755",
    ),
    "YuE2_AR/YuE2_AR_bf16.safetensors": (
        4331951160,
        "59ce30757ae3960405dc6116c3a919541d61e5e55b28c4927dbb879870209366",
    ),
    "YuE2_AR/YuE2_AR_int8_convrot.safetensors": (
        2925023698,
        "02ae63099e82496a6b61ea7ef15aa729bc5503a50ed2d64a1da328dcc40ea5cb",
    ),
    "YuE2_AR/qwen.tiktoken": (2561218, None),
    "yue2/YuE2_VAE_bf16.safetensors": (
        132731320,
        "499714f69404d378b19a6fcbba9a8f99b4babfaed3f706e52c6dfa4e9521db67",
    ),
    "yue2/vae_config.json": (1378, None),
}


class YuE2AssetPinTests(unittest.TestCase):
    def test_license_information_uses_the_upstream_model_weight_license(self):
        defaults = json.loads((APP_ROOT / "defaults/yue2.json").read_text(encoding="utf-8"))
        self.assertEqual(defaults["model"]["license_name"], "CC BY-NC 4.0")
        self.assertEqual(defaults["model"]["license_url"],
            "https://huggingface.co/m-a-p/YuE2-3B/blob/8f24312f187763b9854adeec87e437e2b03bbff1/LICENSE")

    def test_handler_defaults_training_fetches_and_manifest_share_current_snapshot(self):
        self.assertEqual(DOWNLOAD_REVISION, CURRENT_DOWNLOAD_REVISION)
        manifest = json.loads((APP_ROOT / "models/TTS/yue2/assets.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["repo_id"], "DeepBeepMeep/TTS")
        self.assertEqual(manifest["revision"], DOWNLOAD_REVISION)
        self.assertEqual(yue2_handler.REVISION, DOWNLOAD_REVISION)

        defaults = json.loads((APP_ROOT / "defaults/yue2.json").read_text(encoding="utf-8"))
        urls = defaults["model"]["URLs"]
        self.assertEqual(
            urls,
            [
                f"https://huggingface.co/DeepBeepMeep/TTS/resolve/{DOWNLOAD_REVISION}/YuE2_Acoustic_bf16.safetensors",
                f"https://huggingface.co/DeepBeepMeep/TTS/resolve/{DOWNLOAD_REVISION}/YuE2_Acoustic_int8_convrot.safetensors",
            ],
        )

        model_files = yue2_handler.family_handler.query_model_files([], "yue2")
        self.assertEqual(len(model_files), 1)
        self.assertEqual(model_files[0]["repoId"], manifest["repo_id"])
        self.assertEqual(model_files[0]["revision"], DOWNLOAD_REVISION)
        self.assertEqual(model_files[0]["sourceFolderList"], ["yue2", "YuE2_AR"])
        self.assertEqual(model_files[0]["fileList"], [["vae_config.json", "YuE2_VAE_bf16.safetensors"], ["qwen.tiktoken"]])

        for key, remote_path in (
            ("nar_training", "YuE2_Acoustic_bf16.safetensors"),
            ("ar_training", "YuE2_AR/YuE2_AR_bf16.safetensors"),
            ("text_tokenizer", "YuE2_AR/qwen.tiktoken"),
        ):
            with self.subTest(asset=key):
                spec = music_assets.ASSETS[key]
                self.assertEqual(spec["repo_id"], manifest["repo_id"])
                self.assertEqual(spec["revision"], DOWNLOAD_REVISION)
                self.assertEqual(spec["remote_path"], remote_path)

        for remote_path, (size, sha256) in REMOTE_TTS_FILES.items():
            with self.subTest(remote_path=remote_path):
                expected = manifest["files"][remote_path]
                self.assertEqual(expected["size"], size)
                if sha256 is not None:
                    self.assertEqual(expected["sha256"], sha256)

        self.assertEqual(music_assets.ASSETS["nar_training"]["size"], REMOTE_TTS_FILES["YuE2_Acoustic_bf16.safetensors"][0])
        self.assertEqual(music_assets.ASSETS["nar_training"]["sha256"], REMOTE_TTS_FILES["YuE2_Acoustic_bf16.safetensors"][1])
        self.assertEqual(music_assets.ASSETS["ar_training"]["size"], REMOTE_TTS_FILES["YuE2_AR/YuE2_AR_bf16.safetensors"][0])
        self.assertEqual(music_assets.ASSETS["ar_training"]["sha256"], REMOTE_TTS_FILES["YuE2_AR/YuE2_AR_bf16.safetensors"][1])
        self.assertEqual(music_assets.ASSETS["text_tokenizer"]["size"], REMOTE_TTS_FILES["YuE2_AR/qwen.tiktoken"][0])

    def test_download_snapshot_does_not_change_existing_artist_bundle_compatibility(self):
        self.assertEqual(music_styles.BASE_REVISION, OLD_STYLE_COMPAT_REVISION)
        self.assertNotEqual(DOWNLOAD_REVISION, music_styles.BASE_REVISION)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "existing"
            directory.mkdir()
            manifest = {
                "id": "existing",
                "version": 1,
                "architecture": "yue2",
                "base_revision": OLD_STYLE_COMPAT_REVISION,
                "tokenizer_revision": music_styles.TOKENIZER_REVISION,
                "name": "Saved style",
                "trigger": "Saved style",
            }
            for branch in ("ar", "nar"):
                path = directory / f"{branch}.safetensors"
                path.write_bytes(branch.encode("ascii"))
                manifest[branch] = {"file": path.name, "sha256": music_styles.file_digest(path)}
            (directory / "style.json").write_text(json.dumps(manifest), encoding="utf-8")

            loaded = music_styles.load_style("existing", root=root, verify=True)
            self.assertEqual(loaded["base_revision"], OLD_STYLE_COMPAT_REVISION)


if __name__ == "__main__":
    unittest.main()
