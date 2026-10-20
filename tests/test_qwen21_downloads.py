"""Exercise Qwen Image 2.1 downloads without an existing checkpoint cache."""

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.qwen21.qwen21_handler import MODEL_TYPE, family_handler


# The legacy aliases retain the Transformers 4.57-compatible processor export
# at this Ideogram4 revision. The downloader stores them under canonical names.
PROCESSOR_FILES = (
    "added_tokens.json", "chat_template.jinja", "config.json", "merges.txt",
    "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
)
PROCESSOR_FILE_NAME_MAP = {
    "config.json": "config_legacy.json",
    "tokenizer.json": "tokenizer_legacy.json",
    "tokenizer_config.json": "tokenizer_config_legacy.json",
}
PROCESSOR_REPO = "DeepBeepMeep/Ideogram4"
PROCESSOR_REVISION = "a531af16b9cb570479356dae9e3f4a721adc3b7f"
ENCODER_FOLDER = "Qwen3-VL-8B-Instruct"
REMOTE_FILES = {
    ("DeepBeepMeep/Qwen_image_2", None): {"qwen_image_21/qwen_image_21_vae.safetensors"},
    (PROCESSOR_REPO, PROCESSOR_REVISION): {
        f"{ENCODER_FOLDER}/{PROCESSOR_FILE_NAME_MAP.get(name, name)}"
        for name in PROCESSOR_FILES
    },
}
EXPECTED_LOCAL_FILES = {
    "qwen_image_21/qwen_image_21_vae.safetensors",
    *(f"{ENCODER_FOLDER}/{name}" for name in PROCESSOR_FILES),
}
EXPECTED_DOWNLOADS = {
    (repo_id, revision, filename)
    for (repo_id, revision), files in REMOTE_FILES.items()
    for filename in files
}


class Qwen21DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the production download dispatcher without importing the GPU engine.
        source = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "process_files_def")
        cls.dispatcher = compile(ast.Module(body=[function], type_ignores=[]), "wgp.py", "exec")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.remote_calls = []

        def download(*, repo_id, filename, subfolder="", local_dir, revision=None, **_):
            remote = "/".join(part for part in (subfolder, filename) if part)
            self.assertIn(remote, REMOTE_FILES[(repo_id, revision)], f"404: {repo_id}/{remote}")
            self.remote_calls.append((repo_id, revision, remote))
            target = Path(local_dir) / remote
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"download fixture")
            return str(target)

        self.download = Mock(side_effect=download)
        namespace = {"os": os, "hf_download_with_public_fallback": self.download,
                     "fl": SimpleNamespace(get_smart_download_root=lambda _: str(self.root),
                                           get_checkpoints_paths=lambda: [str(self.root)])}
        exec(self.dispatcher, namespace)
        self.process = namespace["process_files_def"]

    def download_manifest(self):
        for definition in family_handler.query_model_files(None, MODEL_TYPE):
            self.process(**definition)

    def test_fresh_install_downloads_every_required_file(self):
        self.download_manifest()
        actual = {file.relative_to(self.root).as_posix() for file in self.root.rglob("*") if file.is_file()}
        self.assertEqual(actual, EXPECTED_LOCAL_FILES)
        self.assertEqual(set(self.remote_calls), EXPECTED_DOWNLOADS)
        self.assertEqual(len(self.remote_calls), len(EXPECTED_DOWNLOADS))

    def test_cached_install_is_reused_and_partial_download_can_resume(self):
        self.download_manifest()
        encoder = self.root / ENCODER_FOLDER
        weights = encoder / "Qwen3-VL-8B-Instruct_int8_convrot.safetensors"
        weights.write_bytes(b"existing weights")
        self.download.reset_mock()
        self.remote_calls.clear()
        self.download_manifest()
        self.download.assert_not_called()

        (encoder / "tokenizer.json").unlink()
        self.remote_calls.clear()
        self.download_manifest()
        self.download.assert_called_once_with(
            repo_id=PROCESSOR_REPO, revision=PROCESSOR_REVISION,
            filename="tokenizer_legacy.json",
            local_dir=str(self.root), subfolder=ENCODER_FOLDER)
        self.assertEqual(self.remote_calls, [
            (PROCESSOR_REPO, PROCESSOR_REVISION, f"{ENCODER_FOLDER}/tokenizer_legacy.json"),
        ])
        self.assertTrue((encoder / "tokenizer.json").is_file())
        self.assertEqual(weights.read_bytes(), b"existing weights")


if __name__ == "__main__":
    unittest.main()
