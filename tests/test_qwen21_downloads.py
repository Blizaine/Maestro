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


# Relevant files verified against the public repositories on 2026-09-20.
# The current image-model revision no longer contains the processor folder;
# its pinned pre-migration export retains the Transformers 4.x tokenizer.
PROCESSOR_FILES = (
    "added_tokens.json", "chat_template.jinja", "config.json", "merges.txt",
    "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
)
REMOTE_FILES = {
    ("DeepBeepMeep/Qwen_image_2", None): {"qwen_image_21/qwen_image_21_vae.safetensors"},
    ("DeepBeepMeep/Qwen_image_2", "55ab7995df218d8f759c6c283ef4c3be66111345"):
        {f"Qwen3-VL-8B-Instruct/{name}" for name in PROCESSOR_FILES},
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

        def download(*, repo_id, filename, subfolder="", local_dir, revision=None, **_):
            remote = "/".join(part for part in (subfolder, filename) if part)
            self.assertIn(remote, REMOTE_FILES[(repo_id, revision)], f"404: {repo_id}/{remote}")
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
        expected = set().union(*REMOTE_FILES.values())
        actual = {file.relative_to(self.root).as_posix() for file in self.root.rglob("*") if file.is_file()}
        self.assertEqual(actual, expected)
        self.assertEqual(self.download.call_count, len(expected))

    def test_cached_install_is_reused_and_partial_download_can_resume(self):
        self.download_manifest()
        encoder = self.root / "Qwen3-VL-8B-Instruct"
        weights = encoder / "Qwen3-VL-8B-Instruct_int8_convrot.safetensors"
        weights.write_bytes(b"existing weights")
        self.download.reset_mock()
        self.download_manifest()
        self.download.assert_not_called()

        (encoder / "tokenizer.json").unlink()
        self.download_manifest()
        self.download.assert_called_once_with(
            repo_id="DeepBeepMeep/Qwen_image_2", revision="55ab7995df218d8f759c6c283ef4c3be66111345",
            filename="tokenizer.json",
            local_dir=str(self.root), subfolder="Qwen3-VL-8B-Instruct")
        self.assertEqual(weights.read_bytes(), b"existing weights")


if __name__ == "__main__":
    unittest.main()
