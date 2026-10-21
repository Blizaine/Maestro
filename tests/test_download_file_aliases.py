"""Download renamed metadata through the real dispatcher without a GPU import."""
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


class DownloadFileAliasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "process_files_def")
        cls.code = compile(ast.Module(body=[function], type_ignores=[]), "wgp.py", "exec")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.roots = [str(self.root)]

        def download(**kwargs):
            if kwargs["filename"] != "tokenizer_legacy.json":
                raise FileNotFoundError("404: requested nonexistent canonical export")
            path = Path(kwargs["local_dir"]) / kwargs.get("subfolder", "") / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"verified legacy tokenizer")
            return str(path)

        self.download = Mock(side_effect=download)
        namespace = {"os": os, "hf_download_with_public_fallback": self.download,
                     "fl": SimpleNamespace(get_smart_download_root=lambda _: str(self.root),
                                           get_checkpoints_paths=lambda: self.roots,
                                           locate_file=lambda name, **_: str(self.root / name)
                                           if (self.root / name).is_file() else None)}
        exec(self.code, namespace)
        self.process = namespace["process_files_def"]

    def run_manifest(self, folder="encoder"):
        self.process(repoId="publisher/model", revision="verified-revision",
                     sourceFolderList=[folder], fileList=[["tokenizer.json"]],
                     fileNameMap={"tokenizer.json": "tokenizer_legacy.json"})

    def test_legacy_download_becomes_canonical_and_is_reused(self):
        self.run_manifest()
        self.assertEqual((self.root / "encoder/tokenizer.json").read_bytes(), b"verified legacy tokenizer")
        self.assertFalse((self.root / "encoder/tokenizer_legacy.json").exists())
        self.download.assert_called_once_with(repo_id="publisher/model", revision="verified-revision",
                                              filename="tokenizer_legacy.json", local_dir=str(self.root),
                                              subfolder="encoder")
        self.download.reset_mock()
        self.run_manifest()
        self.download.assert_not_called()

    def test_root_level_alias(self):
        self.run_manifest(folder="")
        self.assertEqual((self.root / "tokenizer.json").read_bytes(), b"verified legacy tokenizer")
        self.assertNotIn("subfolder", self.download.call_args.kwargs)

    def test_partial_primary_folder_is_completed_without_touching_weights(self):
        secondary = self.root / "linked"
        (secondary / "encoder").mkdir(parents=True)
        (secondary / "encoder/tokenizer.json").write_bytes(b"linked tokenizer")
        self.roots.append(str(secondary))
        (self.root / "encoder").mkdir()
        weights = self.root / "encoder/model.safetensors"
        weights.write_bytes(b"existing model weights")
        self.run_manifest()
        self.assertEqual((self.root / "encoder/tokenizer.json").read_bytes(), b"verified legacy tokenizer")
        self.assertEqual(weights.read_bytes(), b"existing model weights")

    def test_download_failure_does_not_create_canonical_placeholder(self):
        self.download.side_effect = FileNotFoundError("404: legacy export missing")
        with self.assertRaises(FileNotFoundError):
            self.run_manifest()
        self.assertFalse((self.root / "encoder/tokenizer.json").exists())

    def test_aliases_cannot_escape_the_target_directory(self):
        for mapping in ({"../tokenizer.json": "legacy.json"},
                        {"tokenizer.json": "..\\legacy.json"},
                        {"tokenizer.json": "C:legacy.json"}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.process(repoId="publisher/model", sourceFolderList=["encoder"],
                             fileList=[["tokenizer.json"]], fileNameMap=mapping)
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
