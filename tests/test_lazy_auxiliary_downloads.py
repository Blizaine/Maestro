"""Keep shared preprocessing and audio assets lazy without importing the GPU app."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
REPO = "DeepBeepMeep/Wan2.1"


class LazyAuxiliaryDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        wanted = {
            "download_models",
            "ensure_preprocessor_assets",
            "ensure_model_preprocessor_assets",
            "ensure_audio_condition_assets",
        }
        functions = [node for node in source.body
                     if isinstance(node, ast.FunctionDef) and node.name in wanted]
        cls.dispatcher = compile(ast.Module(body=functions, type_ignores=[]), "wgp.py", "exec")

    def setUp(self):
        self.process_files_def = Mock()
        namespace = {
            "process_files_def": self.process_files_def,
            "server_config": {"depth_anything_v2_variant": "vitl"},
            "get_base_model_type": lambda model_type: model_type,
            "get_runtime_model_def": lambda _model_type: {"architecture": "minimax_h3"},
            "get_compatible_local_model_filename": lambda *_args, **_kwargs: None,
            "get_model_recursive_prop": lambda *_args, **_kwargs: [],
            "get_local_model_filename": lambda *_args, **_kwargs: None,
            "fl": SimpleNamespace(get_download_location=lambda *_args: "unused"),
            "model_types_handlers": {
                "minimax_h3": SimpleNamespace(
                    query_model_files=lambda *_args: [{"repoId": REPO, "sourceFolderList": ["vae"], "fileList": [["h3-required-vae.safetensors"]]}]
                )
            },
        }
        exec(self.dispatcher, namespace)
        self.namespace = namespace

    def test_empty_filename_keeps_required_model_files_without_global_auxiliary_prefetch(self):
        self.namespace["download_models"]("", "minimax_h3", 0, -1)

        self.process_files_def.assert_called_once_with(
            repoId=REPO,
            sourceFolderList=["vae"],
            fileList=[["h3-required-vae.safetensors"]],
        )

    def test_preprocessor_assets_are_requested_only_for_the_selected_transform(self):
        ensure = self.namespace["ensure_preprocessor_assets"]

        ensure("pose")
        self.process_files_def.assert_called_once_with(
            repoId=REPO,
            sourceFolderList=["pose"],
            fileList=[["dw-ll_ucoco_384.onnx", "yolox_l.onnx"]],
        )

        self.process_files_def.reset_mock()
        ensure("depth")
        self.process_files_def.assert_called_once_with(
            repoId=REPO,
            sourceFolderList=["depth"],
            fileList=[["depth_anything_v2_vitl.pth"]],
        )

        self.process_files_def.reset_mock()
        self.namespace["server_config"]["depth_anything_v2_variant"] = "vitb"
        ensure("depth")
        self.process_files_def.assert_called_once_with(
            repoId=REPO,
            sourceFolderList=["depth"],
            fileList=[["depth_anything_v2_vitb.pth"]],
        )

        self.process_files_def.reset_mock()
        ensure("raw")
        self.process_files_def.assert_not_called()

    def test_scribble_canny_and_flow_assets_are_lazy(self):
        ensure = self.namespace["ensure_preprocessor_assets"]
        for process_type, folder, filename in (
            ("scribble", "scribble", "netG_A_latest.pth"),
            ("canny", "scribble", "netG_A_latest.pth"),
            ("flow", "flow", "raft-things.pth"),
        ):
            with self.subTest(process_type=process_type):
                self.process_files_def.reset_mock()
                ensure(process_type)
                self.process_files_def.assert_called_once_with(
                    repoId=REPO,
                    sourceFolderList=[folder],
                    fileList=[[filename]],
                )

    def test_model_specific_pose_assets_follow_the_requested_preprocessor(self):
        ensure = self.namespace["ensure_model_preprocessor_assets"]

        ensure("minimax_h3")
        self.process_files_def.assert_not_called()

        ensure("scail")
        self.process_files_def.assert_called_once_with(
            repoId=REPO,
            sourceFolderList=["pose"],
            fileList=[["dw-ll_ucoco_384.onnx", "yolox_l.onnx"]],
        )

        self.process_files_def.reset_mock()
        ensure("steadydancer")
        self.process_files_def.assert_called_once()

        self.process_files_def.reset_mock()
        ensure("scail2_14B", {"scail2_animate_preprocessing": "raw"})
        self.process_files_def.assert_not_called()

        ensure("scail2_14B", {"scail2_animate_preprocessing": "pose"})
        self.process_files_def.assert_called_once_with(
            repoId=REPO,
            sourceFolderList=["pose"],
            fileList=[["dw-ll_ucoco_384.onnx", "yolox_l.onnx"]],
        )

    def test_audio_encoders_are_selected_by_model_architecture(self):
        ensure = self.namespace["ensure_audio_condition_assets"]

        ensure("minimax_h3")
        self.process_files_def.assert_not_called()

        ensure("fantasy")
        self.process_files_def.assert_called_once_with(
            repoId=REPO,
            sourceFolderList=["wav2vec"],
            fileList=[[
                "config.json",
                "feature_extractor_config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.json",
            ]],
        )

        for architecture in (
            "multitalk",
            "infinitetalk",
            "vace_multitalk_14B",
            "i2v_2_2_multitalk",
        ):
            with self.subTest(architecture=architecture):
                self.process_files_def.reset_mock()
                ensure(architecture)
                self.process_files_def.assert_called_once_with(
                    repoId=REPO,
                    sourceFolderList=["chinese-wav2vec2-base"],
                    fileList=[["config.json", "pytorch_model.bin", "preprocessor_config.json"]],
                )

    def test_audio_assets_follow_canonical_architecture_for_imported_model_ids(self):
        aliases = {
            "multitalk_720p": "multitalk",
            "infinitetalk_multi": "infinitetalk",
        }
        self.namespace["get_base_model_type"] = lambda model_id: aliases.get(model_id, model_id)
        self.namespace["get_runtime_model_def"] = lambda _model_id: {}
        for model_id, architecture in aliases.items():
            self.namespace["model_types_handlers"][architecture] = SimpleNamespace(
                query_model_files=lambda *_args: []
            )
            with self.subTest(model_id=model_id):
                self.process_files_def.reset_mock()
                self.namespace["download_models"]("", model_id, 0, -1)
                self.process_files_def.assert_called_once_with(
                    repoId=REPO,
                    sourceFolderList=["chinese-wav2vec2-base"],
                    fileList=[["config.json", "pytorch_model.bin", "preprocessor_config.json"]],
                )


if __name__ == "__main__":
    unittest.main()
