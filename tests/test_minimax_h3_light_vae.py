"""CPU coverage for the WanGP v18.10 MiniMax H3 LightVAE port."""

from __future__ import annotations

from contextlib import nullcontext
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
WGP_PATH = APP / "wgp.py"
WGP_TREE = ast.parse(WGP_PATH.read_text(encoding="utf-8"))
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3 import minimax_h3_handler as h3_handler  # noqa: E402
from models.minimax_h3.minimax_h3_handler import family_handler  # noqa: E402
from models.minimax_h3.video_vae_assets import (  # noqa: E402
    VIDEO_VAE_FULL_FILE,
    VIDEO_VAE_FULL_INT8_FILE,
    VIDEO_VAE_LIGHT_FILE,
    VIDEO_VAE_LIGHT_INT8_FILE,
    VIDEO_VAE_LIGHT_REVISION,
    get_video_vae_variant_choices,
    is_light_video_vae_checkpoint,
)
from models.minimax_h3.vae_upsampler import X2_VAE_FILE  # noqa: E402

h3_main = None


def _wgp_function(name):
    return next(
        node
        for node in WGP_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _run_process_files_def(descriptor, target_root, checkpoint_roots):
    """Run the real downloader folder logic with a local fake HF transfer."""
    process_function = _wgp_function("process_files_def")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = kwargs["local_dir"]
        if kwargs.get("subfolder"):
            destination = os.path.join(destination, kwargs["subfolder"])
        os.makedirs(destination, exist_ok=True)
        path = os.path.join(destination, kwargs["filename"])
        Path(path).write_bytes(b"fake MiniMax H3 decoder")
        return path

    fake_files_locator = SimpleNamespace(
        get_smart_download_root=lambda _target: str(target_root),
        get_checkpoints_paths=lambda: [str(path) for path in checkpoint_roots],
    )
    namespace = {
        "os": os,
        "fl": fake_files_locator,
        "hf_download_with_public_fallback": fake_download,
        "snapshot_download": mock.Mock(),
    }
    exec(
        compile(
            ast.Module(body=[process_function], type_ignores=[]),
            str(WGP_PATH),
            "exec",
        ),
        namespace,
    )
    namespace["process_files_def"](**descriptor)
    return calls


class MiniMaxH3LightVAERoutingTests(unittest.TestCase):
    @staticmethod
    def _downloaded_files(downloads):
        return [
            (download["repoId"], filename)
            for download in downloads
            for group in download["fileList"]
            for filename in group
        ]

    def test_selection_choices_resolve_before_asset_discovery(self):
        definition = family_handler.query_model_def("minimax_h3", {})
        self.assertEqual(
            definition["minimax_h3_video_vae_choices"],
            get_video_vae_variant_choices(),
        )
        expected = {
            "full": VIDEO_VAE_FULL_FILE,
            "full_int8": VIDEO_VAE_FULL_INT8_FILE,
            "lightvae": VIDEO_VAE_LIGHT_FILE,
            "lightvae_int8": VIDEO_VAE_LIGHT_INT8_FILE,
        }

        for variant, filename in expected.items():
            with self.subTest(variant=variant):
                resolved = family_handler.resolve_runtime_model_def(
                    definition,
                    {
                        "minimax_h3_video_vae": variant,
                        "transformer_quantization": "int8",
                    },
                )
                self.assertEqual(
                    resolved["minimax_h3_video_vae_filename"], filename
                )
                self.assertEqual(
                    resolved["minimax_h3_video_vae_variant"], variant
                )
                self.assertFalse(resolved["minimax_h3_video_vae_auto"])

    def test_light_vae_download_uses_public_native_directory_and_shared_repo(self):
        self.assertEqual(
            VIDEO_VAE_LIGHT_REVISION,
            "630975da14fd3c84169d6cd5092da3ce1287d137",
        )
        definition = family_handler.query_model_def("minimax_h3", {})
        for variant, filename in (
            ("lightvae", VIDEO_VAE_LIGHT_FILE),
            ("lightvae_int8", VIDEO_VAE_LIGHT_INT8_FILE),
        ):
            with self.subTest(variant=variant):
                resolved = family_handler.resolve_runtime_model_def(
                    definition,
                    {"minimax_h3_video_vae": variant},
                )
                downloads = family_handler.query_model_files(
                    [], "minimax_h3", resolved
                )
                vae_download = next(
                    download
                    for download in downloads
                    if download["repoId"] == "DeepBeepMeep/MiniMax-H3"
                )
                self.assertEqual(
                    vae_download["sourceFolderList"], ["minimax_h3"]
                )
                self.assertEqual(
                    vae_download["targetFolderList"],
                    [None],
                )
                self.assertEqual(
                    vae_download["revision"], VIDEO_VAE_LIGHT_REVISION
                )
                self.assertEqual(vae_download["fileList"], [[filename]])
                self.assertIn(
                    ("Comfy-Org/MiniMax-H3", "minimax_h3_audio_vae_fp32.safetensors"),
                    self._downloaded_files(downloads),
                )

    def test_light_vae_dispatch_destination_is_found_by_the_production_resolver(self):
        try:
            from models.minimax_h3 import minimax_h3_main as loaded_h3_main
        except ModuleNotFoundError as error:
            if error.name not in {"optimum", "optimum.quanto"}:
                raise
            self.skipTest(
                f"MiniMax H3 runtime dependencies are unavailable: {error}"
            )
        global h3_main
        h3_main = loaded_h3_main
        definition = family_handler.query_model_def("minimax_h3", {})
        for variant, filename in (
            ("lightvae", VIDEO_VAE_LIGHT_FILE),
            ("lightvae_int8", VIDEO_VAE_LIGHT_INT8_FILE),
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                resolved = family_handler.resolve_runtime_model_def(
                    definition, {"minimax_h3_video_vae": variant}
                )
                downloads = family_handler.query_model_files(
                    [], "minimax_h3", resolved
                )
                descriptor = next(
                    item
                    for item in downloads
                    if item["repoId"] == "DeepBeepMeep/MiniMax-H3"
                )
                self.assertEqual(descriptor["targetFolderList"], [None])
                calls = _run_process_files_def(descriptor, root, [root])
                expected_path = root / "minimax_h3" / filename
                self.assertEqual(len(calls), 1)
                self.assertTrue(expected_path.is_file())

                def locate_file(path, error_if_none=True):
                    candidate = root / path
                    if candidate.is_file():
                        return str(candidate)
                    if error_if_none:
                        raise FileNotFoundError(str(candidate))
                    return None

                with mock.patch.object(
                    h3_main.fl, "locate_file", side_effect=locate_file
                ):
                    actual_path = h3_main._locate_video_vae_file(
                        "minimax_h3", filename
                    )
                self.assertEqual(Path(actual_path), expected_path)

    def test_auto_keeps_existing_transformer_quantization_default(self):
        definition = family_handler.query_model_def("minimax_h3", {})
        int8 = family_handler.resolve_runtime_model_def(
            definition,
            {
                "minimax_h3_video_vae": "auto",
                "transformer_quantization": "int8",
            },
        )
        fp8 = family_handler.resolve_runtime_model_def(
            definition,
            {
                "minimax_h3_video_vae": "auto",
                "transformer_quantization": "fp8",
            },
        )
        self.assertEqual(
            int8["minimax_h3_video_vae_filename"], VIDEO_VAE_FULL_INT8_FILE
        )
        self.assertEqual(int8["minimax_h3_video_vae_variant"], "full_int8")
        self.assertEqual(fp8["minimax_h3_video_vae_filename"], VIDEO_VAE_FULL_FILE)
        self.assertEqual(fp8["minimax_h3_video_vae_variant"], "full")

    def test_unknown_runtime_variant_is_rejected(self):
        definition = family_handler.query_model_def("minimax_h3", {})
        with self.assertRaisesRegex(ValueError, "Unsupported MiniMax H3 video VAE variant"):
            family_handler.resolve_runtime_model_def(
                definition, {"minimax_h3_video_vae": "tiny"}
            )

    def test_x2_download_keeps_selected_light_encoder_and_adds_dedicated_decoder(self):
        definition = family_handler.resolve_runtime_model_def(
            family_handler.query_model_def("minimax_h3", {}),
            {"minimax_h3_video_vae": "lightvae"},
        )
        with mock.patch.object(
            h3_handler.fl, "get_checkpoints_paths", return_value=[]
        ), mock.patch(
            "models.minimax_h3.minimax_h3_handler.os.path.isfile",
            return_value=False,
        ), mock.patch(
            "models.minimax_h3.minimax_h3_handler.x2_vae_asset_candidates",
            return_value=[],
        ):
            downloads = family_handler.query_model_files(
                [], "minimax_h3", definition, VAE_upsampling="h3_vae*2"
            )

        files = self._downloaded_files(downloads)
        self.assertIn(("DeepBeepMeep/MiniMax-H3", VIDEO_VAE_LIGHT_FILE), files)
        self.assertIn(
            ("DeepBeepMeep/MiniMax-H3", X2_VAE_FILE),
            files,
        )

    def test_profile_is_only_added_to_fl2va_and_matches_upstream_recipe(self):
        fl2va = family_handler.query_model_def("minimax_h3", {})
        ref2va = family_handler.query_model_def("minimax_h3_ref2va", {})
        self.assertEqual(
            fl2va["profiles_dir"], ["minimax_h3", "minimax_h3_fl2va"]
        )
        self.assertEqual(ref2va["profiles_dir"], ["minimax_h3"])
        profile_path = (
            APP
            / "profiles"
            / "minimax_h3_fl2va"
            / "Turbo Taomate FL2VA 3 steps (2026-10-15).json"
        )
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        self.assertEqual(
            profile["activated_loras"],
            [
                "https://huggingface.co/DeepBeepMeep/MiniMax-H3/resolve/main/loras/TaoMate-H3-FL2VA-3Step-ComfyUI-BF16.safetensors"
            ],
        )
        self.assertEqual(profile["num_inference_steps"], 3)
        self.assertEqual(profile["guidance_scale"], 1)


class MiniMaxH3LightVAELoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global h3_main
        try:
            from models.minimax_h3 import minimax_h3_main as loaded_h3_main
        except ModuleNotFoundError as error:
            if error.name not in {"optimum", "optimum.quanto"}:
                raise
            raise unittest.SkipTest(
                f"MiniMax H3 runtime dependencies are unavailable: {error}"
            ) from error
        h3_main = loaded_h3_main

    class FakeVAE:
        def eval(self):
            return self

        def requires_grad_(self, _requires_grad):
            return self

    def test_light_checkpoint_uses_12_layer_decoder_and_native_encoder_loader(self):
        fake_vae = self.FakeVAE()
        with mock.patch.object(h3_main, "init_empty_weights", return_value=nullcontext()), mock.patch.object(
            h3_main, "AutoencoderKLMiniMaxH3", return_value=fake_vae
        ) as vae_factory, mock.patch.object(
            h3_main, "_video_vae_checkpoint_dtype", return_value=torch.bfloat16
        ), mock.patch.object(h3_main.offload, "load_model_data") as load_model:
            actual = h3_main._load_video_vae(VIDEO_VAE_LIGHT_FILE)

        self.assertIs(actual, fake_vae)
        self.assertEqual(vae_factory.call_args.kwargs["decoder_num_layers"], 12)
        self.assertEqual(vae_factory.call_args.kwargs["out_channels"], 3)
        self.assertIs(load_model.call_args.args[0], fake_vae)
        self.assertIs(
            load_model.call_args.kwargs["preprocess_sd"],
            h3_main.preprocess_native_video_vae_state_dict,
        )
        self.assertEqual(load_model.call_args.kwargs["default_dtype"], torch.bfloat16)
        self.assertEqual(load_model.call_args.kwargs["fp32_dtype"], torch.bfloat16)

    def test_x2_selection_replaces_light_decoder_with_36_layer_x2_decoder(self):
        fake_vae = self.FakeVAE()
        with mock.patch.object(h3_main, "init_empty_weights", return_value=nullcontext()), mock.patch.object(
            h3_main, "AutoencoderKLMiniMaxH3", return_value=fake_vae
        ) as vae_factory, mock.patch.object(
            h3_main, "_video_vae_checkpoint_dtype", return_value=torch.bfloat16
        ), mock.patch.object(
            h3_main, "verify_x2_vae_checkpoint"
        ), mock.patch.object(
            h3_main, "_load_x2_video_vae_weights"
        ) as load_x2:
            actual = h3_main._load_video_vae(
                VIDEO_VAE_LIGHT_FILE,
                "pinned-x2-decoder.safetensors",
            )

        self.assertIs(actual, fake_vae)
        self.assertEqual(vae_factory.call_args.kwargs["decoder_num_layers"], 36)
        self.assertEqual(vae_factory.call_args.kwargs["out_channels"], 12)
        self.assertEqual(load_x2.call_args.kwargs["base_dtype"], torch.bfloat16)
        self.assertEqual(fake_vae.upsampling_set, "h3_vae*2")

    def test_checkpoint_dtype_comes_from_encoder_header_without_loading_weights(self):
        with mock.patch.object(
            h3_main.quant_router,
            "load_metadata_state_dict",
            return_value=({"encoder.conv_in.weight": torch.empty(1, dtype=torch.bfloat16)}, {}),
        ) as load_header:
            dtype = h3_main._video_vae_checkpoint_dtype(VIDEO_VAE_LIGHT_FILE)
        self.assertEqual(dtype, torch.bfloat16)
        load_header.assert_called_once_with(VIDEO_VAE_LIGHT_FILE)

    def test_shared_resolver_checks_existing_maestro_and_upstream_layouts(self):
        resolved_path = "resolved/shared/minimax_h3_lightvae_lite_bf16.safetensors"
        with mock.patch.object(
            h3_main.fl,
            "locate_file",
            side_effect=[None, None, resolved_path],
        ) as locate_file:
            actual = h3_main._locate_video_vae_file(
                "minimax_h3", VIDEO_VAE_LIGHT_FILE
            )
        self.assertEqual(actual, resolved_path)
        self.assertEqual(
            [
                call.args[0].replace("\\", "/")
                for call in locate_file.call_args_list
            ],
            [
                f"minimax_h3/vae/{VIDEO_VAE_LIGHT_FILE}",
                f"minimax_h3/vae/minimax_h3/{VIDEO_VAE_LIGHT_FILE}",
                f"minimax_h3/{VIDEO_VAE_LIGHT_FILE}",
            ],
        )

    def test_lightvae_recognition_uses_basename(self):
        self.assertTrue(is_light_video_vae_checkpoint(VIDEO_VAE_LIGHT_FILE))
        self.assertTrue(
            is_light_video_vae_checkpoint(
                f"minimax_h3/vae/{VIDEO_VAE_LIGHT_INT8_FILE}"
            )
        )
        self.assertFalse(is_light_video_vae_checkpoint(VIDEO_VAE_FULL_FILE))


class MiniMaxH3VideoVAEPipelineBindingTests(unittest.TestCase):
    def test_load_and_download_pass_resolved_definition_to_asset_discovery(self):
        load_models = _wgp_function("load_models")
        resolved_model_def = next(
            node
            for node in ast.walk(load_models)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "model_def"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "get_runtime_model_def"
        )
        self.assertIsInstance(resolved_model_def.value.args[1], ast.Name)
        self.assertEqual(resolved_model_def.value.args[1].id, "model_kwargs")

        download_call = next(
            node
            for node in ast.walk(load_models)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "download_models"
            and any(
                keyword.arg == "model_def_override"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "model_def"
                for keyword in node.keywords
            )
        )
        self.assertIsNotNone(download_call)

        download_models = _wgp_function("download_models")
        query_calls = [
            node
            for node in ast.walk(download_models)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "query_model_files"
        ]
        self.assertTrue(query_calls)
        for call in query_calls:
            self.assertGreaterEqual(len(call.args), 3)
            self.assertIsInstance(call.args[2], ast.Name)
            self.assertEqual(call.args[2].id, "model_def")

        received = []

        class Handler:
            def query_model_files(self, _compute, _model_type, definition, **kwargs):
                received.append((definition, kwargs))
                return []

        resolved = {
            "source": "already-selected-transformer.safetensors",
            "minimax_h3_video_vae_filename": VIDEO_VAE_LIGHT_INT8_FILE,
            "minimax_h3_video_vae_variant": "lightvae_int8",
        }
        namespace = {
            "get_base_model_type": lambda model_type: model_type,
            "get_runtime_model_def": lambda *_args, **_kwargs: self.fail(
                "download_models must reuse its resolved definition override"
            ),
            "ensure_audio_condition_assets": lambda _model_type: None,
            "get_model_recursive_prop": lambda *_args, **_kwargs: [],
            "model_types_handlers": {"minimax_h3": Handler()},
            "process_files_def": lambda **_definition: None,
        }
        exec(
            compile(
                ast.Module(body=[download_models], type_ignores=[]),
                str(WGP_PATH),
                "exec",
            ),
            namespace,
        )
        namespace["download_models"](
            "transformer.safetensors",
            "minimax_h3",
            file_type=0,
            model_def_override=resolved,
        )
        self.assertEqual(len(received), 1)
        self.assertIs(received[0][0], resolved)

    def test_warm_auto_render_reloads_only_when_resolved_variant_changes(self):
        generate_video = _wgp_function("generate_video")
        expected_assignment = next(
            node
            for node in ast.walk(generate_video)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "expected_h3_video_vae"
                for target in node.targets
            )
        )
        self.assertIsInstance(expected_assignment.value, ast.Call)
        self.assertIsInstance(expected_assignment.value.func, ast.Attribute)
        self.assertEqual(expected_assignment.value.func.attr, "get")
        resolver_call = expected_assignment.value.func.value
        self.assertIsInstance(resolver_call, ast.Call)
        self.assertIsInstance(resolver_call.func, ast.Name)
        self.assertEqual(resolver_call.func.id, "get_runtime_model_def")
        self.assertIsInstance(resolver_call.args[1], ast.Name)
        self.assertEqual(resolver_call.args[1].id, "model_kwargs")
        vae_setting = next(
            node
            for node in ast.walk(generate_video)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "model_kwargs"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "minimax_h3_video_vae"
                for target in node.targets
            )
        )
        self.assertEqual(ast.unparse(vae_setting.value), "requested_h3_video_vae")
        self.assertLess(vae_setting.lineno, expected_assignment.lineno)

        reload_guard = next(
            node
            for node in ast.walk(generate_video)
            if isinstance(node, ast.If)
            and "video_vae_variant" in ast.unparse(node.test)
            and any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "reload_needed"
                    for target in child.targets
                )
                for child in ast.walk(node)
            )
        )
        self.assertIn("expected_h3_video_vae", ast.unparse(reload_guard.test))

        def evaluate(warm_variant):
            namespace = {
                "model_type": "minimax_h3",
                "requested_h3_video_vae": "auto",
                "model_kwargs": {},
                "get_runtime_model_def": lambda *_args: {
                    "minimax_h3_video_vae_variant": "full"
                },
                "wan_model": SimpleNamespace(video_vae_variant=warm_variant),
                "reload_needed": False,
            }
            exec(
                compile(
                    ast.Module(
                        body=[vae_setting, expected_assignment, reload_guard],
                        type_ignores=[],
                    ),
                    str(WGP_PATH),
                    "exec",
                ),
                namespace,
            )
            return namespace["reload_needed"]

        self.assertFalse(evaluate("full"))
        self.assertTrue(evaluate("lightvae"))


if __name__ == "__main__":
    unittest.main()
