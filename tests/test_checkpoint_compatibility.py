"""Model-free regressions for safe CivitAI checkpoint imports."""
from __future__ import annotations

import importlib.util
import ast
import json
import os
import struct
import sys
import tempfile
import unittest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODULE_PATH = os.path.join(
    _ROOT, "app", "services", "checkpoint_compatibility.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "maestro_checkpoint_compatibility", _MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Could not load checkpoint compatibility module")
compatibility = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compatibility
_SPEC.loader.exec_module(compatibility)


FLUX1_SHAPES = {
    "img_in.weight": (3072, 64),
    "txt_in.weight": (3072, 4096),
    "double_blocks.0.img_attn.qkv.weight": (9216, 3072),
}
FLUX2_DEV_SHAPES = {
    "img_in.weight": (6144, 128),
    "txt_in.weight": (6144, 15360),
    "double_blocks.0.img_attn.qkv.weight": (18432, 6144),
}
KLEIN4_SHAPES = {
    "img_in.weight": (3072, 128),
    "txt_in.weight": (3072, 7680),
    "double_blocks.0.img_attn.qkv.weight": (9216, 3072),
}
KLEIN9_SHAPES = {
    "img_in.weight": (4096, 128),
    "txt_in.weight": (4096, 12288),
    "double_blocks.0.img_attn.qkv.weight": (12288, 4096),
}


def _write_safetensors(path: str, shapes: dict[str, tuple[int, ...]]) -> None:
    header = {
        key: {
            "dtype": "F16",
            "shape": list(shape),
            "data_offsets": [0, 2],
        }
        for key, shape in shapes.items()
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(b"\0\0")


class TestCheckpointMappings(unittest.TestCase):
    def test_every_allowlisted_target_has_a_matching_shipped_template(self):
        for targets in compatibility._CHECKPOINT_TARGETS.values():
            for target in targets:
                with self.subTest(template=target.template_model_type):
                    path = os.path.join(
                        _ROOT,
                        "app",
                        "defaults",
                        f"{target.template_model_type}.json",
                    )
                    self.assertTrue(os.path.isfile(path))
                    with open(path, "r", encoding="utf-8") as handle:
                        definition = json.load(handle)
                    self.assertEqual(
                        definition["model"]["architecture"],
                        target.architecture,
                    )

    def test_unknown_and_sdxl_bases_are_not_assigned_a_pipeline(self):
        self.assertEqual(
            compatibility.checkpoint_targets_for_base("SDXL 1.0"), ()
        )
        self.assertEqual(
            compatibility.checkpoint_targets_for_base("Unknown Future Model"),
            (),
        )
        self.assertIn(
            "not supported",
            compatibility.unsupported_checkpoint_reason("SDXL 1.0"),
        )

    def test_ltx_versions_map_to_their_actual_generations(self):
        ltx20 = compatibility.checkpoint_targets_for_base("LTXV2")
        ltx23 = compatibility.checkpoint_targets_for_base("LTXV 2.3")
        self.assertEqual(ltx20[0].architecture, "ltx2_19B")
        self.assertEqual(ltx23[0].architecture, "ltx2_22B")

    def test_only_ambiguous_verified_family_requires_user_choice(self):
        krea = compatibility.checkpoint_targets_for_base("Krea 2")
        self.assertEqual(
            {target.architecture for target in krea},
            {"krea2_raw", "krea2_turbo"},
        )
        self.assertIsNone(
            compatibility.suggested_checkpoint_architecture("Krea 2")
        )
        self.assertEqual(
            compatibility.suggested_checkpoint_architecture("Flux.1 D"),
            "flux",
        )

    def test_metadata_gate_rejects_cross_family_selection(self):
        with self.assertRaisesRegex(
            compatibility.CheckpointCompatibilityError,
            "compatible with flux, not 'flux2_dev'",
        ):
            compatibility.ensure_allowed_checkpoint_target(
                "Flux.1 D", "flux2_dev"
            )


class TestCheckpointTensorSignatures(unittest.TestCase):
    def _validate(
        self,
        shapes: dict[str, tuple[int, ...]],
        base_model: str,
        architecture: str,
    ) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.safetensors")
            _write_safetensors(path, shapes)
            return compatibility.validate_checkpoint_file(
                path, base_model, architecture
            )

    def test_flux1_cannot_be_registered_as_flux2(self):
        receipt = self._validate(FLUX1_SHAPES, "Flux.1 D", "flux")
        self.assertEqual(receipt["architecture"], "flux")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.safetensors")
            _write_safetensors(path, FLUX1_SHAPES)
            with self.assertRaisesRegex(
                compatibility.CheckpointCompatibilityError,
                "compatible with flux, not 'flux2_dev'",
            ):
                compatibility.validate_checkpoint_file(
                    path, "Flux.1 D", "flux2_dev"
                )

    def test_flux2_variants_do_not_share_incompatible_shapes(self):
        self._validate(FLUX2_DEV_SHAPES, "Flux.2 D", "flux2_dev")
        self._validate(KLEIN4_SHAPES, "Flux.2 Klein 4B", "flux2_klein_4b")
        self._validate(KLEIN9_SHAPES, "Flux.2 Klein 9B", "flux2_klein_9b")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "klein4.safetensors")
            _write_safetensors(path, KLEIN4_SHAPES)
            with self.assertRaisesRegex(
                compatibility.CheckpointCompatibilityError,
                "matches flux2_klein_4b, not the selected flux2_klein_9b",
            ):
                compatibility.validate_checkpoint_file(
                    path, "Flux.2 Klein 9B", "flux2_klein_9b"
                )

    def test_wrapped_and_quantized_tensor_names_are_normalized(self):
        wrapped = {
            f"model.diffusion_model.{key}._data": shape
            for key, shape in FLUX2_DEV_SHAPES.items()
        }
        self._validate(wrapped, "Flux.2 D", "flux2_dev")

    def test_non_safetensors_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.bin")
            _write_safetensors(path, FLUX1_SHAPES)
            with self.assertRaisesRegex(
                compatibility.CheckpointCompatibilityError,
                "SafeTensor files only",
            ):
                compatibility.validate_checkpoint_file(
                    path, "Flux.1 D", "flux"
                )

    def test_tensor_offsets_cannot_extend_past_downloaded_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "truncated.safetensors")
            header = {
                "img_in.weight": {
                    "dtype": "F16",
                    "shape": [3072, 64],
                    "data_offsets": [0, 100],
                }
            }
            encoded = json.dumps(header).encode("utf-8")
            with open(path, "wb") as handle:
                handle.write(struct.pack("<Q", len(encoded)))
                handle.write(encoded)
                handle.write(b"\0\0")
            with self.assertRaisesRegex(
                compatibility.CheckpointCompatibilityError,
                "outside the downloaded payload",
            ):
                compatibility.read_safetensors_header(path)


class TestH3CheckpointImports(unittest.TestCase):
    @staticmethod
    def header(*, full=False, partition="", dtype="I8", prefix=""):
        # Dimensions cross-checked against pinned DeepBeepMeep/Comfy H3
        # SafeTensor headers, without downloading their 20/33B payloads.
        dim = 2688 if full else 8
        shapes = {
            "video_patch_proj.weight": (5376, 96),
            "audio_patch_proj.weight": (5376, 32),
            "condition_proj.weight": (5376, 5120),
            "final_layer.video_out.weight": (96, 5376),
            "final_layer.audio_out.weight": (32, 5376),
            "final_layer.adaln_proj.linear.weight": (10752, dim),
        }
        if full:
            shapes.update({"time_embedder.proj_in.weight": (5376, 256),
                           "time_embedder.proj_out.weight": (2688, 5376)})
        else:
            shapes["adaln_t_table"] = (1025, 8)
        for block in range(50):
            shapes[f"blocks.{block}.attn.qkv_proj.weight"] = (21504, 5376)
            shapes[f"blocks.{block}.adaln_proj.linear.weight"] = (96768, dim)
        for block in range(2):
            shapes[f"token_refiner.blocks.{block}.attn.qkv_proj.weight"] = (21504, 5376)
        header = {prefix + key: {"shape": list(shape), "dtype": dtype, "data_offsets": [0, 2]}
                  for key, shape in shapes.items()}
        header["__metadata__"] = {"partition": partition}
        return header

    def validate(self, header, architecture="minimax_h3", layout="grouped"):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "h3.safetensors")
            encoded = json.dumps(header).encode()
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", len(encoded)) + encoded + b"\0\0")
            return compatibility.validate_checkpoint_file(
                path, "MiniMax H3", architecture, qkv_layout=layout)

    def test_stream_rejects_wrong_architecture_without_consuming_weights(self):
        header = json.dumps(self.header(partition="REF2VA")).encode()
        def chunks():
            yield struct.pack("<Q", len(header)) + header
            self.fail("Read tensor payload after incompatible header")
        with self.assertRaisesRegex(compatibility.CheckpointCompatibilityError, "matches"):
            list(compatibility.verified_checkpoint_chunks(chunks(), "MiniMax H3", "minimax_h3", filename="model.safetensors", qkv_layout="grouped"))

    def test_stream_preserves_valid_file_across_header_boundaries(self):
        header = json.dumps(self.header(partition="FL2VA")).encode()
        data = struct.pack("<Q", len(header)) + header + b"tensor payload"
        for split in (1, 8, 20, len(data) - 5):
            chunks = [data[:split], data[split:]]
            result = compatibility.verified_checkpoint_chunks(chunks, "MiniMax H3", "minimax_h3", filename="model.safetensors", qkv_layout="grouped")
            self.assertEqual(b"".join(result), data)

    def test_stream_rejects_corrupt_and_truncated_headers_early(self):
        for data in (b"short", struct.pack("<Q", 2**40), struct.pack("<Q", 2) + b"[]", struct.pack("<Q", 2) + b"xx"):
            with self.assertRaises(compatibility.CheckpointCompatibilityError):
                list(compatibility.verified_checkpoint_chunks([data], "MiniMax H3", "minimax_h3", filename="model.safetensors", qkv_layout="grouped"))

    def test_catalog_filters_files_versions_and_preserves_source(self):
        model = {"name": "INT4 collection", "modelVersions": [
            {"id": 1, "baseModel": "MiniMax H3", "files": [
                {"name": "H3_int4.safetensors", "type": "Model"},
                {"name": "H3_int8.safetensors", "type": "Model"},
                {"name": "vae.safetensors", "type": "VAE"},
                {"name": "bundle.zip", "type": "Model"},
                {"name": "unnamed.safetensors", "metadata": {"fp": "nf4"}},
            ]},
            {"id": 2, "baseModel": "MiniMax H3", "files": [{"name": "H3.gguf"}]},
            {"id": 3, "baseModel": "SDXL 1.0", "files": [{"name": "sdxl.safetensors"}]},
        ]}
        original = json.dumps(model)
        result = compatibility.filter_checkpoint_catalog_model(model)
        self.assertEqual([v["id"] for v in result["modelVersions"]], [1])
        self.assertEqual([f["name"] for f in result["modelVersions"][0]["files"]], ["H3_int8.safetensors"])
        self.assertEqual(json.dumps(model), original)

    def test_catalog_hides_unknown_or_missing_metadata(self):
        for model in ({}, {"modelVersions": [{"baseModel": "MiniMax H3"}]}, {"modelVersions": [{"files": [{"name": "weights.safetensors"}]}]}):
            self.assertEqual(compatibility.filter_checkpoint_catalog_model(model)["modelVersions"], [])

    def test_unsupported_filename_does_not_start_stream(self):
        def chunks():
            self.fail("Started unsupported download")
            yield b""
        for name in ("minimaxH3INT4Convrot_fl2vaPrunedInt4.safetensors", "H3_NVFP4.safetensors", "H3_4-bit.safetensors", "H3.gguf"):
            with self.assertRaisesRegex(compatibility.CheckpointCompatibilityError, "different file"):
                list(compatibility.verified_checkpoint_chunks(chunks(), "MiniMax H3", "minimax_h3", filename=name, qkv_layout="grouped"))
        compatibility.validate_checkpoint_filename("H3_pruned_int8_convrot.safetensors", "minimax_h3")
        compatibility.validate_checkpoint_filename("unrelated_int4.safetensors", "flux")

    def test_w4a_exports_are_hidden_and_rejected_before_transfer(self):
        for name in ("minimaxH3ComfyNativeRef2vaW4a8_v10.safetensors", "H3_W4A16.safetensors", "H3_w4_a8.safetensors", "H3_W-4-A-8.safetensors"):
            with self.subTest(name=name):
                for target in compatibility.checkpoint_targets_for_base("MiniMax H3"):
                    with self.assertRaisesRegex(compatibility.CheckpointCompatibilityError, "4-bit"):
                        compatibility.validate_checkpoint_filename(name, target.architecture)
                model = {"modelVersions": [{"baseModel": "MiniMax H3", "files": [{"name": name}]}]}
                self.assertEqual(compatibility.filter_checkpoint_catalog_model(model)["modelVersions"], [])
        metadata_model = {"modelVersions": [{"baseModel": "MiniMax H3", "files": [{"name": "weights.safetensors", "metadata": {"fp": "W4A8"}}]}]}
        self.assertEqual(compatibility.filter_checkpoint_catalog_model(metadata_model)["modelVersions"], [])
        compatibility.validate_checkpoint_filename("H3_W8A8.safetensors", "minimax_h3")

    def test_h3_requires_explicit_workflow_and_size_choice(self):
        self.assertEqual(len(compatibility.checkpoint_targets_for_base(" MiniMax H3 ")), 4)
        self.assertIsNone(compatibility.suggested_checkpoint_architecture("MiniMax H3"))

    def test_supported_precisions_and_loader_namespaces(self):
        for dtype in ("BF16", "F16", "F8_E4M3", "I8"):
            for prefix in ("", "model.diffusion_model.", "diffusion_model."):
                with self.subTest(dtype=dtype, prefix=prefix):
                    result = self.validate(self.header(dtype=dtype, prefix=prefix))
                    self.assertEqual(result["model_options"]["minimax_h3_qkv_layout"], "grouped")

    def test_full_and_pruned_cannot_be_swapped(self):
        for full, correct, wrong in ((True, "minimax_h3_full", "minimax_h3"),
                                     (False, "minimax_h3", "minimax_h3_full")):
            self.validate(self.header(full=full), correct, "interleaved")
            with self.assertRaises(compatibility.CheckpointCompatibilityError):
                self.validate(self.header(full=full), wrong)

    def test_declared_partition_must_match_selected_workflow(self):
        self.validate(self.header(partition="Ref2VA"), "minimax_h3_ref2va")
        self.validate(self.header(full=True, partition="Ref2VA"), "minimax_h3_ref2va_full")
        with self.assertRaises(compatibility.CheckpointCompatibilityError):
            self.validate(self.header(partition="Ref2VA"))
        with self.assertRaises(compatibility.CheckpointCompatibilityError):
            self.validate(self.header(partition="FL2VA"), "minimax_h3_ref2va")

    def test_qkv_order_cannot_be_guessed_from_shapes(self):
        for layout in ("", "auto", "contiguous", None, [], {}):
            with self.subTest(layout=layout), self.assertRaisesRegex(
                    compatibility.CheckpointCompatibilityError, "QKV layout"):
                self.validate(self.header(), layout=layout)

    def test_rejects_missing_blocks_heads_and_invalid_adaln(self):
        for key in ("blocks.49.attn.qkv_proj.weight", "final_layer.audio_out.weight", "adaln_t_table"):
            header = self.header()
            header.pop(key)
            with self.subTest(key=key), self.assertRaises(compatibility.CheckpointCompatibilityError):
                self.validate(header)
        header = self.header()
        header["adaln_t_table"]["shape"] = [1025, 16]
        with self.assertRaises(compatibility.CheckpointCompatibilityError):
            self.validate(header)

    def test_rejects_packed_low_bit_weights_and_unsupported_wrapper(self):
        header = self.header()
        header["blocks.0.attn.qkv_proj.weight"]["shape"] = [21504, 2688]
        with self.assertRaises(compatibility.CheckpointCompatibilityError):
            self.validate(header)
        with self.assertRaises(compatibility.CheckpointCompatibilityError):
            self.validate(self.header(prefix="transformer."))

    def test_unrelated_checkpoint_and_lora_cannot_be_imported_as_h3(self):
        for header in ({}, {"blocks.0.attn.qkv_proj.lora_A.weight": {
                "dtype": "F16", "shape": [16, 5376], "data_offsets": [0, 2]}}):
            with self.assertRaises(compatibility.CheckpointCompatibilityError):
                self.validate(header)


class TestH3CheckpointRegistration(unittest.TestCase):
    def test_registration_preserves_export_layout_and_survives_startup_audit(self):
        source_path = os.path.join(_ROOT, "app", "launch.py")
        with open(source_path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"_ckpt_slugify", "_register_checkpoint_finetune"}]
        for full in (False, True):
            with self.subTest(full=full), tempfile.TemporaryDirectory() as directory:
                os.makedirs(os.path.join(directory, "ckpts"))
                path = os.path.join(directory, "ckpts", "custom.safetensors")
                header = TestH3CheckpointImports.header(full=full, partition="Ref2VA")
                encoded = json.dumps(header).encode()
                with open(path, "wb") as handle:
                    handle.write(struct.pack("<Q", len(encoded)) + encoded + b"\0\0")
                namespace = {
                    "os": os, "json": json,
                    "validate_checkpoint_file": compatibility.validate_checkpoint_file,
                    "checkpoint_template_model_type": compatibility.checkpoint_template_model_type,
                    "CheckpointCompatibilityError": compatibility.CheckpointCompatibilityError,
                    "_DEFAULTS_DIR": os.path.join(_ROOT, "app", "defaults"),
                    "_FINETUNES_DIR": os.path.join(directory, "finetunes"),
                }
                exec(compile(ast.Module(body=functions, type_ignores=[]), source_path, "exec"), namespace)
                architecture = "minimax_h3_ref2va" + ("_full" if full else "")
                _, definition_path = namespace["_register_checkpoint_finetune"](
                    path, {"baseModel": "MiniMax H3", "modelId": 123, "name": "Custom H3"},
                    architecture, qkv_layout="grouped")
                with open(definition_path) as handle:
                    definition = json.load(handle)
                self.assertEqual(definition["model"]["URLs"], ["custom.safetensors"])
                self.assertEqual(definition["model"]["architecture"], architecture)
                # In particular, override the Full template's interleaved default.
                self.assertEqual(definition["model"]["minimax_h3_qkv_layout"], "grouped")
                self.assertEqual(definition["model"]["compatible_model_paths"], {})
                self.assertEqual(definition["model"]["compatible_model_qkv_layouts"], {})
                self.assertIn("num_inference_steps", definition)
                self.assertEqual(compatibility.quarantine_incompatible_checkpoint_definitions(directory), [])
                definition["model"].pop("minimax_h3_qkv_layout")
                with open(definition_path, "w") as handle:
                    json.dump(definition, handle)
                changes = compatibility.quarantine_incompatible_checkpoint_definitions(directory)
                self.assertFalse(changes[0]["compatible"])
                self.assertTrue(os.path.isfile(path))


class TestLegacyCheckpointQuarantine(unittest.TestCase):
    def _definition(
        self,
        *,
        architecture: str,
        base_model: str,
        filename: str,
        visible: bool = True,
    ) -> dict:
        return {
            "model": {
                "name": "Imported checkpoint",
                "architecture": architecture,
                "visible": visible,
                "URLs": [filename],
                "civitai": {
                    "modelType": "Checkpoint",
                    "baseModel": base_model,
                    "filename": filename,
                },
            }
        }

    def test_bad_legacy_registration_is_hidden_without_deleting_weights(self):
        with tempfile.TemporaryDirectory() as app_dir:
            finetunes = os.path.join(app_dir, "finetunes")
            checkpoints = os.path.join(app_dir, "ckpts")
            os.makedirs(finetunes)
            os.makedirs(checkpoints)
            weight_path = os.path.join(checkpoints, "flux1.safetensors")
            _write_safetensors(weight_path, FLUX1_SHAPES)
            definition_path = os.path.join(finetunes, "bad.json")
            with open(definition_path, "w", encoding="utf-8") as handle:
                json.dump(
                    self._definition(
                        architecture="flux2_dev",
                        base_model="Flux.1 D",
                        filename="flux1.safetensors",
                    ),
                    handle,
                )

            changes = compatibility.quarantine_incompatible_checkpoint_definitions(
                app_dir
            )
            self.assertEqual(len(changes), 1)
            self.assertFalse(changes[0]["compatible"])
            self.assertTrue(changes[0]["applied"])
            with open(definition_path, "r", encoding="utf-8") as handle:
                quarantined = json.load(handle)["model"]
            self.assertFalse(quarantined["visible"])
            self.assertEqual(
                quarantined["civitai"]["compatibility_status"], "blocked"
            )
            self.assertIn("maestro_checkpoint_quarantine", quarantined)
            self.assertTrue(os.path.isfile(weight_path))

    def test_valid_definition_is_left_alone_and_old_marker_can_restore(self):
        with tempfile.TemporaryDirectory() as app_dir:
            finetunes = os.path.join(app_dir, "finetunes")
            checkpoints = os.path.join(app_dir, "ckpts")
            os.makedirs(finetunes)
            os.makedirs(checkpoints)
            weight_path = os.path.join(checkpoints, "flux1.safetensors")
            _write_safetensors(weight_path, FLUX1_SHAPES)
            definition = self._definition(
                architecture="flux",
                base_model="Flux.1 D",
                filename="flux1.safetensors",
                visible=False,
            )
            definition["model"]["maestro_checkpoint_quarantine"] = {
                "previous_visible": True,
                "reason": "old block",
            }
            definition["model"]["civitai"]["compatibility_status"] = "blocked"
            definition_path = os.path.join(finetunes, "valid.json")
            with open(definition_path, "w", encoding="utf-8") as handle:
                json.dump(definition, handle)

            changes = compatibility.quarantine_incompatible_checkpoint_definitions(
                app_dir
            )
            self.assertEqual(len(changes), 1)
            self.assertTrue(changes[0]["compatible"])
            self.assertTrue(changes[0]["applied"])
            with open(definition_path, "r", encoding="utf-8") as handle:
                restored = json.load(handle)["model"]
            self.assertTrue(restored["visible"])
            self.assertNotIn("maestro_checkpoint_quarantine", restored)
            self.assertNotIn(
                "compatibility_status", restored["civitai"]
            )

    def test_existing_custom_format_is_preserved_when_metadata_mapping_is_valid(self):
        with tempfile.TemporaryDirectory() as app_dir:
            finetunes = os.path.join(app_dir, "finetunes")
            checkpoints = os.path.join(app_dir, "ckpts")
            os.makedirs(finetunes)
            os.makedirs(checkpoints)
            with open(
                os.path.join(checkpoints, "legacy.gguf"), "wb"
            ) as handle:
                handle.write(b"GGUF")
            definition_path = os.path.join(finetunes, "legacy.json")
            with open(definition_path, "w", encoding="utf-8") as handle:
                json.dump(
                    self._definition(
                        architecture="flux",
                        base_model="Flux.1 D",
                        filename="legacy.gguf",
                    ),
                    handle,
                )

            changes = compatibility.quarantine_incompatible_checkpoint_definitions(
                app_dir
            )
            self.assertEqual(changes, [])
            with open(definition_path, "r", encoding="utf-8") as handle:
                model = json.load(handle)["model"]
            self.assertTrue(model["visible"])

    def test_quarantine_marker_is_not_removed_when_weight_is_missing(self):
        with tempfile.TemporaryDirectory() as app_dir:
            finetunes = os.path.join(app_dir, "finetunes")
            os.makedirs(finetunes)
            os.makedirs(os.path.join(app_dir, "ckpts"))
            definition = self._definition(
                architecture="flux",
                base_model="Flux.1 D",
                filename="missing.safetensors",
                visible=False,
            )
            definition["model"]["maestro_checkpoint_quarantine"] = {
                "previous_visible": True,
                "reason": "awaiting revalidation",
            }
            definition_path = os.path.join(finetunes, "missing.json")
            with open(definition_path, "w", encoding="utf-8") as handle:
                json.dump(definition, handle)

            changes = compatibility.quarantine_incompatible_checkpoint_definitions(
                app_dir
            )
            self.assertEqual(changes, [])
            with open(definition_path, "r", encoding="utf-8") as handle:
                model = json.load(handle)["model"]
            self.assertFalse(model["visible"])
            self.assertIn("maestro_checkpoint_quarantine", model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
