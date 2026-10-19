"""Model-free regressions for safe CivitAI checkpoint imports."""
from __future__ import annotations

import importlib.util
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
KREA2_SHAPES = {
    "first.weight": (6144, 64),
    "blocks.0.attn.wq.weight": (6144, 6144),
    "blocks.0.attn.wk.weight": (1536, 6144),
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


def _krea2_nvfp4_descriptors() -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        "first.weight": ("BF16", (6144, 64)),
        "blocks.0.attn.wq.weight": ("U8", (6144, 3072)),
        "blocks.0.attn.wq.weight_scale": ("F8_E4M3", (6144, 384)),
        "blocks.0.attn.wq.weight_scale_2": ("F32", ()),
        "blocks.0.attn.wk.weight": ("U8", (1536, 3072)),
        "blocks.0.attn.wk.weight_scale": ("F8_E4M3", (1536, 384)),
        "blocks.0.attn.wk.weight_scale_2": ("F32", ()),
    }


def _write_safetensors_descriptors(
    path: str, descriptors: dict[str, tuple[str, tuple[int, ...]]]
) -> None:
    header = {
        key: {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [0, 2],
        }
        for key, (dtype, shape) in descriptors.items()
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

    def test_explicit_krea_turbo_label_selects_only_its_turbo_recipe(self):
        targets = compatibility.checkpoint_targets_for_base("Krea 2 Turbo")
        self.assertEqual([target.architecture for target in targets], ["krea2_turbo"])
        self.assertEqual(compatibility.suggested_checkpoint_architecture("  KREA 2  TURBO "), "krea2_turbo")
        self.assertEqual(compatibility.checkpoint_template_model_type("Krea 2 Turbo", "krea2_turbo"), "krea2_turbo")
        with self.assertRaises(compatibility.CheckpointCompatibilityError):
            compatibility.ensure_allowed_checkpoint_target("Krea 2 Turbo", "krea2_raw")


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

    def _validate_descriptors(
        self,
        descriptors: dict[str, tuple[str, tuple[int, ...]]],
        base_model: str = "Krea 2 Turbo",
        architecture: str = "krea2_turbo",
    ) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.safetensors")
            _write_safetensors_descriptors(path, descriptors)
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

    def test_krea_turbo_label_keeps_tensor_validation_and_format_gate(self):
        receipt = self._validate(KREA2_SHAPES, "Krea 2 Turbo", "krea2_turbo")
        self.assertEqual(receipt["architecture"], "krea2_turbo")
        with self.assertRaisesRegex(compatibility.CheckpointCompatibilityError, "not the selected krea2_turbo"):
            self._validate(FLUX1_SHAPES, "Krea 2 Turbo", "krea2_turbo")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.gguf")
            _write_safetensors(path, KREA2_SHAPES)
            with self.assertRaisesRegex(compatibility.CheckpointCompatibilityError, "SafeTensor files only"):
                compatibility.validate_checkpoint_file(path, "Krea 2 Turbo", "krea2_turbo")

    def test_krea_nvfp4_tensorcore_scales_recover_only_known_logical_anchors(self):
        expected_shapes = {
            "first.weight": (6144, 64),
            "blocks.0.attn.wq.weight": (6144, 6144),
            "blocks.0.attn.wk.weight": (1536, 6144),
        }
        for prefix in ("", "model.diffusion_model."):
            with self.subTest(prefix=prefix):
                descriptors = {
                    prefix + key: descriptor
                    for key, descriptor in _krea2_nvfp4_descriptors().items()
                }
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "checkpoint.safetensors")
                    _write_safetensors_descriptors(path, descriptors)
                    shapes = compatibility._tensor_shapes(
                        compatibility.read_safetensors_header(path)
                    )
                    for key, expected in expected_shapes.items():
                        self.assertEqual(shapes[key], expected)
                    self.assertEqual(
                        set(compatibility.detect_checkpoint_architectures(path)),
                        {"krea2_raw", "krea2_turbo"},
                    )
                    receipt = compatibility.validate_checkpoint_file(
                        path, "Krea 2 Turbo", "krea2_turbo"
                    )
                    self.assertEqual(receipt["architecture"], "krea2_turbo")
                    with self.assertRaisesRegex(
                        compatibility.CheckpointCompatibilityError,
                        "compatible with krea2_turbo, not 'krea2_raw'",
                    ):
                        compatibility.validate_checkpoint_file(
                            path, "Krea 2 Turbo", "krea2_raw"
                        )

    def test_krea_nvfp4_requires_every_exact_sidecar_and_valid_tile_geometry(self):
        cases = {
            "missing weight_scale": lambda d: d.pop("blocks.0.attn.wq.weight_scale"),
            "missing weight_scale_2": lambda d: d.pop("blocks.0.attn.wq.weight_scale_2"),
            "wrong weight_scale dtype": lambda d: d.__setitem__(
                "blocks.0.attn.wq.weight_scale", ("F16", (6144, 384))
            ),
            "wrong weight_scale rows": lambda d: d.__setitem__(
                "blocks.0.attn.wq.weight_scale", ("F8_E4M3", (6016, 384))
            ),
            "wrong weight_scale groups": lambda d: d.__setitem__(
                "blocks.0.attn.wq.weight_scale", ("F8_E4M3", (6144, 383))
            ),
            "non-scalar weight_scale_2": lambda d: d.__setitem__(
                "blocks.0.attn.wq.weight_scale_2", ("F32", (1,))
            ),
            "wrong weight_scale_2 dtype": lambda d: d.__setitem__(
                "blocks.0.attn.wq.weight_scale_2", ("F16", ())
            ),
            "rows not 128 aligned": lambda d: (
                d.__setitem__("blocks.0.attn.wq.weight", ("U8", (6143, 3072))),
                d.__setitem__("blocks.0.attn.wq.weight_scale", ("F8_E4M3", (6143, 384))),
            ),
            "K not 64 aligned": lambda d: (
                d.__setitem__("blocks.0.attn.wq.weight", ("U8", (6144, 3071))),
                d.__setitem__("blocks.0.attn.wq.weight_scale", ("F8_E4M3", (6144, 383))),
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                descriptors = _krea2_nvfp4_descriptors()
                mutate(descriptors)
                with self.assertRaises(
                    compatibility.CheckpointCompatibilityError
                ):
                    self._validate_descriptors(descriptors)

    def test_krea_raw_u8_and_unrelated_u8_shapes_are_not_expanded(self):
        descriptors = _krea2_nvfp4_descriptors()
        descriptors["blocks.0.attn.wq.weight"] = ("U8", (6144, 6144))
        descriptors.pop("blocks.0.attn.wq.weight_scale")
        descriptors.pop("blocks.0.attn.wq.weight_scale_2")
        # An unidentified U8 tensor cannot claim the expected logical size.
        with self.assertRaises(
            compatibility.CheckpointCompatibilityError
        ):
            self._validate_descriptors(descriptors)

        header = {
            "img_in.weight": {
                "dtype": "U8",
                "shape": [6144, 3072],
            },
            "img_in.weight_scale": {
                "dtype": "F8_E4M3",
                "shape": [6144, 384],
            },
            "img_in.weight_scale_2": {
                "dtype": "F32",
                "shape": [],
            },
        }
        self.assertEqual(
            compatibility._tensor_shapes(header)["img_in.weight"],
            (6144, 3072),
        )

    def test_quanto_weight_data_shape_normalization_is_unchanged(self):
        header = {
            "blocks.0.attn.wq.weight._data": {
                "dtype": "U8",
                "shape": [6144, 6144],
            }
        }
        self.assertEqual(
            compatibility._tensor_shapes(header)["blocks.0.attn.wq.weight"],
            (6144, 6144),
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
