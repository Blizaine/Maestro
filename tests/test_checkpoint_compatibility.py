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
