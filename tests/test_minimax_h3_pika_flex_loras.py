"""CPU-only guardrails for experimental Pika H3 Flex LoRAs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class PikaFlexLoRACompatibilityTests(unittest.TestCase):
    @staticmethod
    def _fixed_request(**overrides):
        request = {
            "num_inference_steps": 6,
            "sampling_steps": 6,
            "guidance_scale": 1.0,
            "cfg_scale": 1.0,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "guidance_phases": 1,
            "sample_solver": "euler",
            "denoising_strength": 1.0,
            "masking_strength": 1.0,
            "activated_loras": [],
            "minimax_h3_multi_window": False,
            "minimax_h3_reference_sequence": False,
            "skip_steps_cache_type": "",
        }
        request.update(overrides)
        return request

    def test_request_validator_keeps_recipe_guards_and_accepts_only_opted_in_style_loras(self):
        from models.minimax_h3.pika_flex import validate_pika_flex_request

        with self.assertRaisesRegex(ValueError, "off by default"):
            validate_pika_flex_request(self._fixed_request(
                activated_loras=["character_style.safetensors"],
            ))
        validate_pika_flex_request(self._fixed_request(
            activated_loras=["character_style.safetensors"],
            minimax_h3_experimental_loras=True,
        ))
        with self.assertRaisesRegex(ValueError, "off by default"):
            validate_pika_flex_request(self._fixed_request(
                activated_loras=["character_style.safetensors"],
                minimax_h3_experimental_loras=1,
            ))
        with self.assertRaisesRegex(ValueError, "Turbo and other step-acceleration"):
            validate_pika_flex_request(self._fixed_request(
                activated_loras=["minimax_h3_turbo_style.safetensors"],
                minimax_h3_experimental_loras=True,
            ))
        with self.assertRaisesRegex(ValueError, "does not support additional LoRAs or accelerators"):
            validate_pika_flex_request(self._fixed_request(minimax_h3_turbo_mode=True))
        with self.assertRaisesRegex(ValueError, "First Block"):
            validate_pika_flex_request(self._fixed_request(skip_steps_cache_type="first_block"))

    def test_pika_api_options_mark_style_loras_available_but_experimental(self):
        from models.minimax_h3.pika_flex_assets import (
            pika_flex_capabilities,
            pika_flex_defaults,
        )

        for reference in (False, True):
            with self.subTest(reference=reference):
                capabilities = pika_flex_capabilities(reference=reference)
                self.assertIs(capabilities["loras_disabled"], False)
                self.assertIs(capabilities["experimental_loras_available"], True)
                # Fixed per-job recipe merges must leave a user's explicit
                # generation-level opt-in untouched.
                self.assertNotIn(
                    "minimax_h3_experimental_loras",
                    pika_flex_defaults(reference=reference),
                )
                filename = (
                    "minimax_h3_ref2va_pika_flex.json"
                    if reference
                    else "minimax_h3_pika_flex.json"
                )
                definition = json.loads(
                    (APP / "defaults" / filename).read_text(encoding="utf-8")
                )
                self.assertIs(definition["minimax_h3_experimental_loras"], False)

    def test_ordinary_h3_loras_require_literal_true_opt_in(self):
        from models.minimax_h3.pika_flex_lora_compatibility import (
            validate_pika_flex_additional_loras,
        )

        self.assertEqual(
            validate_pika_flex_additional_loras([], enabled=False),
            (),
        )
        with self.assertRaisesRegex(ValueError, "off by default"):
            validate_pika_flex_additional_loras(
                ["character_style.safetensors"], enabled=False,
            )
        for truthy_or_non_boolean in (1, "true", object()):
            with self.subTest(enabled=truthy_or_non_boolean):
                with self.assertRaisesRegex(ValueError, "off by default"):
                    validate_pika_flex_additional_loras(
                        ["character_style.safetensors"],
                        enabled=truthy_or_non_boolean,
                    )
        self.assertEqual(
            validate_pika_flex_additional_loras(
                ["character_style.safetensors"], enabled=True,
            ),
            ("character_style.safetensors",),
        )

    def test_known_accelerators_and_unsupported_formats_keep_specific_reasons(self):
        from models.minimax_h3.pika_flex_lora_compatibility import (
            validate_pika_flex_additional_loras,
        )

        cases = (
            ("minimax_h3_turbo_style.safetensors", "Turbo and other step-acceleration"),
            ("minimax_h3_ref2va_pdd_style.safetensors", "PDD adapters use"),
            (
                "minimax_h3_ref2va_ptd_refasymopd_8step_hop01_bf16.safetensors",
                "PTD RefAsymOPD hop adapters",
            ),
            ("minimax_h3_vdn_style.safetensors", "VDN adapters require"),
            ("minimax_h3_fasth3_style.safetensors", "special-head H3 recipe"),
            ("minimax_h3_distilled_style.safetensors", "special-head H3 recipe"),
            ("ordinary_dora.safetensors", "DoRA adapters are unsupported"),
        )
        for filename, message in cases:
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, message):
                    validate_pika_flex_additional_loras(
                        [filename], enabled=True,
                    )

        from models.minimax_h3.pika_flex_lora_compatibility import (
            pika_flex_lora_incompatibility,
        )
        pdd_head_bank = {
            "proj_out.weight": {"shape": [32, 96, 64]},
            "audio_proj_out.weight": {"shape": [32, 32, 64]},
        }
        with patch(
            "models.minimax_h3.pika_flex_lora_compatibility._header",
            return_value=pdd_head_bank,
        ):
            self.assertIn("special output-head bank", pika_flex_lora_incompatibility("custom.safetensors"))

        with patch(
            "models.minimax_h3.pika_flex_lora_compatibility._header",
            return_value={"blocks.0.attn.to_gate_compress.weight": {"shape": [4, 4]}},
        ):
            self.assertIn("special-head H3 recipe", pika_flex_lora_incompatibility("custom.safetensors"))

        with patch(
            "models.minimax_h3.pika_flex_lora_compatibility._header",
            return_value={"lora_unet_blocks_0.attn.to_q.lora_magnitude_vector": {"shape": [4]}},
        ):
            self.assertIn("DoRA adapters are unsupported", pika_flex_lora_incompatibility("custom.safetensors"))

    def test_required_pinned_adapter_is_first_unique_verified_and_strength_one(self):
        from models.minimax_h3.pika_flex_assets import PIKA_FLEX_LORA_SPECS
        from models.minimax_h3.pika_flex_lora_compatibility import (
            validate_pika_flex_adapter_strengths,
            validate_pika_flex_lora_selection,
        )

        for reference in (False, True):
            pin = PIKA_FLEX_LORA_SPECS[reference]["file"]
            pinned_path = f"C:/models/{pin}"
            with self.subTest(reference=reference):
                with patch(
                    "models.minimax_h3.pika_flex_assets.verify_pika_flex_loras"
                ) as verify_pin:
                    selected = validate_pika_flex_lora_selection(
                        [pinned_path, "character_style.safetensors"],
                        reference=reference,
                        enabled=True,
                    )
                self.assertEqual(selected, (pinned_path, "character_style.safetensors"))
                verify_pin.assert_called_once_with([pinned_path], reference=reference)
                validate_pika_flex_adapter_strengths(
                    selected,
                    [1.0, [0.6] * 6],
                    reference=reference,
                )
                with self.assertRaisesRegex(ValueError, "strength must remain 1.0"):
                    validate_pika_flex_adapter_strengths(
                        selected,
                        [[0.99] * 6, [0.6] * 6],
                        reference=reference,
                    )
                with self.assertRaisesRegex(ValueError, "pinned adapter as the first"):
                    validate_pika_flex_lora_selection(
                        ["character_style.safetensors", pinned_path],
                        reference=reference,
                        enabled=True,
                    )
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    validate_pika_flex_lora_selection(
                        [pinned_path, pinned_path],
                        reference=reference,
                        enabled=True,
                    )

    def test_model_validator_uses_opt_in_before_the_loader_and_protects_stale_heads(self):
        main_path = APP / "models" / "minimax_h3" / "minimax_h3_main.py"
        tree = ast.parse(main_path.read_text(encoding="utf-8"))
        model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                           and node.name == "MiniMaxH3Model")
        methods = {
            node.name: node for node in model_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"validate_loras", "configure_special_loras"}
        }
        namespace = {"__package__": "models.minimax_h3"}
        for method in methods.values():
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(main_path), "exec"), namespace)

        class ModelStub:
            pika_flex = True
            omni_reference = False
            _turbo_lora_active = True
            _turbo_lora_paths = ("stale-turbo.safetensors",)
            _pdd_lora_active = True
            _pdd_lora_path = "stale-pdd.safetensors"
            _pdd_lora_strength = 0.4
            _ptd_lora_paths = ("stale-ptd.safetensors",)
            validate_loras = namespace["validate_loras"]
            configure_special_loras = namespace["configure_special_loras"]

            def release_special_loras(self):
                self._pdd_controller = None

        from models.minimax_h3.pika_flex_assets import PIKA_FLEX_LORA_SPECS
        model = ModelStub()
        selected = [PIKA_FLEX_LORA_SPECS[False]["file"], "character_style.safetensors"]
        with patch("models.minimax_h3.pika_flex_assets.verify_pika_flex_loras"):
            with self.assertRaisesRegex(ValueError, "off by default"):
                model.validate_loras(selected)
            model.validate_loras(selected, minimax_h3_experimental_loras=True)
        self.assertFalse(model._turbo_lora_active)
        self.assertEqual(model._turbo_lora_paths, ())
        self.assertFalse(model._pdd_lora_active)
        self.assertIsNone(model._pdd_lora_path)
        self.assertEqual(model._pdd_lora_strength, 1.0)
        self.assertEqual(model._ptd_lora_paths, ())
        with self.assertRaisesRegex(ValueError, "strength must remain 1.0"):
            model.configure_special_loras(selected, [0.8, 0.7])
        model.configure_special_loras(selected, [1.0, 0.7])

    def test_wgp_flag_is_signature_filtered_forwarded_and_keeps_standard_loader_list(self):
        wgp_path = APP / "wgp.py"
        tree = ast.parse(wgp_path.read_text(encoding="utf-8"))
        generate = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "generate_video")
        args = generate.args.args + generate.args.kwonlyargs
        names = [argument.arg for argument in args]
        self.assertIn("minimax_h3_experimental_loras", names)
        self.assertIn("minimax_h3_experimental_loras=False", ast.unparse(generate))

        validate_call = next(
            node for node in ast.walk(generate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "validate_loras"
        )
        self.assertTrue(any(
            keyword.arg == "minimax_h3_experimental_loras"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "minimax_h3_experimental_loras"
            for keyword in validate_call.keywords
        ))
        loader_call = next(
            node for node in ast.walk(generate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_loras_into_model"
        )
        self.assertGreater(loader_call.lineno, validate_call.lineno)
        self.assertEqual(ast.unparse(loader_call.args[1]), "loras_selected")
        self.assertEqual(ast.unparse(loader_call.args[2]), "loras_list_mult_choices_nums")
        load_kwargs = next(
            node.value for node in ast.walk(generate)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "lora_load_kwargs"
                    for target in node.targets)
        )
        self.assertIsInstance(load_kwargs, ast.Dict)
        activation = next(
            value for key, value in zip(load_kwargs.keys, load_kwargs.values)
            if isinstance(key, ast.Constant) and key.value == "activate_all_loras"
        )
        self.assertIsInstance(activation, ast.Constant)
        self.assertIs(activation.value, True)

    def test_pika_model_lora_catalog_excludes_both_managed_variants_only(self):
        launch_path = APP / "launch.py"
        tree = ast.parse(launch_path.read_text(encoding="utf-8"))
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_managed_pika_flex_lora_names")
        namespace = {"__name__": "launch"}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(launch_path), "exec"), namespace)

        from models.minimax_h3.pika_flex_assets import PIKA_FLEX_LORA_SPECS
        all_pika = {spec["file"].casefold() for spec in PIKA_FLEX_LORA_SPECS.values()}
        self.assertEqual(namespace["_managed_pika_flex_lora_names"]({}), set())
        self.assertEqual(
            namespace["_managed_pika_flex_lora_names"]({"minimax_h3_pika_flex": True}),
            all_pika,
        )


if __name__ == "__main__":
    unittest.main()
