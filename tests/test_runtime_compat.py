from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.runtime_compat import (  # noqa: E402
    evaluate_runtime,
    is_mmgp_lora_pinning_assertion,
    is_rtx50_capability,
    recommend_h3_lora_pin_budget_mb,
    version_tuple,
)
from services.perf_recommend import recommend_h3_reserved_ram_fraction  # noqa: E402
from services.optional_acceleration import (  # noqa: E402
    flash_attention_wheel_compatibility,
    prepare_optional_flash_attention,
)
from scripts.install_gguf_kernels import _pick_wheel_url  # noqa: E402


class TestRuntimeCompatibility(unittest.TestCase):
    def test_correct_rtx50_runtime_has_no_warning(self):
        self.assertEqual(
            evaluate_runtime(
                python_version="3.11.14",
                torch_version="2.10.0+cu130",
                cuda_version="13.0",
                capability=(12, 0),
                triton_available=True,
                sage_available=True,
                lightx2v_available=True,
            ),
            [],
        )

    def test_public_legacy_stack_explains_the_slow_fallback(self):
        warnings = evaluate_runtime(
            python_version="3.10.11",
            torch_version="2.7.1+cu128",
            cuda_version="12.8",
            capability="sm_120",
            triton_available=True,
            sage_available=True,
            lightx2v_available=False,
        )
        joined = " ".join(warnings)
        self.assertIn("Python 3.11", joined)
        self.assertIn("PyTorch 2.10", joined)
        self.assertIn("CUDA 13", joined)
        self.assertIn("very slow fallback", joined)

    def test_non_blackwell_runtime_is_not_forced_to_migrate(self):
        self.assertEqual(
            evaluate_runtime(
                python_version="3.10",
                torch_version="2.7.1",
                cuda_version="12.8",
                capability=(8, 9),
                triton_available=False,
                sage_available=False,
                lightx2v_available=False,
            ),
            [],
        )

    def test_optional_sage_is_not_a_required_sol_runtime_warning(self):
        self.assertEqual(
            evaluate_runtime(
                python_version="3.11.14",
                torch_version="2.10.0+cu130",
                cuda_version="13.0",
                capability=(12, 0),
                triton_available=True,
                sage_available=False,
                lightx2v_available=True,
            ),
            [],
        )

    def test_capability_and_version_parsing(self):
        self.assertTrue(is_rtx50_capability("sm_120"))
        self.assertFalse(is_rtx50_capability("sm_89"))
        self.assertEqual(version_tuple("2.10.0+cu130"), (2, 10))


class TestOptionalAccelerationCompatibility(unittest.TestCase):
    def test_known_windows_flash_wheels_gate_cuda_architecture(self):
        compatible, detail = flash_attention_wheel_compatibility(
            platform_name="win32",
            package_version="2.7.4.post1",
            cuda_capability=(8, 9),
        )
        self.assertTrue(compatible)
        self.assertIsNone(detail)

        compatible, detail = flash_attention_wheel_compatibility(
            platform_name="win32",
            package_version="2.7.4.post1",
            cuda_capability=(8, 6),
        )
        self.assertFalse(compatible)
        self.assertIn("sm_89", detail)
        self.assertIn("sm_86", detail)

        # The CUDA 13 wheel includes an sm_80 cubin, which CUDA can execute
        # across the Ampere/Ada compute-capability family.
        compatible, detail = flash_attention_wheel_compatibility(
            platform_name="win32",
            package_version="2.8.3",
            cuda_capability=(8, 9),
        )
        self.assertTrue(compatible)
        self.assertIsNone(detail)

    def test_importable_but_incompatible_flash_attention_falls_back(self):
        flash = SimpleNamespace(
            flash_attn_func=object(),
            flash_attn_varlen_func=object(),
        )
        diffusers_imports = SimpleNamespace(_flash_attn_available=True)
        messages = []

        def fake_import(name):
            if name == "flash_attn":
                return flash
            if name == "diffusers.utils.import_utils":
                return diffusers_imports
            raise AssertionError(f"unexpected import: {name}")

        with patch.dict(sys.modules, {}, clear=False):
            self.assertFalse(
                prepare_optional_flash_attention(
                    import_module=fake_import,
                    emit=messages.append,
                    platform_name="win32",
                    package_version="2.7.4.post1",
                    cuda_capability=(8, 6),
                )
            )
            self.assertIsNone(sys.modules.get("flash_attn"))

        self.assertFalse(diffusers_imports._flash_attn_available)
        self.assertEqual(len(messages), 1)
        self.assertIn("sm_86", messages[0])
        self.assertIn("continuing with SageAttention/SDPA", messages[0])

    def test_working_flash_attention_remains_available(self):
        flash = SimpleNamespace(
            flash_attn_func=object(),
            flash_attn_varlen_func=object(),
        )
        messages = []

        self.assertTrue(
            prepare_optional_flash_attention(
                import_module=lambda name: flash,
                emit=messages.append,
            )
        )
        self.assertEqual(messages, [])

    def test_broken_flash_attention_disables_diffusers_and_falls_back(self):
        diffusers_imports = SimpleNamespace(_flash_attn_available=True)
        messages = []

        def fake_import(name):
            if name == "flash_attn":
                raise ImportError("DLL load failed while importing flash_attn_2_cuda")
            if name == "diffusers.utils.import_utils":
                return diffusers_imports
            raise AssertionError(f"unexpected import: {name}")

        with patch.dict(sys.modules, {}, clear=False):
            self.assertFalse(
                prepare_optional_flash_attention(
                    import_module=fake_import,
                    emit=messages.append,
                )
            )
            self.assertIsNone(sys.modules.get("flash_attn"))

        self.assertFalse(diffusers_imports._flash_attn_available)
        self.assertEqual(len(messages), 1)
        self.assertIn("continuing with SageAttention/SDPA", messages[0])


class TestH3ReservedRamRecommendation(unittest.TestCase):
    def test_full_h3_fits_in_128gb_workstation_reserved_ram(self):
        self.assertEqual(
            recommend_h3_reserved_ram_fraction(0, 128, full_checkpoint=True),
            0.5,
        )

    def test_96gb_workstation_keeps_60gb_ceiling(self):
        self.assertEqual(
            recommend_h3_reserved_ram_fraction(0, 96, full_checkpoint=True),
            0.625,
        )

    def test_lower_ram_and_pruned_models_keep_mmgp_default(self):
        self.assertEqual(
            recommend_h3_reserved_ram_fraction(0, 64, full_checkpoint=True),
            0.0,
        )
        self.assertEqual(
            recommend_h3_reserved_ram_fraction(0, 128, full_checkpoint=False),
            0.0,
        )

    def test_explicit_user_value_always_wins(self):
        self.assertEqual(
            recommend_h3_reserved_ram_fraction(0.45, 128, full_checkpoint=True),
            0.45,
        )


class TestH3LoraPinnedMemoryFallback(unittest.TestCase):
    def test_full_h3_uses_regular_ram_for_default_unlimited_lora_budget(self):
        self.assertEqual(
            recommend_h3_lora_pin_budget_mb(
                -1,
                full_checkpoint=True,
                platform_name="nt",
            ),
            0.0,
        )

    def test_pruned_h3_keeps_default_lora_pinning(self):
        self.assertEqual(
            recommend_h3_lora_pin_budget_mb(
                -1,
                full_checkpoint=False,
                platform_name="nt",
            ),
            -1.0,
        )

    def test_non_windows_full_h3_attempts_pinning_before_guarded_fallback(self):
        self.assertEqual(
            recommend_h3_lora_pin_budget_mb(
                -1,
                full_checkpoint=True,
                platform_name="posix",
            ),
            -1.0,
        )

    def test_explicit_lora_pin_budget_still_wins(self):
        self.assertEqual(
            recommend_h3_lora_pin_budget_mb(
                "1024",
                full_checkpoint=True,
                platform_name="nt",
            ),
            1024.0,
        )

    def test_only_mmgp_pin_helper_assertions_are_recoverable(self):
        def _pin_sd_to_memory():
            raise AssertionError()

        try:
            _pin_sd_to_memory()
        except AssertionError as exc:
            self.assertTrue(is_mmgp_lora_pinning_assertion(exc))

        try:
            raise AssertionError()
        except AssertionError as exc:
            self.assertFalse(is_mmgp_lora_pinning_assertion(exc))


class TestCuda13GgufKernelSelection(unittest.TestCase):
    def test_python311_torch210_cuda13_has_both_platform_wheels(self):
        windows = _pick_wheel_url(11, "2.10", "13.0", "windows")
        linux = _pick_wheel_url(11, "2.10", "13.0", "linux")
        self.assertIn("win_amd64.whl", windows or "")
        self.assertIn("linux_x86_64.whl", linux or "")


if __name__ == "__main__":
    unittest.main()
