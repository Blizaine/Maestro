import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'app'))

from app.services.env_snapshot import (
    build_env_snapshot,
    config_digest,
    current_env_snapshot,
    _strip_secrets,
)


class StripSecretsTests(unittest.TestCase):
    def test_secret_shaped_keys_become_set_booleans(self):
        self.assertEqual(
            _strip_secrets({"civitai_api_key": "abc123"}),
            {"civitai_api_key_set": True},
        )
        self.assertEqual(
            _strip_secrets({"civitai_api_key": ""}),
            {"civitai_api_key_set": False},
        )

    def test_match_is_case_insensitive_and_covers_common_names(self):
        stripped = _strip_secrets({
            "OPENAI_API_KEY": "sk-x",
            "auth_token": "t",
            "my_password": "p",
            "session_secret": "s",
            "user_credential": "c",
        })
        self.assertNotIn("OPENAI_API_KEY", stripped)
        for name in ("OPENAI_API_KEY", "auth_token", "my_password",
                     "session_secret", "user_credential"):
            self.assertTrue(stripped[f"{name}_set"])

    def test_nested_dicts_and_lists_are_scrubbed(self):
        stripped = _strip_secrets({
            "services": {"inner": {"api_key": "leak"}},
            "items": [{"token": "leak"}, {"kept": 2}],
        })
        self.assertEqual(
            stripped["services"]["inner"], {"api_key_set": True}
        )
        self.assertEqual(stripped["items"], [{"token_set": True}, {"kept": 2}])

    def test_ordinary_values_survive(self):
        self.assertEqual(
            _strip_secrets({"vram_allocator": "vmm"}),
            {"vram_allocator": "vmm"},
        )


class ConfigDigestTests(unittest.TestCase):
    def test_digest_is_stable_and_key_order_independent(self):
        self.assertEqual(
            config_digest({"a": 1, "b": 2}),
            config_digest({"b": 2, "a": 1}),
        )

    def test_digest_changes_when_an_unlisted_setting_moves(self):
        self.assertNotEqual(
            config_digest({"some_unlisted_setting": 1}),
            config_digest({"some_unlisted_setting": 2}),
        )

    def test_rotating_a_credential_does_not_move_the_digest(self):
        # The digest covers the scrubbed config, so a new key is not a
        # spurious "the environment changed" signal.
        self.assertEqual(
            config_digest({"services": {"civitai_api_key": "one"}}),
            config_digest({"services": {"civitai_api_key": "two"}}),
        )

    def test_switching_studio_mode_does_not_move_the_digest(self):
        # Observed in the wild: generating an image then a video rewrote
        # maestro_studio_preferences and flipped the digest, which made it
        # noise rather than a signal.
        image_session = {
            "vram_allocator": "vmm",
            "maestro_studio_preferences": {
                "generation_mode": "image",
                "selected_model_per_mode": {"image": "qwen_image_21_7B"},
            },
        }
        video_session = {
            "vram_allocator": "vmm",
            "maestro_studio_preferences": {
                "generation_mode": "video",
                "selected_model_per_mode": {"video": "minimax_h3_ref2va"},
            },
        }
        self.assertEqual(
            config_digest(image_session), config_digest(video_session)
        )

    def test_ui_only_state_is_ignored(self):
        for noisy in ("UI_theme", "queue_color_scheme",
                      "maestro_model_visibility", "generation_preview"):
            self.assertEqual(
                config_digest({"vram_allocator": "vmm"}),
                config_digest({"vram_allocator": "vmm", noisy: "anything"}),
                f"{noisy} should not move the digest",
            )

    def test_switching_workspace_is_ignored(self):
        self.assertEqual(
            config_digest({"services": {"active_workspace": "default"}}),
            config_digest({"services": {"active_workspace": "Proofs"}}),
        )

    def test_a_real_performance_setting_still_moves_the_digest(self):
        self.assertNotEqual(
            config_digest({"vram_allocator": "vmm"}),
            config_digest({"vram_allocator": "native"}),
        )

    def test_unserializable_values_still_digest_stably(self):
        first = config_digest({"x": object()})
        second = config_digest({"x": object()})
        self.assertTrue(first)
        self.assertEqual(first, second)


class BuildEnvSnapshotTests(unittest.TestCase):
    def config(self):
        return {
            "vram_allocator": "vmm",
            "ram_allocator": "default",
            "attention_mode": "auto",
            "transformer_quantization": "int8",
            "int8_kernels": "kitchen",
            "vae_config": 1,
            "video_output_codec": "libx264_8",
            "image_output_codec": "png",
            "video_preload_mode": "default",
            "rife_version": "v4",
            "UI_theme": "default",
            "services": {
                "civitai_api_key": "b3624ab2b2435a32",
                "llm_device": "cuda",
                "auto_performance": True,
            },
            "maestro_studio_preferences": {
                "h3_optimizations": {"override_attention": "sla"},
            },
        }

    def test_captures_the_settings_a_run_depends_on(self):
        snapshot = build_env_snapshot(self.config(), "3.2.0")
        self.assertEqual(snapshot["maestro_version"], "3.2.0")
        self.assertEqual(snapshot["vram_allocator"], "vmm")
        self.assertEqual(snapshot["int8_kernels"], "kitchen")
        self.assertEqual(snapshot["video_output_codec"], "libx264_8")
        self.assertEqual(snapshot["services"]["llm_device"], "cuda")
        self.assertEqual(
            snapshot["h3_optimizations"]["override_attention"], "sla"
        )

    def test_credentials_never_reach_the_sidecar(self):
        snapshot = build_env_snapshot(self.config(), "3.2.0")
        blob = json.dumps(snapshot)
        self.assertNotIn("b3624ab2b2435a32", blob)
        # The services allowlist drops the credential outright, so not
        # even its name survives into a sidecar that may be shared.
        self.assertNotIn("civitai_api_key", blob)

    def test_unlisted_keys_are_left_out_but_still_move_the_digest(self):
        # lm_decoder_engine is a real engine setting the allowlist does not
        # name yet -- exactly the case the digest exists to surface.
        base = self.config()
        changed = self.config()
        changed["lm_decoder_engine"] = "something_else"
        self.assertNotIn("lm_decoder_engine", build_env_snapshot(base, "3.2.0"))
        self.assertNotEqual(
            build_env_snapshot(base, "3.2.0")["config_digest"],
            build_env_snapshot(changed, "3.2.0")["config_digest"],
        )

    def test_ui_theme_is_neither_captured_nor_digested(self):
        base = self.config()
        themed = self.config()
        themed["UI_theme"] = "midnight"
        self.assertNotIn("UI_theme", build_env_snapshot(base, "3.2.0"))
        self.assertEqual(
            build_env_snapshot(base, "3.2.0")["config_digest"],
            build_env_snapshot(themed, "3.2.0")["config_digest"],
        )

    def test_absent_keys_are_omitted_rather_than_guessed(self):
        snapshot = build_env_snapshot({"vram_allocator": "vmm"}, "3.2.0")
        self.assertNotIn("attention_mode", snapshot)

    def test_a_broken_config_never_costs_us_the_sidecar(self):
        for bad in (None, {}, {"services": "not-a-dict"}):
            self.assertIsInstance(build_env_snapshot(bad, ""), dict)


class CurrentEnvSnapshotTests(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("wgp")

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("wgp", None)
        else:
            sys.modules["wgp"] = self._saved

    def test_reads_the_loaded_engine_config_and_repo_version(self):
        engine = type(sys)("wgp")
        engine.server_config = {"vram_allocator": "vmm", "openai_api_key": "sk-x"}
        sys.modules["wgp"] = engine
        snapshot = current_env_snapshot()
        version = (Path(__file__).parents[1] / "VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(snapshot["maestro_version"], version)
        self.assertEqual(snapshot["vram_allocator"], "vmm")
        self.assertNotIn("sk-x", json.dumps(snapshot))

    def test_works_without_the_engine_loaded(self):
        sys.modules.pop("wgp", None)
        snapshot = current_env_snapshot()
        self.assertIsInstance(snapshot, dict)
        self.assertIn("config_digest", snapshot)


if __name__ == "__main__":
    unittest.main()
