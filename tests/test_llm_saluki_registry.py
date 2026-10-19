"""Registry and request-mode contracts for the Underdog Saluki GGUF option."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from services import llm_service


SALUKI_REPO = "Kujira/Underdog-Saluki-27B-1.0-MTP-Abliterated-GGUF"
QWEN38_REPO = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
GEMMA_DEFAULT_REPO = "Abhiray/gemma-4-E4B-it-heretic-GGUF"


class TestSalukiRegistry(unittest.TestCase):
    def setUp(self):
        self.saved_model_id = llm_service._model_id
        self.saved_provider = llm_service._provider

    def tearDown(self):
        llm_service._model_id = self.saved_model_id
        llm_service._provider = self.saved_provider

    def test_public_picker_entry_is_adjacent_to_qwen38_and_keeps_gemma_default(self):
        models = llm_service.get_available_models(provider="local")
        ids = [model["id"] for model in models]
        self.assertIn(SALUKI_REPO, ids)
        self.assertEqual(ids[ids.index(QWEN38_REPO) + 1], SALUKI_REPO)

        model = next(item for item in models if item["id"] == SALUKI_REPO)
        self.assertEqual(
            model["label"],
            "Underdog Saluki 27B IQ2-Mix Uncensored (Vision, Thinking)",
        )
        self.assertEqual(model["size_hint"], "12 GB VRAM")
        self.assertEqual(llm_service.DEFAULT_HF_REPO, GEMMA_DEFAULT_REPO)

    def test_registry_uses_the_verified_weights_and_upstream_projector(self):
        entry = llm_service.MODEL_REGISTRY[SALUKI_REPO]
        self.assertEqual(
            entry["gguf_file"],
            "Underdog-Saluki-27B-1.0-IQ2-mix-MTP-abliterated.gguf",
        )
        self.assertEqual(
            entry["mmproj_file"],
            "mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf",
        )
        self.assertEqual(
            entry["mmproj_repo"],
            "ConwayResearch/Underdog-Saluki-27B-1.0",
        )
        # Decimal GB values come from the verified 8,349,689,920-byte
        # weights and 629,246,976-byte projector files.
        self.assertAlmostEqual(entry["weights_gb"], 8.34968992, places=4)
        self.assertAlmostEqual(entry["mmproj_gb"], 0.629246976, places=4)

    def test_32k_q8_profile_and_size_estimate(self):
        entry = llm_service.MODEL_REGISTRY[SALUKI_REPO]
        flags = entry["extra_flags"]
        self.assertEqual(flags[flags.index("-c") + 1], "32768")
        self.assertEqual(flags[flags.index("-np") + 1], "1")
        self.assertEqual(flags[flags.index("-fa") + 1], "on")
        self.assertEqual(flags[flags.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(flags[flags.index("--cache-type-v") + 1], "q8_0")
        self.assertNotIn("--spec-type", flags)
        self.assertNotIn("--spec-draft-n-max", flags)

        kv_gb = llm_service._estimate_kv_gb(entry["arch"], flags)
        self.assertAlmostEqual(kv_gb, 1.0625, places=6)
        self.assertEqual(entry["size_hint"], "12 GB VRAM")

    def test_qwen_thinking_defaults_and_explicit_nonthinking_controls(self):
        llm_service._model_id = SALUKI_REPO
        entry = llm_service.MODEL_REGISTRY[SALUKI_REPO]
        self.assertEqual(entry["default_reasoning_effort"], "xhigh")
        self.assertTrue(entry["enable_thinking_for_prompt_enhancement"])
        self.assertEqual(entry["prompt_enhancement_thinking_budget"], 8192)

        system, enabled, budget = llm_service._prepare_thinking(
            "Plan carefully.", None, 0
        )
        self.assertEqual(system, "Plan carefully.")
        self.assertIs(enabled, True)
        self.assertEqual(budget, 8192)

        reasoning_payload = {}
        effort = llm_service._apply_reasoning_controls(
            reasoning_payload, enable_thinking=True, thinking_budget=budget
        )
        self.assertEqual(effort, "xhigh")
        self.assertEqual(reasoning_payload["reasoning_effort"], "xhigh")
        self.assertEqual(reasoning_payload["thinking_budget_tokens"], 8192)

        thinking_payload = {
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
        }
        with mock.patch(
            "services.llm_sampling.caller_sampling_enabled", return_value=False
        ):
            temperature, top_p = llm_service._apply_model_defaults(
                0.8, 0.9, thinking_payload, enable_thinking=True
            )
        self.assertEqual((temperature, top_p), (1.0, 0.95))
        self.assertEqual(thinking_payload["top_k"], 20)
        self.assertNotIn("frequency_penalty", thinking_payload)
        self.assertEqual(thinking_payload["presence_penalty"], 1.1)
        self.assertNotIn("min_p", thinking_payload)
        self.assertNotIn("repeat_penalty", thinking_payload)

        _, enabled, budget = llm_service._prepare_thinking(
            "Return JSON.", False, 8192
        )
        self.assertIs(enabled, False)
        self.assertEqual(budget, 0)
        direct_payload = {"frequency_penalty": 0.3, "presence_penalty": 0.2}
        with mock.patch(
            "services.llm_sampling.caller_sampling_enabled", return_value=False
        ):
            temperature, top_p = llm_service._apply_model_defaults(
                0.9, 0.95, direct_payload, enable_thinking=False
            )
        self.assertEqual((temperature, top_p), (0.7, 0.8))
        self.assertEqual(direct_payload["top_k"], 20)
        self.assertEqual(direct_payload["frequency_penalty"], 0.3)
        self.assertEqual(direct_payload["presence_penalty"], 0.2)

    def test_json_schema_generation_forces_nonthinking_profile(self):
        llm_service._model_id = SALUKI_REPO
        llm_service._provider = "local"
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

        with (
            mock.patch("services.studio_enhancement.check_cancelled"),
            mock.patch(
                "services.studio_enhancement.is_cancellable", return_value=False
            ),
            mock.patch.object(llm_service, "is_loaded", return_value=True),
            mock.patch.object(llm_service, "_cancel_idle_timer"),
            mock.patch.object(llm_service, "_reset_idle_timer"),
            mock.patch.object(
                llm_service.requests, "post", return_value=response
            ) as post,
        ):
            result = llm_service.generate(
                "Return a JSON object.",
                max_new_tokens=64,
                temperature=0.9,
                top_p=0.95,
                thinking_budget=8192,
                enable_thinking=True,
                reasoning_effort="xhigh",
                frequency_penalty=0.3,
                presence_penalty=0.2,
                json_schema={"type": "object"},
            )

        self.assertEqual(result, "{}")
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["max_tokens"], 64)
        self.assertEqual(sent["temperature"], 0.7)
        self.assertEqual(sent["top_p"], 0.8)
        self.assertEqual(sent["top_k"], 20)
        self.assertEqual(sent["frequency_penalty"], 0.3)
        self.assertEqual(sent["presence_penalty"], 0.2)
        self.assertEqual(
            sent["response_format"],
            {"type": "json_object", "schema": {"type": "object"}},
        )
        self.assertIs(sent["enable_thinking"], False)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_effort", sent)
        self.assertNotIn("thinking_budget_tokens", sent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
