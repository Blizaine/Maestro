"""Registry and runtime contracts for Maestro's Qwen3.8 local LLM option."""

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


QWEN38_REPO = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"


class TestQwen38Registry(unittest.TestCase):
    def setUp(self):
        self.saved_model_id = llm_service._model_id
        self.saved_provider = llm_service._provider

    def tearDown(self):
        llm_service._model_id = self.saved_model_id
        llm_service._provider = self.saved_provider

    def test_model_is_in_the_curated_local_picker(self):
        models = llm_service.get_available_models(provider="local")
        ids = [model["id"] for model in models]
        self.assertIn(QWEN38_REPO, ids)
        self.assertEqual(ids[0], QWEN38_REPO)

        model = next(item for item in models if item["id"] == QWEN38_REPO)
        self.assertIn("Qwen3.8 27B", model["label"])
        self.assertEqual(model["size_hint"], "24 GB VRAM")

    def test_24gb_profile_uses_stable_q4_64k_and_vision(self):
        entry = llm_service.MODEL_REGISTRY[QWEN38_REPO]
        self.assertEqual(
            entry["gguf_file"],
            "Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf",
        )
        self.assertEqual(
            entry["mmproj_file"],
            "Qwen3.8-27B-Uncensored-vision-f16.gguf",
        )

        flags = entry["extra_flags"]
        self.assertEqual(flags[flags.index("-c") + 1], "65536")
        self.assertEqual(flags[flags.index("--cache-type-k") + 1], "q4_0")
        self.assertEqual(flags[flags.index("--cache-type-v") + 1], "q4_0")
        self.assertNotIn("--spec-type", flags)
        self.assertNotIn("--spec-draft-n-max", flags)
        self.assertEqual(entry["default_reasoning_effort"], "xhigh")

    def test_deep_thinking_defaults_and_explicit_opt_out(self):
        llm_service._model_id = QWEN38_REPO

        system, enabled, budget = llm_service._prepare_thinking(
            "Plan carefully.", None, 0
        )
        self.assertEqual(system, "Plan carefully.")
        self.assertIs(enabled, True)
        self.assertEqual(budget, 8192)

        _, enabled, budget = llm_service._prepare_thinking(
            "Return JSON.", False, 8192
        )
        self.assertIs(enabled, False)
        self.assertEqual(budget, 0)

    def test_creative_prompt_enhancement_uses_thinking(self):
        llm_service._model_id = QWEN38_REPO

        with mock.patch.object(
            llm_service,
            "generate",
            return_value="A richly detailed cinematic prompt.",
        ) as generate:
            result = llm_service.enhance_prompt(
                "A traveler enters an impossible city.",
                mode="video",
            )

        self.assertEqual(result, "A richly detailed cinematic prompt.")
        kwargs = generate.call_args.kwargs
        self.assertIs(kwargs["enable_thinking"], True)
        self.assertEqual(kwargs["thinking_budget"], 8192)

    def test_structured_h3_prompt_enhancement_keeps_thinking_off(self):
        llm_service._model_id = QWEN38_REPO
        structured = (
            "integrated_multimodal_description: [Shot 1] A traveler enters a city.\n\n"
            "overall_soundscape: Wind and distant traffic.\n\n"
            "non_diegetic_music: N/A"
        )

        with (
            mock.patch.object(llm_service, "generate", return_value=structured) as generate,
            mock.patch(
                "services.enhance_guides.get_enhance_guide",
                return_value="Write the required MiniMax H3 three-field prompt.",
            ),
        ):
            llm_service.enhance_prompt(
                "A traveler enters an impossible city.",
                mode="video",
                model_type="minimax_h3",
                duration_seconds=8,
            )

        first_call = generate.call_args_list[0].kwargs
        self.assertIs(first_call["enable_thinking"], False)
        self.assertEqual(first_call["thinking_budget"], 0)

    def test_sampling_changes_with_thinking_mode(self):
        llm_service._model_id = QWEN38_REPO

        thinking_payload = {
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
        }
        temperature, top_p = llm_service._apply_model_defaults(
            0.8,
            0.9,
            thinking_payload,
            enable_thinking=True,
        )
        self.assertEqual((temperature, top_p), (1.0, 0.95))
        self.assertEqual(thinking_payload["top_k"], 20)
        self.assertNotIn("frequency_penalty", thinking_payload)
        self.assertNotIn("presence_penalty", thinking_payload)

        direct_payload = {"frequency_penalty": 0.3}
        temperature, top_p = llm_service._apply_model_defaults(
            0.9,
            0.95,
            direct_payload,
            enable_thinking=False,
        )
        self.assertEqual((temperature, top_p), (0.7, 0.8))
        # Maestro's pass-specific structured-output penalty remains intact.
        self.assertEqual(direct_payload["frequency_penalty"], 0.3)

    def test_creative_requests_send_explicit_xhigh_and_budget(self):
        llm_service._model_id = QWEN38_REPO
        payload = {}

        effort = llm_service._apply_reasoning_controls(
            payload,
            enable_thinking=True,
            thinking_budget=16384,
        )

        self.assertEqual(effort, "xhigh")
        self.assertEqual(payload["reasoning_effort"], "xhigh")
        self.assertEqual(payload["thinking_budget_tokens"], 16384)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
        )

    def test_structured_requests_do_not_send_reasoning_controls(self):
        llm_service._model_id = QWEN38_REPO
        payload = {}

        effort = llm_service._apply_reasoning_controls(
            payload,
            enable_thinking=False,
            thinking_budget=0,
            reasoning_effort="xhigh",
        )

        self.assertIsNone(effort)
        self.assertEqual(payload, {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        })

    def test_generate_records_reasoning_usage_and_finish_reason(self):
        llm_service._model_id = QWEN38_REPO
        llm_service._provider = "local"
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "Final screenplay.",
                    "reasoning_content": "I considered the dramatic options.",
                },
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }

        with (
            mock.patch.object(llm_service, "is_loaded", return_value=True),
            mock.patch.object(llm_service, "_cancel_idle_timer"),
            mock.patch.object(llm_service, "_reset_idle_timer"),
            mock.patch.object(llm_service.requests, "post", return_value=response) as post,
        ):
            result = llm_service.generate(
                "Write it.",
                max_new_tokens=200,
                thinking_budget=16384,
            )

        self.assertEqual(result, "Final screenplay.")
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["reasoning_effort"], "xhigh")
        self.assertEqual(sent["thinking_budget_tokens"], 16384)
        self.assertEqual(sent["max_tokens"], 16584)
        self.assertEqual(llm_service._last_generation_metrics, {
            "model_id": QWEN38_REPO,
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "reasoning_tokens": 20,
            "answer_tokens": 10,
            "finish_reason": "stop",
            "truncated": False,
            "reasoning_effort": "xhigh",
            "thinking_budget_tokens": 16384,
            "requested_answer_tokens": 200,
            "request_max_tokens": 16584,
        })

    def test_streaming_collects_final_usage_chunk(self):
        llm_service._model_id = QWEN38_REPO
        llm_service._provider = "local"
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = iter([
            'data: {"choices":[{"delta":{"reasoning_content":"Plan first. "},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"Final answer."},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":18,"completion_tokens_details":{"reasoning_tokens":11}}}',
            "data: [DONE]",
        ])

        with (
            mock.patch.object(llm_service, "is_loaded", return_value=True),
            mock.patch.object(llm_service, "_cancel_idle_timer"),
            mock.patch.object(llm_service, "_reset_idle_timer"),
            mock.patch.object(llm_service.requests, "post", return_value=response) as post,
        ):
            result = llm_service.generate_streaming(
                "Write it.",
                max_new_tokens=100,
                thinking_budget=8192,
            )

        self.assertEqual(result, "Final answer.")
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["stream_options"], {"include_usage": True})
        self.assertEqual(sent["reasoning_effort"], "xhigh")
        metrics = llm_service._last_generation_metrics
        self.assertEqual(metrics["reasoning_tokens"], 11)
        self.assertEqual(metrics["answer_tokens"], 7)
        self.assertEqual(metrics["finish_reason"], "stop")

    def test_llama_runtime_floor_includes_qwen38_cuda_fix(self):
        self.assertGreaterEqual(llm_service.MIN_LLAMA_BUILD, 10450)
        self.assertGreaterEqual(
            llm_service._llama_release_build(llm_service.FALLBACK_LLAMA_TAG),
            10450,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
