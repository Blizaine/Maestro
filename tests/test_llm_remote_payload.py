"""OpenAI-compatible payload translation and failure diagnostics."""

from __future__ import annotations

import os
import sys
import unittest

import requests


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from services import llm_service


def _llama_payload():
    return {
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {"type": "text", "text": "describe"},
            ],
        }],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["<think>"],
        "seed": 42,
        "cache_prompt": False,
        "top_k": 64,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class TestFinalizePayload(unittest.TestCase):
    def setUp(self):
        self.saved = (llm_service._provider, llm_service._model_id)

    def tearDown(self):
        llm_service._provider, llm_service._model_id = self.saved

    def _finalize(self, provider, payload, model_id="remote-model"):
        llm_service._provider = provider
        llm_service._model_id = model_id
        return llm_service._finalize_payload(payload)

    def test_local_payload_is_untouched(self):
        payload = _llama_payload()
        self.assertIs(self._finalize("local", payload), payload)

    def test_remote_adds_model_and_drops_llama_extensions(self):
        output = self._finalize("remote", _llama_payload())
        self.assertEqual(output["model"], "remote-model")
        for field in (
            "cache_prompt",
            "top_k",
            "min_p",
            "repeat_penalty",
            "repeat_last_n",
            "enable_thinking",
            "chat_template_kwargs",
        ):
            with self.subTest(field=field):
                self.assertNotIn(field, output)

    def test_standard_fields_and_multimodal_content_survive(self):
        source = _llama_payload()
        output = self._finalize("remote", source)
        for field in (
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "seed",
        ):
            with self.subTest(field=field):
                self.assertEqual(output[field], source[field])
        self.assertEqual(
            output["messages"][0]["content"][0]["type"],
            "image_url",
        )

    def test_openai_is_translated_but_anthropic_native_payload_is_not(self):
        payload = _llama_payload()
        openai = self._finalize("openai", payload, "gpt-5")
        self.assertEqual(openai["model"], "gpt-5")
        self.assertNotIn("cache_prompt", openai)
        self.assertIs(self._finalize("anthropic", payload), payload)

    def test_translation_does_not_mutate_the_caller(self):
        payload = _llama_payload()
        self._finalize("remote", payload)
        self.assertIn("cache_prompt", payload)
        self.assertNotIn("model", payload)

    def test_every_output_field_is_known(self):
        output = self._finalize("remote", _llama_payload())
        self.assertTrue(set(output).issubset(llm_service._OPENAI_CHAT_FIELDS))


class TestRemoteFailureDetail(unittest.TestCase):
    def setUp(self):
        self.saved = (llm_service._provider, llm_service._process)
        llm_service._provider = "remote"
        llm_service._process = None

    def tearDown(self):
        llm_service._provider, llm_service._process = self.saved

    @staticmethod
    def _error_for(status, body):
        response = requests.Response()
        response.status_code = status
        response.url = "https://api.example.com/v1/chat/completions"
        response._content = body
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            return llm_service._diagnose_llm_request_failure(exc)
        raise AssertionError("expected raise_for_status to fail")

    def test_endpoint_response_body_is_included(self):
        message = str(self._error_for(
            400,
            b'{"error":{"message":"Unrecognized request argument: cache_prompt"}}',
        ))
        self.assertIn("Unrecognized request argument: cache_prompt", message)

    def test_long_response_is_truncated(self):
        message = str(self._error_for(400, b"x" * 5000))
        self.assertIn("truncated", message)
        self.assertLess(len(message), 1200)

    def test_connection_error_without_response_is_safe(self):
        error = requests.exceptions.ConnectionError("connection refused")
        message = str(llm_service._diagnose_llm_request_failure(error))
        self.assertIn("connection refused", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
