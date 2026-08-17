"""Contract tests for remote OpenAI-compatible request payloads.

Payloads are authored in llama-server's dialect: it serves one model, so
no "model" field is sent, and it accepts llama.cpp sampling extensions.
Remote endpoints need the opposite -- "model" is required by the spec and
strict gateways reject unknown fields with 400 Bad Request. These tests
pin the translation between the two.
"""
from __future__ import annotations

import os
import sys
import unittest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)


def _llama_style_payload():
    """What generate() builds for a vision-enabled enhance call."""
    return {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "describe"},
            ],
        }],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["<think>"],
        "seed": 42,
        # llama.cpp-only below this line
        "cache_prompt": False,
        "top_k": 64,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class TestFinalizePayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from services import llm_service
        except ImportError as exc:
            raise unittest.SkipTest("llm_service dependencies unavailable") from exc
        cls.llm_service = llm_service

    def setUp(self):
        self.saved = (self.llm_service._provider, self.llm_service._model_id)

    def tearDown(self):
        self.llm_service._provider, self.llm_service._model_id = self.saved

    def _finalize(self, provider, payload, model_id="claude-opus-5"):
        self.llm_service._provider = provider
        self.llm_service._model_id = model_id
        return self.llm_service._finalize_payload(payload)

    def test_local_payload_is_untouched(self):
        """llama-server wants its own dialect — do not rewrite it."""
        payload = _llama_style_payload()
        self.assertEqual(self._finalize("local", dict(payload)), payload)

    def test_remote_request_carries_a_model(self):
        """Required by the OpenAI spec; a gateway cannot route without it."""
        out = self._finalize("remote", _llama_style_payload())
        self.assertEqual(out.get("model"), "claude-opus-5")

    def test_remote_drops_llama_cpp_extensions(self):
        out = self._finalize("remote", _llama_style_payload())
        for field in (
            "cache_prompt", "top_k", "min_p", "repeat_penalty",
            "repeat_last_n", "enable_thinking", "chat_template_kwargs",
        ):
            with self.subTest(field=field):
                self.assertNotIn(field, out)

    def test_remote_keeps_standard_fields(self):
        source = _llama_style_payload()
        out = self._finalize("remote", dict(source))
        for field in ("messages", "max_tokens", "temperature", "top_p", "stop", "seed"):
            with self.subTest(field=field):
                self.assertEqual(out[field], source[field])

    def test_multimodal_content_survives(self):
        """The image part must reach a vision endpoint intact."""
        out = self._finalize("remote", _llama_style_payload())
        parts = out["messages"][0]["content"]
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertEqual(parts[0]["image_url"]["url"], "data:image/png;base64,AAAA")
        self.assertEqual(parts[1]["text"], "describe")

    def test_openai_provider_translated_too(self):
        """The hosted OpenAI API is as strict as any other gateway."""
        out = self._finalize("openai", _llama_style_payload(), model_id="gpt-5")
        self.assertEqual(out.get("model"), "gpt-5")
        self.assertNotIn("cache_prompt", out)

    def test_caller_payload_not_mutated(self):
        """Translation must not corrupt the caller's dict."""
        payload = _llama_style_payload()
        self._finalize("remote", payload)
        self.assertIn("cache_prompt", payload)
        self.assertNotIn("model", payload)

    def test_every_kept_field_is_a_known_openai_field(self):
        out = self._finalize("remote", _llama_style_payload())
        self.assertTrue(set(out).issubset(self.llm_service._OPENAI_CHAT_FIELDS))


class TestRemoteFailureDetail(unittest.TestCase):
    """A rejected request must quote the endpoint's explanation."""

    @classmethod
    def setUpClass(cls):
        try:
            import requests
            from services import llm_service
        except ImportError as exc:
            raise unittest.SkipTest("llm_service dependencies unavailable") from exc
        cls.requests = requests
        cls.llm_service = llm_service

    def setUp(self):
        self.saved = (self.llm_service._provider, self.llm_service._process)
        self.llm_service._provider = "remote"
        self.llm_service._process = None

    def tearDown(self):
        self.llm_service._provider, self.llm_service._process = self.saved

    def _error_for(self, status, body):
        response = self.requests.Response()
        response.status_code = status
        response.url = "https://api.example.com/v1/chat/completions"
        response._content = body
        try:
            response.raise_for_status()
        except self.requests.exceptions.RequestException as exc:
            return self.llm_service._diagnose_llm_request_failure(exc)
        self.fail("expected raise_for_status to raise")

    def test_response_body_is_quoted(self):
        message = str(self._error_for(
            400, b'{"error":{"message":"Unrecognized request argument: cache_prompt"}}'
        ))
        self.assertIn("Unrecognized request argument: cache_prompt", message)

    def test_long_body_is_truncated(self):
        message = str(self._error_for(400, b"x" * 5000))
        self.assertIn("truncated", message)
        self.assertLess(len(message), 1200)

    def test_empty_body_adds_no_noise(self):
        message = str(self._error_for(500, b""))
        self.assertIn("LLM request failed", message)
        self.assertNotIn("Endpoint response", message)

    def test_connection_error_without_response_is_handled(self):
        """No response object at all — must not raise while building the error."""
        exc = self.requests.exceptions.ConnectionError("connection refused")
        message = str(self.llm_service._diagnose_llm_request_failure(exc))
        self.assertIn("connection refused", message)


if __name__ == "__main__":
    unittest.main()
