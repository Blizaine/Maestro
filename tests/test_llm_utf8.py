"""Exercise Requests' real JSON/SSE decoding without an LLM or network."""
import io
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import requests
from services import llm_service


class TestLlmUtf8(unittest.TestCase):
    def setUp(self):
        for name, value in [("is_loaded", True), ("_cancel_idle_timer", None), ("_reset_idle_timer", None), ("_count_local_tokens", None)]:
            patcher = mock.patch.object(llm_service, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in [("_provider", "local"), ("_model_id", "test-utf8")]:
            patcher = mock.patch.object(llm_service, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def response(self, payload, streaming=False):
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "text/event-stream" if streaming else "application/json; charset=ISO-8859-1"
        response.encoding = requests.utils.get_encoding_from_headers(response.headers)
        if streaming:
            response.raw = io.BytesIO(payload)
        else:
            response._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return response

    def test_openai_stream_preserves_multilingual_content_and_reasoning(self):
        # Arabic meem contains byte 0x85. Unicode separators are also legal in
        # JSON strings; neither should become a phantom SSE line boundary.
        answer = "مرحبا بالعالم — Привет мир — 中文 — café 🎬\u2028more\u0085text"
        reasoning = "تفكير — 思考"
        events = [{"choices": [{"delta": {"reasoning_content": reasoning}}]}]
        events.extend({"choices": [{"delta": {"content": char}}]} for char in answer)
        payload = b"".join(b"data: " + json.dumps(e, ensure_ascii=False).encode("utf-8") + b"\n\n" for e in events)
        payload += b"data: [DONE]\n\n"
        response = self.response(payload, streaming=True)
        # Force transport chunks to split multibyte characters.
        original_iter = response.iter_content
        response.iter_content = lambda chunk_size, decode_unicode: original_iter(1, decode_unicode)
        with mock.patch.object(requests, "post", return_value=response):
            self.assertEqual(llm_service.generate_streaming("test", enable_thinking=False), answer)
        self.assertEqual(llm_service._last_thinking_text, reasoning)
        self.assertIn(answer, llm_service._stream_buffer)

    def test_json_uses_utf8_even_with_incorrect_charset(self):
        answer = "مرحبا Привет 中文 🎬"
        for provider in ("local", "anthropic"):
            payload = ({"choices": [{"message": {"content": answer}}]} if provider == "local"
                       else {"content": [{"type": "text", "text": answer}]})
            with self.subTest(provider=provider), mock.patch.object(llm_service, "_provider", provider), mock.patch.object(requests, "post", return_value=self.response(payload)):
                self.assertEqual(llm_service.generate("test", enable_thinking=False), answer)

    def test_anthropic_stream_preserves_utf8(self):
        answer = "مرحبا Привет 中文 🎬"
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": answer}}
        payload = b"data: " + json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n\n"
        with mock.patch.object(llm_service, "_provider", "anthropic"), mock.patch.object(requests, "post", return_value=self.response(payload, True)):
            self.assertEqual(llm_service.generate_streaming("test", enable_thinking=False), answer)


if __name__ == "__main__":
    unittest.main()
