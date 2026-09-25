"""OpenAI-compatible payload translation and failure diagnostics."""

from __future__ import annotations

import json
import os
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

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

    def test_finalizer_leaves_anthropic_conversion_to_its_native_request_path(self):
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


class _TransportResponse:
    def __init__(self, body=None, *, status_code=200, text=None, lines=()):
        self._body = body or {}
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(self._body)
        self.url = "https://api.example.test/v1/messages"
        self.encoding = None
        self._lines = list(lines)
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            response.url = self.url
            response._content = self.text.encode("utf-8")
            raise requests.HTTPError(f"{self.status_code} Client Error", response=response)

    def json(self):
        return self._body

    def iter_lines(self, **_kwargs):
        yield from self._lines

    def close(self):
        self.closed = True


class TestRemoteVisionRequests(unittest.TestCase):
    PROVIDERS = ("remote", "openai", "anthropic")
    IMAGE_URL = "data:image/jpeg;base64,ZmFrZS1pbWFnZQ=="

    def setUp(self):
        self.saved = {
            name: getattr(llm_service, name)
            for name in (
                "_provider", "_model_id", "_device", "_server_port",
                "_remote_url", "_api_key", "_vision_available", "_process",
                "_stream_buffer", "_stream_done",
            )
        }

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(llm_service, name, value)

    def _prepare_provider(self, provider):
        llm_service._provider = provider
        llm_service._model_id = "text-or-vision-model"
        llm_service._device = provider
        llm_service._server_port = 0
        llm_service._remote_url = "https://api.example.test"
        llm_service._api_key = "test-key"
        llm_service._vision_available = True
        llm_service._process = None

    def test_remote_provider_model_loads_enable_vision_requests(self):
        for provider in self.PROVIDERS:
            with self.subTest(provider=provider):
                llm_service._model_id = ""
                llm_service._process = None
                with patch.object(llm_service, "_reset_idle_timer"):
                    llm_service.load_model(
                        "remote-vision-model",
                        provider=provider,
                        remote_url="https://api.example.test",
                        api_key="test-key",
                    )
                self.assertTrue(llm_service._vision_available)

    @staticmethod
    def _openai_response(streaming):
        if streaming:
            return _TransportResponse(lines=(
                b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                b"data: [DONE]",
            ))
        return _TransportResponse({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        })

    @staticmethod
    def _anthropic_response(streaming):
        if streaming:
            return _TransportResponse(lines=(
                b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}',
                b'data: {"type":"message_stop"}',
            ))
        return _TransportResponse({
            "content": [{"type": "text", "text": "ok"}],
            "usage": {},
        })

    def _run_request(self, provider, streaming, image_paths, response):
        self._prepare_provider(provider)
        captured = {}

        def post(_url, **kwargs):
            captured.update(kwargs)
            return response

        common_patches = (
            patch.object(llm_service, "_cancel_idle_timer"),
            patch.object(llm_service, "_reset_idle_timer"),
            patch.object(llm_service, "_prepare_thinking", side_effect=lambda system, enabled, budget: (system, enabled, budget)),
            patch.object(llm_service, "record_payload", side_effect=lambda payload, **_kwargs: payload),
            patch.object(llm_service, "record_response"),
            patch.object(llm_service, "record_metrics"),
            patch.object(llm_service, "_image_to_data_url", return_value=self.IMAGE_URL),
            patch.object(llm_service.requests, "post", side_effect=post),
        )
        with ExitStack() as stack:
            for patcher in common_patches:
                stack.enter_context(patcher)
            stack.enter_context(patch(
                "services.studio_enhancement.cancellable_lines",
                side_effect=lambda resp, **kwargs: resp.iter_lines(**kwargs),
            ))
            generate = llm_service.generate_streaming if streaming else llm_service.generate
            result = generate(
                "describe this frame",
                system_prompt="Keep the answer concise.",
                max_new_tokens=32,
                image_paths=image_paths,
                enable_thinking=False,
            )
        return result, captured

    def test_unreadable_remote_images_fail_instead_of_being_dropped(self):
        self._prepare_provider("remote")
        with patch.object(llm_service, "_image_to_data_url", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Could not attach an image"):
                llm_service._user_message("describe this frame", ["missing.png"])

    def _assert_stream_preparation_error(
        self,
        provider,
        *,
        image_result=None,
        image_error=None,
        error_type=RuntimeError,
        error_text="Could not attach an image",
    ):
        self._prepare_provider(provider)
        with ExitStack() as stack:
            for patcher in (
                patch.object(llm_service, "_cancel_idle_timer"),
                patch.object(llm_service, "_reset_idle_timer"),
                patch.object(
                    llm_service,
                    "_prepare_thinking",
                    side_effect=lambda system, enabled, budget: (system, enabled, budget),
                ),
                patch.object(llm_service.requests, "post"),
            ):
                stack.enter_context(patcher)
            post = llm_service.requests.post
            image_patcher = (
                patch.object(llm_service, "_image_to_data_url", side_effect=image_error)
                if image_error is not None
                else patch.object(llm_service, "_image_to_data_url", return_value=image_result)
            )
            stack.enter_context(image_patcher)
            with self.assertRaisesRegex(error_type, error_text):
                llm_service.generate_streaming(
                    "describe this frame",
                    image_paths=["broken.png"],
                    enable_thinking=False,
                )
        post.assert_not_called()
        self.assertTrue(llm_service._stream_done)
        self.assertIn(error_text, llm_service._stream_buffer)

    def test_streaming_missing_and_corrupt_images_finish_without_post(self):
        for failure, image_result, image_error in (
            ("missing", None, None),
            ("corrupt", None, OSError("invalid image data")),
        ):
            with self.subTest(failure=failure):
                self._assert_stream_preparation_error(
                    "remote",
                    image_result=image_result,
                    image_error=image_error,
                )

    def test_anthropic_streaming_conversion_error_finishes_without_post(self):
        self._assert_stream_preparation_error(
            "anthropic",
            image_result="data:image/bmp;base64,ZmFrZQ==",
            error_type=ValueError,
            error_text="does not support image media type",
        )

    def test_provider_images_reach_streaming_and_nonstreaming_requests(self):
        for provider in self.PROVIDERS:
            for streaming in (False, True):
                with self.subTest(provider=provider, streaming=streaming):
                    response = (
                        self._anthropic_response(streaming)
                        if provider == "anthropic"
                        else self._openai_response(streaming)
                    )
                    result, request = self._run_request(provider, streaming, ["frame.png"], response)
                    self.assertEqual(result, "ok")
                    body = request["json"]
                    if provider == "anthropic":
                        self.assertEqual(body["system"], "Keep the answer concise.")
                        user_content = body["messages"][0]["content"]
                        self.assertEqual(user_content[0], {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "ZmFrZS1pbWFnZQ==",
                            },
                        })
                    else:
                        messages = body["messages"]
                        self.assertEqual(messages[0], {
                            "role": "system",
                            "content": "Keep the answer concise.",
                        })
                        user_content = messages[1]["content"]
                        self.assertEqual(user_content[0], {
                            "type": "image_url",
                            "image_url": {"url": self.IMAGE_URL},
                        })
                    self.assertEqual(user_content[-1], {
                        "type": "text",
                        "text": "describe this frame",
                    })

    def test_text_only_requests_remain_plain_text_for_every_provider(self):
        for provider in self.PROVIDERS:
            for streaming in (False, True):
                with self.subTest(provider=provider, streaming=streaming):
                    response = (
                        self._anthropic_response(streaming)
                        if provider == "anthropic"
                        else self._openai_response(streaming)
                    )
                    result, request = self._run_request(provider, streaming, None, response)
                    self.assertEqual(result, "ok")
                    body = request["json"]
                    user_content = body["messages"][0]["content"] if provider == "anthropic" else body["messages"][1]["content"]
                    self.assertEqual(user_content, "describe this frame")

    def test_rejected_images_keep_provider_detail_and_explain_vision_requirement(self):
        rejection = '{"error":{"message":"This text-only model does not support image input"}}'
        for provider in self.PROVIDERS:
            for streaming in (False, True):
                with self.subTest(provider=provider, streaming=streaming):
                    self._prepare_provider(provider)
                    error_response = _TransportResponse(
                        status_code=400,
                        text=rejection,
                    )

                    def post(_url, **_kwargs):
                        return error_response

                    with ExitStack() as stack:
                        for patcher in (
                            patch.object(llm_service, "_cancel_idle_timer"),
                            patch.object(llm_service, "_reset_idle_timer"),
                            patch.object(
                                llm_service,
                                "_prepare_thinking",
                                side_effect=lambda system, enabled, budget: (system, enabled, budget),
                            ),
                            patch.object(
                                llm_service,
                                "record_payload",
                                side_effect=lambda payload, **_kwargs: payload,
                            ),
                            patch.object(llm_service, "record_response"),
                            patch.object(llm_service, "record_metrics"),
                            patch.object(llm_service, "_image_to_data_url", return_value=self.IMAGE_URL),
                            patch.object(llm_service.requests, "post", side_effect=post),
                            patch(
                                "services.studio_enhancement.cancellable_lines",
                                side_effect=lambda resp, **kwargs: resp.iter_lines(**kwargs),
                            ),
                        ):
                            stack.enter_context(patcher)
                        generate = llm_service.generate_streaming if streaming else llm_service.generate
                        with self.assertRaisesRegex(RuntimeError, "vision-capable model") as raised:
                            generate(
                                "describe this frame",
                                image_paths=["frame.png"],
                                enable_thinking=False,
                            )
                    self.assertIn("text-only model does not support image input", str(raised.exception))
                    if provider == "anthropic" and streaming:
                        self.assertIn("vision-capable model", llm_service._stream_buffer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
