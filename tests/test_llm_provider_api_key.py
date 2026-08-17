"""Contract tests for LLM provider credential resolution.

Every caller that loads an LLM used to inline its own provider -> key
mapping, and all four copies omitted "remote". Hosted OpenAI-compatible
endpoints were therefore called with no Authorization header and answered
401 Unauthorized. These tests pin the shared mapping so a new provider
cannot be half-wired again.
"""
from __future__ import annotations

import os
import sys
import unittest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)


class TestProviderApiKey(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from services import llm_service
        except ImportError as exc:
            raise unittest.SkipTest("llm_service dependencies unavailable") from exc
        cls.llm_service = llm_service
        cls.services = {
            "llm_remote_api_key": "sk-remote",
            "openai_api_key": "sk-openai",
            "anthropic_api_key": "sk-anthropic",
        }

    def test_remote_provider_resolves_its_own_key(self):
        """The regression this suite exists for."""
        self.assertEqual(
            self.llm_service.provider_api_key("remote", self.services),
            "sk-remote",
        )

    def test_each_provider_gets_its_own_credential(self):
        for provider, expected in (
            ("remote", "sk-remote"),
            ("openai", "sk-openai"),
            ("anthropic", "sk-anthropic"),
        ):
            with self.subTest(provider=provider):
                self.assertEqual(
                    self.llm_service.provider_api_key(provider, self.services),
                    expected,
                )

    def test_local_provider_sends_no_credential(self):
        """llama-server needs no key and must never receive one."""
        self.assertEqual(self.llm_service.provider_api_key("local", self.services), "")

    def test_unknown_provider_is_empty_not_an_error(self):
        for provider in ("", "bogus", None):
            with self.subTest(provider=provider):
                self.assertEqual(
                    self.llm_service.provider_api_key(provider, self.services), ""
                )

    def test_unset_key_is_empty(self):
        """An unconfigured key yields no header rather than KeyError."""
        for provider in self.llm_service.PROVIDER_API_KEY_SETTING:
            with self.subTest(provider=provider):
                self.assertEqual(self.llm_service.provider_api_key(provider, {}), "")

    def test_every_mapped_setting_is_persistable(self):
        """Each mapped setting must survive a settings write.

        A key the settings endpoint refuses to store would resolve to ""
        forever, which is exactly how this bug presented to users.
        """
        launch_source = os.path.join(_APP, "launch.py")
        with open(launch_source, encoding="utf-8") as handle:
            source = handle.read()

        # launch.py declares several ALLOWED_KEYS allowlists (performance,
        # services, ...). A credential may live in any of them, so union them.
        allowed = ""
        marker = "ALLOWED_KEYS = {"
        cursor = source.find(marker)
        self.assertNotEqual(cursor, -1, "no ALLOWED_KEYS allowlist found in launch.py")
        while cursor != -1:
            allowed += source[cursor:source.index("}", cursor)]
            cursor = source.find(marker, cursor + 1)

        for setting in self.llm_service.PROVIDER_API_KEY_SETTING.values():
            with self.subTest(setting=setting):
                self.assertIn(f'"{setting}"', allowed)


class TestApiHeaders(unittest.TestCase):
    """The headers built from a resolved key."""

    @classmethod
    def setUpClass(cls):
        try:
            from services import llm_service
        except ImportError as exc:
            raise unittest.SkipTest("llm_service dependencies unavailable") from exc
        cls.llm_service = llm_service

    def setUp(self):
        self.saved = (
            self.llm_service._provider,
            self.llm_service._api_key,
        )

    def tearDown(self):
        self.llm_service._provider, self.llm_service._api_key = self.saved

    def _headers_for(self, provider, key):
        self.llm_service._provider = provider
        self.llm_service._api_key = key
        return self.llm_service._api_headers()

    def test_remote_sends_bearer_token(self):
        headers = self._headers_for("remote", "sk-remote")
        self.assertEqual(headers.get("Authorization"), "Bearer sk-remote")

    def test_anthropic_uses_its_own_header_scheme(self):
        headers = self._headers_for("anthropic", "sk-anthropic")
        self.assertEqual(headers.get("x-api-key"), "sk-anthropic")
        self.assertNotIn("Authorization", headers)

    def test_local_sends_no_auth(self):
        self.assertNotIn("Authorization", self._headers_for("local", ""))

    def test_remote_without_key_sends_no_auth(self):
        """LM Studio / Ollama on the LAN need no credential."""
        self.assertNotIn("Authorization", self._headers_for("remote", ""))


if __name__ == "__main__":
    unittest.main()
