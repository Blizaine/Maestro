"""Model-free regressions for the August 2026 GitHub quick-win batch."""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import os
import tempfile
import unittest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LAUNCH_PATH = os.path.join(_ROOT, "app", "launch.py")
_LLM_SERVICE_PATH = os.path.join(_ROOT, "app", "services", "llm_service.py")
_WIN_SAFE_FILES_PATH = os.path.join(_ROOT, "app", "services", "win_safe_files.py")
_ADVANCED_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "AdvancedSettings.tsx",
)
_PROMPT_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "PromptInput.tsx",
)
_SERVICES_UI_PATH = os.path.join(
    _ROOT,
    "ui",
    "src",
    "components",
    "SettingsDrawer",
    "ServicesSettingsPanel.tsx",
)
_SERVICES_TYPES_PATH = os.path.join(_ROOT, "ui", "src", "types", "index.ts")
_INDEX_HTML_PATH = os.path.join(_ROOT, "ui", "index.html")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_functions(path: str, names: tuple[str, ...]) -> dict:
    tree = ast.parse(_read(path), filename=os.path.relpath(path, _ROOT))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    if len(selected) != len(names):
        found = {node.name for node in selected}
        raise AssertionError(f"Missing functions: {set(names) - found}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    loaded: dict = {}
    exec(compile(module, os.path.relpath(path, _ROOT), "exec"), loaded)
    return loaded


class TestRemoteOpenAICompatibleCredentials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pick_key = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_llm_api_key_for_provider",),
            )["_llm_api_key_for_provider"]
        )

    def test_each_provider_uses_only_its_own_credential(self):
        services = {
            "llm_remote_api_key": "lan-secret",
            "openai_api_key": "openai-secret",
            "anthropic_api_key": "anthropic-secret",
        }
        self.assertEqual(self.pick_key(services, "remote"), "lan-secret")
        self.assertEqual(self.pick_key(services, "openai"), "openai-secret")
        self.assertEqual(self.pick_key(services, "anthropic"), "anthropic-secret")
        self.assertEqual(self.pick_key(services, "local"), "")

    def test_remote_key_is_exposed_masked_and_editable(self):
        launch = _read(_LAUNCH_PATH)
        panel = _read(_SERVICES_UI_PATH)
        types = _read(_SERVICES_TYPES_PATH)
        self.assertIn(
            '"llm_remote_api_key": _mask_key(services.get("llm_remote_api_key", ""))',
            launch,
        )
        self.assertIn('"llm_remote_api_key"', launch)
        self.assertIn('label="Server API Key"', panel)
        self.assertIn("llm_remote_api_key_set: boolean", types)

    def test_remote_connection_reloads_when_url_or_key_changes(self):
        source = _read(_LLM_SERVICE_PATH)
        self.assertIn("_remote_url == remote_url", source)
        self.assertIn("_api_key == api_key", source)


class TestCivitAIVariantUpdates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.select = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_select_latest_compatible_civitai_version",),
            )["_select_latest_compatible_civitai_version"]
        )

    def test_turbo_install_tracks_newest_turbo_not_newer_base_release(self):
        versions = [
            {"id": 400, "baseModel": "Z-Image Base", "baseModelType": "Standard"},
            {"id": 350, "baseModel": "Z-Image Turbo", "baseModelType": "Standard"},
            {"id": 300, "baseModel": "Z-Image Turbo", "baseModelType": "Standard"},
        ]
        self.assertEqual(self.select(versions, 300)["id"], 350)

    def test_base_install_stays_on_base_branch(self):
        versions = [
            {"id": 400, "baseModel": "Z-Image Turbo", "baseModelType": "Standard"},
            {"id": 390, "baseModel": "Z-Image Base", "baseModelType": "Standard"},
            {"id": 200, "baseModel": "Z-Image Base", "baseModelType": "Standard"},
        ]
        self.assertEqual(self.select(versions, 200)["id"], 390)

    def test_unknown_local_version_retains_api_newest_fallback(self):
        versions = [{"id": 2}, {"id": 1}]
        self.assertEqual(self.select(versions, 999)["id"], 2)

    def test_old_false_positive_manifest_is_invalidated(self):
        self.assertIn("LORA_MANIFEST_VERSION = 2", _read(_LAUNCH_PATH))


class TestStudioInterfaceQuickWins(unittest.TestCase):
    def test_lora_selector_is_near_top_of_advanced_panel(self):
        source = _read(_ADVANCED_PATH)
        self.assertEqual(source.count("<LoraSelector />"), 1)
        self.assertLess(
            source.index("<LoraSelector />"),
            source.index("H3 Text Encoder"),
        )
        self.assertLess(
            source.index("<LoraSelector />"),
            source.index("<PresetManager />"),
        )

    def test_browser_spellcheck_is_enabled_for_text_inputs(self):
        self.assertIn('<body spellcheck="true">', _read(_INDEX_HTML_PATH))

    def test_main_prompt_grows_automatically_with_its_content(self):
        source = _read(_PROMPT_PATH)
        self.assertIn(
            "const promptTextareaRef = useAutoGrowingTextarea(prompt)",
            source,
        )
        self.assertIn("ref={promptTextareaRef}", source)
        self.assertIn("resize-none overflow-hidden", source)
        self.assertNotIn("resize: 'vertical'", source)
        self.assertNotIn("maxHeight: '70vh'", source)


class TestDisconnectedMediaStreams(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "maestro_win_safe_files_quick_win_test",
            _WIN_SAFE_FILES_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load win_safe_files.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    async def test_client_disconnect_cancels_file_stream(self):
        payload = b"x" * (1024 * 1024)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(payload)
            path = handle.name
        try:
            response = self.module.ShareDeleteFileResponse(path)
            response.chunk_size = 1024
            first_chunk = asyncio.Event()
            sent: list[dict] = []

            async def send(message: dict) -> None:
                sent.append(message)
                if message.get("type") == "http.response.body" and message.get("body"):
                    first_chunk.set()

            async def receive() -> dict:
                await first_chunk.wait()
                return {"type": "http.disconnect"}

            await asyncio.wait_for(
                response({"type": "http", "headers": []}, receive, send),
                timeout=2,
            )
            bytes_sent = sum(
                len(message.get("body", b""))
                for message in sent
                if message.get("type") == "http.response.body"
            )
            self.assertGreater(bytes_sent, 0)
            self.assertLess(bytes_sent, len(payload))
        finally:
            os.remove(path)

    async def test_completed_stream_cancels_disconnect_listener(self):
        payload = b"complete response"
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(payload)
            path = handle.name
        try:
            response = self.module.ShareDeleteFileResponse(path)
            sent: list[dict] = []
            never = asyncio.Event()

            async def send(message: dict) -> None:
                sent.append(message)

            async def receive() -> dict:
                await never.wait()
                return {"type": "http.disconnect"}

            await asyncio.wait_for(
                response({"type": "http", "headers": []}, receive, send),
                timeout=2,
            )
            body = b"".join(
                message.get("body", b"")
                for message in sent
                if message.get("type") == "http.response.body"
            )
            self.assertEqual(body, payload)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
