"""Enhancement overrides keep the queued/configured provider and credentials."""
import ast
import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class TestEnhancementProviderRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "app/launch.py"
        nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name in {"_ensure_llm_loaded", "_llm_enhance_prompt_payload", "_active_lora_hint"}]
        for node in nodes:
            node.decorator_list = []
        cls.code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")

    def test_speech_writer_controls_reach_native_writer_even_with_legacy_enhancer_enabled(self):
        from services import llm_service

        namespace = {
            "wgp": SimpleNamespace(server_config={"enhancer_enabled": 1, "services": {}},
                                   get_model_def=lambda _: {}),
            "enhancement_settings": lambda settings: settings,
            "enhancement_warnings": lambda: [],
            "_PUBLIC_LLM_PROVIDERS": {"openai", "anthropic"},
        }
        exec(self.code, namespace)
        legacy = AsyncMock(return_value={"enhanced": "Wrong cinematic path"})
        namespace.update(_ensure_llm_loaded=Mock(), _active_lora_hint=lambda *args: "",
                         _enhance_with_wangp=legacy)
        with patch.object(llm_service, "enhance_prompt", return_value="Speaker 1: Hello.") as writer:
            result = asyncio.run(namespace["_llm_enhance_prompt_payload"]({
                "prompt": "A lighthouse story.", "mode": "audio", "model_type": "fish_drama3_int4",
                "tts_enhance_mode": "monologue", "tts_voice_count": 1,
                "tts_writer_instructions": "A warm narrator", "tts_target_minutes": 12,
            }))
        legacy.assert_not_called()
        self.assertEqual(result["enhanced"], "Speaker 1: Hello.")
        self.assertEqual(writer.call_args.kwargs["tts_writer_instructions"], "A warm narrator")
        self.assertEqual(writer.call_args.kwargs["tts_target_minutes"], 12)
        self.assertEqual(writer.call_args.kwargs["max_new_tokens"], 2048)

    def test_speech_writer_receives_selected_duration_without_an_override(self):
        from services import llm_service

        namespace = {
            "wgp": SimpleNamespace(server_config={"enhancer_enabled": 0, "services": {}},
                                   get_model_def=lambda _: {}),
            "enhancement_settings": lambda settings: settings,
            "enhancement_warnings": lambda: [],
            "_PUBLIC_LLM_PROVIDERS": {"openai", "anthropic"},
        }
        exec(self.code, namespace)
        namespace.update(_ensure_llm_loaded=Mock(), _active_lora_hint=lambda *args: "")
        with patch.object(llm_service, "enhance_prompt", return_value="A lighthouse story.") as writer:
            asyncio.run(namespace["_llm_enhance_prompt_payload"]({
                "prompt": "Tell a story about a lighthouse.", "mode": "audio",
                "model_type": "fish_drama3_int4", "tts_enhance_mode": "monologue",
                "duration_seconds": 1200, "tts_target_minutes": 0, "max_new_tokens": 1600,
            }))
        self.assertEqual(writer.call_args.kwargs["duration_seconds"], 1200)
        self.assertEqual(writer.call_args.kwargs["tts_target_minutes"], 0)
        self.assertEqual(writer.call_args.kwargs["max_new_tokens"], 1600)

    def test_flux3_uses_native_guide_when_legacy_cinematic_enhancer_is_enabled(self):
        from services import llm_service

        namespace = {
            "wgp": SimpleNamespace(server_config={"enhancer_enabled": 1, "services": {}},
                                   get_model_def=lambda _: {}),
            "enhancement_settings": lambda settings: settings,
            "enhancement_warnings": lambda: [],
            "_PUBLIC_LLM_PROVIDERS": {"openai", "anthropic"},
        }
        exec(self.code, namespace)
        loader = Mock()
        legacy = AsyncMock(return_value={"enhanced": "Legacy draft"})
        namespace.update(_ensure_llm_loaded=loader, _active_lora_hint=lambda *args: "",
                         _enhance_with_wangp=legacy)
        with patch.object(llm_service, 'enhance_prompt', return_value='Native caption') as writer:
            result = asyncio.run(namespace['_llm_enhance_prompt_payload']({
                'prompt': 'A woodcut fox.', 'mode': 'image', 'model_type': 'flux3_klein',
                'image_paths': ['first.png', 'second.png'],
            }))
        self.assertEqual(result['enhanced'], 'Native caption')
        self.assertEqual(writer.call_args.kwargs['model_type'], 'flux3_klein')
        self.assertEqual(writer.call_args.kwargs['image_paths'], ['first.png', 'second.png'])
        loader.assert_called_once_with()
        legacy.assert_not_called()

        loader.reset_mock()
        result = asyncio.run(namespace['_llm_enhance_prompt_payload']({
            'prompt': 'A woodcut fox.', 'mode': 'image', 'model_type': 'flux2_klein_9b',
        }))
        self.assertEqual(result['enhanced'], 'Legacy draft')
        legacy.assert_awaited_once()
        loader.assert_not_called()

    def test_override_keeps_each_remote_provider_and_its_key(self):
        for provider in ("remote", "openai", "anthropic"):
            with self.subTest(provider=provider):
                config = {"llm_provider": provider, "llm_model_id": "director-model",
                          "llm_remote_url": "http://localhost:1234/v1", "llm_device": "cpu"}
                loader = Mock()
                service = SimpleNamespace(load_model=loader, provider_api_key=lambda p, s: f"key-for-{p}")
                package = ModuleType("services")
                package.llm_service = service
                namespace = {"wgp": SimpleNamespace(server_config={"services": config}),
                             "enhancement_settings": lambda s: dict(s), "_DEFAULT_LLM_REPO": "default",
                             "_llm_default_device": lambda: "cpu"}
                exec(self.code, namespace)
                with patch.dict(sys.modules, {"services": package}):
                    namespace["_ensure_llm_loaded"](model_id="writer:27b", device="cuda")
                loader.assert_called_once_with(model_id="writer:27b", device="cuda", provider=provider,
                                               remote_url=config["llm_remote_url"], api_key=f"key-for-{provider}",
                                               mtp_acceleration=False)

    def test_queue_snapshot_and_local_defaults_are_respected(self):
        loader = Mock()
        service = SimpleNamespace(load_model=loader, provider_api_key=lambda p, s: "")
        package = ModuleType("services")
        package.llm_service = service
        snapshot = {"llm_model_id": "queued-local-model", "llm_device": "cpu", "llm_provider": "local"}
        namespace = {"wgp": SimpleNamespace(server_config={"services": {"llm_model_id": "changed-model"}}),
                     "enhancement_settings": lambda s: snapshot, "_DEFAULT_LLM_REPO": "default",
                     "_llm_default_device": lambda: "cuda"}
        exec(self.code, namespace)
        with patch.dict(sys.modules, {"services": package}):
            namespace["_ensure_llm_loaded"]()
        loader.assert_called_once_with(model_id="queued-local-model", device="cpu", provider="local",
                                       remote_url="", api_key="", mtp_acceleration=False)

    def test_enhance_endpoint_keeps_remote_override_and_skips_local_dedicated_model(self):
        from services import llm_service

        class ReachedModelSelection(Exception):
            pass

        for provider in ("remote", "openai", "anthropic"):
            with self.subTest(provider=provider):
                config = {"llm_provider": provider, "llm_model_id": "director-model",
                          "enhance_llm_model_id": "remote-writer", "enhance_llm_device": "cpu",
                          "llm_remote_url": "http://localhost:1234/v1",
                          "llm_remote_api_key": "remote-key", "openai_api_key": "openai-key",
                          "anthropic_api_key": "anthropic-key"}
                lookup = Mock(return_value={"prompt_enhancer_model": "local/dedicated-writer"})
                namespace = {"wgp": SimpleNamespace(server_config={"services": config}, get_model_def=lookup),
                             "enhancement_settings": lambda settings: settings,
                             "_PUBLIC_LLM_PROVIDERS": {"openai", "anthropic"},
                             "_DEFAULT_LLM_REPO": "default", "_llm_default_device": lambda: "cuda"}
                exec(self.code, namespace)
                with patch.object(llm_service, "load_model", side_effect=ReachedModelSelection) as loader:
                    with self.assertRaises(ReachedModelSelection):
                        asyncio.run(namespace["_llm_enhance_prompt_payload"]({
                            "prompt": "Describe a rainy street", "mode": "image", "model_type": "image-model",
                        }))
                lookup.assert_not_called()
                loader.assert_called_once_with(model_id="remote-writer", device="cpu", provider=provider,
                                               remote_url=config["llm_remote_url"], api_key=f"{provider}-key",
                                               mtp_acceleration=False)


if __name__ == "__main__":
    unittest.main()
