"""Enhancement overrides keep the queued/configured provider and credentials."""
import ast
import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class TestEnhancementProviderRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "app/launch.py"
        nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name in {"_ensure_llm_loaded", "_llm_enhance_prompt_payload"}]
        for node in nodes:
            node.decorator_list = []
        cls.code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")

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
                                               remote_url=config["llm_remote_url"], api_key=f"key-for-{provider}")

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
                                       remote_url="", api_key="")

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
                                               remote_url=config["llm_remote_url"], api_key=f"{provider}-key")


if __name__ == "__main__":
    unittest.main()
