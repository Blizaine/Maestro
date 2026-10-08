"""Model pre-download must use the same H3 encoder default as generation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def helpers(model_def, recommend="gguf_q2_k"):
    calls = []
    selectors = []
    hardware = {"ram_gb": 31.8, "gpu_vram_gb": 12}

    def filename(**kwargs):
        selectors.append(kwargs)
        urls = kwargs.get("URLs")
        return urls[0] if urls else "https://example.test/transformer.safetensors"

    def recursive(_model, prop, **_kwargs):
        return model_def.get(prop, [])

    def recommendation(hw, definition):
        calls.append(("recommend", hw, definition))
        if isinstance(recommend, Exception):
            raise recommend
        return recommend

    wgp = SimpleNamespace(
        transformer_quantization="int8", transformer_dtype_policy="bf16",
        get_model_def=lambda _: model_def,
        get_transformer_dtype=lambda *_: "bf16",
        get_model_filename=filename,
        get_model_recursive_prop=recursive,
        get_model_handler=lambda _: SimpleNamespace(recommend_text_encoder=recommendation),
        get_base_model_type=lambda _: "minimax_h3",
        download_models=lambda *args, **kwargs: calls.append(("download", args, kwargs)),
        get_compatible_local_model_filename=lambda filename, *_args, **_kwargs: filename,
        text_encoder_quantization="int8",
    )
    wanted = {"_download_model_files", "_recommended_minimax_h3_encoder"}
    tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"wgp": wgp, "_get_cached_hardware": lambda: hardware,
                 "check_download_cancelled": lambda: None, "_check_model_downloaded": lambda _: True}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_helpers", "exec"), namespace)
    return namespace["_download_model_files"], calls, selectors


class TestModelPredownload(unittest.TestCase):
    def h3(self):
        return {"minimax_h3_text_encoder_default": "nvfp4_awq",
                "text_encoder_URLs": ["https://example.test/native.safetensors"],
                "minimax_h3_text_encoder_variants": {
                    "nvfp4_awq": {"URLs": ["https://example.test/native.safetensors"]},
                    "gguf_q2_k": {"URLs": ["https://example.test/Q2_K.gguf"]}}}

    def test_low_memory_h3_downloads_recommended_encoder_only(self):
        run, calls, selectors = helpers(self.h3())
        run("minimax_h3")
        encoder = [call for call in calls if call[0] == "download" and call[1][2] == 2]
        self.assertEqual(encoder[0][1][0], "https://example.test/Q2_K.gguf")
        self.assertEqual(len(encoder), 1)
        self.assertEqual(selectors[-1]["URLs"], ["https://example.test/Q2_K.gguf"])
        self.assertTrue(any(call[0] == "recommend" for call in calls))

    def test_hardware_probe_failure_uses_existing_h3_fallback(self):
        run, calls, _ = helpers(self.h3(), RuntimeError("probe unavailable"))
        run("minimax_h3")
        encoder = [call for call in calls if call[0] == "download" and call[1][2] == 2]
        self.assertEqual(encoder[0][1][0], "https://example.test/native.safetensors")

    def test_other_family_keeps_forced_encoder_quantization(self):
        definition = {"text_encoder_URLs": ["https://example.test/encoder.safetensors"],
                      "text_encoder_quantization": "bf16", "text_encoder_folder": "special"}
        run, calls, selectors = helpers(definition)
        run("other_model")
        self.assertEqual(selectors[-1]["quantization"], "bf16")
        self.assertEqual(selectors[-1]["URLs"], definition["text_encoder_URLs"])
        self.assertFalse(any(call[0] == "recommend" for call in calls))
        encoder = [call for call in calls if call[0] == "download" and call[1][2] == 2]
        self.assertEqual(encoder[0][2]["force_path"], "special")


if __name__ == "__main__":
    unittest.main()