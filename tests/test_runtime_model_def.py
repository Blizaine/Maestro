"""Exercise engine asset resolution and canvas sizing without starting Maestro."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image


ENGINE = Path(__file__).resolve().parents[1] / "app" / "wgp.py"
TREE = ast.parse(ENGINE.read_text(encoding="utf-8"))


def function(name):
    return next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, namespace):
    with redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ENGINE), "exec"), namespace)
    return namespace


class RuntimeModelDefinitionTests(unittest.TestCase):
    def test_runtime_resolution_does_not_mutate_registered_assets(self):
        registered = {"vae": "fp16"}

        def resolve(definition, context):
            definition["vae"] = context["transformer_quantization"]
            return definition

        namespace = {
            "get_model_def": lambda _: registered,
            "get_base_model_type": lambda _: "test",
            "model_types_handlers": {"test": SimpleNamespace(resolve_runtime_model_def=resolve)},
            "transformer_quantization": "int8",
        }
        execute([function("get_runtime_model_def")], namespace)
        self.assertEqual(namespace["get_runtime_model_def"]("test"), {"vae": "int8"})
        self.assertEqual(registered, {"vae": "fp16"})
        for name in ("download_models", "load_models"):
            self.assertTrue(any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "get_runtime_model_def"
                for node in ast.walk(function(name))
            ), name)

    def resolve_canvas(self, runtime, control, reference, active=True):
        definition = {
            "block_size": 32,
            "auto_resolution_budgets": {"auto_2k": 2048 * 2048},
            "auto_resolution_fallbacks": {"auto_2k": "2048x2048"},
        }
        common = {"resolution": "auto_2k", "model_def": definition, "os": os}
        if runtime:
            block = next(node for node in function("generate_video").body
                         if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                         and node.test.id == "_auto_aspect")
            common.update(_auto_aspect=True, is_image=True, block_size=32,
                          image_guide=None, video_guide=control, image_refs=[reference] if reference else [],
                          image_start=None, minimax_h3_references=[], video_prompt_type="VI" if active else "I")
        else:
            block = next(node for node in function("validate_settings").body
                         if isinstance(node, ast.If) and 'resolution.startswith' in ast.unparse(node.test))
            common.update(image_outputs=True, inputs={
                "image_guide": control, "image_refs": [reference] if reference else [],
                "video_prompt_type": "VI" if active else "I",
            })
        return execute([block], common)["resolution"]

    def test_auto_2k_uses_active_canvas_before_identity_reference_in_both_paths(self):
        canvas = Image.new("RGB", (96, 32))
        reference = Image.new("RGB", (32, 96))
        for runtime in (False, True):
            with self.subTest(runtime=runtime):
                width, height = map(int, self.resolve_canvas(runtime, canvas, reference).split("x"))
                self.assertGreater(width, height)
                self.assertAlmostEqual(width * height / (2048 * 2048), 1, delta=0.05)
                width, height = map(int, self.resolve_canvas(runtime, canvas, reference, active=False).split("x"))
                self.assertLess(width, height)

    def test_auto_2k_without_source_uses_square_fallback_in_both_paths(self):
        for runtime in (False, True):
            with self.subTest(runtime=runtime):
                self.assertEqual(self.resolve_canvas(runtime, None, None), "2048x2048")


if __name__ == "__main__":
    unittest.main()
