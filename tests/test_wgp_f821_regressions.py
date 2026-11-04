"""CPU-only regressions for repaired WGP name-resolution paths."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


_WGP_PATH = Path(__file__).resolve().parents[1] / "app" / "wgp.py"
_WGP_TREE = ast.parse(_WGP_PATH.read_text(encoding="utf-8"), filename=str(_WGP_PATH))


def _function(name):
    return next(
        node for node in _WGP_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _names(node, name, context_type=None):
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Name)
        and item.id == name
        and (context_type is None or isinstance(item.ctx, context_type))
    ]


class WgpF821RegressionTests(unittest.TestCase):
    def test_aligned_pose_validation_uses_the_standard_info_return_contract(self):
        validate = _function("validate_settings")
        self.assertFalse(_names(validate, "err", ast.Load))
        cases = (
            (
                "Aligned Pose transfer requires a Start Image or Source Video to continue to be used",
                {
                    "image_start": None,
                    "video_source": None,
                    "video_prompt_type": "",
                    "all_letters": lambda _value, _letters: False,
                },
            ),
            (
                "Aligned Pose transfer supports only Inpainting process outside the masked area",
                {
                    "video_prompt_type": "AY",
                    "any_letters": lambda _value, _letters: True,
                },
            ),
        )
        for message, locals_ in cases:
            with self.subTest(message=message):
                branch = next(
                    node for node in ast.walk(validate)
                    if isinstance(node, ast.If)
                    and any(
                        isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Attribute)
                        and isinstance(statement.value.func.value, ast.Name)
                        and statement.value.func.value.id == "gr"
                        and statement.value.func.attr == "Info"
                        and statement.value.args
                        and isinstance(statement.value.args[0], ast.Constant)
                        and statement.value.args[0].value == message
                        for statement in node.body
                    )
                )
                function = ast.FunctionDef(
                    name="_run_validation_branch",
                    args=ast.arguments(
                        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
                    ),
                    body=[copy.deepcopy(branch)],
                    decorator_list=[],
                )
                namespace = {
                    **locals_,
                    "gr": SimpleNamespace(Info=lambda text: notices.append(text)),
                    "ret": lambda: (None, None, None, None),
                }
                notices = []
                module = ast.Module(body=[function], type_ignores=[])
                exec(compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"), namespace)
                self.assertEqual(
                    namespace["_run_validation_branch"](),
                    (None, None, None, None),
                )
                self.assertEqual(notices, [message])

    def test_transformer_dtype_policy_returns_without_dead_undefined_name(self):
        function = _function("get_transformer_dtype")
        torch = SimpleNamespace(float16=object(), bfloat16=object())
        namespace = {
            "torch": torch,
            "get_base_model_type": lambda model_type: model_type,
            "get_model_def": lambda _model_type: {},
            "get_model_family": lambda _model_type: "wan",
            "bfloat16_supported": True,
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"), namespace)
        resolve = namespace["get_transformer_dtype"]
        self.assertIs(resolve("test", ""), torch.bfloat16)
        self.assertIs(resolve("test", "fp16"), torch.float16)
        self.assertIs(resolve("test", "bf16"), torch.bfloat16)

    def test_h3_outpaint_fit_guard_runs_in_preprocessing_not_face_extraction(self):
        preprocess = _function("preprocess_video_with_mask")
        extract_faces = _function("extract_faces_from_video_with_mask")
        self.assertFalse(_names(extract_faces, "outpainting_dims", ast.Load))
        self.assertFalse(_names(extract_faces, "outpainting_quantize_margins", ast.Load))

        guard = next(
            node for node in ast.walk(preprocess)
            if isinstance(node, ast.If)
            and {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
            >= {"outpainting_dims", "outpainting_quantize_margins"}
            and any(
                isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "fit_canvas" for target in item.targets)
                for item in ast.walk(node)
            )
        )
        fit_crop_guard = next(
            node for node in ast.walk(preprocess)
            if isinstance(node, ast.If)
            and "fit_crop" in {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
            and "outpainting_dims" in {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
        )
        self.assertLess(guard.lineno, fit_crop_guard.lineno)

        for dims, quantize, initial_fit, expected_fit in (
            ((256, 512, 0, 0), 8, 1, None),
            ((256, 512, 0, 0), 0, 1, 1),
            (None, 8, 1, 1),
        ):
            namespace = {
                "outpainting_dims": dims,
                "outpainting_quantize_margins": quantize,
                "fit_canvas": initial_fit,
            }
            module = ast.Module(body=[copy.deepcopy(guard)], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"), namespace)
            self.assertEqual(namespace["fit_canvas"], expected_fit)

    def test_audio_overlap_is_initialized_before_multitalk_reads_it(self):
        generate = _function("generate_video")
        reuse_reads = _names(generate, "reuse_frames", ast.Load)
        first_read_line = min(item.lineno for item in reuse_reads)
        setup = next(
            node for node in ast.walk(generate)
            if isinstance(node, ast.If)
            and node.lineno < first_read_line
            and any(
                isinstance(item, ast.Name)
                and item.id == "reuse_frames"
                and isinstance(item.ctx, ast.Store)
                for item in ast.walk(node)
            )
        )
        self.assertLess(setup.lineno, first_read_line)

        for supports_sliding, expected in ((True, 17), (False, 0)):
            namespace = {
                "test_any_sliding_window": lambda _model, enabled=supports_sliding: enabled,
                "model_type": "fixture",
                "sliding_window_size": 81,
                "sliding_window_overlap": 17,
                "latent_size": 5,
            }
            module = ast.Module(body=[copy.deepcopy(setup)], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"), namespace)
            self.assertEqual(namespace["reuse_frames"], expected)

    def test_temporal_upscale_reuses_state_between_windows_and_resets_per_repeat(self):
        generate = _function("generate_video")
        call = next(
            node for node in ast.walk(generate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "perform_temporal_upsampling"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.IfExp)
        )
        state_init = next(
            node for node in ast.walk(generate)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "previous_last_frame"
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        )
        self.assertLess(state_init.lineno, call.lineno)
        containing_loops = [
            node for node in ast.walk(generate)
            if isinstance(node, ast.While)
            and node.lineno < call.lineno <= node.end_lineno
        ]
        outer_loop = min(containing_loops, key=lambda node: node.lineno)
        window_loop = max(containing_loops, key=lambda node: node.lineno)
        self.assertLess(outer_loop.lineno, state_init.lineno)
        self.assertLess(state_init.lineno, window_loop.lineno)

        continuation_arg = ast.Expression(body=copy.deepcopy(call.args[1]))
        first_window = eval(
            compile(ast.fix_missing_locations(continuation_arg), str(_WGP_PATH), "eval"),
            {"previous_last_frame": None, "sliding_window": True, "window_no": 1},
        )
        carried = object()
        later_window = eval(
            compile(ast.fix_missing_locations(continuation_arg), str(_WGP_PATH), "eval"),
            {"previous_last_frame": carried, "sliding_window": True, "window_no": 2},
        )
        non_sliding = eval(
            compile(ast.fix_missing_locations(continuation_arg), str(_WGP_PATH), "eval"),
            {"previous_last_frame": carried, "sliding_window": False, "window_no": 2},
        )
        self.assertIsNone(first_window)
        self.assertIs(later_window, carried)
        self.assertIsNone(non_sliding)


if __name__ == "__main__":
    unittest.main()
