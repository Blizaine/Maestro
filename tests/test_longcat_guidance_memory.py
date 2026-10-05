"""CPU-only execution of LongCat Avatar's production guidance branch."""
from __future__ import annotations

import ast
from pathlib import Path
import textwrap
import unittest
from types import SimpleNamespace


_MAIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "models"
    / "longcat"
    / "longcat_main.py"
)
_HELPER_NAMES = {
    "_should_use_joint_pass",
    "_build_avatar_guidance_branches",
    "_combine_avatar_guidance",
}


def _load_generate_guidance_branch():
    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"), filename=str(_MAIN_PATH))
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LongCatModel")
    generate = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    route = next(
        node
        for node in generate.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "use_joint_pass" for target in node.targets)
    )
    abort_check = next(
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.FunctionDef) and node.name == "_aborted"
    )
    avatar_branch = next(
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.is_avatar and audio_emb is not None and any_guidance"
    )
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _HELPER_NAMES
    ]
    helper_source = "\n\n".join(ast.unparse(node) for node in helpers)
    branch_source = "\n".join(ast.unparse(node) for node in (route, abort_check, avatar_branch))
    arguments = ", ".join(
        (
            "self", "joint_pass", "any_guidance", "latents", "audio_emb", "prompt_embeds",
            "neg_embeds", "prompt_mask", "neg_mask", "ref_target_masks", "timestep",
            "num_cond_latents", "ref_kwargs", "guide_scale", "audio_cfg_scale", "torch",
        )
    )
    wrapper = f"def run_generate_guidance_branch({arguments}):\n"
    wrapper += textwrap.indent(branch_source + "\nreturn noise_pred", "    ")
    namespace = {}
    exec(compile(helper_source + "\n\n" + wrapper, str(_MAIN_PATH), "exec"), namespace)
    return namespace["run_generate_guidance_branch"]


class _FakeAudio:
    def to(self, device, dtype):
        return self


class _FakeTransformer:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.outputs)


class LongCatGuidanceMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_guidance_branch = _load_generate_guidance_branch()

    def _run_avatar_guidance(self, outputs):
        transformer = _FakeTransformer(outputs)
        audio_cond = _FakeAudio()
        audio_uncond = object()
        values = {name: object() for name in (
            "latents", "prompt", "negative", "prompt_mask", "negative_mask", "refs", "timestep"
        )}
        result = type(self).run_guidance_branch(
            self=SimpleNamespace(is_avatar=True, device="cpu", dtype="cpu", transformer=transformer),
            joint_pass=True,
            any_guidance=True,
            latents=values["latents"],
            audio_emb=audio_cond,
            prompt_embeds=values["prompt"],
            neg_embeds=values["negative"],
            prompt_mask=values["prompt_mask"],
            neg_mask=values["negative_mask"],
            ref_target_masks=values["refs"],
            timestep=values["timestep"],
            num_cond_latents=0,
            ref_kwargs={},
            guide_scale=2.5,
            audio_cfg_scale=1.5,
            torch=SimpleNamespace(zeros_like=lambda _: audio_uncond),
        )
        return result, transformer.calls, values, audio_cond, audio_uncond

    def test_guided_avatar_joint_request_runs_three_ordered_scalar_calls_and_blends(self):
        result, calls, values, audio_cond, audio_uncond = self._run_avatar_guidance([7.0, 4.0, 1.0])

        self.assertEqual(result, 13.0)
        self.assertEqual(len(calls), 3)
        expected = (
            (values["prompt"], values["prompt_mask"], audio_cond),
            (values["negative"], values["negative_mask"], audio_cond),
            (values["negative"], values["negative_mask"], audio_uncond),
        )
        for call, (context, mask, audio) in zip(calls, expected):
            self.assertIs(call["hidden_states"], values["latents"])
            self.assertIs(call["encoder_hidden_states"], context)
            self.assertIs(call["encoder_attention_mask"], mask)
            self.assertIs(call["audio_embs"], audio)
            self.assertIs(call["ref_target_masks"], values["refs"])

    def test_avatar_branch_propagates_transformer_abort(self):
        result, calls, *_ = self._run_avatar_guidance([7.0, None, 1.0])

        self.assertIsNone(result)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
