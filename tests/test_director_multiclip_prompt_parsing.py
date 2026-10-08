"""Regression coverage for Director and legacy multi-clip prompt parsing."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "app" / "launch.py"


def load_parser():
    tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_split_multi_clip_prompt_text"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(LAUNCH), "exec"), namespace)
    return namespace[function.name]


class DirectorMultiClipPromptParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parse_prompts = staticmethod(load_parser())

    def test_explicit_single_clip_keeps_complete_multiline_context_ir(self):
        prompt = (
            "subject_definitions: <Subject 1> is the singer.\n\n"
            "summary: Stormy field.\n\n"
            "retention_analysis: preserve identity.\n\n"
            "detailed_description: The singer moves with the verse.\n\n"
            "overall_soundscape: Rain and wind.\n"
            "non_diegetic_music: use the driving audio."
        )
        prompts = self.parse_prompts(prompt, [243], [243])
        self.assertEqual(prompts, [prompt])
        self.assertEqual(len(prompts), len([243]))

    def test_explicit_single_clip_accepts_either_duration_array(self):
        prompt = "summary: keep these\nsections together."
        self.assertEqual(self.parse_prompts(prompt, [243]), [prompt])
        self.assertEqual(self.parse_prompts(prompt, None, [243]), [prompt])

    def test_director_clip_separator_still_splits_multiple_clips(self):
        prompt = "first multiline\nsection\n---CLIP_BOUNDARY---\nsecond multiline\nsection"
        self.assertEqual(
            self.parse_prompts(prompt, [124, 124], [122, 121]),
            ["first multiline\nsection", "second multiline\nsection"],
        )

    def test_metadata_free_studio_prompt_keeps_legacy_line_splitting(self):
        self.assertEqual(
            self.parse_prompts("first paragraph\n\nsecond paragraph"),
            ["first paragraph", "second paragraph"],
        )

    def test_conflicting_clip_metadata_does_not_claim_one_clip(self):
        self.assertEqual(
            self.parse_prompts("first\nsecond", [243], [122, 121]),
            ["first", "second"],
        )

    def test_generation_branch_uses_the_tested_parser(self):
        tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
        generation = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_generation"
        )
        branch = next(
            node for node in ast.walk(generation)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "raw_params.get('multi_prompts_gen_type') == 3"
        )
        self.assertTrue(any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_split_multi_clip_prompt_text"
            for node in ast.walk(branch)
        ))


if __name__ == "__main__":
    unittest.main()