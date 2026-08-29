"""Model-free regressions for Studio's capability-aware Image workspace."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import tempfile
import unittest
import uuid

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = ROOT / "app" / "launch.py"
STORE_PATH = ROOT / "ui" / "src" / "stores" / "useStore.ts"
SELECTOR_PATH = (
    ROOT / "ui" / "src" / "components" / "Sidebar"
    / "ImageWorkflowSelector.tsx"
)
CONTROLS_PATH = (
    ROOT / "ui" / "src" / "components" / "Sidebar"
    / "ImageWorkflowControls.tsx"
)
TOOLS_PATH = (
    ROOT / "ui" / "src" / "components" / "Sidebar" / "ToolsPanel.tsx"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_launch_function(name: str, namespace: dict):
    tree = ast.parse(_read(LAUNCH_PATH), filename=str(LAUNCH_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LAUNCH_PATH), "exec"), namespace)
    return namespace[name]


class StudioImageWorkflowUiTests(unittest.TestCase):
    def test_compact_image_workflow_suite_is_exposed(self):
        selector = _read(SELECTOR_PATH)
        for value in ("new", "edit", "inpaint", "outpaint", "upscale"):
            self.assertIn(f"value: '{value}'", selector)
        self.assertIn("modelSupportsImageWorkflow", selector)

    def test_inpaint_and_outpaint_have_distinct_native_inputs(self):
        controls = _read(CONTROLS_PATH)
        self.assertIn("Edit Mask", controls)
        self.assertIn("Expand Canvas", controls)
        self.assertIn("White areas are regenerated", controls)
        self.assertIn("Use selected gallery image", controls)

    def test_image_upscale_has_still_safe_methods(self):
        tools = _read(TOOLS_PATH)
        self.assertIn("imageUpscaleMethods", tools)
        self.assertIn("FlashVSR 2x (AI detail)", tools)
        self.assertIn("Lanczos 4x (fast)", tools)
        self.assertIn("storedUpscaleMedia", tools)

    def test_image_inpaint_cannot_enter_video_multiclip_branch(self):
        store = _read(STORE_PATH)
        guarded_branch = (
            "state.generationMode === 'video'\n"
            "      && !isOmniReference\n"
            "      && state.params.image_mode === 2"
        )
        self.assertIn(guarded_branch, store)


class StudioImageOutpaintContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare = staticmethod(
            _load_launch_function(
                "_prepare_studio_image_outpaint_mask",
                {
                    "os": os,
                    "uuid": uuid,
                    "_get_active_workspace": lambda: "default",
                    "_workspace_dir": lambda _workspace: os.getcwd(),
                },
            )
        )

    def test_outpaint_builds_black_source_protection_mask(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                source = Path(temp_dir) / "source.png"
                Image.new("RGB", (37, 23), (180, 100, 40)).save(source)
                body = {
                    "_studio_image_workflow": "outpaint",
                    "image_mode": 2,
                    "image_guide": str(source),
                    "video_prompt_type": "",
                }

                self.prepare(body, {"inpaint_video_prompt_type": "VAG"})

                mask_path = Path(body["image_mask"])
                self.assertTrue(mask_path.is_file())
                with Image.open(mask_path) as mask:
                    self.assertEqual(mask.size, (37, 23))
                    self.assertEqual(mask.convert("RGB").getbbox(), None)
                self.assertIn("V", body["video_prompt_type"])
                self.assertIn("A", body["video_prompt_type"])
            finally:
                os.chdir(old_cwd)

    def test_non_outpaint_workflow_is_untouched(self):
        body = {
            "_studio_image_workflow": "inpaint",
            "image_mode": 2,
            "image_guide": "missing.png",
            "video_prompt_type": "VAG",
        }
        self.prepare(body, {"inpaint_video_prompt_type": "VAG"})
        self.assertNotIn("image_mask", body)


if __name__ == "__main__":
    unittest.main()
