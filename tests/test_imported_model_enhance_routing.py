"""Imported checkpoints route prompt enhancement by architecture, not by ID."""

import ast
import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import enhance_guides, llm_service
from services.enhance_guides import get_enhance_guide, model_architecture

IMPORTED_FRAMES = "civitai_h3_2877206_3392061_3281858_frames"
IMPORTED_OMNI = "civitai_h3_2877206_3392061_3281858_references"
ARCHITECTURES = {
    IMPORTED_FRAMES: "minimax_h3",
    IMPORTED_OMNI: "minimax_h3_ref2va",
    "minimax_h3_fasth3_omniref": "minimax_h3_ref2va",
    "moviigen": "t2v",
}
CONTEXT_IR_GUIDE = "You are Maestro's context planner for MiniMax H3"
OMNI_GUIDE = "You are Maestro's prompt writer for MiniMax H3 Ref2VA"
SHARED_VIDEO_RULES = "CHARACTER REFERENCES — describe people by APPEARANCE"
INVENTORY = "\n".join((
    "<Picture 1>: visual identity/appearance reference for Alex; retention=reference",
    "<Picture 2>: visual identity/appearance reference for Sam; retention=reference",
    "<Audio 1>: background music; intent=AUDIO REFERENCE; retention=weak_reference",
))
OMNI_DRAFT = "\n".join((
    "subject_definitions: <Subject 1> is Alex from <Picture 1>. "
    "<Subject 2> is Sam from <Picture 2>. <Audio 1> supplies music style.",
    "summary: [reference generation + audio reference] Two dancers in a courtyard.",
    "retention_analysis: <Picture 1>: fully_preserved. "
    "<Picture 2>: fully_preserved. <Audio 1>: weak_reference.",
    "detailed_description: [Shot 1] Alex and Sam dance together in the courtyard.",
    "overall_soundscape: Footsteps on stone. No speech.",
    "non_diegetic_music: Use the rhythm and texture of <Audio 1>.",
))
CONTEXT_IR_DRAFT = "\n".join((
    "integrated_multimodal_description: [Shot 1] A lighthouse keeper climbs the "
    "spiral stairs at dusk, lantern swinging.",
    "overall_soundscape: Wind against the glass and footsteps on iron. No speech.",
    "non_diegetic_music: N/A",
))


def _loaded_wgp():
    return SimpleNamespace(get_model_def=lambda model_type: (
        {"architecture": ARCHITECTURES[model_type]} if model_type in ARCHITECTURES else None
    ))


def _launch_functions(*names):
    path = ROOT / "app/launch.py"
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")


class ModelArchitectureTests(unittest.TestCase):
    def test_loaded_definition_supplies_the_architecture(self):
        with patch.dict(sys.modules, {"wgp": _loaded_wgp()}):
            self.assertEqual(model_architecture(IMPORTED_OMNI), "minimax_h3_ref2va")
            self.assertEqual(model_architecture("minimax_h3_fasth3_omniref"), "minimax_h3_ref2va")
            self.assertEqual(model_architecture("unregistered_model"), "unregistered_model")
            self.assertEqual(model_architecture(""), "")

    def test_id_is_kept_without_a_usable_definition(self):
        with patch.dict(sys.modules):
            sys.modules.pop("wgp", None)
            self.assertEqual(model_architecture(IMPORTED_OMNI), IMPORTED_OMNI)
        failing = SimpleNamespace(get_model_def=Mock(side_effect=RuntimeError("registry unavailable")))
        with patch.dict(sys.modules, {"wgp": failing}):
            self.assertEqual(model_architecture(IMPORTED_FRAMES), IMPORTED_FRAMES)


class ImportedModelGuideTests(unittest.TestCase):
    def _guide(self, model_type, *, has_images=False):
        with patch.dict(sys.modules, {"wgp": _loaded_wgp()}):
            return get_enhance_guide(model_type, "video", has_images=has_images)

    def test_imported_h3_checkpoints_use_their_architecture_guides(self):
        self.assertTrue(self._guide(IMPORTED_FRAMES).startswith(CONTEXT_IR_GUIDE))
        self.assertTrue(self._guide(IMPORTED_OMNI, has_images=True).startswith(OMNI_GUIDE))

    def test_fasth3_omniref_uses_the_omni_guide(self):
        self.assertTrue(self._guide("minimax_h3_fasth3_omniref", has_images=True).startswith(OMNI_GUIDE))

    def test_wan_finetune_with_unprefixed_id_uses_the_wan_guide(self):
        self.assertTrue(self._guide("moviigen").startswith(
            "Rewrite the user's prompt for the Wan video generation model."))

    def test_stock_ids_without_a_definition_keep_prefix_routing(self):
        self.assertTrue(self._guide("minimax_h3_fused_turbo").startswith(CONTEXT_IR_GUIDE))

    def test_moss_editors_keep_the_edit_guide_under_the_shared_architecture(self):
        # MOSS Edit registers the moss_image architecture; its moss_edit flag,
        # not the prefix map, has to keep it off the layout guide.
        wgp = SimpleNamespace(get_model_def=lambda model_type: {
            "architecture": "moss_image", "moss_edit": model_type == "moss_image_edit_hq",
        })
        with patch.dict(sys.modules, {"wgp": wgp}):
            self.assertTrue(get_enhance_guide("moss_image_edit_hq", "image").startswith(
                "# MOSS Image Edit prompt enhancement"))
            self.assertTrue(get_enhance_guide("moss_image_sft", "image").startswith(
                "# MOSS Image prompt enhancement"))


class ImportedModelEnhancePromptTests(unittest.TestCase):
    def _system_prompt(self, draft, prompt, model_type, **kwargs):
        with patch.dict(sys.modules, {"wgp": _loaded_wgp()}), \
                patch.object(llm_service, "generate", return_value=draft) as generate:
            llm_service.enhance_prompt(prompt, mode="video", model_type=model_type,
                                       duration_seconds=10, **kwargs)
        return generate.call_args_list[0].kwargs["system_prompt"]

    def test_imported_omni_checkpoint_gets_the_six_field_contract(self):
        system = self._system_prompt(
            OMNI_DRAFT, "Alex and Sam dance together. Music only, no dialogue.",
            IMPORTED_OMNI, reference_context=INVENTORY,
        )
        self.assertIn(OMNI_GUIDE, system)
        self.assertIn("CRITICAL MINIMAX H3 REF2VA OUTPUT CONTRACT", system)
        self.assertNotIn(SHARED_VIDEO_RULES, system)

    def test_imported_frames_checkpoint_gets_the_context_ir_contract(self):
        system = self._system_prompt(
            CONTEXT_IR_DRAFT, "A lighthouse keeper climbs the stairs at dusk. No dialogue.",
            IMPORTED_FRAMES,
        )
        self.assertIn(CONTEXT_IR_GUIDE, system)
        self.assertIn("CRITICAL MINIMAX H3 OUTPUT CONTRACT", system)
        self.assertNotIn("REF2VA OUTPUT CONTRACT", system)
        self.assertNotIn(SHARED_VIDEO_RULES, system)


class ImportedModelEndpointTests(unittest.TestCase):
    def test_imported_h3_bypasses_the_legacy_cinematic_enhancer(self):
        namespace = {
            "wgp": SimpleNamespace(server_config={"enhancer_enabled": 1, "services": {}},
                                   get_model_def=lambda _: {}),
            "enhancement_settings": lambda settings: settings,
            "enhancement_warnings": lambda: [],
            "_PUBLIC_LLM_PROVIDERS": {"openai", "anthropic"},
        }
        exec(_launch_functions("_llm_enhance_prompt_payload"), namespace)
        legacy = AsyncMock(return_value={"enhanced": "Generic cinematic paragraph"})
        namespace.update(_ensure_llm_loaded=Mock(), _active_lora_hint=lambda *args: "",
                         _enhance_with_wangp=legacy)
        request = {"prompt": "A keeper climbs the stairs. No dialogue.", "mode": "video",
                   "model_type": IMPORTED_FRAMES, "duration_seconds": 5}

        with patch.dict(sys.modules, {"wgp": _loaded_wgp()}), \
                patch.object(llm_service, "enhance_prompt", return_value=CONTEXT_IR_DRAFT) as writer:
            result = asyncio.run(namespace["_llm_enhance_prompt_payload"](request))
        legacy.assert_not_called()
        self.assertEqual(result["enhanced"], CONTEXT_IR_DRAFT)
        # Guides, LoRA folders and per-model notes still resolve from the real ID.
        self.assertEqual(writer.call_args.kwargs["model_type"], IMPORTED_FRAMES)

    def test_deferred_enhancement_keeps_imported_h3_context_ir_as_one_prompt(self):
        namespace = {}
        exec(_launch_functions("_apply_deferred_prompt_enhancement"), namespace)
        apply_deferred = namespace["_apply_deferred_prompt_enhancement"]

        async def fake_enhance(payload):
            return {"original": payload["prompt"], "enhanced": CONTEXT_IR_DRAFT}

        fake_llm = SimpleNamespace(is_loaded=lambda: False, unload_model=Mock())
        services = ModuleType("services")
        services.llm_service = fake_llm
        namespace.update(asyncio=asyncio, _llm_enhance_prompt_payload=fake_enhance,
                         update_job=lambda job, **fields: None)
        params = {
            "model_type": IMPORTED_FRAMES, "prompt": "Raw idea", "multi_prompts_gen_type": 0,
            "_deferred_prompt_enhance": {"prompt": "Raw idea", "mode": "video",
                                         "model_type": IMPORTED_FRAMES, "window_count": 1},
        }
        job = {"id": "queued-imported", "params": dict(params)}
        raw_params = dict(params)

        with patch.dict(sys.modules, {"services": services, "services.enhance_guides": enhance_guides,
                                      "wgp": _loaded_wgp()}):
            apply_deferred(job, raw_params)
        self.assertEqual(raw_params["prompt"], CONTEXT_IR_DRAFT)
        self.assertEqual(raw_params["multi_prompts_gen_type"], 2)
        self.assertEqual(job["params"]["multi_prompts_gen_type"], 2)
        self.assertEqual(raw_params["_h3_original_prompt"], "Raw idea")


class ImportedModelDirectorTests(unittest.TestCase):
    def test_imported_h3_skips_the_creative_third_pass(self):
        from services.director.prompt_polish import should_polish_director_video_prompts

        with patch.dict(sys.modules, {"wgp": _loaded_wgp()}):
            self.assertFalse(should_polish_director_video_prompts(IMPORTED_FRAMES))
            self.assertFalse(should_polish_director_video_prompts(IMPORTED_OMNI))
            self.assertTrue(should_polish_director_video_prompts("moviigen"))

    def test_imported_h3_plans_with_the_h3_shot_breakdown(self):
        from services.director.planners.short_film import _route_video_pass2_guide

        with patch.dict(sys.modules, {"wgp": _loaded_wgp()}):
            guide = _route_video_pass2_guide(IMPORTED_FRAMES)
        self.assertIn("integrated_multimodal_description", guide)
        self.assertNotIn("for the LTX-2 video model", guide)

    def test_polish_guides_follow_the_architecture(self):
        from services.director.prompt_polish import get_video_guide

        with patch.dict(sys.modules, {"wgp": _loaded_wgp()}):
            self.assertTrue(get_video_guide(IMPORTED_OMNI, "full").startswith(OMNI_GUIDE))
            self.assertTrue(get_video_guide(IMPORTED_FRAMES, "light").startswith(
                "MINIMAX H3 CONTEXT-IR RULES"))
            self.assertTrue(get_video_guide("minimax_h3_fasth3_omniref", "light").startswith(
                "MINIMAX H3 REF2VA RULES"))

    def test_imported_omni_prompts_compile_as_ref2va_before_rendering(self):
        from services import director_pipeline

        plans = [{"video_prompt": OMNI_DRAFT}]
        with patch.dict(sys.modules, {"wgp": _loaded_wgp()}), \
                patch("services.director.h3_dialogue.compile_h3_clip_plans") as compile_plans, \
                patch("services.director.h3_dialogue.validate_final_h3_subject_bindings", return_value=[]):
            director_pipeline._preflight_h3_director_prompts(IMPORTED_OMNI, plans)
        self.assertEqual(compile_plans.call_args.kwargs["prompt_modes"], ["ref2va"])


if __name__ == "__main__":
    unittest.main()
