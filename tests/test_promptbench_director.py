"""Director bench preserves the final generation handoff and cache isolation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from promptbench.checks import evaluate
from promptbench.director import prepare_director
from services.director.planners.short_film import ShortFilmPlanner
from services import director_pipeline


class DirectorBenchTests(unittest.TestCase):
    def test_final_preflight_prompts_and_effective_timing_are_reported_without_a_project(self):
        params = {"prompt": "A silent walk.", "model_type": "minimax_h3_fused_turbo", "video_length": 345}
        original = deepcopy(params)
        clips = [{"video_prompt": "raw planning draft"}]
        timeline = [{"duration_sec": 14.375}]
        compiled = "integrated_multimodal_description: Walking.\noverall_soundscape: Steps.\nnon_diegetic_music: None."
        def preflight(model, plans, **kwargs):
            self.assertEqual(kwargs["durations"], [14.375])
            plans[0]["video_prompt"] = compiled
        def plan(pid, request, skill):
            self.assertNotIn(pid, director_pipeline._pipelines)
            self.assertEqual(request["llm_model_id"], "local-test-writer")
            self.assertEqual(director_pipeline._director_effective_shot_image_policy(request), "prompt_only")
            return clips, timeline
        with patch.object(director_pipeline, "_run_planning", side_effect=plan), \
             patch.object(director_pipeline, "prepare_director_timeline", return_value=(clips, timeline)), \
             patch.object(director_pipeline, "_preflight_h3_director_prompts", side_effect=preflight):
            output = prepare_director(params, {"fps": 24}, {"llm_model_id": "local-test-writer"})
        self.assertEqual(params, original)
        checks = evaluate({"params": params, "checks": {"expected_windows": 1}},
                          {"status": "complete", "prepared": output})
        self.assertEqual(checks["native_prompts"], [compiled])
        self.assertEqual(next(check for check in checks["checks"] if check["check"] == "draft_review")["status"], "review")

    def test_writer_contract_and_optional_content_setting_invalidate_resume_cache(self):
        planner = ShortFilmPlanner()
        fingerprints = []
        for kwargs in ({"polish_block": "old"}, {"polish_block": "new"}, {"polish_block": "new", "nsfw": True}):
            planner._configure_planning_runtime(kwargs, kind="story", fingerprint_payload={"brief": "Same brief"})
            fingerprints.append(planner._planning_checkpoint_fingerprint)
        self.assertEqual(len(set(fingerprints)), 3)


if __name__ == "__main__":
    unittest.main()
