"""Static regression coverage for gallery Save/Load Settings round trips.

The browser store is TypeScript and is validated by the UI build.  These
assertions protect the durable contracts between Python sidecars and the
workflow-specific UI state that historically regressed when an engine-facing
payload normalized fields such as ``image_mode``.
"""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORE = (ROOT / "ui/src/stores/useStore.ts").read_text(encoding="utf-8")
SELECTOR = (
    ROOT / "ui/src/components/Sidebar/VideoWorkflowSelector.tsx"
).read_text(encoding="utf-8")
INPUTS = (
    ROOT / "ui/src/components/Sidebar/InputsPanel.tsx"
).read_text(encoding="utf-8")
MIXER = (
    ROOT / "ui/src/components/Sidebar/MixerControls.tsx"
).read_text(encoding="utf-8")
LAUNCH = (ROOT / "app/launch.py").read_text(encoding="utf-8")


class SettingsRoundTripTests(unittest.TestCase):
    def test_extend_and_blend_restore_workflow_and_source_media(self):
        self.assertIn("p.video_source ? 'extend'", STORE)
        self.assertIn("restoredVideoWorkflow === 'extend'", STORE)
        self.assertIn("newParams.image_mode = 3", STORE)
        self.assertIn("newParams.image_mode = 4", STORE)
        self.assertIn("continueVideoPath: restoredContinuePath", STORE)
        self.assertIn("continueVideoUrl: restoredContinueName", STORE)
        self.assertIn("_probeRestoredVideo(", STORE)
        self.assertIn("get().setContinueVideo", STORE)
        self.assertIn("blendClipAPath: restoredBlendAPath", STORE)
        self.assertIn("blendClipBPath: restoredBlendBPath", STORE)
        self.assertIn("rememberedWorkflow === 'extend'", SELECTOR)
        self.assertIn("rememberedWorkflow === 'blend'", SELECTOR)

    def test_specialized_video_sidecars_keep_ui_recipe(self):
        for marker in (
            '"_studio_video_workflow": "blend"',
            '"_blend_transition_sec"',
            '"_blend_motion_prefix_sec"',
            '"_blend_motion_suffix_sec"',
            '"edit_prompt_strength"',
            '"edit_sam_target"',
            '"edit_invert_mask"',
            '"outpaint_canvas_w"',
            '"outpaint_lock_source_pixels"',
            '"outpaint_trim_smear"',
        ):
            self.assertIn(marker, LAUNCH)

    def test_transform_modes_restore_their_sources_and_controls(self):
        for mode in (
            "'retake'", "'inpaint'", "'restyle'", "'outpaint'",
            "'edit_anything'", "'recast'",
        ):
            self.assertIn(mode, STORE)
        for state_field in (
            "editVideoPath", "editPromptStrength", "editSamTarget",
            "editInvertMask", "editAnythingStartAnchor",
            "editAnythingEndAnchor", "editRepaintMappings",
            "editRecastMappings", "outpaintVideoBox",
            "outpaintPreserveSourceAudio", "outpaintLockSourcePixels",
        ):
            self.assertIn(state_field, STORE)

    def test_image_and_keyframe_inputs_use_general_file_restore(self):
        self.assertIn("newParams.image_guide", STORE)
        self.assertIn("newParams.image_mask", STORE)
        self.assertIn("imageWorkflowSourcePath: String(p.image_guide", STORE)
        self.assertIn("imageWorkflowMaskPath: String(p.image_mask", STORE)
        self.assertIn("api.getFileUrl(restoredImageGuide)", STORE)
        self.assertIn("api.getFileUrl(restoredImageMask)", STORE)
        self.assertIn("previewUrl: api.getFileUrl(filename)", INPUTS)

    def test_audio_workflows_restore_submode_and_all_reference_slots(self):
        for guide in range(2, 7):
            self.assertIn(f"newParams.audio_guide{guide}", STORE)
        self.assertIn("audioSubMode: subMode", STORE)
        self.assertIn("musicDescription", STORE)
        self.assertIn("musicInstrumental", STORE)
        self.assertIn("voiceCloneEnabled", STORE)
        self.assertIn("voiceCloneRefs: restoredVoiceCloneRefs", STORE)
        self.assertIn("toolsRevoiceRefs", STORE)
        self.assertIn('"voice_ref_paths": voice_refs', LAUNCH)

    def test_mixer_and_editor_have_dedicated_restore_paths(self):
        self.assertIn("p.model_type === 'editor'", STORE)
        self.assertIn("loadProject(p.editor_project_id)", STORE)
        self.assertIn("p.model_type === 'audio_mixer'", STORE)
        self.assertIn("audio_mixer_tracks", STORE)
        self.assertIn("restoredTracks", MIXER)
        self.assertIn('"model_type": "audio_mixer"', LAUNCH)

    def test_regular_restore_clears_cross_output_state(self):
        self.assertIn("multi_prompts_gen_type: Number(p.multi_prompts_gen_type) || 0", STORE)
        self.assertIn("h3_window_prompts = restoredH3WindowPrompts", STORE)
        self.assertIn("h3_window_plan_signature", STORE)
        self.assertIn("removeBackgroundRefs", STORE)
        self.assertIn("outputCount: newParams.repeat_generation || 1", STORE)
        self.assertIn("p.duration_seconds ?? p._duration_seconds", STORE)

    def test_sidecars_capture_scalar_and_array_asset_names(self):
        self.assertIn('"voice_clone_refs"', LAUNCH)
        self.assertIn('"image_refs"', LAUNCH)
        self.assertIn("elif val and isinstance(val, list):", LAUNCH)
        self.assertIn("elif isinstance(value, list):", LAUNCH)


if __name__ == "__main__":
    unittest.main()
