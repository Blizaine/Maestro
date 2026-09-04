import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LongFormDurationUiTests(unittest.TestCase):
    def test_shared_control_has_full_preset_ladder_and_timecode(self):
        planning = (
            ROOT / "ui/src/lib/durationPlanning.ts"
        ).read_text(encoding="utf-8")
        control = (
            ROOT / "ui/src/components/Sidebar/DurationPresetControl.tsx"
        ).read_text(encoding="utf-8")
        for label in ("30s", "1m", "2m", "3m", "4m", "5m", "10m", "15m", "30m", "60m"):
            self.assertIn(f"label: '{label}'", planning)
        self.assertIn("LONG_FORM_MAX_SECONDS = 60 * 60", planning)
        self.assertIn("1 window", control)
        self.assertIn("HH:MM:SS", control)
        self.assertIn("final output", control)
        self.assertIn("trim", control)

    def test_duration_windows_and_bounded_auto_planning(self):
        planning = (
            ROOT / "ui/src/lib/durationPlanning.ts"
        ).read_text(encoding="utf-8")
        control = (
            ROOT / "ui/src/components/Sidebar/DurationPresetControl.tsx"
        ).read_text(encoding="utf-8")
        studio = (
            ROOT / "ui/src/components/Sidebar/DurationSlider.tsx"
        ).read_text(encoding="utf-8")
        director = (
            ROOT / "ui/src/components/Sidebar/DirectorChat.tsx"
        ).read_text(encoding="utf-8")
        store = (
            ROOT / "ui/src/stores/useStore.ts"
        ).read_text(encoding="utf-8")

        self.assertIn("DurationPlanningMode = 'duration' | 'windows' | 'auto'", planning)
        self.assertIn("wholeWindowDuration", planning)
        self.assertIn("maximumWholeWindowCount", planning)
        self.assertIn("recommendAutoDuration", planning)
        self.assertIn("continuationFirstWindowSeconds", planning)
        self.assertIn("firstWindowSeconds?: number | null", planning)
        self.assertIn("maximumInferredWindows ?? 8", planning)
        self.assertIn("manual prompt", planning)
        self.assertIn("visibleActionBeatCount", planning)
        self.assertIn("const nonDialoguePrompt = prompt.replace", planning)
        self.assertIn("dialogueAttributionOnly", planning)
        self.assertIn("productionHeadingOnly", planning)
        self.assertIn("looksLikeSpeakerLabel", planning)
        self.assertIn("analyzePromptTiming", planning)
        self.assertIn("dialogueWords / 2.15", planning)
        self.assertIn("timing.hasScreenplayDialogue", planning)
        self.assertIn("dialogueRequiredWindows", planning)
        self.assertIn("Vague inferred concepts are capped", control)
        self.assertNotIn("const sentences = prompt", planning)
        self.assertIn("['auto', 'duration', 'windows']", control)
        self.assertIn("mode === 'duration' ? 'Time' : 'Window'", control)
        self.assertNotIn("Auto · Beta", control)
        self.assertIn("_duration_planning_mode: 'auto'", store)
        self.assertIn("s.params._duration_planning_mode ?? 'auto'", studio)
        self.assertIn("useState<'duration' | 'windows' | 'auto'>('auto')", director)
        self.assertIn("WINDOW_COUNT_PRESETS = [1, 2, 3, 4, 6, 8]", control)
        self.assertIn("Exact window count", control)
        self.assertIn(
            "Timed media, explicit durations, manual prompt lines, and explicit screenplay dialogue",
            control,
        )
        self.assertIn("enablePlanningModes={isH3 || isLtx || supportsSlidingWindows}", studio)
        self.assertIn("studioVideoWorkflow === 'extend'", studio)
        self.assertIn("firstWindowSeconds={firstWindowSeconds}", studio)
        self.assertIn("autoSourceLabel={autoSourceLabel}", studio)
        self.assertIn("s.params._h3_original_prompt", studio)
        self.assertIn("s.params._ltx_original_prompt", studio)
        self.assertIn("autoPrompt={durationPlanningPrompt}", studio)
        self.assertNotIn("autoPrompt={prompt}", studio)
        self.assertIn("durationPlanningMode === 'auto'", studio)
        self.assertIn("duration <= windowSize + 0.05", studio)
        self.assertIn("? automaticWindowMax", studio)
        self.assertIn("duration, durationPlanningMode, locked", studio)
        self.assertNotIn("s.params.minimax_h3_references ?? []", studio)
        self.assertIn("s.params.minimax_h3_references)", studio)
        self.assertIn("enablePlanningModes", director)
        self.assertIn('autoPlanningStyle="creative"', director)

    def test_studio_and_director_share_duration_control(self):
        studio = (
            ROOT / "ui/src/components/Sidebar/DurationSlider.tsx"
        ).read_text(encoding="utf-8")
        director = (
            ROOT / "ui/src/components/Sidebar/DirectorChat.tsx"
        ).read_text(encoding="utf-8")
        song = (
            ROOT / "ui/src/components/Sidebar/DirectorSongSetup.tsx"
        ).read_text(encoding="utf-8")
        audio = (
            ROOT / "ui/src/components/Sidebar/AudioDurationControl.tsx"
        ).read_text(encoding="utf-8")
        store = (ROOT / "ui/src/stores/useStore.ts").read_text(encoding="utf-8")
        self.assertIn("<DurationPresetControl", studio)
        self.assertIn("<DirectorTargetDurationControl />", director)
        self.assertIn("<DurationPresetControl", song)
        self.assertIn("quantizeToWindows={false}", song)
        self.assertIn("quantizeToWindows={false}", audio)
        self.assertIn("Math.min(totalDuration, 60 * 60)", store)
        self.assertIn("wantsSequence", store)

    def test_chunked_speech_models_publish_one_hour_ceiling(self):
        for relative in (
            "app/models/TTS/qwen3_handler.py",
            "app/models/TTS/index_tts2_handler.py",
            "app/models/TTS/kugelaudio_handler.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('"max": 3600', source, relative)


if __name__ == "__main__":
    unittest.main()
