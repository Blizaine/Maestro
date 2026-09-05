import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GalleryActiveMediaTests(unittest.TestCase):
    def test_gallery_uses_one_selected_output_source_of_truth(self):
        main_content = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MainContent.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("const activeIndex = useStore(s => s.selectedOutput)", main_content)
        self.assertIn("selectViewportCenteredItem", main_content)
        self.assertIn("viewportCenterY", main_content)
        self.assertIn("[data-feed-index]", main_content)
        self.assertIn("!media.paused && !media.ended", main_content)
        self.assertIn("feedEl.scrollTop <= boundaryTolerance", main_content)
        self.assertIn("feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight", main_content)
        self.assertIn("Math.min(...visibleItems.map", main_content)
        self.assertIn("Math.max(...visibleItems.map", main_content)
        self.assertNotIn("const [activeIndex, setActiveIndex]", main_content)
        self.assertNotIn("handleItemVisible", main_content)

    def test_playing_media_activates_and_unmutes_exclusively(self):
        main_content = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MainContent.tsx"
        ).read_text(encoding="utf-8")
        feed_item = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MediaFeedItem.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("activateIndex(index)", main_content)
        self.assertIn("candidate.pause()", main_content)
        self.assertIn("candidate.muted = true", main_content)
        self.assertIn("media.muted = false", main_content)
        self.assertIn('data-gallery-media="true"', feed_item)
        self.assertIn("onPlay={handlePlaybackStart}", feed_item)
        self.assertIn("event.currentTarget.muted = false", feed_item)

    def test_generation_details_drawer_exposes_searchable_run_metadata(self):
        feed_item = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MediaFeedItem.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("Show generation details", feed_item)
        self.assertIn("Active LoRAs", feed_item)
        self.assertIn("Optimizations", feed_item)
        self.assertIn("Turbo preset", feed_item)
        self.assertIn("First Block Cache", feed_item)
        self.assertIn("Sol Engine", feed_item)
        self.assertIn("whitespace-pre-wrap", feed_item)
        self.assertIn("handleCopyOriginalPrompt", feed_item)
        self.assertIn('aria-label="Copy original prompt"', feed_item)
        self.assertIn("copiedOriginalPrompt ? 'Copied' : 'Copy'", feed_item)

    def test_generation_details_expose_multi_window_summary_and_exact_timings(self):
        feed_item = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MediaFeedItem.tsx"
        ).read_text(encoding="utf-8")
        types = (
            ROOT / "ui" / "src" / "types" / "index.ts"
        ).read_text(encoding="utf-8")

        self.assertIn("multi_window_timing", feed_item)
        self.assertIn("Window timing", feed_item)
        self.assertIn("Scene duration", feed_item)
        self.assertIn("Total render", feed_item)
        self.assertIn("window_generation_seconds", types)
        self.assertIn("...(turboEnabled ? ['Turbo'] : [])", feed_item)

    def test_generation_details_expose_single_window_render_time(self):
        feed_item = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MediaFeedItem.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("singleWindowGenerationSeconds", feed_item)
        self.assertIn('>Generation time</dt>', feed_item)
        self.assertIn(
            "Generation time excluding queue wait and model loading",
            feed_item,
        )

    def test_clip_footer_uses_compact_persistent_controls_and_labeled_more_menu(self):
        feed_item = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MediaFeedItem.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("Four persistent controls", feed_item)
        self.assertIn('aria-label="More clip actions"', feed_item)
        self.assertIn('aria-label="Clip actions"', feed_item)
        self.assertIn("spaceBelow > spaceAbove", feed_item)
        self.assertIn("actionMenuOpensDown ? 'top-full mt-2' : 'bottom-full mb-2'", feed_item)
        for label in (
            "Save as Recipe",
            "Regenerate with same settings",
            "Retake a time region",
            "Extend this video",
            "Copy prompt",
            "Use current frame as reference",
            "Download",
            "Move to workspace",
            "Delete output",
        ):
            self.assertIn(label, feed_item)

        self.assertIn("Click again to delete", feed_item)
        self.assertIn("No other workspaces", feed_item)

    def test_extend_action_routes_to_named_workflow_before_attaching_source(self):
        feed_item = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MediaFeedItem.tsx"
        ).read_text(encoding="utf-8")

        route = "setStudioVideoWorkflow('extend')"
        attach = "setContinueVideo(videoFile, uploaded.path, url, duration)"
        self.assertIn("const setStudioVideoWorkflow = useStore", feed_item)
        self.assertIn("setSidebarMode('studio')", feed_item)
        self.assertIn(route, feed_item)
        self.assertIn(attach, feed_item)
        self.assertLess(feed_item.index(route), feed_item.index(attach))
        self.assertNotIn("setParam('image_mode', 3)", feed_item)

    def test_card_height_measurements_immediately_reflow_virtualized_offsets(self):
        main_content = (
            ROOT / "ui" / "src" / "components" / "MainContent" / "MainContent.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("const [measureEpoch, setMeasureEpoch] = useState(0)", main_content)
        self.assertIn("estimatedItemHeight, measureEpoch]", main_content)
        self.assertIn("setMeasureEpoch(e => e + 1)", main_content)


if __name__ == "__main__":
    unittest.main()
