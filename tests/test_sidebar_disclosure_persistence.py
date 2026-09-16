import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SidebarDisclosurePersistenceTests(unittest.TestCase):
    def test_duration_presets_use_two_balanced_rows(self):
        source = (
            ROOT / "ui/src/components/Sidebar/DurationPresetControl.tsx"
        ).read_text(encoding="utf-8")
        self.assertIn("grid-cols-[repeat(14,minmax(0,1fr))]", source)
        self.assertIn("col-span-4 w-full", source)
        self.assertIn("'col-span-2'", source)
        self.assertIn("grid-cols-6", source)

    # Character placement, library scrolling and reference drop-zone behavior
    # are covered by tests/ui/sidebar_redesign.cjs against the rendered sidebar.

    def test_character_and_h3_disclosures_persist(self):
        helper = (
            ROOT / "ui/src/lib/persistentDisclosure.ts"
        ).read_text(encoding="utf-8")
        characters = (
            ROOT / "ui/src/components/Sidebar/OmniReferenceSection.tsx"
        ).read_text(encoding="utf-8")
        optimizations = (
            ROOT / "ui/src/components/Sidebar/MiniMaxH3Optimizations.tsx"
        ).read_text(encoding="utf-8")
        self.assertIn("localStorage.getItem", helper)
        self.assertIn("localStorage.setItem", helper)
        self.assertIn("maestro-omni-characters-expanded", characters)
        self.assertIn("writePersistentDisclosure", characters)
        self.assertIn("maestro-h3-optimizations-expanded", optimizations)
        self.assertIn("writePersistentDisclosure", optimizations)


if __name__ == "__main__":
    unittest.main()
