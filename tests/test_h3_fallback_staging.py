"""Source-based reference fallbacks must retain entrances and opening state."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import llm_service


REFERENCES = (
    "<Subject 1> is Alex from <Picture 1>, preserving identity.\n"
    "<Subject 2> is Sam from <Picture 2>, preserving identity."
)


class FallbackStagingTests(unittest.TestCase):
    def test_fallback_does_not_place_a_later_entrant_in_opening_composition(self):
        source = (
            'Alex stands beside a closed door. The door opens and Sam walks in. '
            'Alex says, "Welcome." Sam replies, "Thanks."'
        )
        fallback = llm_service._build_h3_ref2va_tagged_fallback(
            source, REFERENCES, duration_seconds=10, planning_style="adaptive",
        )
        self.assertIn("The door opens and Sam walks in.", fallback)
        self.assertNotIn("visible in the opening composition", fallback)
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(fallback), ["Welcome.", "Thanks."])
        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
            source, fallback, REFERENCES,
        ))

    def test_silent_fallback_keeps_source_opening_and_retains_reference_bindings(self):
        source = "Alex waits alone at the table. Later, Sam enters through the doorway. No dialogue."
        fallback = llm_service._build_h3_ref2va_tagged_fallback(
            source, REFERENCES, duration_seconds=10, planning_style="adaptive",
        )
        self.assertIn(source, fallback)
        self.assertNotIn("visible in the opening composition", fallback)
        self.assertIn("<Subject 1>", fallback)
        self.assertIn("<Subject 2>", fallback)
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(fallback), [])
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(fallback, REFERENCES))


if __name__ == "__main__":
    unittest.main()
