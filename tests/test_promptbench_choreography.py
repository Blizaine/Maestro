"""Candidate craft advice stays scoped and does not affect unrelated scenes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from promptbench.experiments import experiment_context
from services.adaptive_enhancement import adaptive_writing_guide


class ChoreographyExperimentTests(unittest.TestCase):
    def test_fight_candidate_is_scoped_and_office_guide_is_identical(self):
        fight = "A kung fu fight in the mountains."
        office = "Pam asks if the visitor needs help. Michael invites her into his office."
        baseline = adaptive_writing_guide(fight)
        office_baseline = adaptive_writing_guide(office)
        with experiment_context("fight_choreography"):
            self.assertNotEqual(adaptive_writing_guide(fight), baseline)
            self.assertEqual(adaptive_writing_guide(office), office_baseline)
        self.assertEqual(adaptive_writing_guide(fight), baseline)


if __name__ == "__main__":
    unittest.main()
