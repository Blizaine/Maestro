"""Authored speech takes precedence over automatic dialogue-writing targets."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import llm_service
from services.dialogue_writing import DialogueBudget


SOURCE = (
    'Mira says <d>Wait here for the signal.</d>\n'
    'Tavi replies <d>I will watch the door.</d>\n'
    'Mira says <d>Then tell me what you see.</d>\n'
    'Tavi answers <d>The corridor is quiet now.</d>\n'
    'Juno says <d>We can leave when everyone is ready.</d>\n'
    'Mira replies <d>All right, let us go.</d>\n'
    'Tavi says something encouraging as an unscripted closing reply.'
)


class AuthoredDialogueGuidanceTests(unittest.TestCase):
    def test_automatic_target_does_not_compete_with_supplied_lines(self):
        budget = DialogueBudget(15, 20, 43)
        instruction = llm_service._enhance_dialogue_budget_instruction(
            SOURCE, budget, 'minimax_h3_ref2va',
        )
        self.assertIn('6 spoken lines containing 33 spoken words', instruction)
        self.assertIn('at most 43 spoken words', instruction)
        self.assertIn('Preserve every supplied line verbatim', instruction)
        self.assertIn('only where the brief explicitly requests', instruction)
        self.assertNotIn('Aim for 20', instruction)

    def test_adaptive_request_preserves_script_before_allocating_extra_turn(self):
        request = llm_service._build_enhance_user_prompt(
            SOURCE, 'video', 14.375, 1, 14.375, 'minimax_h3_ref2va', 'adaptive',
        )
        self.assertIn('6 spoken lines', request)
        self.assertIn('Preserve every supplied line verbatim', request)
        self.assertNotIn('duration-derived minimum', request)
        self.assertIn(SOURCE, request)

    def test_open_dialogue_brief_retains_automatic_budget(self):
        budget = DialogueBudget(15, 20, 43)
        self.assertEqual(llm_service._enhance_dialogue_budget_instruction(
            'Two friends discuss their journey.', budget, 'minimax_h3_ref2va',
        ), budget.instruction())

    def test_other_model_instruction_retains_existing_budget(self):
        budget = DialogueBudget(15, 20, 43)
        self.assertEqual(llm_service._enhance_dialogue_budget_instruction(
            SOURCE, budget, 'ltx2',
        ), budget.instruction())

    def test_multi_window_guidance_distributes_instead_of_repeating_the_script(self):
        request = llm_service._build_enhance_user_prompt(
            SOURCE, 'video', 30, 3, 10, 'minimax_h3_ref2va', 'adaptive',
        )
        self.assertIn('across the 3 windows, never repeating', request)
        self.assertIn('Each window allows at most 30 spoken words', request)
        self.assertNotIn('at most 30 spoken words in total', request)

    def test_focused_repair_receives_actual_validation_failures(self):
        with patch.object(llm_service, 'generate', return_value='invalid draft') as writer, \
                patch('services.enhance_guides.get_enhance_guide', return_value='Use native H3 fields.'), \
                patch('services.studio_enhancement.fidelity_retry_limit', return_value=1):
            llm_service.enhance_prompt(
                'Mira says <d>Ready.</d>', mode='video', model_type='minimax_h3_ref2va',
                reference_context='No reference media were supplied.', duration_seconds=8,
            )
        self.assertGreaterEqual(writer.call_count, 2)
        retry = writer.call_args_list[1].kwargs['prompt']
        self.assertIn('VALIDATION FAILURES TO FIX:', retry)
        self.assertIn('missing required field subject_definitions:', retry)
        self.assertIn('requested dialogue is missing', retry)
        self.assertIn('Ready.', retry)


if __name__ == '__main__':
    unittest.main()
