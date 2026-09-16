"""Studio's image enhancement choice reaches guided and raw LLM providers."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import llm_service


class ImageEnhancementChoiceTests(unittest.TestCase):
    def test_explicit_style_reaches_provider_without_changing_source_or_refs(self):
        source = 'Replace the main character with Blaine. Keep the pose and background.'
        for raw in (False, True):
            for style in ('faithful', 'creative'):
                with (
                    self.subTest(raw=raw, style=style),
                    mock.patch.object(llm_service, 'generate', return_value='An edited portrait.') as generate,
                    mock.patch('services.enhance_guides.get_enhance_guide', return_value='Write a still image prompt.'),
                ):
                    result = llm_service.enhance_prompt(
                        source, mode='image', model_type='flux2_klein_9b',
                        image_paths=['source.png', 'blaine.png'],
                        raw_enhancer_mode=raw, planning_style=style,
                    )
                self.assertEqual(result, 'An edited portrait.')
                request = generate.call_args.kwargs
                self.assertEqual(request['image_paths'], ['source.png', 'blaine.png'])
                self.assertIn(source, request['prompt'])
                self.assertIn(style.upper() + ' IMAGE WRITING', request['prompt'])
                self.assertIn('reference constraints and edit boundaries', request['prompt'])
                self.assertIn('one still image, not a sequence', request['prompt'])


if __name__ == '__main__':
    unittest.main()
