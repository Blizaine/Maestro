"""Studio's image enhancement choice reaches guided and raw LLM providers."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import llm_service


class ImageEnhancementChoiceTests(unittest.TestCase):
    def test_flux3_receives_exact_ordered_native_reference_map(self):
        paths = [f"ref-{index}.png" for index in range(10)]
        source = 'Combine these characters, preserving the monkey as a monkey. The banner reads "WELCOME".'
        with (
            mock.patch.object(llm_service, 'generate', return_value='A group portrait.') as generate,
            mock.patch('services.enhance_guides.get_enhance_guide', return_value='Write a FLUX 3 caption.'),
        ):
            result = llm_service.enhance_prompt(
                source, mode='image', model_type='flux3_klein', image_paths=paths,
            )
        self.assertEqual(result, 'A group portrait.')
        payload = generate.call_args.kwargs
        self.assertEqual(payload['image_paths'], paths)
        self.assertIn(source, payload['prompt'])
        for index in range(10):
            self.assertIn(f"Image {index + 1} = <ref_image_{index}>", payload['prompt'])
        self.assertNotIn('<ref_image_10>', payload['prompt'])
        self.assertIn('not the final subject count', payload['prompt'])
        self.assertIn('allowed model syntax', payload['system_prompt'])
        self.assertFalse(payload['enable_thinking'])
        self.assertEqual(payload['thinking_budget'], 0)

    def test_flux3_cleanup_keeps_all_ten_bindings_and_exact_lettering(self):
        caption = ('A group portrait: ' + ', '.join(
            f'the character from <ref_image_{index}>' for index in range(10)
        ) + '. A banner reads "WELCOME".')
        with (
            mock.patch.object(llm_service, 'generate', return_value=caption),
            mock.patch('services.enhance_guides.get_enhance_guide', return_value='Write a FLUX 3 caption.'),
        ):
            result = llm_service.enhance_prompt(
                'A group portrait.', mode='image', model_type='flux3_klein',
                image_paths=[f'{index}.png' for index in range(10)],
            )
        self.assertEqual(result, caption)

    def test_flux3_text_only_has_no_invented_reference_map(self):
        with (
            mock.patch.object(llm_service, 'generate', return_value='A fox in a woodcut illustration.') as generate,
            mock.patch('services.enhance_guides.get_enhance_guide', return_value='Write a FLUX 3 caption.'),
        ):
            llm_service.enhance_prompt('A woodcut fox.', mode='image', model_type='flux3_klein')
        payload = generate.call_args.kwargs
        self.assertNotIn('Image 1 =', payload['prompt'])
        self.assertIn('No reference image is attached', payload['system_prompt'])

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
