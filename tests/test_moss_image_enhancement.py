"""MOSS's guided image rewrite keeps layout syntax and selected-canvas context."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import llm_service
from services.studio_enhancement import enhancement_request


class MossEnhancementTests(unittest.TestCase):
    def test_actual_guided_provider_keeps_whole_layout_and_rectangle(self):
        layout = {'summary': {'meta': ['poster'], 'canvas': 'single', 'caption': 'A travel poster.'},
                  'items': [{'label': 'text', 'caption': 'Navy heading.', 'text': 'MAESTRO',
                             'bbox': [-480, -700, -280, 700], 'custom': 'keep'}],
                  'bbox_coord_space': 'abs_centralized', 'bbox_format': 'yxyx'}
        text = json.dumps(layout, indent=2)
        with patch.dict(sys.modules, {'wgp': SimpleNamespace(get_model_def=lambda _: {})}), \
                patch.object(llm_service, 'generate', return_value=text) as generate:
            result = llm_service.enhance_prompt(text, mode='image', model_type='moss_image_sft', resolution='1536x1024', raw_enhancer_mode=True)
        self.assertEqual(json.loads(result), layout)
        arguments = generate.call_args.kwargs
        self.assertIn('X bounds are -768 to 768', arguments['prompt'])
        self.assertIn('Y bounds are -512 to 512', arguments['prompt'])
        self.assertIn('abs_centralized', arguments['system_prompt'])
        self.assertIn('MOSS IMAGE OUTPUT CONTRACT', arguments['system_prompt'])
        self.assertNotIn('FLUX 3 OUTPUT CONTRACT', arguments['system_prompt'])
        self.assertGreaterEqual(arguments['max_new_tokens'], 4096)
        self.assertFalse(arguments['enable_thinking'])

    def test_queued_image_enhancement_freezes_selected_resolution(self):
        original = {'model_type': 'moss_image_sft', 'generation_mode': 'image',
                    'image_mode': 1, 'prompt': 'A mountain poster.', 'resolution': '1536x1024'}
        payload, sequence = enhancement_request(original, {'architecture': 'moss_image', 'image_outputs': True})
        self.assertFalse(sequence)
        self.assertEqual(payload['resolution'], '1536x1024')
        self.assertEqual(payload['image_paths'], [])
        self.assertEqual(original['prompt'], 'A mountain poster.')


if __name__ == '__main__':
    unittest.main()
