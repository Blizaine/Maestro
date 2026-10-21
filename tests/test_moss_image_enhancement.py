"""MOSS's guided image rewrite keeps layout syntax and selected-canvas context."""
import ast
import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import llm_service
from services.studio_enhancement import enhancement_request
from services.moss_image_prompt import finalize_enhancement_output


def layout_draft():
    return {'width': 1536, 'height': 1024, 'bbox_coord_space': 'abs', 'bbox_format': 'xyxy',
            'summary': {'canvas': 'multi_panel', 'caption': 'A cup and its maker.'},
            'items': [{'label': 'character', 'caption': 'The potter.', 'bbox': [16, 16, 400, 900]},
                      {'label': 'text', 'caption': 'Exact heading.', 'text': '**MAESTRO**\n你好',
                       'bbox': [800.5, 100.5, 1500.5, 400.5]}]}


class MossEnhancementTests(unittest.TestCase):
    def test_guided_provider_compiles_new_draft_without_changing_text(self):
        with patch.dict(sys.modules, {'wgp': SimpleNamespace(get_model_def=lambda _: {})}), \
                patch.object(llm_service, 'generate', return_value=json.dumps(layout_draft())) as generate:
            result = llm_service.enhance_prompt('A two-panel layout with a potter and exact heading.',
                                              mode='image', model_type='moss_image', resolution='1536x1024')
        parsed = json.loads(result)
        self.assertEqual(parsed['summary']['canvas'], 'multi_panel')
        self.assertEqual(parsed['items'][0]['label'], 'character')
        self.assertEqual(parsed['items'][1]['text'], '**MAESTRO**\n你好')
        self.assertEqual(parsed['items'][1]['bbox'], [-412, 32, -111, 733])
        self.assertEqual(parsed['bbox_coord_space'], 'abs_centralized')
        self.assertNotIn('width', parsed)
        self.assertEqual(generate.call_count, 1)
        self.assertIn('do not manually center', generate.call_args.kwargs['system_prompt'])

    def test_invalid_draft_gets_one_bounded_non_thinking_repair(self):
        invalid = layout_draft()
        invalid['items'][0]['bbox'][0] = -10
        with patch.dict(sys.modules, {'wgp': SimpleNamespace(get_model_def=lambda _: {})}), \
                patch.object(llm_service, 'generate', side_effect=[json.dumps(invalid), json.dumps(layout_draft())]) as generate:
            result = llm_service.enhance_prompt('A two-panel layout.', mode='image',
                                              model_type='moss_image_sft', resolution='1536x1024')
        self.assertEqual(json.loads(result)['bbox_format'], 'yxyx')
        self.assertEqual(generate.call_count, 2)
        repair = generate.call_args.kwargs
        self.assertIn('exceeds the 1536x1024 canvas', repair['prompt'])
        self.assertFalse(repair['enable_thinking'])

    def test_repeated_invalid_output_is_never_forwarded_as_model_prompt(self):
        with patch.dict(sys.modules, {'wgp': SimpleNamespace(get_model_def=lambda _: {})}), \
                patch.object(llm_service, 'generate', return_value='{"summary":') as generate:
            with self.assertRaisesRegex(ValueError, 'original prompt is unchanged'):
                llm_service.enhance_prompt('A layout.', mode='image', model_type='moss_image')
        self.assertEqual(generate.call_count, 2)

    def test_caption_and_comments_do_not_go_through_prose_cleaner(self):
        caption = 'Output a sign reading "**PROMPT**".\n\n\nKeep its whitespace.'
        with patch.dict(sys.modules, {'wgp': SimpleNamespace(get_model_def=lambda _: {})}), \
                patch.object(llm_service, 'generate', return_value=caption):
            self.assertEqual(llm_service.enhance_prompt('A sign.', mode='image', model_type='moss_image'), caption)
        self.assertEqual(finalize_enhancement_output(json.dumps({'caption': caption})), caption)
        source = '# keep exact note\n' + json.dumps({
            'summary': {'caption': 'A sign.', 'canvas': 'collage', 'custom': 42},
            'items': [{'label': 'text', 'text': '**MAESTRO**\n#2', 'caption': 'Heading.',
                       'bbox': [-100, -100, 100, 100], 'custom': {'keep': True}}],
            'bbox_coord_space': 'abs_centralized', 'bbox_format': 'yxyx', 'extra_root': [1, 2]})
        result = finalize_enhancement_output('```json\n' + source + '\n```')
        self.assertTrue(result.startswith('# keep exact note\n'))
        self.assertEqual(json.loads(result.split('\n', 1)[1]), json.loads(source.split('\n', 1)[1]))

    def test_draft_cannot_override_default_canvas_and_centered_boxes_are_not_clipped(self):
        with self.assertRaisesRegex(ValueError, 'does not match selected canvas'):
            finalize_enhancement_output(json.dumps(layout_draft()))
        incomplete = layout_draft()
        del incomplete['bbox_coord_space']
        del incomplete['bbox_format']
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            finalize_enhancement_output(json.dumps(incomplete), '1536x1024')
        source = {'summary': {'caption': 'A sign.'}, 'items': [{'bbox': [-512, -769, 512, 768]}]}
        with self.assertRaisesRegex(ValueError, 'exceeds the selected'):
            finalize_enhancement_output(json.dumps(source), '1536x1024')

    def test_zero_repair_preference_is_respected(self):
        with patch.dict(sys.modules, {'wgp': SimpleNamespace(get_model_def=lambda _: {})}), \
                patch('services.studio_enhancement.fidelity_retry_limit', return_value=0), \
                patch.object(llm_service, 'generate', return_value='{"summary":') as generate:
            with self.assertRaisesRegex(ValueError, 'original prompt is unchanged'):
                llm_service.enhance_prompt('A layout.', mode='image', model_type='moss_image')
        self.assertEqual(generate.call_count, 1)

    def test_classic_profile_compiles_draft_before_returning_to_generation(self):
        # Isolate the real legacy function from wgp's GPU/server initialization.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app/wgp.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_prompt_enhancer')
        code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<classic enhancer>', 'exec')
        generated = unittest.mock.Mock(return_value=[json.dumps(layout_draft())])
        namespace = {'re': re, 'enhancer_offloadobj': None, 'prompt_enhancer_image_caption_model': None,
                     'prompt_enhancer_image_caption_processor': None, 'prompt_enhancer_llm_model': None,
                     'prompt_enhancer_llm_tokenizer': None, 'server_config': {'prompt_enhancer_randomize_seed': False}}
        with patch.dict(sys.modules, {'shared.prompt_enhancer.prompt_enhance_utils':
                                     SimpleNamespace(generate_cinematic_prompt=generated)}):
            exec(code, namespace)
            results = namespace['process_prompt_enhancer'](
                {'architecture': 'moss_image', 'text_prompt_enhancer_instructions1': 'Layout {{width}}'},
                'T1', ['A layout.'], None, None, True, False, 42, resolution='1536x1024')
        self.assertEqual(json.loads(results[0])['bbox_coord_space'], 'abs_centralized')
        self.assertIn('Layout 1536', generated.call_args.kwargs['prompt_enhancer_instructions'])
        self.assertTrue(generated.call_args.kwargs['literal_prompt'])

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
