"""Exercise shared prompt/catalog integration without importing the GPU server."""
import ast
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from services.enhance_guides import get_enhance_guide
from services.studio_enhancement import enhancement_request
from services.h3_runtime_configuration import two_pass_load_context


class PortIntegrationTests(unittest.TestCase):
    def test_two_pass_assets_and_reload_follow_explicit_opt_in(self):
        md = {'architecture': 'minimax_h3_ref2va', 'minimax_h3_two_pass_compatible': True}
        self.assertEqual(two_pass_load_context(md, None), ({'minimax_h3_two_pass': False}, False))
        loaded = SimpleNamespace(two_pass_enabled=False)
        self.assertEqual(two_pass_load_context(md, {'h3_two_pass_enabled': 'on'}, loaded),
                         ({'minimax_h3_two_pass': True}, True))
        loaded.two_pass_enabled = True
        self.assertEqual(two_pass_load_context(md, {'h3_two_pass_enabled': 'on'}, loaded),
                         ({'minimax_h3_two_pass': True}, False))
        self.assertEqual(two_pass_load_context(md, {}, loaded), ({'minimax_h3_two_pass': False}, True))
        for invalid in ('false', '', 'enabled', 1, 0, [], {}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                two_pass_load_context(md, {'h3_two_pass_enabled': invalid}, loaded)
        with self.assertRaises(ValueError):
            two_pass_load_context({'architecture': 'minimax_h3', 'minimax_h3_two_pass_compatible': False},
                                  {'h3_two_pass_enabled': 'on'})
        self.assertEqual(two_pass_load_context({'architecture': 'moss_image'}, {'h3_two_pass_enabled': 'on'}), ({}, False))

    def test_fibo_prose_and_reference_guides_preserve_different_contracts(self):
        with patch.dict(sys.modules, {'wgp': SimpleNamespace(get_model_def=lambda _: {})}):
            caption = get_enhance_guide('fibo2_dev', 'image', False)
            edit = get_enhance_guide('fibo2_edit_turbo', 'image', True)
        self.assertIn('detailed visual caption', caption)
        self.assertIn('image 1, image 2, image 3 and image 4', edit)
        self.assertIn('requested changes', edit)
        self.assertNotIn('integrated_multimodal_description:', edit)

    def test_studio_preserves_all_fibo_reference_order_and_requested_roles(self):
        params = {'model_type': 'fibo2_dev', 'generation_mode': 'image', 'image_mode': 1,
                  'prompt': 'Use the subject from image 2 in image 1.', 'image_refs': ['scene.png', 'subject.png'],
                  'video_prompt_type': 'I', 'resolution': '512x512'}
        result, _ = enhancement_request(params, {'image_outputs': True, 'instruction_image_edit': True})
        self.assertEqual(result['image_paths'], ['scene.png', 'subject.png'])
        self.assertIn('attachment order', result['reference_context'])
        self.assertEqual(result['prompt'], params['prompt'])

    def test_native_structured_image_enhancement_uses_selected_profile_and_all_refs(self):
        tree = ast.parse((ROOT / 'app/wgp.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_prompt_enhancer')
        namespace = dict(re=re, Image=SimpleNamespace(open=lambda value: value), enhancer_offloadobj=None,
                         server_config={}, prompt_enhancer_image_caption_model=None,
                         prompt_enhancer_image_caption_processor=None, prompt_enhancer_llm_model=None,
                         prompt_enhancer_llm_tokenizer=None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<native enhancer>', 'exec'), namespace)
        generated = Mock(return_value=['structured edit'])
        module = ModuleType('shared.prompt_enhancer.prompt_enhance_utils')
        module.generate_cinematic_prompt = generated
        references = [object(), object(), object(), object()]
        md = {'architecture': 'fibo2_dev', 'instruction_image_edit': True, 'max_image_refs': 4,
              'image_prompt_enhancer_instructions': 'Plain edit', 'image_prompt_enhancer_max_tokens': 512,
              'image_prompt_enhancer_instructions1': 'Structured edit', 'image_prompt_enhancer_max_tokens1': 4096}
        with patch.dict(sys.modules, {'shared.prompt_enhancer.prompt_enhance_utils': module}):
            result = namespace['process_prompt_enhancer'](md, 'TI1', ['Edit'], None, references, True, False, 42)
            self.assertEqual(result, ['structured edit'])
            self.assertEqual(generated.call_args.args[5], references)
            self.assertEqual(generated.call_args.kwargs['prompt_enhancer_instructions'], 'Structured edit')
            self.assertEqual(generated.call_args.kwargs['max_new_tokens'], 4096)
            namespace['process_prompt_enhancer'](md, 'TI', ['Edit'], None, references, True, False, 42)
            self.assertEqual(generated.call_args.kwargs['prompt_enhancer_instructions'], 'Plain edit')


if __name__ == '__main__':
    unittest.main()
