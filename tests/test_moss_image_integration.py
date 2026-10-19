"""MOSS catalog, raw layout and enhancer contracts without model downloads."""
import ast
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))

from services.enhance_guides import get_enhance_guide
from services.moss_image_prompt import enhancement_canvas_context, render_layout_instructions


class MossImageIntegrationTests(unittest.TestCase):
    def test_catalogs_use_same_native_recipe_and_pinned_public_formats(self):
        revision = '0a508e986e920c8051b99df98d27329742fc3d0e'
        for model_id, variant in (('moss_image', 'base'), ('moss_image_sft', 'sft')):
            with self.subTest(model=model_id):
                catalog = json.loads((ROOT / f'app/defaults/{model_id}.json').read_text())
                model = catalog['model']
                self.assertEqual(model['architecture'], 'moss_image')
                self.assertEqual((catalog['num_inference_steps'], catalog['guidance_scale'], catalog['flow_shift']), (50, 3, 5))
                self.assertEqual((catalog['image_mode'], catalog['video_prompt_type'], catalog['prompt_enhancer']), (1, '', ''))
                self.assertEqual(set(model['URLs']), {f'moss_image_{variant}_{fmt}.safetensors' for fmt in ('bf16', 'int8_convrot')})
                for name, source in model['download_sources'].items():
                    self.assertEqual(source['url'], f'https://huggingface.co/DeepBeepMeep/Moss/resolve/{revision}/{name}')
                    self.assertRegex(source['sha256'], r'^[0-9a-f]{64}$')
                    self.assertGreater(source['size_bytes'], 5_000_000_000)
                    self.assertTrue(source['anonymous'])

    def test_raw_json_bypasses_prompt_templates_without_changing_other_models(self):
        tree = ast.parse((ROOT / 'app/wgp.py').read_text(encoding='utf-8'))
        layout = json.dumps({'summary': {'caption': 'A {literal} sign'}, 'items': [{'text': '[MAESTRO]', 'bbox': [-100, -200, 100, 200]}]}, indent=2)
        # Execute the actual admission statements, not a duplicate parser.
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets)
                          and isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Name) and n.value.value.id == 'inputs'
                          for n in node.body)
                  and any(isinstance(n, ast.If) and 'skip_prompt_template' in ast.unparse(n.test) for n in node.body))
        start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and ast.unparse(n).startswith("prompt = inputs['prompt']"))
        statements = fn.body[start:start+3]
        self.assertEqual(len(statements), 3)
        code = compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), '<native prompt admission>', 'exec')
        parser = Mock(return_value=('expanded', ''))
        namespace = {'inputs': {'prompt': layout}, 'model_def': {'skip_prompt_template': True}, 'prompt_parser': SimpleNamespace(process_template=parser)}
        exec(code, namespace)
        self.assertEqual(namespace['prompt'], layout)
        parser.assert_not_called()
        namespace['model_def'] = {}
        exec(code, namespace)
        self.assertEqual(namespace['prompt'], 'expanded')
        parser.assert_called_once()

    def test_native_layout_guide_uses_centered_pixels_and_no_reference_editing(self):
        fake_wgp = SimpleNamespace(get_model_def=lambda _: {})
        for model_id in ('moss_image', 'moss_image_sft'):
            with patch.dict(sys.modules, {'wgp': fake_wgp}):
                guide = get_enhance_guide(model_id, 'image')
            self.assertIn('MOSS Image', guide)
            self.assertIn('abs_centralized', guide)
            self.assertIn('[y_min, x_min, y_max, x_max]', guide)
            self.assertIn('do not accept reference images', guide)
            self.assertNotIn('ref_image_', guide)

    def test_rectangular_canvas_context_uses_actual_x_and_y_bounds(self):
        context = enhancement_canvas_context('1536x1024')
        self.assertIn('1536x1024', context)
        self.assertIn('X bounds are -768 to 768', context)
        self.assertIn('Y bounds are -512 to 512', context)
        self.assertIn('plain caption request stays', context)
        self.assertIn('1024x1024', enhancement_canvas_context())

    def test_bad_layout_dimensions_fail_before_loading_an_llm(self):
        for value in ('1025x1024', '0x1024', '4096x4096'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'divisible by 16'):
                enhancement_canvas_context(value)

    def test_classic_enhancer_resolves_upstream_canvas_placeholders(self):
        prompt = render_layout_instructions('{{width}}x{{height}}; {{half_width}}, {{half_height}}, {{aspect_ratio}}', '1536x1024')
        self.assertIn('1536x1024; 768, 512, 1536:1024', prompt)
        self.assertNotIn('{{', prompt)

    def test_model_handler_and_react_capabilities_are_registered(self):
        source = (ROOT / 'app/wgp.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        handlers = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'family_handlers' for t in n.targets))
        self.assertIn('models.moss_image.moss_image_handler', ast.literal_eval(handlers))
        launch = (ROOT / 'app/launch.py').read_text(encoding='utf-8')
        self.assertIn('"moss_image_prompt_helper": md.get("moss_image_prompt_helper", False)', launch)
        self.assertIn('needs_moss_image_guide', launch)


if __name__ == '__main__':
    unittest.main()
