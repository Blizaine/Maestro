"""Exercise production image handoff blocks without loading model checkpoints."""
import ast
import copy
import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
WGP = Path(os.environ.get('MAESTRO_TEST_WGP_SOURCE', ROOT / 'app/wgp.py'))


def source_tree(path):
    return ast.parse(path.read_text(encoding='utf-8'))


def execute(nodes, namespace, filename):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(filename), 'exec'), namespace)


class TestZImageVaePrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'app/models/z_image/pipeline_z_image.py'
        tree = source_tree(path)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__call__')
        # Execute the real final output block, including latent-only bypass.
        branch = next(i for i, n in enumerate(method.body)
                      if isinstance(n, ast.If) and ast.unparse(n.test) == "output_type == 'latent'")
        wrapper = ast.parse('def output(self, latents, dtype, output_type):\n    pass').body[0]
        wrapper.body = method.body[branch - 1:]
        namespace = {}
        execute([wrapper], namespace, path)
        cls.output = staticmethod(namespace['output'])

    def pipe(self, dtype):
        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 3, 1).to(dtype=dtype)
                self.config = SimpleNamespace(scaling_factor=0.5, shift_factor=0.1)

            def decode(self, latents, return_dict=False):
                self.received_dtype = latents.dtype
                return (self.conv(latents),)

        return SimpleNamespace(vae=Vae(), image_processor=SimpleNamespace(postprocess=lambda x, **_: x))

    def test_bfloat_denoiser_decodes_with_half_vae(self):
        pipe = self.pipe(torch.float16)
        result = self.output(pipe, torch.ones(1, 4, 2, 2), torch.bfloat16, 'pt')
        self.assertEqual(pipe.vae.received_dtype, torch.float16)
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(tuple(result.shape), (1, 3, 2, 2))

    def test_half_denoiser_decodes_with_bfloat_vae(self):
        pipe = self.pipe(torch.bfloat16)
        self.output(pipe, torch.ones(1, 4, 2, 2), torch.float16, 'pt')
        self.assertEqual(pipe.vae.received_dtype, torch.bfloat16)

    def test_latent_output_retains_denoiser_precision(self):
        pipe = self.pipe(torch.float16)
        result = self.output(pipe, torch.ones(1, 4, 2, 2), torch.bfloat16, 'latent')
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertFalse(hasattr(pipe.vae, 'received_dtype'))


class TestLiveModelArchitectureReload(unittest.TestCase):
    def setUp(self):
        def family(encoder, **extra):
            return SimpleNamespace(query_model_def=lambda *args: {'text_encoder_folder': encoder, **extra})

        self.ns = dict(glob=glob, os=os, json=json, models_def={}, reload_needed=False, model_types_handlers={
            'krea2': family('Qwen3-VL', vision_encoder_URLs=['vision.safetensors']),
            'z_image': family('Qwen3'),
        })
        names = {'get_model_def', 'get_base_model_type', 'init_model_def', 'load_model_definitions'}
        execute([n for n in source_tree(WGP).body if isinstance(n, ast.FunctionDef) and n.name in names], self.ns, WGP)

    def test_architecture_change_removes_old_defaults_but_preserves_registry_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'civitai_same_id.json'
            with patch.object(glob, 'glob', side_effect=lambda pattern: [str(path)] if pattern.startswith('defaults') else []):
                path.write_text(json.dumps({'model': {'architecture': 'krea2'}, 'steps': 4}), encoding='utf-8')
                self.ns['load_model_definitions']()
                original = self.ns['models_def']['civitai_same_id']
                self.assertEqual(original['text_encoder_folder'], 'Qwen3-VL')
                self.ns['transformer_type'] = 'civitai_same_id'
                path.write_text(json.dumps({'model': {'architecture': 'z_image'}, 'steps': 8}), encoding='utf-8')
                self.ns['load_model_definitions']()
        rebuilt = self.ns['models_def']['civitai_same_id']
        self.assertIs(rebuilt, original)
        self.assertEqual(rebuilt['text_encoder_folder'], 'Qwen3')
        self.assertNotIn('vision_encoder_URLs', rebuilt)
        self.assertEqual(rebuilt['settings'], {'steps': 8})
        self.assertIn('civitai_same_id', self.ns['displayed_model_types'])
        self.assertTrue(self.ns['reload_needed'])

    def test_unchanged_active_definition_does_not_reload_warm_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'civitai_same_id.json'
            path.write_text(json.dumps({'model': {'architecture': 'krea2'}, 'steps': 4}), encoding='utf-8')
            with patch.object(glob, 'glob', side_effect=lambda pattern: [str(path)] if pattern.startswith('defaults') else []):
                self.ns['load_model_definitions']()
                self.ns['transformer_type'] = 'civitai_same_id'
                self.ns['load_model_definitions']()
        self.assertFalse(self.ns['reload_needed'])

    def test_inactive_definition_change_does_not_reload_another_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'civitai_same_id.json'
            with patch.object(glob, 'glob', side_effect=lambda pattern: [str(path)] if pattern.startswith('defaults') else []):
                path.write_text(json.dumps({'model': {'architecture': 'krea2'}}), encoding='utf-8')
                self.ns['load_model_definitions']()
                self.ns['transformer_type'] = 'other_active_id'
                path.write_text(json.dumps({'model': {'architecture': 'z_image'}}), encoding='utf-8')
                self.ns['load_model_definitions']()
        self.assertFalse(self.ns['reload_needed'])

    def test_failed_family_initialization_preserves_previous_definition(self):
        original = {'architecture': 'krea2', 'settings': {'steps': 4}}
        self.ns['models_def']['civitai_same_id'] = original
        self.ns['transformer_type'] = 'civitai_same_id'
        def fail(*args):
            raise ValueError('invalid family definition')
        self.ns['model_types_handlers']['z_image'].query_model_def = fail
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'civitai_same_id.json'
            path.write_text(json.dumps({'model': {'architecture': 'z_image'}}), encoding='utf-8')
            with patch.object(glob, 'glob', side_effect=lambda pattern: [str(path)] if pattern.startswith('defaults') else []):
                with self.assertRaises(ValueError):
                    self.ns['load_model_definitions']()
        self.assertIs(self.ns['models_def']['civitai_same_id'], original)
        self.assertEqual(original['architecture'], 'krea2')
        self.assertFalse(self.ns['reload_needed'])


class TestStillImagePromptRouting(unittest.TestCase):
    def route(self, image_mode, prompt_mode=0, definition=None):
        tree = source_tree(WGP)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_video')
        block = next(n for n in fn.body if isinstance(n, ast.If) and 'isinstance(h3_window_prompts' in ast.unparse(n.test))
        ns = dict(model_def=definition or {}, h3_window_prompts=None, image_mode=image_mode,
                  multi_prompts_gen_type=prompt_mode, prompt='natural window light\na bright red hat')
        execute([block], ns, WGP)
        return ns['prompts']

    def test_image_and_inpaint_keep_all_description_lines(self):
        for image_mode in (1, 2):
            for prompt_mode in (0, 1, 2):
                with self.subTest(image_mode=image_mode, prompt_mode=prompt_mode):
                    self.assertEqual(self.route(image_mode, prompt_mode), ['natural window light\na bright red hat'])

    def test_video_extend_and_blend_keep_window_routing(self):
        for mode in (0, 3, 4):
            self.assertEqual(self.route(mode), ['natural window light', 'a bright red hat'])

    def test_legacy_image_prompt_pairing_still_creates_distinct_tasks(self):
        fn = next(n for n in source_tree(WGP).body if isinstance(n, ast.FunctionDef) and n.name == 'process_prompt_and_add_tasks')
        block = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'multi_prompts_gen_type == 3')
        for pairing, expected in ((0, [('first', 'A'), ('second', 'A'), ('first', 'B'), ('second', 'B')]),
                                  (1, [('first', 'A'), ('second', 'B')])):
            tasks = []
            ns = dict(multi_prompts_gen_type=0, inputs={'multi_images_gen_type': pairing},
                      prompts=['first', 'second'], image_start=['A', 'B'], image_end=None,
                      add_video_task=lambda **kwargs: tasks.append(kwargs.copy()))
            arguments = ns.copy()
            wrapper = ast.parse('def expand(multi_prompts_gen_type, inputs, prompts, image_start, image_end, add_video_task):\n    pass').body[0]
            wrapper.body = [block]
            execute([wrapper], ns, WGP)
            ns['expand'](**arguments)
            self.assertEqual([(t['prompt'], t['image_start']) for t in tasks], expected)


if __name__ == '__main__':
    unittest.main()
