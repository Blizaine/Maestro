from pathlib import Path
import ast, asyncio, threading, types, unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'_normalize_model_visibility_ids', '_normalize_model_visibility_overrides', '_model_visibility_response', 'update_model_visibility'}

class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail
        super().__init__(detail)

class ModelVisibilityTests(unittest.TestCase):
    def setUp(self):
        source = ast.parse((ROOT / 'app/launch.py').read_text(encoding='utf-8'))
        nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in NAMES]
        for node in nodes:
            node.decorator_list = []
        self.engine = types.SimpleNamespace(server_config={})
        self.persist = Mock()
        self.ns = {'wgp': self.engine, '_MODEL_VISIBILITY_CONFIG_KEY': 'visibility',
                   '_MODEL_VISIBILITY_WRITE_LOCK': threading.RLock(), '_persist_model_visibility_config': self.persist,
                   'HTTPException': HTTPException, 'Request': object}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / 'app/launch.py'), 'exec'), self.ns)

    def put(self, body):
        async def json(): return body
        return asyncio.run(self.ns['update_model_visibility'](types.SimpleNamespace(json=json)))

    def test_scoped_choices_preserve_shared_baseline(self):
        result = self.put({'enabled_models': [' h3 ', 'h3'], 'mode_overrides': {
            'video:frames': {'h3': False}, 'video:references': {' h3 ': True}}, 'defaults_version': 4})
        self.assertEqual(result['enabled_models'], ['h3'])
        self.assertEqual(result['mode_overrides'], {'video:frames': {'h3': False}, 'video:references': {'h3': True}})
        self.assertEqual(self.persist.call_count, 1)
        self.assertEqual(result, self.ns['_model_visibility_response']())

    def test_legacy_payload_preserves_overrides_and_explicit_reset_clears(self):
        self.put({'enabled_models': ['h3'], 'mode_overrides': {'video:frames': {'h3': False}}})
        self.assertEqual(self.put({'enabled_models': ['h3', 'other']})['mode_overrides'], {'video:frames': {'h3': False}})
        self.assertEqual(self.put({'enabled_models': ['h3'], 'mode_overrides': {}})['mode_overrides'], {})

    def test_legacy_config_hydrates_without_overrides(self):
        self.engine.server_config['visibility'] = {'enabled_models': ['h3'], 'defaults_version': 2}
        self.assertEqual(self.ns['_model_visibility_response']()['mode_overrides'], {})
        self.assertTrue(self.ns['_model_visibility_response']()['configured'])

    def test_corrupt_overrides_do_not_discard_existing_baseline(self):
        self.engine.server_config['visibility'] = {'enabled_models': ['h3'], 'mode_overrides': {'unknown': {'h3': True}}}
        result = self.ns['_model_visibility_response']()
        self.assertTrue(result['configured'])
        self.assertEqual(result['enabled_models'], ['h3'])
        self.assertEqual(result['mode_overrides'], {})

    def test_invalid_choices_rejected_without_mutating(self):
        self.put({'enabled_models': ['h3']})
        original = self.engine.server_config['visibility'].copy()
        for overrides in [[], {'unknown': {}}, {'video:frames': []}, {'video:frames': {'h3': 1}},
                          {'image:generate': {'__proto__': True}}, {'audio:music': {'x' * 201: True}}]:
            with self.subTest(overrides=overrides), self.assertRaises(HTTPException) as error:
                self.put({'enabled_models': [], 'mode_overrides': overrides})
            self.assertEqual(error.exception.status_code, 400)
            self.assertEqual(self.engine.server_config['visibility'], original)
        self.assertEqual(self.persist.call_count, 1)

    def test_all_scopes_and_entry_limit(self):
        scopes = ['video:frames', 'video:references', 'video:avatar', 'video:extend', 'video:blend', 'video:animate',
                  'video:retake', 'video:prompt_edit', 'video:outpaint', 'video:repaint', 'video:recast', 'video:inpaint',
                  'image:generate', 'image:inpaint', 'image:outpaint', 'audio:speech', 'audio:music', 'audio:sfx']
        result = self.put({'enabled_models': [], 'mode_overrides': {scope: {'h3': True} for scope in scopes}})
        self.assertEqual(len(result['mode_overrides']), len(scopes))
        with self.assertRaises(ValueError):
            self.ns['_normalize_model_visibility_overrides']({'video:frames': {str(i): True for i in range(5000)}, 'video:references': {'extra': False}})

if __name__ == '__main__': unittest.main()
