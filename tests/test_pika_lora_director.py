"""Director adapter opt-in survives child jobs without importing the engine."""
import ast
import unittest
from pathlib import Path


class PikaLoraDirectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'app/services/director_pipeline.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == '_apply_director_h3_optimizations')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        cls.apply = staticmethod(namespace[function.name])

    def test_only_literal_opt_in_reaches_child(self):
        for value in (True, False, None, 'true', 1, 0):
            with self.subTest(value=value):
                params = {'num_inference_steps': 6, 'loras_multipliers': '1 0.65'}
                self.apply(params, {'minimax_h3_experimental_loras': value},
                           {'is_minimax_h3': True})
                self.assertIs(params['minimax_h3_experimental_loras'], value is True)
                self.assertEqual(params['num_inference_steps'], 6)
                self.assertEqual(params['loras_multipliers'], '1 0.65')

    def test_absent_opt_in_is_off(self):
        params = {}
        self.apply(params, {}, {'is_minimax_h3': True})
        self.assertIs(params['minimax_h3_experimental_loras'], False)

    def test_other_architectures_unchanged(self):
        params = {'other': True}
        self.apply(params, {'minimax_h3_experimental_loras': True}, {})
        self.assertEqual(params, {'other': True})


if __name__ == '__main__':
    unittest.main()
