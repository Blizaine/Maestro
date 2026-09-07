"""Run real URL-import handlers with provider requests and downloads isolated."""
from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid


class Response(dict):
    def __init__(self, content, status_code=200):
        super().__init__(content)
        self.status_code = status_code


class TestLoraUrlImport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app_dir = Path(self.temp.name) / 'app'
        self.app_dir.mkdir()
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app/launch.py').read_text(encoding='utf-8'))
        names = {'hf_import_lora', '_safe_join', '_is_safe_path_component',
                 '_hf_disk_filename', '_is_minimax_h3_identity', '_new_download_record'}
        constants = {'HF_BASE_TO_LOCAL_DIR', '_GENERIC_HF_LORA_FILENAMES'}
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                node.decorator_list = []
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
                nodes.append(node)
        self.repo = {'siblings': [{'rfilename': 'film.safetensors'}],
                     'cardData': {'base_model': 'Lightricks/LTX-2.3'}}
        self.http = SimpleNamespace(get=Mock(side_effect=self.provider_response), RequestException=OSError)
        self.namespace = {
            'os': os, 'PureWindowsPath': PureWindowsPath, 'time': time,
            'threading': threading, 'uuid': uuid, 'json': json,
            'Request': object, 'JSONResponse': Response,
            '__file__': str(self.app_dir / 'launch.py'), 'requests': self.http,
            'wgp': SimpleNamespace(server_config={}),
            '_civitai_downloads': {}, '_civitai_download_lock': threading.Lock(),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app/launch.py', 'exec'), self.namespace)
        self.worker = patch('threading.Thread').start()
        self.addCleanup(patch.stopall)

    def provider_response(self, url, **kwargs):
        if '/api/models/' in url:
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: self.repo)
        if url.endswith('/README.md'):
            return SimpleNamespace(ok=False)
        self.fail(f'Unexpected provider request: {url}')

    def run_import(self, destination='', url='https://huggingface.co/creator/Film'):
        async def body():
            return {'url': url, 'target_dir': destination}
        return asyncio.run(self.namespace['hf_import_lora'](SimpleNamespace(json=body)))

    def assert_destination(self, result, expected):
        self.assertEqual(result['status'], 'downloading', result)
        record = self.namespace['_civitai_downloads'][result['download_id']]
        self.assertEqual(os.path.normcase(record['target_dir']), os.path.normcase(str(expected.resolve())))
        self.assertTrue(expected.is_dir())
        self.assertEqual(list(expected.iterdir()), [])  # No weights downloaded by these tests.
        self.worker.return_value.start.assert_called_once()

    def test_manual_folder_beats_conflicting_provider_metadata(self):
        self.assert_destination(self.run_import('minimax_h3'), self.app_dir / 'loras/minimax_h3')

    def test_direct_file_url_does_not_pick_first_file_in_collection(self):
        self.repo['siblings'] = [{'rfilename': 'Other.safetensors'}, {'rfilename': 'people/Blaine refmod.safetensors'}]
        result = self.run_import('minimax_h3', 'https://huggingface.co/author/collection/blob/main/people/Blaine%20refmod.safetensors')
        record = self.namespace['_civitai_downloads'][result['download_id']]
        self.assertEqual(record['filename'], 'Blaine refmod.safetensors')

    def test_missing_direct_file_fails_instead_of_importing_another_character(self):
        result = self.run_import('', 'https://huggingface.co/author/collection/blob/main/Missing.safetensors')
        self.assertEqual(result.status_code, 400)
        self.worker.return_value.start.assert_not_called()

    def test_absolute_custom_root(self):
        custom = Path(self.temp.name) / 'Custom LoRAs'
        self.namespace['wgp'].server_config['loras_root'] = str(custom)
        self.assert_destination(self.run_import('minimax_h3'), custom / 'minimax_h3')
        self.assertFalse((self.app_dir / 'loras').exists())

    def test_relative_custom_root(self):
        self.namespace['wgp'].server_config['loras_root'] = 'custom_loras'
        self.assert_destination(self.run_import('my_adapters'), self.app_dir / 'custom_loras/my_adapters')

    def test_auto_still_uses_repository_metadata(self):
        self.repo['cardData']['base_model'] = 'MiniMaxAI/MiniMax-H3'
        result = self.run_import()
        self.assert_destination(result, self.app_dir / 'loras/minimax_h3')

    def test_manual_folder_with_missing_metadata(self):
        self.repo['cardData'] = None
        self.assert_destination(self.run_import('minimax_h3'), self.app_dir / 'loras/minimax_h3')

    def test_invalid_folders_rejected_before_provider_requests(self):
        for directory in ['../outside', r'..\outside', 'C:\\outside', 'nested/folder', 'NUL', ['invalid']]:
            with self.subTest(directory=directory):
                result = self.run_import(directory)
                self.assertEqual(result.status_code, 400)
        self.http.get.assert_not_called()
        self.worker.assert_not_called()

    def test_civitai_dispatch_preserves_manual_folder(self):
        dispatch = Mock(return_value=Response({'status': 'downloading'}))
        self.namespace['_import_civitai_lora_by_url'] = dispatch
        url = 'https://civitai.com/models/123'
        self.run_import('minimax_h3', url)
        dispatch.assert_called_once_with(url, 'minimax_h3')
        self.http.get.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
