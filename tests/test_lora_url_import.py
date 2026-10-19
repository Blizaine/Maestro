"""Run real URL-import handlers with provider requests and downloads isolated."""
from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))

from services.download_control import create_download_control, download_scope


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
        tree = ast.parse((ROOT / 'app/launch.py').read_text(encoding='utf-8'))
        names = {'hf_import_lora', '_safe_join', '_is_safe_path_component',
                 '_hf_disk_filename', '_is_minimax_h3_identity', '_is_qwen21_identity',
                 '_new_download_record', '_civitai_lora_arch', '_import_civitai_lora_by_url',
                 '_update_download_record', '_start_import_download_worker', '_fail_download_record',
                 'civitai_download', '_is_safe_civitai_url', 'civitai_model_detail',
                 'civitai_search', 'civitai_base_models'}
        constants = {'HF_BASE_TO_LOCAL_DIR', '_GENERIC_HF_LORA_FILENAMES',
                     'CIVIT_TO_LOCAL_ARCH', 'CIVITAI_MODEL_FILTERS', '_CIVITAI_ALLOWED_HOSTS'}
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                node.decorator_list = []
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
                nodes.append(node)
        self.repo = {'siblings': [{'rfilename': 'film.safetensors'}],
                     'cardData': {'base_model': 'Lightricks/LTX-2.3'}}
        self.civit_model = {'id': 123, 'name': 'Film LoRA', 'type': 'LORA', 'modelVersions': [
            {'id': 21, 'name': 'Qwen 2.1', 'baseModel': 'Qwen 2',
             'files': [{'primary': True, 'name': 'new.safetensors',
                        'downloadUrl': 'https://civitai.com/api/download/models/21'}]},
            {'id': 20, 'name': 'Qwen 2512', 'baseModel': 'Qwen',
             'files': [{'primary': True, 'name': 'old.safetensors',
                        'downloadUrl': 'https://civitai.com/api/download/models/20'}]},
        ]}
        self.http = SimpleNamespace(
            get=Mock(side_effect=self.provider_response),
            RequestException=requests.RequestException,
            Timeout=requests.Timeout,
        )
        lora_dir = Mock(side_effect=lambda arch: str(self.app_dir / 'loras' / {
            'qwen_image_21_7B': 'qwen21', 'qwen_image_20B': 'qwen',
            'flux3_klein': 'flux3_klein',
            'moss_image': 'moss_image', 'krea2': 'krea2',
        }[arch]))
        self.namespace = {
            'os': os, 'PureWindowsPath': PureWindowsPath, 'time': time,
            'threading': threading, 'uuid': uuid, 'json': json,
            'Request': object, 'JSONResponse': Response,
            '__file__': str(self.app_dir / 'launch.py'), 'requests': self.http,
            'create_download_control': create_download_control, 'download_scope': download_scope,
            'wgp': SimpleNamespace(server_config={}, get_lora_dir=lora_dir),
            '_civitai_downloads': {}, '_civitai_download_lock': threading.Lock(),
            'CIVITAI_BASE_URL': 'https://civitai.com/api/v1',
            '_civitai_headers': lambda: {}, '_civitai_cache_get': lambda key: None,
            '_civitai_cache_put': Mock(), '_fix_civitai_images': Mock(),
            '_run_civitai_download': Mock(),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app/launch.py', 'exec'), self.namespace)
        self.worker = patch('threading.Thread').start()
        self.addCleanup(patch.stopall)

    def provider_response(self, url, **kwargs):
        if url.startswith('https://civitai.com/api/v1/models'):
            data = self.civit_model if url.endswith('/123') else {
                'items': [self.civit_model], 'metadata': {'nextCursor': 'next-page'},
            }
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: data)
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
        if record.get('_model_id') == self.civit_model['id']:
            selected = next(version for version in self.civit_model['modelVersions']
                            if version['id'] == record['_version_id'])
            self.assertEqual(record['_version_name'], selected['name'])
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

    def test_qwen21_hf_import_uses_its_own_library(self):
        self.repo['cardData']['base_model'] = 'Qwen/Qwen-Image-2.1'
        self.assert_destination(self.run_import(), self.app_dir / 'loras/qwen21')

    def test_qwen21_repo_name_without_base_metadata(self):
        self.repo['cardData'] = None
        self.assert_destination(
            self.run_import(url='https://huggingface.co/creator/Qwen2.1-Image-Film'),
            self.app_dir / 'loras/qwen21',
        )

    def test_older_qwen_metadata_is_not_mistaken_for_qwen21(self):
        self.repo['cardData']['base_model'] = 'Qwen/Qwen-Image-Edit-2511'
        self.assert_destination(self.run_import(), self.app_dir / 'loras/qwen')
        detect = self.namespace['_is_qwen21_identity']
        for identity in ['Qwen-Image-2512', 'Qwen-Image-v2.1', 'Qwen2.5', 'Qwen-Image-2.10']:
            self.assertFalse(detect(identity), identity)

    def test_qwen21_filter_uses_civitai_category_and_keeps_pagination(self):
        filters = self.namespace['civitai_base_models']()['filters']
        selected = next(f for f in filters if f['label'] == 'Qwen Image 2.1')
        result = self.namespace['civitai_search'](baseModels=selected['civitai_base'], cursor='page-2')
        params = self.http.get.call_args.kwargs['params']
        self.assertEqual(params['baseModels'], 'Qwen 2')
        self.assertEqual(params['cursor'], 'page-2')
        self.assertEqual(params['types'], 'LORA')
        self.assertEqual(result['metadata']['nextCursor'], 'next-page')
        self.assertEqual(selected['default_dir'], 'qwen21')

    def test_flux3_klein_filter_uses_civitai_lora_labels_and_keeps_pagination(self):
        filters = self.namespace['civitai_base_models']()['filters']
        selected = next(f for f in filters if f['label'] == 'Flux.3 Klein')
        self.assertEqual(selected['civitai_base'], 'Flux.3 Klein,Flux.3 Klein-base')
        self.assertEqual(selected['default_dir'], 'flux3_klein')
        self.assertNotIn('Flux.3', [
            base.strip()
            for entry in filters
            for base in entry['civitai_base'].split(',')
        ])

        result = self.namespace['civitai_search'](
            baseModels=selected['civitai_base'], cursor='page-2',
        )
        params = self.http.get.call_args.kwargs['params']
        self.assertEqual(params['baseModels'], ['Flux.3 Klein', 'Flux.3 Klein-base'])
        self.assertEqual(params['cursor'], 'page-2')
        self.assertEqual(params['types'], 'LORA')
        self.assertEqual(result['metadata']['nextCursor'], 'next-page')

    def test_flux3_klein_civitai_versions_map_to_shared_native_model_key(self):
        variants = ['Flux.3 Klein', 'Flux.3 Klein-base']
        arch = self.namespace['_civitai_lora_arch']
        for base_model in variants:
            with self.subTest(base_model=base_model):
                self.assertEqual(arch(base_model), 'flux3_klein')
        # CivitAI's plain Flux.3 entry is checkpoint-only and must not be
        # admitted to the LoRA browser or auto-import mapping.
        self.assertEqual(arch('Flux.3'), '')

        self.civit_model['modelVersions'] = [
            {'id': 31, 'name': 'Klein Base', 'baseModel': variants[1], 'files': []},
            {'id': 30, 'name': 'Klein Distilled', 'baseModel': variants[0], 'files': []},
        ]
        versions = self.namespace['civitai_model_detail'](123)['modelVersions']
        self.assertEqual([version['localArch'] for version in versions], ['flux3_klein', 'flux3_klein'])

    def test_civitai_flux3_klein_import_routes_both_variants_to_native_lora_directory(self):
        variants = ['Flux.3 Klein', 'Flux.3 Klein-base']
        for index, base_model in enumerate(variants, start=30):
            with self.subTest(base_model=base_model):
                self.civit_model['modelVersions'] = [{
                    'id': index, 'name': f'Klein {index}', 'baseModel': base_model,
                    'files': [{'primary': True, 'name': 'film.safetensors',
                               'downloadUrl': f'https://civitai.com/api/download/models/{index}'}],
                }]
                result = self.namespace['_import_civitai_lora_by_url']('https://civitai.com/models/123')
                self.assertEqual(result['status'], 'downloading', result)
                record = self.namespace['_civitai_downloads'][result['download_id']]
                self.assertEqual(record['_base_model'], base_model)
                self.assertEqual(
                    os.path.normcase(record['target_dir']),
                    os.path.normcase(str((self.app_dir / 'loras/flux3_klein').resolve())),
                )
                self.assertEqual(list((self.app_dir / 'loras/flux3_klein').iterdir()), [])
                self.worker.return_value.start.assert_called_once()
                self.namespace['_civitai_downloads'].clear()
                self.worker.return_value.start.reset_mock()

    def test_flux3_huggingface_base_repositories_use_native_lora_directory(self):
        repos = [
            'black-forest-labs/flux-3-image-klein',
            'black-forest-labs/flux-3-image-klein-base',
        ]
        for repo_id in repos:
            with self.subTest(repo_id=repo_id):
                self.assertEqual(self.namespace['HF_BASE_TO_LOCAL_DIR'][repo_id], 'flux3_klein')

        self.repo['cardData']['base_model'] = repos[1]
        self.assert_destination(self.run_import(), self.app_dir / 'loras/flux3_klein')

    def test_model_detail_maps_each_version_separately(self):
        versions = self.namespace['civitai_model_detail'](123)['modelVersions']
        self.assertEqual([v['localArch'] for v in versions], ['qwen_image_21_7B', 'qwen_image_20B'])

    def test_moss_browser_filter_searches_base_and_sft_with_pagination(self):
        selected = next(f for f in self.namespace['civitai_base_models']()['filters'] if f['label'] == 'MOSS Image')
        result = self.namespace['civitai_search'](baseModels=selected['civitai_base'], cursor='moss-page-2')
        params = self.http.get.call_args.kwargs['params']
        self.assertEqual(params['baseModels'], ['MOSS Image', 'MOSS Image Base'])
        self.assertEqual(params['cursor'], 'moss-page-2')
        self.assertEqual(result['metadata']['nextCursor'], 'next-page')
        self.assertEqual(selected['default_dir'], 'moss_image')

    def test_moss_base_and_sft_civitai_imports_use_native_lora_folder(self):
        for base in ['MOSS Image', 'MOSS Image Base']:
            with self.subTest(base=base):
                self.civit_model['modelVersions'] = [{
                    'id': 40, 'name': 'MOSS adapter', 'baseModel': base,
                    'files': [{'primary': True, 'name': 'moss.safetensors',
                               'downloadUrl': 'https://civitai.com/api/download/models/40'}],
                }]
                self.assertEqual(self.namespace['_civitai_lora_arch'](base), 'moss_image')
                result = self.namespace['_import_civitai_lora_by_url']('https://civitai.com/models/123')
                self.assert_destination(result, self.app_dir / 'loras/moss_image')
                self.namespace['_civitai_downloads'].clear()
                self.worker.return_value.start.reset_mock()

    def test_moss_huggingface_base_metadata_routes_to_native_lora_folder(self):
        for base in ['OpenMOSS-Team/MOSS-Image-Base', 'OpenMOSS-Team/MOSS-Image-SFT']:
            with self.subTest(base=base):
                self.repo['cardData']['base_model'] = base
                self.assert_destination(self.run_import(), self.app_dir / 'loras/moss_image')
                self.namespace['_civitai_downloads'].clear()
                self.worker.return_value.start.reset_mock()

    def test_krea_turbo_filter_and_version_metadata_use_shared_lora_folder(self):
        filters = self.namespace['civitai_base_models']()['filters']
        combined = next(f for f in filters if f['label'] == 'Krea 2')
        turbo = next(f for f in filters if f['label'] == 'Krea 2 Turbo')
        self.namespace['civitai_search'](baseModels=combined['civitai_base'])
        self.assertEqual(self.http.get.call_args.kwargs['params']['baseModels'], ['Krea 2', 'Krea 2 Turbo'])
        self.assertEqual(turbo['civitai_base'], 'Krea 2 Turbo')
        self.civit_model['modelVersions'] = [{
            'id': 41, 'name': 'Turbo adapter', 'baseModel': 'Krea 2 Turbo',
            'files': [{'primary': True, 'name': 'krea.safetensors',
                       'downloadUrl': 'https://civitai.com/api/download/models/41'}],
        }]
        result = self.namespace['_import_civitai_lora_by_url']('https://civitai.com/models/123')
        self.assert_destination(result, self.app_dir / 'loras/krea2')

    def test_civitai_url_import_routes_new_qwen(self):
        result = self.namespace['_import_civitai_lora_by_url']('https://civitai.com/models/123')
        self.assert_destination(result, self.app_dir / 'loras/qwen21')

    def test_civitai_version_id_beats_model_title(self):
        self.civit_model['name'] = 'Film - Qwen Image 2.1'
        result = self.namespace['_import_civitai_lora_by_url'](
            'https://civitai.com/models/123/film-qwen-2.1?modelVersionId=20',
        )
        self.assert_destination(result, self.app_dir / 'loras/qwen')

    def test_civitai_browser_download_uses_version_over_stale_arch(self):
        async def body():
            return {'download_url': 'https://civitai.com/api/download/models/21',
                    'filename': 'film.safetensors', 'base_model': 'Qwen 2',
                    'target_arch': 'qwen_image_20B'}
        result = asyncio.run(self.namespace['civitai_download'](SimpleNamespace(json=body)))
        self.assert_destination(result, self.app_dir / 'loras/qwen21')

    def test_qwen21_manual_destination_still_wins(self):
        self.repo['cardData']['base_model'] = 'Qwen/Qwen-Image-2.1'
        self.assert_destination(self.run_import('my_adapters'), self.app_dir / 'loras/my_adapters')


if __name__ == '__main__':
    unittest.main(verbosity=2)
