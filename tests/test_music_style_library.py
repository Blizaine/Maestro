"""Music LoRA removal is reversible and keeps queued generation assets valid."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import music_styles
from services.music_training_api import create_router
from fastapi import FastAPI
from fastapi.testclient import TestClient


class MusicStyleLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        directory = self.root / 'existing'
        directory.mkdir()
        manifest = {'id': 'existing', 'version': 1, 'architecture': 'yue2',
                    'base_revision': music_styles.BASE_REVISION,
                    'tokenizer_revision': music_styles.TOKENIZER_REVISION,
                    'name': 'Saved singer', 'trigger': 'My singer'}
        for branch in ('ar', 'nar'):
            path = directory / f'{branch}.safetensors'
            path.write_bytes(branch.encode())
            manifest[branch] = {'file': path.name, 'sha256': music_styles.file_digest(path)}
        (directory / 'style.json').write_text(json.dumps(manifest))

    def test_remove_restore_preserves_weights_and_existing_job_lookup(self):
        before = music_styles.load_style('existing', root=self.root, verify=True)
        created = music_styles.list_styles(root=self.root)[0]['created_at']
        removed = music_styles.set_style_archived('existing', True, root=self.root)
        self.assertTrue(removed['archived'])
        self.assertEqual(music_styles.list_styles(root=self.root), [])
        self.assertEqual(len(music_styles.list_styles(root=self.root, include_archived=True)), 1)
        # Jobs hold an ID and must still resolve the exact old weights.
        queued = music_styles.load_style('existing', root=self.root, verify=True)
        self.assertEqual((queued['ar'], queued['nar']), (before['ar'], before['nar']))
        restored = music_styles.set_style_archived('existing', False, root=self.root)
        self.assertFalse(restored['archived'])
        self.assertEqual(music_styles.list_styles(root=self.root)[0]['created_at'], created)
        self.assertEqual(sorted(p.name for p in (self.root / 'existing').iterdir()),
                         ['ar.safetensors', 'nar.safetensors', 'style.json'])

    def test_bad_archive_values_and_paths_leave_library_unchanged(self):
        before = (self.root / 'existing/style.json').read_bytes()
        for value in (None, 'true', 1, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                music_styles.set_style_archived('existing', value, root=self.root)
        for name in ('../existing', '..', 'a/b', 'C:/outside'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                music_styles.set_style_archived(name, True, root=self.root)
        self.assertEqual((self.root / 'existing/style.json').read_bytes(), before)

    def test_api_hides_removed_by_default_and_can_restore_without_gpu_jobs(self):
        def no_submit(*args, **kwargs):
            self.fail('Library management must not start or interrupt GPU work')
        app = FastAPI()
        app.include_router(create_router(no_submit, {}))
        with patch.object(music_styles, 'STYLE_ROOT', self.root), TestClient(app) as client:
            response = client.patch('/api/v1/music-styles/existing', json={'archived': True})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get('/api/v1/music-styles').json()['styles'], [])
            rows = client.get('/api/v1/music-styles?include_archived=true').json()['styles']
            self.assertTrue(rows[0]['archived'])
            self.assertEqual(client.patch('/api/v1/music-styles/existing', json={'archived': 'yes'}).status_code, 400)
            self.assertEqual(client.patch('/api/v1/music-styles/existing', json={'name': 'changed'}).status_code, 400)
            self.assertEqual(client.patch('/api/v1/music-styles/missing', json={'archived': True}).status_code, 400)
            self.assertEqual(client.patch('/api/v1/music-styles/existing', json={'archived': False}).status_code, 200)
            self.assertEqual(client.get('/api/v1/music-styles').json()['styles'][0]['name'], 'Saved singer')

    def test_shortlist_is_opt_in_persistent_and_separate_from_archive(self):
        before = music_styles.load_style('existing', root=self.root, verify=True)
        self.assertFalse(music_styles.list_styles(root=self.root)[0]['in_selector'])
        created = music_styles.list_styles(root=self.root)[0]['created_at']
        music_styles.update_style_library('existing', {'in_selector': True}, root=self.root)
        self.assertTrue(music_styles.load_style('existing', root=self.root)['in_selector'])
        music_styles.set_style_archived('existing', True, root=self.root)
        music_styles.set_style_archived('existing', False, root=self.root)
        self.assertTrue(music_styles.list_styles(root=self.root)[0]['in_selector'])
        music_styles.update_style_library('existing', {'in_selector': False}, root=self.root)
        listed = music_styles.list_styles(root=self.root)
        self.assertEqual(len(listed), 1, 'Unlisted LoRAs remain in My music')
        self.assertFalse(listed[0]['in_selector'])
        self.assertEqual(listed[0]['created_at'], created)
        queued = music_styles.load_style('existing', root=self.root, verify=True)
        self.assertEqual((queued['ar'], queued['nar']), (before['ar'], before['nar']))
        self.assertNotIn('artist_id', queued, 'Listing is not generation activation')

    def test_api_shortlist_validation_and_queue_independence(self):
        def no_submit(*args, **kwargs):
            self.fail('Shortlisting must never submit GPU work')
        app = FastAPI()
        app.include_router(create_router(no_submit, {}))
        with patch.object(music_styles, 'STYLE_ROOT', self.root), TestClient(app) as client:
            self.assertEqual(client.patch('/api/v1/music-styles/existing', json={'in_selector': True}).status_code, 200)
            self.assertTrue(client.get('/api/v1/music-styles').json()['styles'][0]['in_selector'])
            before = (self.root / 'existing/style.json').read_bytes()
            for body in ({}, {'in_selector': 'false'}, {'in_selector': 1}, {'in_selector': None},
                         {'in_selector': True, 'ar': {}}, {'in_selector': False, 'archived': 'yes'}):
                with self.subTest(body=body):
                    self.assertEqual(client.patch('/api/v1/music-styles/existing', json=body).status_code, 400)
                    self.assertEqual((self.root / 'existing/style.json').read_bytes(), before)
            self.assertEqual(client.patch('/api/v1/music-styles/existing', json={'in_selector': False}).status_code, 200)
            self.assertFalse(client.get('/api/v1/music-styles').json()['styles'][0]['in_selector'])


if __name__ == '__main__':
    unittest.main()
