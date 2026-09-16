import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.gallery_library import GalleryLibrary


class GalleryLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folders = [(name, str(self.root / name)) for name in ("default", "Folder-B")]
        for _, path in self.folders:
            Path(path).mkdir()
        self.library = GalleryLibrary()

    def media(self, folder, name, *, stamp=100, params=None, content=b"media"):
        path = self.root / folder / name
        path.write_bytes(content)
        os.utime(path, (stamp, stamp))
        if params is not None:
            path.with_suffix('.meta.json').write_text(json.dumps({"params": params, "generation_mode": "video"}), encoding="utf-8")
        return path

    def test_same_name_origins_and_combined_filters(self):
        self.media("default", "same.mp4", params={"prompt": "Enhanced beach", "_image_original_prompt": "Old harbor"})
        self.media("Folder-B", "same.mp4", params={"prompt": "Enhanced mountain"})
        self.media("Folder-B", "same.flac", params={"prompt": "harbor music"})
        result = self.library.list(self.folders)
        self.assertEqual(result['total'], 3)
        self.assertEqual(len({o['id'] for o in result['outputs']}), 3)
        for item in result['outputs']:
            self.assertIn('workspace=' + item['workspace'], item['url'])
            self.assertEqual(Path(item['path']).parent.name, item['workspace'])
        result = self.library.list(self.folders, media_filter="videos", search="harbor", favorites_only=True,
                                   favorites=lambda folder: {"same.mp4"} if folder == "default" else set())
        self.assertEqual([o['id'] for o in result['outputs']], ['default/same.mp4'])
        result = self.library.list(self.folders, media_filter="audio")
        self.assertEqual([o['type'] for o in result['outputs']], ['audio'])

    def test_old_matches_are_filtered_before_paging(self):
        for i in range(213):
            self.media('default', f'{i:04}.png', stamp=i+1,
                       params={'prompt': 'old needle' if i < 103 else 'new image'})
        result = self.library.list(self.folders, limit=100, media_filter='images', search='needle')
        self.assertEqual((len(result['outputs']), result['total']), (100, 103))
        second = self.library.list(self.folders, limit=100, media_filter='images', search='needle', cursor=result['next_cursor'])
        self.assertEqual(len(second['outputs']), 3)
        self.assertIsNone(second['next_cursor'])
        self.assertFalse({o['id'] for o in result['outputs']} & {o['id'] for o in second['outputs']})

    def test_cursor_survives_new_output_and_deleted_previous_page(self):
        for i in range(6):
            self.media('default', f'{i}.png', stamp=i+1)
        first = self.library.list(self.folders, limit=2)
        (self.root / 'default' / '5.png').unlink()
        self.media('Folder-B', 'new.png', stamp=1000)
        second = self.library.list(self.folders, limit=2, cursor=first['next_cursor'])
        self.assertEqual([o['name'] for o in second['outputs']], ['3.png', '2.png'])
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.library.list(self.folders, media_filter='images', cursor=first['next_cursor'])

    def test_cached_search_tracks_late_changed_and_deleted_sidecars(self):
        path = self.media('default', 'draft.png')
        self.assertEqual(self.library.list(self.folders, search='garden')['total'], 0)
        meta = path.with_suffix('.meta.json')
        meta.write_text(json.dumps({'params': {'prompt': 'garden'}}), encoding='utf-8')
        self.assertEqual(self.library.list(self.folders, search='garden')['total'], 1)
        # An unchanged sidecar must not be reparsed on each keystroke.
        with patch('app.services.gallery_library.json.load', side_effect=AssertionError('cache miss')):
            self.assertEqual(self.library.list(self.folders, search='garden')['total'], 1)
        meta.write_text(json.dumps({'params': {'prompt': 'winter town'}}), encoding='utf-8')
        self.assertEqual(self.library.list(self.folders, search='garden')['total'], 0)
        self.assertEqual(self.library.list(self.folders, search='winter')['total'], 1)
        meta.unlink()
        self.assertEqual(self.library.list(self.folders, search='winter')['total'], 0)
        path.unlink()
        self.assertEqual(self.library.list(self.folders)['total'], 0)
        self.assertFalse(self.library._cache)

    def test_filters_compose_with_multiclip_and_corrupt_metadata(self):
        self.media('default', 'multiclip.mp4', params={'prompt': 'first city'})
        self.media('Folder-B', 'multiclip.mp4', params={'prompt': 'second city'})
        self.media('default', 'other.mp4', params={'multi_clip_info': {'index': None, 'total': 'broken'}})
        result = self.library.list(self.folders, multiclip_only=True, search='second', favorites_only=True,
                                   favorites=lambda _: {'multiclip.mp4'})
        self.assertEqual([o['id'] for o in result['outputs']], ['Folder-B/multiclip.mp4'])

    def test_qualified_metadata_reads_exact_folder(self):
        # Exercise the real route without importing/loading generation engines.
        source = (Path(__file__).parents[1] / 'app' / 'launch.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        route = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_output_metadata')
        route.decorator_list = []
        namespace = {'os': os, 'json': json, '_gallery_directory': lambda ws: str(self.root / ws),
                     '_safe_join': lambda folder, name: os.path.join(folder, name)}
        exec(compile(ast.Module(body=[route], type_ignores=[]), 'launch.py', 'exec'), namespace)
        self.media('default', 'same.png', params={'prompt': 'first', 'seed': 1})
        self.media('Folder-B', 'same.png', params={'prompt': 'second', 'seed': 2})
        self.assertEqual(namespace['get_output_metadata']('same.png', 'Folder-B')['params']['prompt'], 'second')
        self.assertEqual(namespace['get_output_metadata']('same.png', 'default')['params']['prompt'], 'first')


if __name__ == '__main__':
    unittest.main()
