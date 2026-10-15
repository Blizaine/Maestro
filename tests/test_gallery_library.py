import ast
import json
import os
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'app'))

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

    def media(self, folder, name, *, stamp=100, params=None, metadata=None, content=b"media"):
        path = self.root / folder / name
        path.write_bytes(content)
        os.utime(path, (stamp, stamp))
        if params is not None or metadata is not None:
            path.with_suffix('.meta.json').write_text(json.dumps({
                "params": params or {}, "generation_mode": "video", **(metadata or {}),
            }), encoding="utf-8")
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

    def test_director_keeps_completed_shots_before_and_after_join(self):
        for index in range(9):
            self.media('default', f'shot-{index}.mp4', stamp=index + 1,
                       params={'prompt': f'Battle scene {index}', 'multi_clip_info': {
                           'group_id': 'director-run', 'index': index, 'total': 9,
                           'omni_sequence_continuity': False,
                       }}, metadata={'director_pipeline_id': 'pipeline-a',
                                     'director_artifact_kind': 'clip'})
            result = self.library.list(self.folders, media_filter='videos')
            self.assertEqual(result['total'], index + 1,
                             'Completed Director shots remain visible as later shots finish')
        self.media('default', 'final_multiclip.mp4', stamp=10,
                   metadata={'director_pipeline_id': 'pipeline-a',
                             'director_artifact_kind': 'joined'})
        result = self.library.list(self.folders, media_filter='videos', limit=4)
        self.assertEqual((result['total'], len(result['outputs'])), (10, 4))
        self.assertEqual(result['outputs'][0]['name'], 'final_multiclip.mp4')
        self.assertIsNotNone(result['next_cursor'])
        result = self.library.list(self.folders, favorites_only=True, search='battle',
                                   favorites=lambda _: {'shot-3.mp4'})
        self.assertEqual([o['name'] for o in result['outputs']], ['shot-3.mp4'])
        result = self.library.list(self.folders, multiclip_only=True)
        self.assertEqual([o['name'] for o in result['outputs']], ['final_multiclip.mp4'])

    def test_legacy_director_sidecars_keep_independent_shots(self):
        for index in range(2):
            self.media('default', f'old-shot-{index}.mp4', params={
                '_director_pipeline_id': 'legacy-pipeline',
                'multi_clip_info': {'group_id': 'old-run', 'index': index, 'total': 2},
            })
        self.assertEqual(self.library.list(self.folders)['total'], 2)

    def test_cumulative_director_and_studio_segments_still_collapse(self):
        for director in (False, True):
            with self.subTest(director=director):
                group = 'seamless-director' if director else 'studio'
                for index in range(2):
                    self.media('default', f'{group}-{index}.mp4', params={
                        'multi_clip_info': {'group_id': group, 'index': index, 'total': 3,
                                            'omni_sequence_continuity': director},
                    }, metadata=({'director_pipeline_id': 'pipeline-b',
                                  'director_artifact_kind': 'clip'} if director else None))
                result = self.library.list(self.folders)
                names = {o['name'] for o in result['outputs']}
                self.assertNotIn(f'{group}-0.mp4', names)
                self.assertIn(f'{group}-1.mp4', names)
                self.media('default', f'{group}-2.mp4', params={
                    'multi_clip_info': {'group_id': group, 'index': 2, 'total': 3,
                                        'omni_sequence_continuity': director},
                }, metadata=({'director_pipeline_id': 'pipeline-b',
                              'director_artifact_kind': 'clip'} if director else None))
                names = {o['name'] for o in self.library.list(self.folders)['outputs']}
                self.assertFalse(any(name.startswith(group) for name in names))

    def test_late_director_ownership_refreshes_collapsed_listing(self):
        path = self.media('default', 'shot.mp4', params={'multi_clip_info': {
            'group_id': 'late-run', 'index': 0, 'total': 1,
        }})
        self.assertEqual(self.library.list(self.folders)['total'], 0)
        sidecar = path.with_suffix('.meta.json')
        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        metadata.update(director_pipeline_id='pipeline-c', director_artifact_kind='clip')
        sidecar.write_text(json.dumps(metadata), encoding='utf-8')
        self.assertEqual(self.library.list(self.folders)['total'], 1)

    def director_media(self, folder, name, *, pid='pipeline-a', index=None, stamp=100, prompt=''):
        return self.media(folder, name, stamp=stamp, params={'prompt': prompt}, metadata={
            'director_pipeline_id': pid,
            'director_artifact_kind': 'joined' if index is None else 'clip',
            **({'director_clip_index': index} if index is not None else {}),
        })

    def test_director_groups_before_pagination_and_keeps_ungrouped_api(self):
        for index in range(9):
            self.director_media('default', f'shot-{index}.mp4', index=index, stamp=index + 10)
        self.director_media('default', 'final_multiclip.mp4', stamp=30)
        self.media('default', 'older.png', stamp=1)
        result = self.library.list(self.folders, group_director_clips=True, limit=1)
        self.assertEqual(result['total'], 2)
        self.assertEqual(result['outputs'][0]['director_group'], {'pipeline_id': 'pipeline-a', 'clip_count': 9})
        second = self.library.list(self.folders, group_director_clips=True, limit=1, cursor=result['next_cursor'])
        self.assertEqual([item['name'] for item in second['outputs']], ['older.png'])
        self.assertEqual(self.library.list(self.folders)['total'], 11, 'Editor and legacy callers retain individual files')

    def test_expansion_is_ordered_paged_and_workspace_qualified(self):
        for folder in ('default', 'Folder-B'):
            for index in range(4):
                self.director_media(folder, f'shot-{index}.mp4', index=index, stamp=20-index)
            self.director_media(folder, 'final_multiclip.mp4', stamp=30)
        first = self.library.list([self.folders[0]], director_pipeline_id='pipeline-a', limit=2)
        self.assertEqual(first['total'], 4)
        self.assertEqual([item['director_clip_index'] for item in first['outputs']], [0, 1])
        second = self.library.list([self.folders[0]], director_pipeline_id='pipeline-a', limit=2, cursor=first['next_cursor'])
        self.assertEqual([item['director_clip_index'] for item in second['outputs']], [2, 3])
        self.assertTrue(all(item['workspace'] == 'default' for item in first['outputs'] + second['outputs']))
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.library.list([self.folders[1]], director_pipeline_id='pipeline-a', cursor=first['next_cursor'])
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.library.list([self.folders[0]], group_director_clips=True, cursor=first['next_cursor'])
        grouped = self.library.list(self.folders, group_director_clips=True)
        self.assertEqual(grouped['total'], 2)
        self.assertEqual([item['director_group']['clip_count'] for item in grouped['outputs']], [4, 4])

    def test_search_favorites_and_unjoined_runs_do_not_hide_individual_shots(self):
        shot = self.director_media('default', 'shot.mp4', index=0, stamp=10, prompt='unique forest')
        self.assertEqual(self.library.list(self.folders, group_director_clips=True)['total'], 1)
        parent = self.director_media('default', 'final_multiclip.mp4', stamp=20)
        found = self.library.list(self.folders, group_director_clips=True, search='forest')
        self.assertEqual([item['name'] for item in found['outputs']], [shot.name])
        favorite = self.library.list(self.folders, group_director_clips=True, media_filter='favorites',
                                     favorites=lambda _: {shot.name})
        self.assertEqual([item['name'] for item in favorite['outputs']], [shot.name])
        parent.unlink()
        self.assertEqual(self.library.list(self.folders, group_director_clips=True)['total'], 1)

    def test_latest_join_owns_group_without_hiding_older_masters(self):
        self.director_media('default', 'shot.mp4', index=0, stamp=10)
        self.director_media('default', 'original_multiclip.mp4', stamp=20)
        self.director_media('default', 'rejoin_multiclip.mp4', stamp=30)
        result = self.library.list(self.folders, group_director_clips=True)
        self.assertEqual([item['name'] for item in result['outputs']], ['rejoin_multiclip.mp4', 'original_multiclip.mp4'])
        self.assertIn('director_group', result['outputs'][0])
        self.assertNotIn('director_group', result['outputs'][1])

    def test_studio_and_cumulative_renders_do_not_become_director_groups(self):
        self.director_media('default', 'final_multiclip.mp4', stamp=30)
        self.media('default', 'seamless.mp4', params={'multi_clip_info': {
            'group_id': 'cumulative', 'index': 0, 'total': 2, 'omni_sequence_continuity': True,
        }}, metadata={'director_pipeline_id': 'pipeline-a', 'director_artifact_kind': 'clip'})
        self.media('default', 'studio_multiclip.mp4', stamp=20)
        result = self.library.list(self.folders, group_director_clips=True)
        self.assertEqual(result['total'], 3)
        self.assertFalse(any('director_group' in item for item in result['outputs']))
        self.assertEqual(self.library.list(self.folders, director_pipeline_id='pipeline-a')['total'], 0)

    def test_legacy_clip_and_rejoin_ownership_recovers_from_saved_plan_read_only(self):
        shot = self.media('default', 'old-shot.mp4', metadata={'director_pipeline_id': 'legacy'})
        joined = self.media('default', 'old_rejoin_multiclip.mp4', stamp=200)
        plan = self.root / 'default' / '_director_pipeline_legacy.json'
        plan.write_text(json.dumps({'seamless': False, 'clips': [{'index': 0, 'video_filename': shot.name}],
                                    'output_files': [shot.name, joined.name]}), encoding='utf-8')
        original = shot.with_suffix('.meta.json').read_bytes()
        result = self.library.list(self.folders, group_director_clips=True)
        self.assertEqual([item['name'] for item in result['outputs']], [joined.name])
        self.assertEqual(result['outputs'][0]['director_group'], {'pipeline_id': 'legacy', 'clip_count': 1})
        with patch('app.services.gallery_library.json.load', side_effect=AssertionError('cache miss')):
            members = self.library.list(self.folders, director_pipeline_id='legacy')
        self.assertEqual([item['name'] for item in members['outputs']], [shot.name])
        self.assertEqual(shot.with_suffix('.meta.json').read_bytes(), original)
        self.assertFalse(joined.with_suffix('.meta.json').exists())
        plan.write_text(json.dumps({'clips': [], 'output_files': [joined.name]}), encoding='utf-8')
        self.assertEqual(self.library.list(self.folders, group_director_clips=True)['total'], 2,
                         'Changed plan ownership must not keep a stale hidden child')

    def test_malformed_legacy_plan_does_not_hide_media(self):
        self.media('default', 'shot.mp4', metadata={'director_pipeline_id': 'legacy'})
        self.media('default', 'old_multiclip.mp4')
        (self.root / 'default' / '_director_pipeline_legacy.json').write_text('{broken', encoding='utf-8')
        self.assertEqual(self.library.list(self.folders, group_director_clips=True)['total'], 2)

    def test_grouped_route_and_lazy_members_keep_requested_folder(self):
        source = (Path(__file__).parents[1] / 'app' / 'launch.py').read_text(encoding='utf-8')
        route = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'list_outputs')
        route.decorator_list = []
        namespace = {'_get_active_workspace': lambda: 'default',
                     '_gallery_directory': lambda workspace: str(self.root / workspace),
                     '_load_favorites': lambda workspace: set()}
        exec(compile(ast.Module(body=[route], type_ignores=[]), 'launch.py', 'exec'), namespace)
        for folder in ('default', 'Folder-B'):
            self.director_media(folder, 'shot.mp4', index=0)
            self.director_media(folder, 'final_multiclip.mp4', stamp=200)
        grouped = namespace['list_outputs'](workspace='Folder-B', group_director_clips=True, limit=1)
        self.assertEqual(grouped['total'], 1)
        self.assertEqual(grouped['outputs'][0]['workspace'], 'Folder-B')
        self.assertEqual(grouped['outputs'][0]['director_group']['clip_count'], 1)
        members = namespace['list_outputs'](workspace='Folder-B', director_pipeline_id='pipeline-a', limit=100)
        self.assertEqual([item['id'] for item in members['outputs']], ['Folder-B/shot.mp4'])

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
