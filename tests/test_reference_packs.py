"""Pack storage, approval, API and H3 identity contracts (no GPU/model downloads)."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from PIL import Image
from services import character_library as library, reference_packs as packs
from models.minimax_h3.reference_manifest import validate_reference_manifest


class ReferencePackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.cwd)
        Path('uploads').mkdir()
        self.images = {}
        for i, view in enumerate(packs.VIEWS):
            path = Path('uploads', view + '.png').resolve()
            Image.new('RGB', (64, 96), (i * 35, 60, 80)).save(path)
            self.images[view] = str(path)

    def saved(self, approve=True):
        result = packs.create_revision(name='Test person', images=self.images, label='Original')
        if approve:
            result = packs.review_revision(result['id'], result['reference_pack']['revisions'][0]['id'], True)
        return result

    def refs(self, saved, index=0):
        revision = saved['reference_pack']['revisions'][index]
        return [dict(type='image', path=image['path'], image_intent='identity',
                     role=saved['name'], character_name=saved['name'], library_character_id=saved['id'],
                     reference_pack_revision_id=revision['id'], reference_pack_view=view)
                for view, image in revision['images'].items()]

    def test_immutable_revision_and_explicit_approval(self):
        original = self.saved()
        refs = self.refs(original)
        validate_reference_manifest(refs)
        changed = Path('uploads/replacement.png').resolve()
        Image.new('RGB', (80, 96), 'red').save(changed)
        updated = packs.create_revision(character_id=original['id'], base_revision_id=refs[0]['reference_pack_revision_id'],
                                        images={'face_closeup': str(changed)}, label='New face')
        revisions = updated['reference_pack']['revisions']
        self.assertEqual(len(revisions), 2)
        self.assertTrue(revisions[0]['approved'])
        self.assertFalse(revisions[1]['approved'])
        self.assertEqual(revisions[1]['parent_revision_id'], revisions[0]['id'])
        for view in packs.VIEWS:
            self.assertNotEqual(revisions[0]['images'][view]['path'], revisions[1]['images'][view]['path'])
            self.assertEqual(revisions[1]['images'][view]['inherited'], view != 'face_closeup')
        with self.assertRaisesRegex(ValueError, 'approve'):
            validate_reference_manifest(self.refs(updated, 1))
        validate_reference_manifest(refs)  # new draft never changes active selection
        packs.review_revision(original['id'], revisions[0]['id'], False)
        with self.assertRaisesRegex(ValueError, 'approve'):
            validate_reference_manifest(refs)

    def test_atomic_import_rollback_and_path_boundary(self):
        with self.assertRaises(ValueError):
            packs.create_revision(name='Missing', images={'face_closeup': self.images['face_closeup']})
        self.assertEqual(library.list_characters(), [])
        invalid = {**self.images, 'full_body_back': '/etc/passwd'}
        with self.assertRaises(ValueError):
            packs.create_revision(name='Outside', images=invalid)
        with patch.object(library, '_save_index', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.saved(False)
        self.assertEqual(library.list_characters(), [])
        self.assertEqual(list(Path('uploads/characters').glob('*/packs/*')), [])

    def test_partial_mixed_and_modified_packs_rejected(self):
        saved = self.saved()
        refs = self.refs(saved)
        variants = [refs[:-1], refs + [dict(refs[0])]]
        for field, value in [('reference_pack_revision_id', 'other'), ('reference_pack_view', {}),
                             ('image_intent', 'scene'), ('remove_background', True), ('path', self.images['face_closeup'])]:
            changed = deepcopy(refs); changed[-1][field] = value; variants.append(changed)
        stripped = deepcopy(refs)
        for item in stripped:
            item.pop('reference_pack_revision_id'); item.pop('reference_pack_view')
        variants.append(stripped)
        for changed in variants:
            with self.subTest(changed=changed[-1]):
                with self.assertRaises(ValueError):
                    validate_reference_manifest(changed)
        Image.new('RGB', (64, 96), 'white').save(refs[0]['path'])
        with self.assertRaisesRegex(ValueError, 'match'):
            validate_reference_manifest(refs)

    def test_remaining_references_and_image_limits(self):
        refs = self.refs(self.saved())
        other = dict(type='image', path=self.images['face_closeup'], image_intent='scene', role='Park')
        result = validate_reference_manifest([other] + refs + [other] * 3)
        self.assertEqual(len(result), 9)
        with self.assertRaisesRegex(ValueError, '9 image'):
            validate_reference_manifest(refs + [other] * 5)

    def test_runtime_and_enhancer_use_same_single_subject_with_picture_offsets(self):
        from models.minimax_h3.ref2va import ensure_ref2va_prompt_relationships
        from services.h3_sequence_planner import _reference_context
        refs = self.refs(self.saved())
        scene = dict(type='image', path=self.images['face_closeup'], image_intent='scene', role='Park')
        other = dict(type='image', path=self.images['full_body_front'], role='Companion')
        voice = dict(type='audio', path='voice.wav', role='Test person', library_character_id=refs[0]['library_character_id'])
        references = [voice, scene] + refs + [other]
        relationships, retention, _ = _reference_context(references)
        raw = ensure_ref2va_prompt_relationships('Test person waves to Companion in the park.', references)
        for prompt in [relationships, raw]:
            self.assertEqual(prompt.count('ONE character jointly defined'), 1)
            self.assertIn('<Subject 1> is Test person', prompt)
            self.assertIn('<Subject 2> is Companion', prompt)
            self.assertIn('<Subject 3> is the environment', prompt)
            for number, label in enumerate(packs.VIEWS.values(), 2):
                self.assertIn(f'<Picture {number}>: {label}', prompt)
            self.assertIn('<Audio 1>', prompt)
            self.assertNotIn('<Subject 4>', prompt)
        self.assertEqual(retention.count('<Subject 1>'), 1)

    def test_actual_h3_media_preparation_keeps_all_five_pack_images(self):
        from models.minimax_h3.ref2va import prepare_references
        refs = self.refs(self.saved())
        scene = dict(type='image', path=self.images['face_closeup'], role='Park', image_intent='scene')
        prepared = prepare_references([scene] + refs, num_frames=24, target_height=64, target_width=64)
        self.assertEqual(len(prepared), 6)
        self.assertEqual(prepared[0].image_intent, 'scene')
        for i, reference in enumerate(prepared[1:]):
            self.assertEqual(reference.kind, 'image')
            self.assertEqual(reference.role, 'Test person')
            self.assertEqual(reference.image_intent, 'identity')
            self.assertEqual(reference.image.getpixel((reference.image.width // 2, reference.image.height // 2)), (i * 35, 60, 80))
        packs.review_revision(refs[0]['library_character_id'], refs[0]['reference_pack_revision_id'], False)
        with self.assertRaisesRegex(ValueError, 'approve'):
            prepare_references(refs, num_frames=24, target_height=64, target_width=64)

    def test_enhancement_manifest_retains_five_views_and_voice_binding(self):
        from services import llm_service
        from services.h3_sequence_planner import _reference_context
        from services.studio_enhancement import enhancement_request
        refs = self.refs(self.saved())
        refs.append(dict(type='audio', path='voice.wav', role='Test person', library_character_id=refs[0]['library_character_id']))
        relations, retention, _ = _reference_context(refs)
        context = relations + "\n" + retention
        manifest = llm_service._parse_h3_ref2va_subject_manifest(context)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]['pictures'], [f'<Picture {i}>' for i in range(1, 6)])
        self.assertEqual(manifest[0]['audios'], ['<Audio 1>'])
        source = 'Test person says, "Hello there."'
        draft = llm_service._build_h3_ref2va_tagged_fallback(source, context, duration_seconds=5)
        guide = (Path(__file__).resolve().parents[1] / 'app/services/llm_guides/enhance/minimax_h3_ref2va_video.md').read_text()
        with patch.object(llm_service, 'generate', return_value=draft) as generate, patch('services.enhance_guides.get_enhance_guide', return_value=guide):
            enhanced = llm_service.enhance_prompt(source, model_type='minimax_h3_ref2va', reference_context=context, duration_seconds=5)
        self.assertIn('ONE character and ONE Subject', generate.call_args.kwargs['system_prompt'])
        self.assertIn('<Subject 1> (S1)', enhanced)
        for i in range(1, 6):
            self.assertIn(f'<Picture {i}>', enhanced)
        self.assertNotIn('<Subject 2>', enhanced)
        self.assertIn('ONE character jointly defined', enhanced)
        payload, _ = enhancement_request(dict(prompt=source, minimax_h3_references=refs, model_type='minimax_h3_ref2va'),
                                         dict(architecture='minimax_h3', omni_reference=True))
        self.assertEqual(len(payload['minimax_h3_references']), 6)
        self.assertEqual(payload['image_paths'], [r['path'] for r in refs[:5]])

    def test_api_create_review_revision_media_and_errors(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI(); app.include_router(packs.create_router())
        with TestClient(app) as client:
            response = client.post('/api/v1/characters/reference-packs', json=dict(name='API person', images=self.images))
            self.assertEqual(response.status_code, 200, response.text)
            saved = response.json(); revision = saved['reference_pack']['revisions'][0]
            base = f"/api/v1/characters/{saved['id']}/reference-packs"
            self.assertEqual(client.get(revision['images']['face_closeup']['url']).status_code, 200)
            self.assertEqual(client.put(f"{base}/{revision['id']}/review", json={'approved': 'yes'}).status_code, 400)
            self.assertEqual(client.put(f"{base}/{revision['id']}/review", json={'approved': True}).status_code, 200)
            result = client.post(base, json=dict(base_revision_id=revision['id'], images={'face_closeup': self.images['face_closeup']}))
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(len(result.json()['reference_pack']['revisions']), 2)
            self.assertEqual(client.get(f'{base}/unknown/images/face_closeup').status_code, 404)


if __name__ == '__main__':
    unittest.main()
