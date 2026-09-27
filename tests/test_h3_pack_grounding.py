"""Five-view vision, immutable identity mapping and enhancement integration."""
import json
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import h3_pack_grounding as grounding, llm_service
from services.reference_packs import VIEWS


class PackGroundingTests(TestCase):
    def setUp(self):
        self.context = '<Subject 2> is Alex, ONE character jointly defined by five views: ' + '; '.join(
            f'<Picture {n}>: {label}' for n, label in enumerate(VIEWS.values(), 2)
        ) + '. The close-up defines facial identity.\n<Subject 2>: Fully retained.'
        self.paths = [f'image-{n}.png' for n in range(1, 7)]
        self.observations = {f'picture_{n}': f'Visible detail from view {n}.' for n in range(2, 7)}
        self.observations.update(identity_summary='Short auburn hair, square glasses, green jacket and dark trousers.', inconsistencies='')
        self.generate = Mock(return_value=json.dumps(self.observations))

    def test_five_views_keep_offsets_and_grounded_identity_through_repairs(self):
        context, warning = grounding.ground_context(self.generate, self.context, self.paths, vision_available=True)
        call = self.generate.call_args.kwargs
        self.assertEqual(call['image_paths'], self.paths[1:])
        for number, label in enumerate(call['image_labels'], 2):
            self.assertIn(f'<Picture {number}>', label)
            self.assertIn('<Subject 2>', label)
        self.assertIsNone(warning)
        source = 'Alex walks through a garden.'
        draft = llm_service._build_h3_ref2va_tagged_fallback(source, context, duration_seconds=5)
        repaired = llm_service._canonicalize_h3_ref2va_reference_fields(draft, context, source)
        self.assertIn(self.observations['identity_summary'], repaired)
        self.assertNotIn('picture_2', repaired)
        clean = grounding.binding_context(context)
        self.assertIn(self.observations['identity_summary'], clean)
        self.assertNotIn('picture_2', clean)

    def test_text_only_does_not_claim_inspection(self):
        context, warning = grounding.ground_context(self.generate, self.context, self.paths, vision_available=False)
        self.assertEqual(context, self.context)
        self.generate.assert_not_called()
        self.assertIn('not inspected', warning)

    def test_incomplete_or_control_injected_observations_fail_visibly(self):
        variants = ['not JSON', json.dumps({}), json.dumps({**self.observations, 'picture_3': ''}),
                    json.dumps({**self.observations, 'identity_summary': '<Subject 7> is someone else.'})]
        for response in variants:
            with self.subTest(response=response):
                self.generate.return_value = response
                with self.assertRaisesRegex(ValueError, 'inspection was incomplete'):
                    grounding.ground_context(self.generate, self.context, self.paths, vision_available=True)

    def test_missing_attachment_cannot_shift_picture_numbers(self):
        with self.assertRaisesRegex(ValueError, 'numbering'):
            grounding.ground_context(self.generate, self.context, self.paths[:-1], vision_available=True)
        self.generate.assert_not_called()
        with patch.object(llm_service, '_vision_available', True), patch.object(llm_service, '_image_to_data_url', side_effect=['data:image/jpeg;base64,YQ==', None]):
            with self.assertRaisesRegex(ValueError, 'could not be attached'):
                llm_service._user_message('prompt', ['one', 'two'], ['<Picture 1>', '<Picture 2>'])

    def test_captions_are_adjacent_to_actual_image_bytes_for_both_providers(self):
        labels = grounding.image_labels(self.context, self.paths)
        with patch.object(llm_service, '_vision_available', True), patch.object(llm_service, '_image_to_data_url', side_effect=lambda p: 'data:image/jpeg;base64,'+p):
            message = llm_service._user_message('prompt', self.paths, labels)
        content = message['content']
        for i, path in enumerate(self.paths):
            self.assertEqual(content[i*2], dict(type='text', text=labels[i]))
            self.assertTrue(content[i*2+1]['image_url']['url'].endswith(path))
        _, messages = llm_service._anthropic_messages([message])
        for i, path in enumerate(self.paths):
            self.assertEqual(messages[0]['content'][i*2]['text'], labels[i])
            self.assertEqual(messages[0]['content'][i*2+1]['source']['data'], path)

    def test_actual_enhancer_inspects_then_writes_with_all_labels(self):
        source = 'Alex walks through a garden.'
        draft = llm_service._build_h3_ref2va_tagged_fallback(source, self.context, duration_seconds=5)
        def generate(**kwargs):
            if kwargs.get('json_schema'):
                return json.dumps(self.observations)
            self.assertIn(self.observations['identity_summary'], kwargs['prompt'])
            self.assertEqual(kwargs['image_paths'], self.paths)
            self.assertEqual(len(kwargs['image_labels']), 6)
            return draft
        guide = (Path(__file__).resolve().parents[1] / 'app/services/llm_guides/enhance/minimax_h3_ref2va_video.md').read_text()
        with patch.object(llm_service, '_vision_available', True), patch.object(llm_service, 'generate', side_effect=generate) as calls, patch('services.enhance_guides.get_enhance_guide', return_value=guide):
            result = llm_service.enhance_prompt(source, model_type='minimax_h3_ref2va', reference_context=self.context, image_paths=self.paths, duration_seconds=5)
        self.assertGreaterEqual(calls.call_count, 2)
        self.assertIn(self.observations['identity_summary'], result)
        self.assertNotIn('picture_2', result)
