"""Exact soundtrack performances must not become generated dialogue jobs."""
from copy import deepcopy
import ast
import asyncio
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from services import llm_service
from services.h3_performance_audio import (
    PERFORMANCE_AUDIO_DIRECTION, enforce_h3_performance_audio, has_h3_performance_audio,
)
from services.h3_sequence_planner import _reference_context, plan_h3_reference_sequence
from services.h3_story_ledger import (
    _canonicalize_story_ledger, _deterministic_ledger, _prepare_h3_story_context,
    extract_h3_source_intent, extract_source_events, ledger_violations,
)
from services.h3_prompt_budget import H3PromptBudgetError
from services.studio_enhancement import enhancement_context, enhancement_warnings, enhancement_request


SOURCE = (
    'Alex, the man from reference image 1, dances and sings to the reference music '
    'on the Cedar Park Lawn while people watch and cheer. '
    'Handheld phone-camera style with natural shake.'
)
REFERENCES = [
    {'type': 'image', 'role': 'Alex', 'path': 'alex.png'},
    {'type': 'audio', 'role': 'music', 'path': 'music.wav', 'audio_intent': 'drive'},
    {'type': 'image', 'role': 'Cedar Park Lawn', 'path': 'park.png', 'image_intent': 'scene'},
]


def context():
    return _reference_context(REFERENCES)[0]


def story(count=8):
    ledger = _deterministic_ledger(
        SOURCE, segment_count=count, segment_durations=[10.125] * count,
        locked_dialogue=[], camera_coverage='multi_shot', reference_context=context(),
    )
    # A writer may cite the single sustained activity on every window. Keep
    # the visual progression; record ownership once instead of rejecting it.
    ledger['beats'] = [{
        'beat_id': f'B{i}', 'segment': i, 'source_event_ids': ['E1'],
        'dialogue_ids': [], 'description': (
            'Alex, the man from reference image 1, dances and sings to the reference music '
            'on the Cedar Park Lawn while people watch and cheer. '
            f'He develops rhythmic phrase {i} with a new footwork pattern.'
        ),
        'state_after': 'Alex remains mid-dance with the crowd beside the lawn.',
        'sound_effects': 'N/A',
    } for i in range(1, count + 1)]
    return ledger


class PerformanceAudioTests(unittest.TestCase):
    def test_driver_is_distinct_from_voice_style_and_paired_video_audio(self):
        self.assertTrue(has_h3_performance_audio(context()))
        self.assertTrue(has_h3_performance_audio(
            'Exact target soundtrack: Music; intent=AUDIO REUSE / PERFORMANCE DRIVER; retention=fully_preserved'
        ))
        for reference in (
            {'type': 'audio', 'path': 'voice.wav', 'audio_intent': 'voice', 'role': 'Alex'},
            {'type': 'audio', 'path': 'style.wav', 'audio_intent': 'style', 'role': 'music'},
            {'type': 'video', 'path': 'clip.mp4', 'has_audio': True, 'role': 'clip'},
        ):
            with self.subTest(reference=reference):
                self.assertFalse(has_h3_performance_audio(_reference_context([reference])[0]))
        self.assertFalse(has_h3_performance_audio(
            '<Audio 1>: soundtrack paired with <Video 1>; intent=AUDIO REUSE / PERFORMANCE DRIVER'
        ))

    def test_camera_style_is_persistent_and_location_is_not_a_character(self):
        self.assertEqual(len(extract_source_events(SOURCE)), 1)
        intent = extract_h3_source_intent(SOURCE)
        self.assertNotIn('Cedar Park Lawn', intent['cast_names'])
        self.assertEqual(story()['source_intent']['cast_names'], ['Alex'])
        self.assertIn('Handheld phone-camera style', intent['perspective_contract'])
        self.assertIn('Handheld phone-camera style', intent['global_instructions'])
        for action in ('The phone camera reveals Alex behind a tree.',
                       'Alex drops the phone camera on the lawn.'):
            self.assertIn(action.rstrip('.'), extract_source_events(action)[0]['text'])

    def test_eight_window_performance_keeps_ai_plan_and_singing(self):
        calls = []

        def writer(**kwargs):
            calls.append(kwargs)
            fields = kwargs['json_schema']['properties']
            self.assertIn('EXACT SOUNDTRACK PERFORMANCE', kwargs['system_prompt'])
            if 'beats' in fields:
                self.assertNotIn('spoken words', kwargs['prompt'])
                return json.dumps(story())
            number = fields['segment']['minimum']
            events = json.loads(kwargs['prompt'].split(
                'Assigned chronological events (depict each once, in order):\n', 1,
            )[1].split('\n\nImmutable dialogue performances', 1)[0])
            return json.dumps({
                'segment': number, 'title': 'The performance continues',
                'coverage': 'Handheld phone camera facing Alex across the lawn and cheering crowd.',
                'pacing': 'Keep time with the supplied music',
                'event_cards': {f'event_{i}': {'phases': [{
                    'action': event['staging_draft'], 'framing': 'Medium full shot of Alex',
                    'camera': 'Handheld phone camera follows his footwork with natural shake',
                    'transition': 'continuous reframe', 'sound_effects': 'Crowd cheers',
                }]} for i, event in enumerate(events, start=1)},
                'closing_state': 'Alex balances on one foot with the crowd beside the lawn.',
            })

        with patch.object(llm_service, 'generate', side_effect=writer):
            plan = plan_h3_reference_sequence(
                SOURCE, model_type='minimax_h3_ref2va_fused_turbo', resolution='704x1280',
                total_frames=1620, min_clip_frames=5, max_clip_frames=243,
                frame_step=1, fps=24, references=REFERENCES,
                native_continuation=True, overlap_frames=18, planning_style='adaptive',
            )
        self.assertEqual(plan['window_count'], 8)
        self.assertEqual(plan['planned_by'], 'llm')
        self.assertEqual(plan['planning_warnings'], [])
        self.assertEqual(plan['planning_diagnostics'], [])
        self.assertEqual(len(calls), 9)  # Story and eight camera calls, no dialogue repairs.
        self.assertEqual(plan['source_intent']['cast_names'], ['Alex'])
        self.assertEqual(plan['story_ledger']['generated_dialogue'], [])
        for prompt in plan['window_prompts']:
            self.assertIn('lip movement', prompt)
            self.assertNotIn('<d>', prompt)
            self.assertNotIn('remain silent', prompt)
            self.assertNotIn('No words are spoken or mouthed', prompt)
            self.assertNotIn('Silent visual action', prompt)
            self.assertNotIn('exactly one <Subject 2>', prompt)
            self.assertNotIn('prompt-native recurring character', prompt)
            self.assertIn('original vocals, music and timing', prompt)
            self.assertNotIn('Synchronized practical sound: Crowd cheers', prompt)

    def test_missing_real_plot_event_still_fails_visual_checks(self):
        source = 'Alex dances to the music. Alex switches off the lights.'
        canonical = _deterministic_ledger(
            source, segment_count=2, segment_durations=[10, 10], locked_dialogue=[],
            camera_coverage='auto', reference_context=context(),
        )
        broken = story(2)
        normalized = _canonicalize_story_ledger(
            source, canonical, broken, locked_dialogue=[], segment_count=2,
            preserve_adaptation=True,
        )
        errors = ledger_violations(source, normalized, segment_count=2,
                                   locked_dialogue=[], expect_dialogue=False)
        self.assertIn('source event IDs are missing, foreign, or repeated', errors)

    def test_quoted_lyrics_have_no_generated_speech_clock(self):
        source = 'Alex sings, "' + 'Keep moving to the music ' * 6 + '"'
        response = story(1)
        response['beats'][0]['description'] = 'Alex performs the supplied song.'
        response['beats'][0]['dialogue_ids'] = ['D1']  # Ignore an invented transcript binding.
        prepared = _prepare_h3_story_context(
            source, segment_durations=[3], mode='reference_sequence', camera_coverage='auto',
            reference_context=context(), expect_dialogue=True, planning_style='adaptive',
            image_paths=None, has_start_image=False, nsfw=False,
            llm_generate=lambda **kwargs: json.dumps(response),
        )
        self.assertFalse(prepared['allow_generated_dialogue'])
        self.assertEqual(prepared['locked_dialogue'], [])
        self.assertEqual(prepared['catalog'], [])
        self.assertEqual(prepared['dialogue_fragments'], [])
        self.assertEqual(prepared['planning_warnings'], [])

    def test_single_window_enhance_now_and_queued_skip_speech_repairs(self):
        source = 'Alex sings, "' + 'Keep moving to the music ' * 6 + '"'
        payload, sequence = enhancement_request({
            'prompt': source, 'model_type': 'minimax_h3_ref2va_fused_turbo',
            'video_length': 124, 'minimax_h3_references': REFERENCES,
        }, {'architecture': 'minimax_h3_ref2va', 'omni_reference': True, 'fps': 24})
        self.assertFalse(sequence)
        interactive_context = (
            '<Picture 1>: Alex; intent=IDENTITY REFERENCE; retention=fully_preserved\n'
            'Exact target soundtrack: music; intent=AUDIO REUSE / PERFORMANCE DRIVER; retention=fully_preserved'
        )
        valid = (
            'subject_definitions: Alex.\nsummary: [reference generation + audio reuse] Alex performs.\n'
            'retention_analysis: N/A\ndetailed_description: Realistic phone footage. '
            '[Shot 1] Alex dances and sings to the supplied audio on the lawn.\n'
            'overall_soundscape: A new score and cheering.\nnon_diegetic_music: A new orchestral song.'
        )
        for ref_context in (payload['reference_context'], interactive_context):
            with self.subTest(context=ref_context), enhancement_context({}, lambda: False), patch.object(
                llm_service, 'generate', return_value=valid,
            ) as writer, patch.object(
                llm_service, 'validate_h3_source_dialogue_duration', side_effect=AssertionError('speech gate'),
            ), patch('services.adaptive_enhancement.draft_spoken_exchange', side_effect=AssertionError('dialogue writing')), patch(
                'services.enhance_guides.get_enhance_guide', return_value='H3 reference guide',
            ):
                result = llm_service.enhance_prompt(
                    source, model_type='minimax_h3_ref2va_fused_turbo', duration_seconds=3,
                    planning_style='adaptive', reference_context=ref_context,
                )
                self.assertEqual(writer.call_count, 1)
                self.assertEqual(enhancement_warnings(), [])
                self.assertIn('EXACT SOUNDTRACK PERFORMANCE', writer.call_args.kwargs['system_prompt'])
                self.assertIn('original vocals, music and timing', result)
                self.assertNotIn('orchestral', result)
                self.assertNotIn('<d>', result)
                self.assertNotIn('closed mouths', result)
                self.assertNotIn('no human voices', result)

    def test_voice_and_style_samples_retain_dialogue_duration_checks(self):
        source = 'Alex says, "' + 'This is a very long sentence. ' * 6 + '"'
        for intent in ('voice', 'style'):
            references = deepcopy(REFERENCES)
            references[1]['audio_intent'] = intent
            with self.subTest(intent=intent), patch.object(llm_service, 'generate') as writer:
                with self.assertRaises(H3PromptBudgetError):
                    llm_service.enhance_prompt(source, duration_seconds=3,
                        model_type='minimax_h3_ref2va_fused_turbo', planning_style='adaptive',
                        reference_context=_reference_context(references)[0])
                writer.assert_not_called()

    def test_invalid_visual_structure_still_requires_review_with_target_audio(self):
        with enhancement_context({}, lambda: False), patch.object(
            llm_service, 'generate', return_value='An incomplete response',
        ), patch('services.enhance_guides.get_enhance_guide', return_value='H3 reference guide'):
            result = llm_service.enhance_prompt(
                SOURCE, model_type='minimax_h3_ref2va_fused_turbo', duration_seconds=10,
                planning_style='adaptive', reference_context=context(),
            )
            self.assertTrue(enhancement_warnings())
            self.assertIn('valid H3 reference prompt', enhancement_warnings()[0])
            self.assertNotIn('remain silent', result)
            self.assertNotIn('mouths remain closed', result)
            self.assertEqual(result.count(PERFORMANCE_AUDIO_DIRECTION), 1)
            self.assertEqual(enforce_h3_performance_audio(result), result)

    def test_api_preflight_skips_speech_gate_only_for_target_audio(self):
        # Load the real API function without starting the diffusion engine.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app/launch.py').read_text(encoding='utf-8'))
        node = deepcopy(next(item for item in tree.body if isinstance(item, ast.AsyncFunctionDef)
                             and item.name == '_llm_enhance_prompt_payload'))
        node.decorator_list = []

        class ReachedModelLoad(Exception):
            pass

        def stop_before_loading():
            raise ReachedModelLoad()

        namespace = {
            'wgp': SimpleNamespace(server_config={}, get_model_def=lambda _: {}),
            'enhancement_settings': lambda services: services, '_PUBLIC_LLM_PROVIDERS': set(),
            '_ensure_llm_loaded': stop_before_loading,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'launch.py', 'exec'), namespace)
        for drive in (True, False):
            with self.subTest(drive=drive), patch.object(
                llm_service, 'validate_h3_source_dialogue_duration', side_effect=RuntimeError('speech gate'),
            ) as gate:
                with self.assertRaises(ReachedModelLoad if drive else RuntimeError):
                    asyncio.run(namespace['_llm_enhance_prompt_payload']({
                        'prompt': SOURCE, 'model_type': 'minimax_h3_ref2va_fused_turbo',
                        'duration_seconds': 10, 'reference_context': context() if drive else '',
                    }))
                self.assertEqual(gate.call_count, 0 if drive else 1)


if __name__ == '__main__':
    unittest.main()
