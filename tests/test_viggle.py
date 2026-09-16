"""Viggle's immutable recipe and control-window conditioning contracts."""
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import torch
import numpy as np
from PIL import Image
from models.minimax_h3 import viggle
from models.minimax_h3.minimax_h3_handler import family_handler
from models.minimax_h3.scheduler import MiniMaxH3Scheduler


class ViggleTests(unittest.TestCase):
    def test_discovery_and_no_text_encoder_download(self):
        self.assertIn('viggle_animate', family_handler.query_supported_types())
        model = family_handler.query_model_def('viggle_animate', {})
        self.assertFalse(model['omni_reference'])
        self.assertTrue(model['minimax_h3_viggle'])
        self.assertEqual(model['text_encoder_URLs'], [])
        self.assertEqual(model['sliding_window_defaults']['window_max'], 124)
        self.assertTrue(model['extract_guide_from_window_start'])
        assets = family_handler.query_model_files(None, 'viggle_animate', model)
        self.assertNotIn('processor', str(assets))
        self.assertIn(viggle.REVISION, str(assets))
        self.assertIn(viggle.PROMPT_FILE, str(assets))

    def test_stale_settings_become_three_step_recipe_without_shortening_timeline(self):
        settings = dict(video_guide='source.mp4', image_refs=['edit.png'], video_length=1000,
            num_inference_steps=30, sliding_window_size=345, video_prompt_type='GV',
            prompt='This text must not condition Viggle', image_start='old.png',
            override_attention='sol',
            audio_prompt_type='K', custom_settings={'audio_refinement': 'enabled'})
        viggle.normalize_settings(settings)
        self.assertEqual(settings['video_length'], 1000)
        self.assertEqual(settings['sliding_window_size'], 124)
        self.assertEqual(settings['sliding_window_overlap'], 18)
        self.assertEqual(settings['num_inference_steps'], 3)
        self.assertEqual(settings['flow_shift'], 3)
        self.assertEqual(settings['video_prompt_type'], 'IVU')
        self.assertIsNone(settings['image_start'])
        self.assertEqual(settings['image_refs'], ['edit.png'])
        self.assertEqual(settings['audio_prompt_type'], 'K')
        self.assertEqual(settings['custom_settings'], {})
        self.assertEqual(settings['override_attention'], '')
        again = dict(settings)
        viggle.normalize_settings(again)
        self.assertEqual(again, settings)

    def test_missing_inputs_and_invalid_durations(self):
        for patch_values in ({'video_guide': ''}, {'image_refs': []}, {'image_refs': ['a.png', 'b.png']},
                             {'video_length': float('nan')}, {'video_length': 86401},
                             {'audio_prompt_type': 'A'}, {'audio_prompt_type': 'AB'}):
            settings = dict(video_guide='source.mp4', image_refs=['edit.png'], video_length=124)
            settings.update(patch_values)
            with self.subTest(patch_values=patch_values), self.assertRaises(ValueError):
                viggle.normalize_settings(settings)

    def test_legacy_audio_migration_and_conditioning_device(self):
        settings = {'audio_prompt_type': 'KV', 'resolution': 'auto480p'}
        viggle.normalize_settings(settings, validate_media=False)
        self.assertEqual(settings['audio_prompt_type'], 'K')
        self.assertGreaterEqual(settings['settings_version'], 2.58)
        self.assertEqual(settings['resolution'], 'auto_480p')
        tensors = {'prompt_embeds': torch.zeros(1, 362, 5120, dtype=torch.bfloat16),
                   'text_token_tags': torch.zeros(362, dtype=torch.int64)}
        with patch('shared.utils.files_locator.locate_file', return_value='fixed.safetensors'), \
             patch('safetensors.torch.load_file', return_value=tensors):
            conditioner = viggle.load_conditioner()
        embeds, tags = conditioner.forward_ref2va('', torch.device('meta'), [])
        self.assertEqual(embeds.device, tags.device)
        self.assertEqual(tags.device.type, 'meta')

    def test_control_window_order_and_history_offset(self):
        video = torch.linspace(-1, 1, 124)[None, :, None, None].expand(3, -1, 32, 64)
        edited = Image.new('RGB', (64, 32), 'red')
        for history, count in ((0, 124), (17, 107)):
            refs = viggle.prepare_window_references(video, [edited], history, 124, 32, 64)
            self.assertEqual([ref.kind for ref in refs], ['video', 'image'])
            self.assertEqual(refs[0].frames.shape, (count, 32, 64, 3))
            self.assertEqual(refs[0].frames[0, 0, 0, 0], round(history / 123 * 255))
            self.assertFalse(refs[0].has_audio)
            self.assertEqual(refs[1].image.size, (64, 32))

    def test_last_window_padding_and_invalid_reference_count(self):
        video = torch.zeros(3, 90, 32, 64)
        image = Image.new('RGB', (64, 32))
        refs = viggle.prepare_window_references(video, [image], 17, 124, 32, 64)
        self.assertEqual(len(refs[0].frames), 107)
        with self.assertRaises(ValueError):
            viggle.prepare_window_references(video, [], 0, 124, 32, 64)

    def test_bounded_pixel_conversion_preserves_existing_frames_and_input(self):
        from models.minimax_h3.minimax_h3_main import _prepare_control_video_tensor

        rng = torch.Generator().manual_seed(371)
        full_range = torch.rand(3, 90, 12, 18, generator=rng).mul(2.4).sub(1.2)
        # Range detection must cover the whole source, not each individual
        # chunk: later chunks here contain only values in [0, 1].
        mixed_range = torch.rand(3, 90, 12, 18, generator=rng)
        mixed_range[:, 0] = -0.5
        unit_range = torch.rand(3, 90, 12, 18, generator=rng)
        uint8 = torch.randint(0, 256, (3, 90, 12, 18), generator=rng, dtype=torch.uint8)
        for source in (full_range, mixed_range, unit_range, uint8, full_range.half(),
                       full_range[:, :, :, ::2]):
            original = source.clone()
            for history in (0, 17, 102):
                for size in ((12, source.shape[-1]), (16, 24)):
                    with self.subTest(dtype=source.dtype, history=history, size=size):
                        expected = _prepare_control_video_tensor(source, *size)
                        expected = torch.cat([expected, expected[:, -1:].expand(
                            -1, 124 - expected.shape[1], -1, -1)], dim=1)
                        expected = expected[:, history:124].add(1).mul(127.5).round().clamp(0, 255)
                        expected = expected.permute(1, 2, 3, 0).numpy().astype(np.uint8)
                        actual = viggle._window_pixels(source, history, 124, *size, buffer_bytes=8192)
                        np.testing.assert_array_equal(actual, expected)
                        torch.testing.assert_close(source, original, rtol=0, atol=0)

    def test_pixel_resize_scratch_stays_bounded_as_window_grows(self):
        import torch.nn.functional as F

        source = torch.zeros(3, 124, 32, 48)
        frame_counts = []
        resize = F.interpolate
        def record(input, *args, **kwargs):
            frame_counts.append(input.shape[0])
            return resize(input, *args, **kwargs)

        with patch.object(F, 'interpolate', side_effect=record):
            result = viggle._window_pixels(source, 0, 124, 64, 96, buffer_bytes=256 * 1024)
        self.assertEqual(result.shape, (124, 64, 96, 3))
        self.assertGreater(len(frame_counts), 1)
        self.assertLessEqual(max(frame_counts) * 3 * 4 * (32 * 48 + 64 * 96), 256 * 1024)
        np.testing.assert_array_equal(result, np.zeros(result.shape, dtype=np.uint8))

    def test_euler_three_evaluations_matches_published_sigma_grid(self):
        scheduler = MiniMaxH3Scheduler(shift=3.0, solver='euler')
        scheduler.set_timesteps(4, device='cpu')
        self.assertEqual(len(scheduler.timesteps), 3)
        torch.testing.assert_close(scheduler.sigmas, torch.tensor([1., 6/7, .6, 0.]))


if __name__ == '__main__': unittest.main()
