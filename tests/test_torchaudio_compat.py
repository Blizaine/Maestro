"""Restore removed metadata APIs without replacing existing audio behavior."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from shared.torchaudio_compat import ensure_pyannote_audio_compat


class AudioCompatibilityTests(unittest.TestCase):
    def test_existing_functions_and_types_are_untouched(self):
        original = {name: object() for name in ('AudioMetaData', 'info', 'list_audio_backends', 'load', 'save')}
        module = SimpleNamespace(**original)
        ensure_pyannote_audio_compat(module)
        for key, value in original.items():
            self.assertIs(getattr(module, key), value)

    def test_removed_metadata_apis_report_real_soundfile_properties(self):
        module = SimpleNamespace(load=object())
        ensure_pyannote_audio_compat(module)
        with patch.dict(sys.modules, {'soundfile': SimpleNamespace(info=lambda uri: SimpleNamespace(
                samplerate=48000, frames=4800, channels=2, subtype='PCM_16'))}):
            value = module.info('recording.wav', backend='soundfile')
        self.assertEqual((value.sample_rate, value.num_frames, value.num_channels, value.bits_per_sample, value.encoding),
                         (48000, 4800, 2, 16, 'PCM_S'))
        self.assertEqual(module.list_audio_backends(), ['soundfile'])
        first = module.info
        ensure_pyannote_audio_compat(module)
        self.assertIs(first, module.info)


if __name__ == '__main__':
    unittest.main()
