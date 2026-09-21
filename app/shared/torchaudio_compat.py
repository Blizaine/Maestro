"""Legacy metadata adapters for Pyannote 3 on TorchAudio 2.9 and newer.

Only supply removed APIs. Existing audio loaders and supported TorchAudio
versions are left alone; Maestro passes decoded waveforms to diarization.
"""
from dataclasses import dataclass


@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def ensure_pyannote_audio_compat(module=None):
    if module is None:
        import torchaudio as module
    if not hasattr(module, 'AudioMetaData'):
        module.AudioMetaData = AudioMetaData
    if not hasattr(module, 'list_audio_backends'):
        module.list_audio_backends = lambda: ['soundfile']
    if not hasattr(module, 'info'):
        def info(uri, format=None, buffer_size=4096, backend=None):
            import soundfile as sf
            metadata = sf.info(uri)
            bits = {'PCM_U8': 8, 'PCM_S8': 8, 'PCM_16': 16, 'PCM_24': 24,
                    'PCM_32': 32, 'FLOAT': 32, 'DOUBLE': 64}.get(metadata.subtype, 0)
            encoding = ('PCM_F' if metadata.subtype in {'FLOAT', 'DOUBLE'} else
                        'PCM_U' if metadata.subtype == 'PCM_U8' else
                        'PCM_S' if metadata.subtype.startswith('PCM_') else metadata.subtype)
            return module.AudioMetaData(metadata.samplerate, metadata.frames, metadata.channels, bits, encoding)
        module.info = info
