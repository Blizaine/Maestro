"""Lightweight video codec settings shared by native saves and joins."""


def get_imageio_codec_params(codec_type, container):
    """Return the imageio-ffmpeg writer options for one saved-video setting."""
    if codec_type == 'libx264_8':
        return {'codec': 'libx264', 'quality': 8, 'pixelformat': 'yuv420p'}
    elif codec_type == 'libx264_10':
        return {'codec': 'libx264', 'quality': 10, 'pixelformat': 'yuv420p'}
    elif codec_type == 'libx265_28':
        return {
            'codec': 'libx265',
            'pixelformat': 'yuv420p',
            'output_params': [
                '-crf', '28', '-x265-params', 'log-level=none',
                '-hide_banner', '-nostats',
            ],
        }
    elif codec_type == 'libx265_8':
        return {
            'codec': 'libx265',
            'pixelformat': 'yuv420p',
            'output_params': [
                '-crf', '8', '-x265-params', 'log-level=none',
                '-hide_banner', '-nostats',
            ],
        }
    elif codec_type == 'libx264_lossless':
        if container == 'mkv':
            return {'codec': 'ffv1', 'pixelformat': 'rgb24'}
        else:  # mp4
            return {
                'codec': 'libx264',
                'output_params': ['-crf', '0'],
                'pixelformat': 'yuv444p',
            }
    else:  # Match save_video's compatibility fallback.
        return {'codec': 'libx264', 'pixelformat': 'yuv420p'}


def get_video_encoding(codec_type='libx264_8', container='mp4'):
    """Resolve a saved-video setting to its container and ffmpeg arguments.

    imageio-ffmpeg maps H.264 quality to CRF as
    ``int((1 - quality / 10) * 51)``. Keep joined videos on the same path as
    native saves, including the container-specific lossless format.
    """
    codec_type = codec_type or 'libx264_8'
    container = str(container or 'mp4').lower().lstrip('.')
    if container not in ('mp4', 'mkv'):
        raise ValueError(f"Unsupported video container: {container!r}")
    if codec_type not in (
        'libx264_8', 'libx264_10', 'libx264_lossless', 'libx265_8', 'libx265_28',
    ):
        raise ValueError(f"Unsupported video codec: {codec_type!r}")

    params = get_imageio_codec_params(codec_type, container)
    ffmpeg_args = [
        '-c:v', params['codec'],
        '-pix_fmt', params['pixelformat'],
    ]
    if 'quality' in params:
        # This matches imageio-ffmpeg's H.264 quality-to-CRF conversion.
        crf = int((1 - params['quality'] / 10.0) * 51)
        ffmpeg_args.extend(['-crf', str(crf)])
    ffmpeg_args.extend(params.get('output_params') or [])

    return {
        'codec_type': codec_type,
        'container': container,
        'extension': f'.{container}',
        'ffmpeg_args': ffmpeg_args,
    }
