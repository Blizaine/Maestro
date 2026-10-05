"""CPU-only contract tests for TinyVAE previews and their bounded session."""
import io
import hashlib
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from shared.tinyvae.decoder import TinyVAE, frame_indices
from shared.tinyvae.media import VideoPreview, encode_video
from shared.tinyvae.session import PreviewSession
import shared.tinyvae.decoder as decoder_module
import shared.tinyvae.session as session_module


def test_decoder_download_uses_maestro_helper_signature_and_checks_weight_hash(tmp_path, monkeypatch):
    from shared.utils import files_locator as fl
    checkpoint = tmp_path / 'preview_decoders' / 'test.safetensors'
    data = b'checked-preview-weights'
    monkeypatch.setitem(decoder_module.REGISTRY['decoders'], 'test', {
        'filename': checkpoint.name, 'repo': 'owner/models', 'sha256': hashlib.sha256(data).hexdigest(),
    })
    monkeypatch.setattr(fl, 'locate_file', lambda *args, **kwargs: None)
    monkeypatch.setattr(fl, 'get_smart_download_location', lambda *args: str(checkpoint))
    def download(url, path):
        Path(path).write_bytes(data)
    with mock.patch('shared.utils.download.download_file', autospec=True, side_effect=download) as downloader:
        assert decoder_module.prepare_decoder('test', gen={}) == str(checkpoint)
        downloader.assert_called_once_with('https://huggingface.co/owner/models/resolve/main/preview_decoders/test.safetensors',
                                          str(checkpoint))
    checkpoint.write_bytes(b'bad-checksum')
    monkeypatch.setattr(fl, 'locate_file', lambda *args, **kwargs: str(checkpoint))
    with pytest.raises(ValueError, match='checksum mismatch'):
        decoder_module.prepare_decoder('test')


def test_decoder_download_honors_abort_before_and_after_local_helper(tmp_path, monkeypatch):
    from shared.utils import files_locator as fl
    gen = {'abort': True}
    with mock.patch('shared.utils.download.download_file', autospec=True) as downloader:
        with pytest.raises(RuntimeError, match='cancelled'):
            decoder_module.prepare_decoder('taeh3', gen)
        downloader.assert_not_called()
    checkpoint = tmp_path / 'preview_decoders' / 'taeh3.safetensors'
    monkeypatch.setattr(fl, 'locate_file', lambda *args, **kwargs: None)
    monkeypatch.setattr(fl, 'get_smart_download_location', lambda *args: str(checkpoint))
    gen['abort'] = False
    def cancel_download(url, path):
        gen['abort'] = True
    with mock.patch('shared.utils.download.download_file', autospec=True, side_effect=cancel_download):
        with pytest.raises(RuntimeError, match='cancelled'):
            decoder_module.prepare_decoder('taeh3', gen)


class _PixelDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, value):
        self.seen.append(float(value[0, 0, 0, 0]))
        return value[:, :3]


class _TemporalDecoder(torch.nn.Module):
    t_upscale = 1

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen_indices = None

    def decode_video(self, value, *, output_indices, abort_check, output_transform, parallel):
        assert not parallel
        if abort_check is not None and abort_check():
            return None
        self.seen_indices = list(output_indices)
        return output_transform(value[0, output_indices, :3])[None]


class _RecordingDecoder:
    def __init__(self, image=None, video=None):
        self.image_result = image or Image.new("RGB", (8, 8), "red")
        self.video_result = video or ([Image.new("RGB", (8, 8), "blue")], 12.0)
        self.calls = []

    def __call__(self, latent, **kwargs):
        self.calls.append(kwargs)
        return self.video_result if kwargs["video"] else self.image_result


def _events():
    values = []
    return values, lambda *args: values.append(args)


def test_image_decoder_builds_a_four_frame_strip_from_bounded_samples():
    assert frame_indices(10) == [0, 2, 5, 7]
    net = _PixelDecoder()
    decoder = TinyVAE("taef1", net)
    latents = torch.zeros((16, 10, 2, 2))
    latents[0, :, :, :] = torch.arange(10, dtype=torch.float32)[:, None, None]

    preview = decoder(latents, image=False, video=False)

    assert net.seen == [0.0, 2.0, 5.0, 7.0]
    assert preview.size == (800, 200)


def test_video_decoder_caps_samples_dimensions_and_rate(monkeypatch):
    requested_sizes = []

    def cheap_interpolate(value, size, **kwargs):
        requested_sizes.append(size)
        return value.new_zeros((value.shape[0], value.shape[1], 2, 2))

    monkeypatch.setattr(decoder_module.F, "interpolate", cheap_interpolate)
    net = _TemporalDecoder()
    decoder = TinyVAE("taew2_1", net)
    latents = torch.zeros((16, 300, 10, 20))

    frames, fps = decoder(latents, image=False, video=True, duration=20)

    assert len(frames) == 240
    assert len(net.seen_indices) == 240
    assert net.seen_indices[0] == 0
    assert net.seen_indices[-1] == 299
    assert fps == 12
    assert requested_sizes
    assert all(height <= 200 and width <= 384 for height, width in requested_sizes)


def test_invalid_video_duration_and_image_mode_disable_video_encoding():
    events, send_cmd = _events()
    session = PreviewSession(_RecordingDecoder(), send_cmd, {}, image=True, video=True, duration=0)

    assert session.video is False
    assert session.executor is None
    session.capture(None, 0, 1, 0)
    assert events == [("preview", session.decoder.image_result)]
    session.close()


def test_capture_cadence_and_due_check_are_side_effect_free():
    events, send_cmd = _events()
    decoder = _RecordingDecoder()
    session = PreviewSession(decoder, send_cmd, {}, image=True)

    assert session.wants_capture(0, 30, 0) is True
    assert session.last_step == -1
    due_steps = []
    for step in range(30):
        if session.wants_capture(step, 30, 0):
            due_steps.append(step)
        session.capture(None, step, 30, 0)

    assert due_steps == [0, 5, 10, 15, 20, 25, 29]
    assert len(decoder.calls) == 7
    assert len(events) == 7
    assert session.wants_capture(0, 30, 1) is True
    session.close()


def test_encoder_busy_skips_intermediate_work_and_allows_final_capture(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    encoded = VideoPreview(Image.new("RGB", (8, 8), "green"), b"mp4")

    def slow_encode(frames, fps, cancelled):
        entered.set()
        release.wait(timeout=5)
        return None if cancelled() else encoded

    monkeypatch.setattr(session_module, "encode_video", slow_encode)
    events, send_cmd = _events()
    decoder = _RecordingDecoder()
    session = PreviewSession(decoder, send_cmd, {}, image=False, video=True, duration=1)

    session.capture(None, 0, 20, 0)
    assert entered.wait(timeout=2)
    first_future = session.future

    assert session.wants_capture(4, 20, 0) is False
    session.capture(None, 4, 20, 0)
    assert session.future is first_future
    assert len(decoder.calls) == 1

    assert session.wants_capture(19, 20, 0) is True
    release.set()
    session.capture(None, 19, 20, 0)
    session.close()

    assert len(decoder.calls) == 2
    assert [kind for kind, _ in events] == ["preview", "preview"]


def test_decode_and_encode_errors_are_swallowed_and_noticed_once(monkeypatch):
    events, send_cmd = _events()

    def broken_decoder(*args, **kwargs):
        raise RuntimeError("optional decoder failed")

    session = PreviewSession(broken_decoder, send_cmd, {}, image=True)
    session.capture(None, 0, 1, 0)
    session.capture(None, 0, 1, 0)
    assert session.failed is True
    assert [kind for kind, _ in events] == ["preview_notice"]
    session.close()

    events, send_cmd = _events()
    monkeypatch.setattr(session_module, "encode_video", lambda *args: (_ for _ in ()).throw(RuntimeError("encode failed")))
    video_session = PreviewSession(_RecordingDecoder(), send_cmd, {}, image=False, video=True, duration=1)
    video_session.capture(None, 0, 2, 0)
    video_session.close()

    assert video_session.failed is True
    assert [kind for kind, _ in events] == ["preview_notice"]


def test_future_error_is_swallowed_and_noticed_once():
    events, send_cmd = _events()
    session = PreviewSession(_RecordingDecoder(), send_cmd, {}, image=False, video=True, duration=1)
    failed_future = Future()
    failed_future.set_exception(RuntimeError("future failed"))
    session.future = failed_future

    session.close()
    session.close()

    assert session.failed is True
    assert session.closed is True
    assert [kind for kind, _ in events] == ["preview_notice"]


def test_cancelled_decode_cannot_publish_after_close():
    started = threading.Event()
    release = threading.Event()
    events, send_cmd = _events()

    class BlockingDecoder:
        def __call__(self, latent, **kwargs):
            started.set()
            release.wait(timeout=5)
            return Image.new("RGB", (8, 8), "red")

    session = PreviewSession(BlockingDecoder(), send_cmd, {}, image=True)
    worker = threading.Thread(target=lambda: session.capture(None, 0, 1, 0))
    worker.start()
    assert started.wait(timeout=2)

    session.close(cancel=True)
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert events == []


def test_stale_video_encoder_does_not_publish_old_context(monkeypatch):
    old_entered = threading.Event()
    cancelled_old = threading.Event()
    result = VideoPreview(Image.new("RGB", (8, 8), "green"), b"mp4")
    count = {"value": 0}

    def encode(frames, fps, cancelled):
        count["value"] += 1
        if count["value"] == 1:
            old_entered.set()
            while not cancelled():
                threading.Event().wait(0.005)
            cancelled_old.set()
            return None
        return result

    monkeypatch.setattr(session_module, "encode_video", encode)
    events, send_cmd = _events()
    session = PreviewSession(_RecordingDecoder(), send_cmd, {}, image=False, video=True, duration=1)

    session.capture(None, 0, 5, "old")
    assert old_entered.wait(timeout=2)
    session.capture(None, 4, 5, "new")
    session.close()

    assert cancelled_old.is_set()
    assert len(events) == 1
    assert events[0] == ("preview", result)


def test_close_swallows_shutdown_errors_and_notifies_once():
    events, send_cmd = _events()
    session = PreviewSession(_RecordingDecoder(), send_cmd, {}, image=False, video=True, duration=1)

    class BrokenShutdown:
        def shutdown(self, **kwargs):
            raise RuntimeError("shutdown failed")

    session.executor.shutdown(wait=False, cancel_futures=True)
    session.executor = BrokenShutdown()
    session.close()

    assert session.failed is True
    assert session.closed is True
    assert [kind for kind, _ in events] == ["preview_notice"]
    session.close()
    assert [kind for kind, _ in events] == ["preview_notice"]


def test_pyav_encodes_a_tiny_bounded_mp4_when_libx264_is_available():
    av = pytest.importorskip("av")
    try:
        av.Codec("libx264", "w")
    except Exception:
        pytest.skip("PyAV is available but libx264 is not")

    result = encode_video(
        [Image.new("RGB", (16, 16), (255, 0, 0)), Image.new("RGB", (16, 16), (0, 255, 0))],
        12,
        lambda: False,
    )

    assert isinstance(result, VideoPreview)
    assert result.video
    with av.open(io.BytesIO(result.video), mode="r") as container:
        assert len(list(container.decode(video=0))) == 2
