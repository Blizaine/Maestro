"""CPU-only behavior tests for optional generation previews."""
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.generation_preview import (
    DEFAULT_PREVIEW_MODE,
    PREVIEW_MODES,
    GenerationPreviewCache,
    PreviewMedia,
    configured_preview_mode,
    preview_mode,
    preview_response,
)
from shared.preview_runtime import RGBPreviewSession, decoder_key, prepare_preview_decoder
import shared.preview_runtime as preview_runtime
import shared.tinyvae.decoder as tinyvae_decoder


def _context(*, serial=1, window=1, total_windows=2, clip=1, total_clips=3, mode="rgb"):
    return {
        "serial": serial,
        "window": window,
        "total_windows": total_windows,
        "clip": clip,
        "total_clips": total_clips,
        "mode": mode,
    }


def _video(data):
    return SimpleNamespace(video=data, image=Image.new("RGB", (4, 4), "black"))


@pytest.mark.parametrize("value", PREVIEW_MODES)
def test_preview_mode_accepts_only_supported_values(value):
    assert preview_mode(value) == value


@pytest.mark.parametrize("value", [None, "", "RGB", "tinyvae", 1, object()])
def test_preview_mode_normalizes_invalid_values_to_off(value):
    assert preview_mode(value) == "off"


def test_missing_config_defaults_to_live_video_but_preserves_explicit_modes():
    assert DEFAULT_PREVIEW_MODE == "tiny_vae_video"
    assert configured_preview_mode(None) == "tiny_vae_video"
    assert configured_preview_mode("off") == "off"
    assert configured_preview_mode("tiny_vae_frames") == "tiny_vae_frames"
    assert configured_preview_mode("invalid") == "off"


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [("serial", 2), ("window", 2), ("clip", 2)],
)
def test_late_window_clip_or_serial_media_cannot_replace_current_preview(field, changed_value):
    cache = GenerationPreviewCache()
    cache.begin("job")
    old_context = _context(serial=1, window=1, clip=1, mode="tiny_vae_video")
    current_context = dict(old_context, **{field: changed_value})
    cache.set_context("job", old_context)
    prior = cache.publish("job", _video(b"old-video"), old_context)
    assert prior is not None

    cache.set_context("job", current_context)

    assert cache.fields({"id": "job", "status": "running"})["preview"] is None
    assert cache.publish("job", _video(b"late-old-video"), old_context) is None
    current = cache.publish("job", Image.new("RGB", (8, 8), "red"), current_context)
    assert current is not None
    assert current["window"] == current_context["window"]
    assert current["clip"] == current_context["clip"]
    assert cache.fields({"id": "job", "status": "running"})["preview"] == current


def test_rgb_fallback_can_replace_tiny_media_for_the_same_callback():
    cache = GenerationPreviewCache()
    cache.begin("job")
    tiny_context = _context(serial=8, mode="tiny_vae_video")
    rgb_context = dict(tiny_context, mode="rgb")
    cache.set_context("job", tiny_context)

    tiny = cache.publish("job", _video(b"tiny"), tiny_context)
    rgb = cache.publish("job", Image.new("RGB", (8, 8), "blue"), rgb_context)

    assert tiny is not None and tiny["kind"] == "video"
    assert rgb is not None and rgb["kind"] == "image"
    assert rgb["mode"] == "rgb"
    assert cache.fields({"id": "job", "status": "running"})["preview"] == rgb


def test_preview_media_is_isolated_per_job():
    cache = GenerationPreviewCache()
    cache.begin("first")
    cache.begin("second")
    first_context = _context(serial=11)
    second_context = _context(serial=12)
    cache.set_context("first", first_context)
    cache.set_context("second", second_context)

    first = cache.publish("first", _video(b"first"), first_context)
    second = cache.publish("second", _video(b"second"), second_context)

    assert first is not None and second is not None
    assert first["revision"] != second["revision"]
    first_media = cache.get("first", first["revision"])
    second_media = cache.get("second", second["revision"])
    assert first_media is not None and first_media.data == b"first"
    assert second_media is not None and second_media.data == b"second"
    assert cache.fields({"id": "first", "status": "running"})["preview"] == first
    assert cache.fields({"id": "second", "status": "running"})["preview"] == second


def test_cache_keeps_only_two_revisions_per_job():
    cache = GenerationPreviewCache(max_bytes=100, max_media_bytes=50)
    cache.begin("job")
    context = _context()
    cache.set_context("job", context)

    revisions = [
        cache.publish("job", _video(bytes([index]) * 8), context)
        for index in (1, 2, 3)
    ]

    assert all(revisions)
    assert cache.get("job", revisions[0]["revision"]) is None
    assert cache.get("job", revisions[1]["revision"]) is not None
    assert cache.get("job", revisions[2]["revision"]) is not None
    assert len(cache._jobs["job"].media) == 2
    assert cache._bytes == 16


def test_cache_enforces_total_bytes_and_evicts_oldest_media():
    cache = GenerationPreviewCache(max_bytes=5, max_media_bytes=5, max_jobs=4)
    cache.begin("a")
    cache.begin("b")
    context_a = _context(serial=1)
    context_b = _context(serial=2)
    cache.set_context("a", context_a)
    cache.set_context("b", context_b)

    a_first = cache.publish("a", _video(b"a" * 3), context_a)
    a_latest = cache.publish("a", _video(b"b" * 3), context_a)
    assert a_first is not None and a_latest is not None
    assert cache.get("a", a_first["revision"]) is None
    assert cache._bytes == 3

    b_latest = cache.publish("b", _video(b"c" * 4), context_b)

    assert b_latest is not None
    assert cache._bytes <= cache.max_bytes
    assert cache.get("a", a_latest["revision"]) is None
    assert cache.fields({"id": "a", "status": "running"})["preview"] is None
    assert cache.get("b", b_latest["revision"]).data == b"c" * 4


def test_cache_evicts_oldest_job_when_job_limit_is_reached():
    cache = GenerationPreviewCache(max_jobs=2)
    cache.begin("a")
    cache.begin("b")
    context_a = _context(serial=1)
    context_b = _context(serial=2)
    cache.set_context("a", context_a)
    cache.set_context("b", context_b)
    b = cache.publish("b", _video(b"b"), context_b)
    a = cache.publish("a", _video(b"a"), context_a)
    assert a is not None and b is not None

    cache.begin("c")

    assert list(cache._jobs) == ["a", "c"]
    assert cache.get("a", a["revision"]) is not None
    assert cache.get("b", b["revision"]) is None
    assert cache._bytes == 1


def test_terminal_and_cancelled_jobs_do_not_expose_preview_fields():
    cache = GenerationPreviewCache()
    cache.begin("job")
    context = _context()
    cache.set_context("job", context)
    cache.set_notice("job", "preview fallback")
    preview = cache.publish("job", _video(b"video"), context)
    assert preview is not None

    assert cache.fields({"id": "job", "status": "running"}) == {
        "preview": preview,
        "preview_notice": "preview fallback",
    }
    for status in ("completed", "failed", "cancelled"):
        assert cache.fields({"id": "job", "status": status}) == {
            "preview": None,
            "preview_notice": None,
        }
    assert cache.fields({
        "id": "job", "status": "running", "cancel_requested": True,
    }) == {"preview": None, "preview_notice": None}


def test_publish_serializes_rgb_as_jpeg_and_serves_video_ranges_and_etags():
    cache = GenerationPreviewCache()
    cache.begin("jpeg")
    image_context = _context()
    cache.set_context("jpeg", image_context)
    image_info = cache.publish("jpeg", Image.new("RGB", (12, 8), (200, 40, 10)), image_context)
    assert image_info is not None
    image_media = cache.get("jpeg", image_info["revision"])
    assert image_media is not None
    image_response = preview_response(image_media)

    assert image_response.status_code == 200
    assert image_response.media_type == "image/jpeg"
    assert Image.open(io.BytesIO(image_response.body)).format == "JPEG"
    assert image_response.headers["ETag"] == f'"preview-{image_info["revision"]}"'
    assert image_response.headers["Accept-Ranges"] == "bytes"
    assert image_response.headers["X-Content-Type-Options"] == "nosniff"

    video_media = PreviewMedia(b"0123456789", "video/mp4", 77)
    full = preview_response(video_media)
    unchanged = preview_response(video_media, if_none_match='"preview-77"')
    partial = preview_response(video_media, range_header="bytes=2-5")
    suffix = preview_response(video_media, range_header="bytes=-3")
    unsatisfiable = preview_response(video_media, range_header="bytes=10-")

    assert full.status_code == 200 and full.body == b"0123456789"
    assert full.media_type == "video/mp4"
    assert unchanged.status_code == 304
    assert unchanged.headers["ETag"] == '"preview-77"'
    assert partial.status_code == 206 and partial.body == b"2345"
    assert partial.headers["Content-Range"] == "bytes 2-5/10"
    assert suffix.status_code == 206 and suffix.body == b"789"
    assert suffix.headers["Content-Range"] == "bytes 7-9/10"
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["Content-Range"] == "bytes */10"


def test_rgb_session_captures_sparse_steps_and_honors_abort():
    captured = []
    commands = []
    gen = {}
    session = RGBPreviewSession(captured.append, lambda *args: commands.append(args), gen)

    assert session.wants_capture(0, 30, 0)
    assert session.last_step == -1
    due = []
    for step in range(30):
        if session.wants_capture(step, 30, 0):
            due.append(step)
        session.capture(step, step, 30, 0)

    assert due == [0, 5, 10, 15, 20, 25, 29]
    assert captured == due
    assert commands == []
    gen["abort"] = True
    assert session.wants_capture(0, 30, 1) is False
    session.capture(100, 0, 30, 1)
    assert captured == due
    session.close()
    assert session.wants_capture(0, 30, 1) is False


def test_rgb_capture_failure_disables_session_and_notice_failure_is_swallowed(capsys):
    notices = []

    def bad_capture(latent):
        raise RuntimeError("capture unavailable")

    def broken_send(*args):
        notices.append(args)
        raise RuntimeError("closed preview channel")

    session = RGBPreviewSession(bad_capture, broken_send, {})
    session.capture(None, 0, 1)

    assert session.failed is True
    assert notices == [(
        "preview_notice",
        "Live previews are unavailable for this render. Generation will continue.",
    )]
    session.capture(None, 0, 1)
    assert len(notices) == 1
    output = capsys.readouterr().out
    assert "Fast preview disabled" in output
    assert "Could not publish preview notice" in output


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("minimax_h3", "taeh3"),
        ("minimax_h3_full", "taeh3"),
        ("minimax_h3_ref2va_full", "taeh3"),
        ("ltx2", "taeltx_2"),
        ("ltx2_25", "taeltx2_3"),
    ],
)
def test_preview_decoder_model_aliases(architecture, expected):
    assert decoder_key("tiny_vae_frames", architecture, {}) == expected


def test_preview_decoder_key_respects_model_contract_exclusions():
    assert decoder_key("tiny_vae_video", "ltx2", {"ltx2_msr": True}) is None
    assert decoder_key("tiny_vae_video", "flux", {"external_runtime": True}) is None
    assert decoder_key("tiny_vae_video", "minimax_h3", {"audio_only": True}) is None
    assert prepare_preview_decoder("tiny_vae_video", "minimax_h3", {"audio_only": True}) == (None, None, None)


def test_off_and_rgb_modes_do_not_resolve_or_load_decoders(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fast preview modes must not resolve TinyVAE decoders")

    monkeypatch.setattr(preview_runtime, "decoder_key", forbidden)
    assert prepare_preview_decoder("off", "flux", {}) == (None, None, None)
    assert prepare_preview_decoder("rgb", "flux", {}) == (None, None, None)
    assert prepare_preview_decoder("invalid", "flux", {}) == (None, None, None)


def _stub_mmgp(monkeypatch, supports=True):
    module = ModuleType("mmgp")
    module.offload = SimpleNamespace(
        offload=SimpleNamespace(supports_cotenant_wildcards=supports)
    )
    monkeypatch.setitem(sys.modules, "mmgp", module)


def test_preview_decoder_setup_failure_is_nonfatal_and_preserves_cpu_rng(monkeypatch, capsys):
    _stub_mmgp(monkeypatch)
    monkeypatch.setattr(tinyvae_decoder, "prepare_decoder", lambda key, gen=None: "checkpoint")
    original_load = tinyvae_decoder.load_decoder

    def failing_load(key, path):
        torch.rand(4)
        raise RuntimeError("mock decoder load failure")

    monkeypatch.setattr(tinyvae_decoder, "load_decoder", failing_load)
    rng_state = torch.random.get_rng_state().clone()
    try:
        decoder, key, notice = prepare_preview_decoder("tiny_vae_frames", "flux", {})
        assert torch.equal(torch.random.get_rng_state(), rng_state)
    finally:
        torch.random.set_rng_state(rng_state)
        monkeypatch.setattr(tinyvae_decoder, "load_decoder", original_load)

    assert decoder is None
    assert key == "taef1"
    assert notice and "could not start" in notice
    assert "mock decoder load failure" in capsys.readouterr().out


def test_preview_decoder_setup_failure_when_mmgp_capability_is_missing(monkeypatch):
    _stub_mmgp(monkeypatch, supports=False)
    monkeypatch.setattr(preview_runtime, "decoder_key", lambda *args: "taeh3")
    decoder, key, notice = prepare_preview_decoder("tiny_vae_frames", "minimax_h3", {})

    assert decoder is None
    assert key == "taeh3"
    assert notice and "could not start" in notice


def test_preview_decoder_success_restores_rng_after_loading(monkeypatch):
    _stub_mmgp(monkeypatch)
    monkeypatch.setattr(tinyvae_decoder, "prepare_decoder", lambda key, gen=None: "checkpoint")

    def consume_rng(key, path):
        torch.rand(4)
        return {"loaded": key, "path": path}

    monkeypatch.setattr(tinyvae_decoder, "load_decoder", consume_rng)
    rng_state = torch.random.get_rng_state().clone()
    try:
        decoder, key, notice = prepare_preview_decoder("tiny_vae_frames", "flux", {})
        assert torch.equal(torch.random.get_rng_state(), rng_state)
    finally:
        torch.random.set_rng_state(rng_state)

    assert decoder == {"loaded": "taef1", "path": "checkpoint"}
    assert key == "taef1"
    assert notice is None
