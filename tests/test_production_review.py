from __future__ import annotations

import copy
import json
import multiprocessing as mp
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

import services.production_review as production_review
from services.production_review import (
    BudgetExceededError,
    PolishBudgetReviewStore,
    RequestCollisionError,
    ReviewStoreCorruptError,
    ReviewStoreError,
    ReviewValidationError,
    director_take_signature,
    inspect_take,
    normalize_take_review,
)


def _review(**overrides):
    body = {
        "verdict": "keep",
        "summary": "The take has clear action and stable framing.",
        "observations": [
            {
                "category": "action",
                "severity": "low",
                "text": "The runner stays centered while crossing the bridge.",
            }
        ],
    }
    body.update(overrides)
    return body


def _claim_once_in_process(directory, barrier, results):
    store = PolishBudgetReviewStore(
        directory, max_renders=2, max_retakes_per_clip=2
    )
    barrier.wait(timeout=15)
    intent, created = store.claim_once(
        "shared-request", "clip-a", "source-v1", "request-v1"
    )
    results.put((intent, created))


def test_normalize_take_review_returns_explicit_assessment_and_cleans_text():
    body = _review(
        summary="  The take has clear action\nand stable framing.  ",
        observations=[{
            "category": "composition",
            "severity": "medium",
            "text": "  The subject remains centered through the turn.  ",
        }],
    )

    result = normalize_take_review(body)

    assert result == {
        "schema_version": 1,
        "creative_status": "assessed",
        "verdict": "keep",
        "summary": "The take has clear action and stable framing.",
        "observations": [{
            "category": "composition",
            "severity": "medium",
            "text": "The subject remains centered through the turn.",
        }],
    }


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        _review(verdict=""),
        _review(verdict="automatic"),
        _review(observations=[]),
        _review(observations=[{"category": "unknown", "severity": "low", "text": "The subject remains centered."}]),
        _review(observations=[{"category": "action", "severity": "urgent", "text": "The subject remains centered."}]),
        _review(observations=[{"category": "action", "severity": "low", "text": "The take looks good overall."}]),
        _review(extra="not allowed"),
        _review(observations=[{"category": "action", "severity": "low", "text": "Specific action detail.", "timestamp": 1}]),
    ],
)
def test_normalize_take_review_rejects_malformed_or_empty_assessments(body):
    with pytest.raises(ReviewValidationError):
        normalize_take_review(body)


def test_normalize_take_review_enforces_text_and_observation_limits():
    with pytest.raises(ReviewValidationError):
        normalize_take_review(_review(summary="x" * 321))
    with pytest.raises(ReviewValidationError):
        normalize_take_review(_review(observations=[
            {"category": "action", "severity": "low", "text": "specific detail " * 40}
        ]))
    with pytest.raises(ReviewValidationError):
        normalize_take_review(_review(observations=[
            {"category": "action", "severity": "low", "text": "The subject stays centered."}
        ] * 13))


def test_director_take_signature_tracks_creative_state_not_injected_paths():
    state = {
        "clips": [{
            "video_prompt": "A runner crosses a bridge.",
            "planned_clip": {"duration_sec": 4},
            "video_path": "injected/current.mp4",
            "video_file": "injected/legacy.mp4",
            "output_path": "injected/output.mp4",
        }],
        "_params_snapshot": {"resolution": "960x544"},
        "video_model": "model-a",
        "pipeline_type": "short_film_story",
    }
    original = director_take_signature(state, 0)
    path_change = copy.deepcopy(state)
    path_change["clips"][0]["video_path"] = "injected/other.mp4"
    assert director_take_signature(path_change, 0) == original

    prompt_change = copy.deepcopy(state)
    prompt_change["clips"][0]["video_prompt"] = "A runner stops at the bridge."
    assert director_take_signature(prompt_change, 0) != original
    with pytest.raises(ValueError):
        director_take_signature(state, True)


def test_polish_store_claim_is_persistent_idempotent_and_receipt_bound(tmp_path: Path):
    store = PolishBudgetReviewStore(
        tmp_path, max_renders=3, max_retakes_per_clip=2
    )
    claim = store.claim("req-1", "clip-a", "source-v1", "request-v1")

    assert claim["status"] == "claimed"
    assert claim["receipt_id"] is None
    assert store.claim("req-1", "clip-a", "source-v1", "request-v1") == claim
    assert store.snapshot()["renders_claimed"] == 1

    submitted = store.bind_receipt("req-1", "renderer-job-7")
    assert submitted["status"] == "submitted"
    assert submitted["receipt_id"] == "renderer-job-7"
    assert store.bind_receipt("req-1", "renderer-job-7") == submitted
    assert store.get("req-1") == submitted
    with pytest.raises(RequestCollisionError):
        store.bind_receipt("req-1", "renderer-job-other")

    store.claim("req-2", "clip-b", "source-v1", "request-v2")
    with pytest.raises(RequestCollisionError):
        store.bind_receipt("req-2", "renderer-job-7")

    reopened = PolishBudgetReviewStore(
        tmp_path, max_renders=3, max_retakes_per_clip=2
    )
    assert reopened.list_intents()[0] == submitted
    assert json.loads((tmp_path / "polish-review-ledger.json").read_text(encoding="utf-8"))["intents"] == reopened.list_intents()


def test_polish_store_rejects_request_id_collision_and_changed_persisted_limits(tmp_path: Path):
    store = PolishBudgetReviewStore(tmp_path, max_renders=2, max_retakes_per_clip=2)
    store.claim("req-1", "clip-a", "source-v1", "request-v1")
    ledger_path = tmp_path / "polish-review-ledger.json"
    original_ledger = ledger_path.read_text(encoding="utf-8")

    with pytest.raises(RequestCollisionError):
        store.claim("req-1", "clip-a", "source-v2", "request-v1")
    with pytest.raises(RequestCollisionError):
        store.claim("req-1", "clip-a", "source-v1", "request-v2")
    with pytest.raises(ReviewStoreError):
        PolishBudgetReviewStore(tmp_path, max_renders=3, max_retakes_per_clip=2)

    assert ledger_path.read_text(encoding="utf-8") == original_ledger
    assert store.snapshot()["renders_claimed"] == 1


def test_polish_store_keeps_failed_or_ambiguous_intents_counted_and_enforces_limits(tmp_path: Path):
    store = PolishBudgetReviewStore(tmp_path, max_renders=2, max_retakes_per_clip=1)
    store.claim("req-1", "clip-a", "source-v1", "request-v1")

    # Re-opening after an ambiguous renderer result does not free the pre-POST claim.
    reopened = PolishBudgetReviewStore(tmp_path, max_renders=2, max_retakes_per_clip=1)
    with pytest.raises(BudgetExceededError, match="Per-clip"):
        reopened.claim("req-2", "clip-a", "source-v1", "request-v2")

    reopened.claim("req-2", "clip-b", "source-v1", "request-v2")
    with pytest.raises(BudgetExceededError, match="Project"):
        reopened.claim("req-3", "clip-c", "source-v1", "request-v3")
    assert reopened.snapshot()["renders_claimed"] == 2


def test_polish_store_serializes_concurrent_claims_across_instances(tmp_path: Path):
    stores = [
        PolishBudgetReviewStore(tmp_path, max_renders=1, max_retakes_per_clip=1),
        PolishBudgetReviewStore(tmp_path, max_renders=1, max_retakes_per_clip=1),
    ]
    gate = Barrier(2)

    def claim(index: int):
        gate.wait(timeout=5)
        try:
            return stores[index].claim(
                f"req-{index}", f"clip-{index}", "source-v1", f"request-{index}"
            )
        except BudgetExceededError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))

    assert sum(result is not None for result in results) == 1
    assert stores[0].snapshot()["renders_claimed"] == 1


def test_polish_store_claim_once_is_atomic_across_processes(tmp_path: Path):
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    workers = [
        context.Process(
            target=_claim_once_in_process,
            args=(str(tmp_path), barrier, results),
        )
        for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        outcomes = [results.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=15)
        assert [worker.exitcode for worker in workers] == [0, 0]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()
        results.join_thread()

    intents = [intent for intent, _created in outcomes]
    created_flags = [created for _intent, created in outcomes]
    assert sum(created_flags) == 1
    assert intents[0] == intents[1]
    reopened = PolishBudgetReviewStore(
        tmp_path, max_renders=2, max_retakes_per_clip=2
    )
    assert reopened.snapshot()["renders_claimed"] == 1


def test_polish_store_fails_closed_on_malformed_persisted_intent(tmp_path: Path):
    ledger_path = tmp_path / "polish-review-ledger.json"
    malformed = {
        "schema_version": 1,
        "max_renders": 2,
        "max_retakes_per_clip": 2,
        "intents": [{"request_id": "partial"}],
    }
    ledger_path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(ReviewStoreCorruptError):
        PolishBudgetReviewStore(tmp_path, max_renders=2, max_retakes_per_clip=2)

    assert json.loads(ledger_path.read_text(encoding="utf-8")) == malformed


@pytest.mark.parametrize(
    ("max_renders", "max_retakes"),
    [(0, 1), (13, 1), (True, 1), (1, 0), (1, 4), (1, False)],
)
def test_polish_store_validates_limits(tmp_path: Path, max_renders, max_retakes):
    with pytest.raises(ValueError):
        PolishBudgetReviewStore(
            tmp_path / str(max_renders) / str(max_retakes),
            max_renders=max_renders,
            max_retakes_per_clip=max_retakes,
        )


def test_inspect_take_rejects_missing_or_non_file_media_and_bad_duration(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        inspect_take(tmp_path / "missing.mp4", tmp_path / "out")

    media_dir = tmp_path / "media-dir"
    media_dir.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        inspect_take(media_dir, tmp_path / "out")

    media = tmp_path / "empty.mp4"
    media.write_bytes(b"")
    with pytest.raises(ValueError, match="expected_duration"):
        inspect_take(media, tmp_path / "out", expected_duration=float("nan"))


def test_inspect_take_reports_unknown_technical_checks_without_ffmpeg(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(production_review.shutil, "which", lambda _name: None)
    media = tmp_path / "opaque-media"
    media.write_bytes(b"not a supported container")

    packet = inspect_take(media, tmp_path / "review")

    assert packet["creative_status"] == "not_assessed"
    assert packet["creative_assessment"] is None
    assert packet["technical"]["probe"]["status"] == "unavailable"
    assert packet["technical"]["duration_seconds"] is None
    assert packet["technical"]["audio"]["present"] is None
    assert packet["technical"]["software_decode"]["status"] == "unavailable"
    assert packet["artifacts"]["stills"]["status"] == "unavailable"
    packet_path = Path(packet["artifacts"]["packet"]["path"])
    assert packet_path.is_file()
    assert json.loads(packet_path.read_text(encoding="utf-8"))["creative_status"] == "not_assessed"


def test_inspect_take_creates_cpu_review_artifacts_when_ffmpeg_is_available(tmp_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg/ffprobe are not installed")

    media = tmp_path / "fixture.mp4"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
        "-t", "1", "-c:v", "mpeg4", "-q:v", "10", "-c:a", "aac", "-y", str(media),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=20)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Installed FFmpeg cannot create the tiny fixture: {exc}")

    packet = inspect_take(
        media, tmp_path / "review", ffmpeg_path=ffmpeg, ffprobe_path=ffprobe,
        expected_duration=1.0,
    )

    assert packet["creative_status"] == "not_assessed"
    assert packet["technical"]["probe"]["status"] == "available"
    assert packet["technical"]["duration_seconds"] == pytest.approx(1.0, abs=0.2)
    assert packet["technical"]["geometry"]["width"] == 64
    assert packet["technical"]["geometry"]["height"] == 48
    assert packet["technical"]["audio"]["present"] is True
    assert packet["technical"]["software_decode"]["status"] == "passed"
    stills = packet["artifacts"]["stills"]
    assert stills["status"] == "complete"
    assert len(stills["paths"]) == 5
    assert all(Path(path).is_file() for path in stills["paths"])
    assert packet["artifacts"]["contact_sheet"]["status"] in {"created", "unavailable"}
    assert packet["artifacts"]["audio_preview"]["status"] == "created"
    assert Path(packet["artifacts"]["packet"]["path"]).is_file()
