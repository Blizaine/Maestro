from __future__ import annotations

import asyncio
import base64
import hashlib

import pytest

from services.agent_access.transfers import TransferStore, TransferStoreError


def _run(awaitable):
    return asyncio.run(awaitable)


async def _pieces(piece: bytes, count: int):
    for _ in range(count):
        yield piece


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_streams_large_binary_into_private_store_and_reads_bounded_chunks(tmp_path):
    store = TransferStore(tmp_path / "transfers", origin="https://maestro.local:8765")
    piece = b"0123456789abcdef" * 4096
    count = 145  # 9.06 MiB: larger than a typical in-memory API response cap.
    record = _run(store.capture("agent-1", "coast.mp4", "video/mp4", _pieces(piece, count), access_scope="observe"))

    assert record == {
        "transfer_id": record["transfer_id"],
        "filename": "coast.mp4",
        "content_type": "video/mp4",
        "size": len(piece) * count,
        "sha256": hashlib.sha256(piece * count).hexdigest(),
        "url": f"https://maestro.local:8765/api/v1/agent-access/transfers/{record['transfer_id']}",
    }
    assert not any("path" in item for item in record)
    first = store.read_chunk(record["transfer_id"], "agent-1", max_bytes=17)
    assert base64.b64decode(first["data_base64"]) == piece[:17]
    assert first["offset"] == 0
    assert first["bytes_read"] == 17
    assert first["eof"] is False
    assert first["sha256"] == record["sha256"]

    remaining = _run(_collect(store.iter_bytes(record["transfer_id"], "agent-1", offset=17, chunk_size=128 * 1024)))
    assert len(remaining) == record["size"] - 17
    assert remaining == (piece * count)[17:]


async def _collect(chunks):
    result = bytearray()
    async for chunk in chunks:
        result.extend(chunk)
    return bytes(result)


def test_transfer_urls_follow_validated_request_origin_without_persisting_it(tmp_path):
    from services.agent_access.context import public_origin_context

    store = TransferStore(tmp_path / "transfers", origin="https://maestro.local:8765")
    token = public_origin_context.set("https://192.168.1.20:8765")
    try:
        result = _run(store.capture("agent-1", "clip.mp4", "video/mp4", _pieces(b"clip", 1), access_scope="observe"))
        assert result["url"].startswith("https://192.168.1.20:8765/")
    finally:
        public_origin_context.reset(token)
    assert store.metadata(result["transfer_id"], "agent-1")["url"].startswith("https://maestro.local:8765/")


def test_browser_recovery_selects_only_published_certificate_addresses():
    from services.agent_access.boundary import browser_login_origin

    status = {"controller_url": "https://maestro.local:8765", "local_setup_url": "https://127.0.0.1:8765", "lan_mcp_urls": ["https://192.168.1.20:8765/mcp"]}
    assert browser_login_origin(status, {"headers": [(b"host", b"192.168.1.20:42015")]}) == "https://192.168.1.20:8765"
    assert browser_login_origin(status, {"headers": [(b"host", b"attacker.test")]}) == status["controller_url"]


def test_owner_scope_access_and_opaque_metadata(tmp_path):
    store = TransferStore(tmp_path / "transfers")
    record = _run(store.capture("agent-a", "private.png", "image/png", _pieces(b"image", 1), access_scope="manage"))

    with pytest.raises(TransferStoreError) as foreign:
        store.metadata(record["transfer_id"], "agent-b")
    assert foreign.value.status == 404
    assert store.metadata(record["transfer_id"], "agent-b", admin=True) == record
    assert store.access_scope(record["transfer_id"], "agent-a") == "manage"
    assert store.access_scope(record["transfer_id"], "agent-b", owner=True) == "manage"


def test_binary_request_receipt_prevents_duplicate_dispatch_and_checks_fingerprint(tmp_path):
    store = TransferStore(tmp_path / "transfers")
    fingerprint = hashlib.sha256(b"GET binary?" ).hexdigest()
    assert store.reserve_request("agent-1", "same-id", fingerprint) is None
    record = _run(store.capture(
        "agent-1", "output.zip", "application/zip", _pieces(b"zip-data", 1),
        request_id="same-id", fingerprint=fingerprint, access_scope="generate",
    ))
    assert store.reserve_request("agent-1", "same-id", fingerprint) == record

    with pytest.raises(TransferStoreError, match="different operation inputs") as changed:
        store.reserve_request("agent-1", "same-id", hashlib.sha256(b"other").hexdigest())
    assert changed.value.status == 409


def test_lost_binary_stream_becomes_uncertain_and_never_reuses_request_id(tmp_path):
    store = TransferStore(tmp_path / "transfers")
    fingerprint = hashlib.sha256(b"render").hexdigest()
    store.reserve_request("agent-1", "render-1", fingerprint)

    async def interrupted():
        yield b"partial"
        raise OSError("connection reset")

    with pytest.raises(TransferStoreError, match="interrupted") as failure:
        _run(store.capture(
            "agent-1", "movie.mp4", "video/mp4", interrupted(),
            request_id="render-1", fingerprint=fingerprint, access_scope="generate",
        ))
    assert failure.value.status == 503
    with pytest.raises(TransferStoreError, match="uncertain") as retry:
        store.reserve_request("agent-1", "render-1", fingerprint)
    assert retry.value.status == 409


def test_expired_artifacts_are_removed_but_request_id_stays_reserved(tmp_path):
    clock = _Clock()
    store = TransferStore(
        tmp_path / "transfers", clock=clock, complete_ttl_seconds=10,
    )
    fingerprint = hashlib.sha256(b"export").hexdigest()
    store.reserve_request("agent-1", "export-1", fingerprint)
    record = _run(store.capture(
        "agent-1", "export.zip", "application/zip", _pieces(b"file", 1),
        request_id="export-1", fingerprint=fingerprint, access_scope="manage",
    ))
    clock.now += 11
    with pytest.raises(TransferStoreError) as expired:
        store.metadata(record["transfer_id"], "agent-1")
    assert expired.value.status == 410
    assert store.cleanup() == 1
    with pytest.raises(TransferStoreError, match="expired") as retry:
        store.reserve_request("agent-1", "export-1", fingerprint)
    assert retry.value.status == 410


def test_orphan_pending_receipt_ages_to_uncertain_without_becoming_reusable(tmp_path):
    clock = _Clock()
    store = TransferStore(tmp_path / "transfers", clock=clock, unfinished_ttl_seconds=5)
    fingerprint = hashlib.sha256(b"pending").hexdigest()
    store.reserve_request("agent-1", "never-dispatched", fingerprint)
    clock.now += 6
    store.cleanup()
    with pytest.raises(TransferStoreError, match="uncertain"):
        store.reserve_request("agent-1", "never-dispatched", fingerprint)


def test_stream_quota_and_filename_are_bounded(tmp_path):
    store = TransferStore(tmp_path / "transfers", max_file_bytes=4)
    with pytest.raises(TransferStoreError) as size_limit:
        _run(store.capture("agent-1", "../private.bin", "application/octet-stream", _pieces(b"12345", 1)))
    assert size_limit.value.status == 413
