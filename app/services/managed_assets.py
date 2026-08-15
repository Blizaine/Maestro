"""Integrity receipts for large assets that Maestro downloads on demand."""

from __future__ import annotations

import hashlib
import json
import os
import uuid


_RECEIPT_SCHEMA_VERSION = 1
_RECEIPT_SUFFIX = ".maestro-managed.json"


def managed_asset_receipt_path(path: str) -> str:
    return os.fspath(path) + _RECEIPT_SUFFIX


def _spec_identity(spec: dict) -> dict:
    return {
        "repo_id": str(spec.get("repo_id") or ""),
        "revision": str(spec.get("revision") or "main"),
        "remote_path": str(spec.get("remote_path") or ""),
        "sha256": str(spec.get("sha256") or "").lower(),
        "size": int(spec["size"]) if spec.get("size") is not None else None,
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_managed_asset_receipt(
    path: str,
    spec: dict,
    *,
    actual_sha256: str | None = None,
) -> None:
    """Atomically record the exact source and local stat of a verified asset."""

    stat = os.stat(path)
    identity = _spec_identity(spec)
    receipt = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        **identity,
        "file_size": int(stat.st_size),
        "file_mtime_ns": int(stat.st_mtime_ns),
        "verified_sha256": str(actual_sha256 or identity["sha256"]).lower(),
    }
    receipt_path = managed_asset_receipt_path(path)
    temporary = f"{receipt_path}.{uuid.uuid4().hex[:8]}.part"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, receipt_path)
    finally:
        try:
            if os.path.isfile(temporary):
                os.remove(temporary)
        except OSError:
            pass


def managed_asset_matches(path: str, spec: dict) -> bool:
    """Verify a managed asset, hashing it only when its receipt is stale/missing."""

    if not os.path.isfile(path):
        return False
    try:
        stat = os.stat(path)
    except OSError:
        return False

    identity = _spec_identity(spec)
    if identity["size"] is not None and int(stat.st_size) != identity["size"]:
        return False
    expected_sha256 = identity["sha256"]
    if not expected_sha256:
        return True

    receipt_path = managed_asset_receipt_path(path)
    try:
        with open(receipt_path, "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, json.JSONDecodeError):
        receipt = None
    if isinstance(receipt, dict):
        receipt_identity = {
            key: receipt.get(key) for key in identity
        }
        if (
            receipt.get("schema_version") == _RECEIPT_SCHEMA_VERSION
            and receipt_identity == identity
            and int(receipt.get("file_size", -1)) == int(stat.st_size)
            and int(receipt.get("file_mtime_ns", -1)) == int(stat.st_mtime_ns)
            and str(receipt.get("verified_sha256") or "").lower() == expected_sha256
        ):
            return True

    try:
        actual_sha256 = _sha256(path)
    except OSError:
        return False
    if actual_sha256.lower() != expected_sha256:
        return False
    try:
        write_managed_asset_receipt(
            path,
            spec,
            actual_sha256=actual_sha256,
        )
    except OSError:
        # A linked installation may be read-only. The content was still fully
        # verified, so use it for this process even without a cached receipt.
        pass
    return True


__all__ = [
    "managed_asset_matches",
    "managed_asset_receipt_path",
    "write_managed_asset_receipt",
]
