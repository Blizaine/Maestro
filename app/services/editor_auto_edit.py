"""Bounded, review-only Editor auto-edit proposal service.

This module deliberately does not call a model, create a generation job, or
modify an Editor project. Callers provide a JSON writer; the service gives it a
path-free project ledger and validates the returned edit suggestions before
persisting them for user review.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping


MAX_ASSETS = 128
MAX_CLIPS = 200
MAX_TRACKS = 100
MAX_ACTIONS = 64
MAX_MISSING_MEDIA = 8
MAX_BRIEF_CHARS = 4000
MAX_TARGET_SECONDS = 86400
MAX_DRAFT_CHARS = 65536
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,160}$")
_MODES = {"director", "assemble", "polish", "revise"}
_ACTION_TYPES = {"insert", "move", "trim", "remove", "set-volume", "crossfade", "close-gap"}
_STATUSES = {"running", "ready", "needs_review", "failed", "accepted", "rejected", "superseded"}
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|authorization|credential)\b"
    r"\s*(?:is\s+|[:=]\s*)(?:Bearer\s+)?[^\s,;]+"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_URL_RE = re.compile(r"(?i)\b(?:https?|file)://[^\s<>\"']+")
_WINDOWS_PATH_RE = re.compile(r"(?<![\w])(?:[A-Za-z]:\\|\\\\)[^\s<>\"']+")
_POSIX_PATH_RE = re.compile(r"(?<![\w:])/(?:[^\s<>\"']+/)*[^\s<>\"']+")
_lock = threading.RLock()


class EditorAutoEditError(ValueError):
    """An Editor proposal request or stored proposal is invalid."""


def _safe_id(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not _ID_RE.fullmatch(result):
        raise EditorAutoEditError(f"Invalid {label}.")
    return result


def _finite(value: Any, label: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise EditorAutoEditError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EditorAutoEditError(f"{label} must be a finite number.") from error
    if not math.isfinite(result):
        raise EditorAutoEditError(f"{label} must be a finite number.")
    if minimum is not None and result < minimum:
        raise EditorAutoEditError(f"{label} must be at least {minimum:g}.")
    if maximum is not None and result > maximum:
        raise EditorAutoEditError(f"{label} must be at most {maximum:g}.")
    return result


def _redact_text(value: Any, limit: int) -> str:
    text = str(value or "")
    text = _URL_RE.sub("[redacted URL]", text)
    text = _BEARER_RE.sub("[redacted credential]", text)
    text = _AWS_KEY_RE.sub("[redacted credential]", text)
    text = _WINDOWS_PATH_RE.sub("[redacted path]", text)
    text = _POSIX_PATH_RE.sub("[redacted path]", text)
    text = _SECRET_RE.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    return text[:limit]


def _redact_value(value: Any, depth: int = 0) -> Any:
    """Keep small useful drafts while dropping path/credential-bearing keys."""
    if depth > 8:
        return "[depth limit]"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            name = str(key)[:100]
            if re.search(r"(?i)(^|_)(path|url|uri|token|secret|password|credential)(_|$)", name):
                continue
            result[name] = _redact_value(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, depth + 1) for item in list(value)[:100]]
    if isinstance(value, str):
        return _redact_text(value, 1200)
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return _redact_text(value, 300)


def _settings_root(settings_dir: str | os.PathLike[str]) -> Path:
    base = Path(settings_dir)
    base.mkdir(parents=True, exist_ok=True)
    resolved_base = base.resolve(strict=True)
    target = resolved_base / "editor_auto_edit"
    target.mkdir(mode=0o700, exist_ok=True)
    resolved_target = target.resolve(strict=True)
    if resolved_target.parent != resolved_base or target.is_symlink():
        raise EditorAutoEditError("Editor proposal storage escaped the settings directory.")
    return resolved_target


def _proposal_path(settings_dir: str | os.PathLike[str], proposal_id: Any) -> Path:
    safe_id = _safe_id(proposal_id, "proposal id")
    root = _settings_root(settings_dir)
    path = root / f"{safe_id}.json"
    if path.resolve(strict=False).parent != root:
        raise EditorAutoEditError("Editor proposal path escaped its storage directory.")
    return path


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    root = path.parent.resolve(strict=True)
    if path.resolve(strict=False).parent != root:
        raise EditorAutoEditError("Editor proposal path escaped its storage directory.")
    descriptor, temp_name = tempfile.mkstemp(prefix=".editor-proposal-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _read_record(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise EditorAutoEditError("Editor proposal record cannot be a symbolic link.")
    try:
        if path.stat().st_size > 1024 * 1024:
            raise EditorAutoEditError("Editor proposal record exceeds the size limit.")
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EditorAutoEditError("Editor proposal record is unreadable.") from error
    if not isinstance(value, dict) or value.get("proposal_id") != path.stem:
        raise EditorAutoEditError("Editor proposal record is malformed.")
    return value


def _project_ledger(project: Mapping[str, Any], workspace: str,
                    pinned_item_ids: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    project_id = str(project.get("id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", project_id):
        raise EditorAutoEditError("A saved Editor project id is required.")
    if workspace in {".", ".."} or any(character in workspace for character in ("/", "\\", ":", "\0")):
        raise EditorAutoEditError("Editor workspace must be a simple workspace name.")
    raw_assets = project.get("assets")
    raw_tracks = project.get("tracks")
    if not isinstance(raw_assets, Mapping):
        raw_assets = {}
    if not isinstance(raw_tracks, list):
        raw_tracks = []
    if len(raw_assets) > MAX_ASSETS:
        raise EditorAutoEditError(f"Auto-edit supports at most {MAX_ASSETS} project assets.")
    if len(raw_tracks) > MAX_TRACKS:
        raise EditorAutoEditError(f"Auto-edit supports at most {MAX_TRACKS} project tracks.")

    assets: dict[str, dict[str, Any]] = {}
    for key, raw_asset in raw_assets.items():
        if not isinstance(raw_asset, Mapping):
            continue
        asset_id = str(raw_asset.get("id") or key)
        media_type = str(raw_asset.get("type") or "").lower()
        if not asset_id or media_type not in {"video", "audio", "image"}:
            continue
        try:
            duration = _finite(raw_asset.get("duration", 0), "Asset duration", minimum=0)
        except EditorAutoEditError as error:
            raise EditorAutoEditError(f"Asset {asset_id} has an invalid duration.") from error
        assets[asset_id] = {
            "id": asset_id,
            "type": media_type,
            "duration": duration,
            "name": _redact_text(raw_asset.get("display_name") or raw_asset.get("name") or "", 160),
        }

    tracks: list[dict[str, Any]] = []
    items_by_id: dict[str, dict[str, Any]] = {}
    director_pipeline_ids: dict[str, str | None] = {}
    clip_count = 0
    track_ids: set[str] = set()
    for raw_track in raw_tracks:
        if not isinstance(raw_track, Mapping):
            continue
        track_id = str(raw_track.get("id") or "")
        track_type = str(raw_track.get("type") or "").lower()
        if not track_id or track_type not in {"video", "audio", "text"}:
            continue
        if track_id in track_ids:
            raise EditorAutoEditError("Saved project contains duplicate track ids.")
        track_ids.add(track_id)
        raw_items = raw_track.get("items")
        if not isinstance(raw_items, list):
            raw_items = []
        clip_count += len(raw_items)
        if clip_count > MAX_CLIPS:
            raise EditorAutoEditError(f"Auto-edit supports at most {MAX_CLIPS} timeline clips.")
        track_items: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            item_id = str(raw_item.get("id") or "")
            if not item_id:
                continue
            if item_id in items_by_id:
                raise EditorAutoEditError("Saved project contains duplicate clip ids.")
            asset_id = str(raw_item.get("asset_id") or "")
            if track_type != "text" and asset_id not in assets:
                # A missing media reference remains visible in the ledger but
                # cannot be used by a proposed edit.
                asset_id = asset_id or None
            try:
                start = _finite(raw_item.get("start", 0), "Clip start", minimum=0)
                duration = _finite(raw_item.get("duration", 0), "Clip duration", minimum=0)
                source_in = _finite(raw_item.get("source_in", 0), "Clip source in", minimum=0)
                speed = _finite(raw_item.get("speed", 1), "Clip speed", minimum=0.000001)
                volume = _finite(raw_item.get("volume", 1), "Clip volume", minimum=0, maximum=4)
            except EditorAutoEditError as error:
                raise EditorAutoEditError(f"Clip {item_id} has invalid timing or volume.") from error
            director = raw_item.get("director") if isinstance(raw_item.get("director"), Mapping) else {}
            clip_index: int | None = None
            try:
                candidate_index = int(director.get("clip_index"))
                if candidate_index >= 0:
                    clip_index = candidate_index
            except (TypeError, ValueError, OverflowError):
                pass
            item_locked = bool(raw_item.get("locked"))
            locked = bool(raw_track.get("locked")) or item_locked
            pinned = (
                bool(raw_track.get("pinned"))
                or bool(raw_item.get("pinned"))
                or bool(director.get("pinned"))
                or item_id in pinned_item_ids
            )
            item = {
                "id": item_id,
                "asset_id": asset_id,
                "type": assets.get(str(asset_id), {}).get("type", "text" if track_type == "text" else "unknown"),
                "track_id": track_id,
                "track_type": track_type,
                "start": start,
                "duration": duration,
                "source_in": source_in,
                "speed": speed,
                "volume": volume,
                "locked": locked,
                "item_locked": item_locked,
                "pinned": pinned,
                "link_group_id": str(raw_item.get("link_group_id") or "") or None,
            }
            if clip_index is not None:
                director_pipeline_ids[item_id] = (
                    str(director.get("pipeline_id") or "").strip() or None
                )
                item["director"] = {
                    "clip_index": clip_index,
                    "video_prompt": _redact_text(director.get("video_prompt") or "", 320),
                }
            items_by_id[item_id] = item
            track_items.append(item)
        tracks.append({
            "id": track_id,
            "type": track_type,
            "locked": bool(raw_track.get("locked")),
            "pinned": bool(raw_track.get("pinned")),
            "items": track_items,
        })

    ledger = {
        "project_id": project_id,
        "workspace": workspace,
        "revision": project.get("revision") if isinstance(project.get("revision"), int) else None,
        "canvas": {
            key: project.get("canvas", {}).get(key)
            for key in ("width", "height", "fps")
            if isinstance(project.get("canvas"), Mapping) and key in project.get("canvas", {})
        },
        "assets": list(assets.values()),
        "tracks": tracks,
    }
    digest = hashlib.sha256(
        json.dumps(ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    # Lookup maps are server-only and never sent to the writer or persisted.
    context = {
        "assets": assets,
        "tracks": tracks,
        "items": items_by_id,
        "director_pipeline_ids": director_pipeline_ids,
    }
    return ledger, {"sha256": digest, **context}


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "actions", "missing_media"],
    "properties": {
        "summary": {"type": "string", "maxLength": 400},
        "missing_media": {"type": "array", "maxItems": MAX_MISSING_MEDIA, "items": {"type": "string", "maxLength": 240}},
        "actions": {
            "type": "array",
            "maxItems": MAX_ACTIONS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "type", "item_id", "asset_id", "track_id", "start", "duration", "source_in", "volume", "seconds", "reason"],
                "properties": {
                    "id": {"type": "null", "description": "Must be null; the server assigns action ids."},
                    "type": {"type": "string", "enum": sorted(_ACTION_TYPES)},
                    "item_id": {"type": ["string", "null"], "maxLength": 160},
                    "asset_id": {"type": ["string", "null"], "maxLength": 160},
                    "track_id": {"type": ["string", "null"], "maxLength": 160},
                    "start": {"type": ["number", "null"]},
                    "duration": {"type": ["number", "null"]},
                    "source_in": {"type": ["number", "null"]},
                    "volume": {"type": ["number", "null"]},
                    "seconds": {"type": ["number", "null"]},
                    "reason": {"type": ["string", "null"], "maxLength": 500},
                },
            },
        },
    },
}


def _new_action(proposal_id: str, index: int, action_type: str, **fields: Any) -> dict[str, Any]:
    action: dict[str, Any] = {
        "id": f"{proposal_id}-act-{index:03d}",
        "type": action_type,
        "item_id": None,
        "asset_id": None,
        "track_id": None,
        "start": None,
        "duration": None,
        "source_in": None,
        "volume": None,
        "seconds": None,
        "reason": None,
    }
    action.update(fields)
    return action


def _director_actions(proposal_id: str, ledger: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str], str]:
    actions: list[dict[str, Any]] = []
    warnings: list[str] = ["Review timing after the Director shot order is changed."]
    index = 1
    for track in context["tracks"]:
        if track["type"] != "video":
            continue
        directors = [item for item in track["items"] if isinstance(item.get("director"), Mapping)]
        if len(directors) < 2:
            continue
        pipeline_ids = {
            context["director_pipeline_ids"].get(item["id"])
            for item in directors
        }
        if None in pipeline_ids or len(pipeline_ids) > 1:
            warnings.append(
                f"Director order was left unchanged on track {track['id']} because it contains clips from different or unidentified Director pipelines."
            )
            continue
        movable = [item for item in directors if not item["locked"] and not item["pinned"] and not item["link_group_id"]]
        fixed = [item for item in directors if item not in movable]
        if fixed:
            warnings.append(f"Linked, locked, or pinned Director clips were left fixed on track {track['id']}.")
        if len(movable) < 2:
            continue
        original_order = sorted(movable, key=lambda item: (item["start"], item["id"]))
        desired = sorted(movable, key=lambda item: (item["director"]["clip_index"], item["id"]))
        if [item["id"] for item in original_order] == [item["id"] for item in desired]:
            continue
        cursor = min(item["start"] for item in movable)
        targets: dict[str, float] = {}
        for item in desired:
            targets[item["id"]] = cursor
            cursor += item["duration"]
        other_items = [
            item for item in track["items"]
            if item["id"] not in {candidate["id"] for candidate in movable}
        ]
        collision = False
        for item in movable:
            new_start = targets[item["id"]]
            new_end = new_start + item["duration"]
            if any(new_start < other["start"] + other["duration"] - 1e-7 and other["start"] < new_end - 1e-7 for other in other_items):
                collision = True
                break
        if collision:
            warnings.append(f"Director order was left unchanged on track {track['id']} because the new timing overlaps another clip on that track.")
            continue
        for item in movable:
            new_start = targets[item["id"]]
            if abs(new_start - item["start"]) <= 1e-7:
                continue
            actions.append(_new_action(
                proposal_id, index, "move", item_id=item["id"], track_id=track["id"],
                start=new_start, reason=f"Place Director shot {item['director']['clip_index']} in clip order.",
            ))
            index += 1
    summary = "Director shots were ordered by their saved clip index." if actions else "No safe Director shot reorder was needed."
    return actions, warnings, summary


def _strict_root(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None, "Writer response was not valid JSON."
    if not isinstance(raw, Mapping):
        return None, "Writer response must be a JSON object."
    errors: list[str] = []
    if set(raw.keys()) != {"summary", "actions", "missing_media"}:
        errors.append("Writer response must contain only summary, actions, and missing_media.")
    summary = raw.get("summary")
    if not isinstance(summary, str):
        errors.append("Writer response summary must be text.")
        summary = ""
    actions = raw.get("actions")
    if not isinstance(actions, list):
        errors.append("Writer response actions must be an array.")
        return None, " ".join(errors)
    missing_media = raw.get("missing_media")
    if not isinstance(missing_media, list):
        errors.append("Writer response missing_media must be an array.")
        missing_media = []
    projected = {"summary": summary, "actions": actions, "missing_media": missing_media}
    return projected, " ".join(errors) if errors else None


def _read_action_row(row: Any, index: int, proposal_id: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(row, Mapping):
        return None, "Action must be an object."
    required = set(ACTION_SCHEMA["properties"]["actions"]["items"]["required"])
    if set(row.keys()) != required:
        return None, "Action fields do not match the strict action schema."
    action_type = str(row.get("type") or "")
    if action_type not in _ACTION_TYPES:
        return None, "Action type is unsupported."
    if row.get("id") is not None:
        return None, "Action ids are assigned by the server and must be null in writer output."
    action = _new_action(proposal_id, index, action_type)
    for key in ("item_id", "asset_id", "track_id"):
        value = row.get(key)
        if value is not None:
            value = str(value).strip()
            if not value or len(value) > 160:
                return None, f"Action {key} is invalid."
        action[key] = value
    for key in ("start", "duration", "source_in", "volume", "seconds"):
        value = row.get(key)
        if value is not None:
            try:
                action[key] = _finite(value, f"Action {key}")
            except EditorAutoEditError as error:
                return None, str(error)
    reason = row.get("reason")
    if reason is not None and not isinstance(reason, str):
        return None, "Action reason must be text or null."
    action["reason"] = _redact_text(reason, 300) if reason else None
    fields_by_type = {
        "insert": {"asset_id", "track_id", "start", "duration", "source_in"},
        "move": {"item_id", "track_id", "start"},
        "trim": {"item_id", "duration", "source_in"},
        "remove": {"item_id"},
        "set-volume": {"item_id", "volume"},
        "crossfade": {"item_id", "seconds"},
        "close-gap": {"item_id"},
    }
    if any(action.get(key) is not None for key in set(action) - {"id", "type", "reason"} - fields_by_type[action_type]):
        return None, f"Action {action_type} contains fields that do not apply to that action."
    return action, None


def _source_range_valid(asset: Mapping[str, Any], source_in: float, duration: float, speed: float) -> bool:
    if source_in < 0 or duration <= 0 or speed <= 0:
        return False
    if asset.get("type") == "image":
        return True
    if float(asset.get("duration") or 0) <= 0:
        return False
    return source_in + duration * speed <= float(asset["duration"]) + 1e-6


def _validate_action(action: dict[str, Any], context: Mapping[str, Any], pinned_ids: set[str]) -> str | None:
    kind = action["type"]
    items = context["items"]
    assets = context["assets"]
    tracks = {track["id"]: track for track in context["tracks"]}
    item_id = action.get("item_id")
    asset_id = action.get("asset_id")
    track_id = action.get("track_id")
    item = items.get(str(item_id)) if item_id else None
    target_track = tracks.get(str(track_id)) if track_id else None
    if kind == "insert":
        if item_id is not None:
            return "Insert item ids are assigned by the server."
        if not asset_id or asset_id not in assets:
            return "Insert must reference a known asset."
        if not target_track:
            return "Insert must reference a known track."
        if target_track["locked"] or target_track["pinned"]:
            return "Insert target track is locked or pinned."
        start, duration = action.get("start"), action.get("duration")
        source_in = action.get("source_in") if action.get("source_in") is not None else 0.0
        if start is None or duration is None or start < 0 or duration <= 0 or source_in < 0:
            return "Insert needs a non-negative start and a positive duration/source range."
        asset = assets[str(asset_id)]
        if target_track["type"] == "text" or (target_track["type"] == "audio" and asset["type"] == "image"):
            return "Asset type does not fit the target track."
        if target_track["type"] == "video" and asset["type"] == "audio":
            return "Audio assets cannot be inserted on a video track."
        if not _source_range_valid(asset, source_in, duration, 1.0):
            return "Insert exceeds the source asset duration."
        action["item_id"] = f"{action['id']}-item"
        action["source_in"] = source_in
        return None
    if not item:
        return "Action references an unknown clip."
    if item["item_locked"] or item["pinned"] or item["id"] in pinned_ids:
        return "Action targets a locked or pinned clip."
    source_track = tracks[item["track_id"]]
    if source_track["locked"] or source_track["pinned"]:
        return "Action targets a locked or pinned track."
    structural = kind in {"move", "trim", "remove", "crossfade", "close-gap"}
    if structural and item.get("link_group_id"):
        return "Linked clips are preserved as a group; this proposal cannot change only one linked clip."

    if kind == "move":
        if action.get("start") is None or action["start"] < 0:
            return "Move needs a non-negative start time."
        if track_id is not None:
            if not target_track:
                return "Move target track is unknown."
            if target_track["type"] != item["track_type"]:
                return "Move must keep the clip on a track of the same type."
            if target_track["locked"] or target_track["pinned"]:
                return "Move target track is locked or pinned."
    elif kind == "trim":
        duration, source_in = action.get("duration"), action.get("source_in")
        asset = assets.get(str(item.get("asset_id")))
        if duration is None or duration <= 0 or source_in is None or source_in < 0:
            return "Trim needs a positive duration and non-negative source in."
        if not asset or not _source_range_valid(asset, source_in, duration, item["speed"]):
            return "Trim exceeds the source asset duration."
    elif kind == "remove":
        pass
    elif kind == "set-volume":
        if item["track_type"] == "text":
            return "Text clips do not have a volume control."
        if action.get("volume") is None or not 0 <= action["volume"] <= 4:
            return "Volume must be between 0 and 4."
    elif kind == "crossfade":
        if item["track_type"] != "video":
            return "Crossfades require a visual track."
        seconds = action.get("seconds")
        if seconds is None or seconds < 0:
            return "Crossfade duration must be zero or greater."
        previous = [candidate for candidate in source_track["items"] if candidate["start"] < item["start"] - 1e-7]
        previous_item = max(previous, key=lambda candidate: candidate["start"], default=None)
        if not previous_item:
            return "Crossfade needs a preceding clip on the same track."
        if previous_item["locked"] or previous_item["pinned"] or previous_item.get("link_group_id"):
            return "Crossfade cannot change a locked, pinned, or linked neighboring clip."
        if seconds > min(item["duration"], previous_item["duration"]):
            return "Crossfade is longer than one of its clips."
        following = [candidate for candidate in source_track["items"] if candidate["start"] >= item["start"] - 1e-7]
        if any(candidate["locked"] or candidate["pinned"] or candidate.get("link_group_id") for candidate in following):
            return "Crossfade cannot shift locked, pinned, or linked following clips."
    elif kind == "close-gap":
        previous_end = max((candidate["start"] + candidate["duration"]
                            for candidate in source_track["items"]
                            if candidate["id"] != item["id"]
                            and candidate["start"] + candidate["duration"] <= item["start"] + 1e-6),
                           default=0.0)
        gap = item["start"] - previous_end
        if gap <= 1e-6:
            return "There is no open gap before this clip."
        moving = [candidate for candidate in source_track["items"]
                  if candidate["start"] >= item["start"] - 1e-7]
        moving_ids = {candidate["id"] for candidate in moving}
        for affected in moving:
            if affected["locked"] or affected["pinned"] or affected.get("link_group_id"):
                return "Closing this gap would move a locked, pinned, or linked clip."
            shifted_start = affected["start"] - gap
            if shifted_start < -1e-6 or any(
                shifted_start < other["start"] + other["duration"] - 1e-6
                and shifted_start + affected["duration"] > other["start"] + 1e-6
                for other in source_track["items"] if other["id"] not in moving_ids
            ):
                return "Closing this gap would overlap another clip on this track."
    else:
        return "Unsupported action."
    return None


def _action_dedupe_key(action: Mapping[str, Any]) -> str:
    fields = ("type", "item_id", "asset_id", "track_id", "start", "duration", "source_in", "volume", "seconds")
    values = {key: action.get(key) for key in fields}
    if action.get("type") == "insert":
        values["item_id"] = None
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _normalize_model_plan(raw: Any, *, proposal_id: str, context: Mapping[str, Any],
                          pinned_ids: set[str], start_index: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str], list[str], list[str]]:
    draft = _redact_text(raw, MAX_DRAFT_CHARS + 1) if isinstance(raw, str) else _redact_value(raw)
    if isinstance(draft, dict):
        # Retain only the proposal vocabulary; unknown root keys may contain
        # arbitrary tool-call payloads and cannot help a focused repair.
        draft = {key: draft[key] for key in ("summary", "actions", "missing_media") if key in draft}
        if isinstance(draft.get("actions"), list):
            allowed_action_fields = set(ACTION_SCHEMA["properties"]["actions"]["items"]["required"])
            draft["actions"] = [
                {key: row[key] for key in allowed_action_fields if isinstance(row, dict) and key in row}
                if isinstance(row, dict) else row
                for row in draft["actions"][:MAX_ACTIONS]
            ]
        if isinstance(draft.get("missing_media"), list):
            draft["missing_media"] = draft["missing_media"][:MAX_MISSING_MEDIA]
    draft_text = draft if isinstance(draft, str) else json.dumps(draft, ensure_ascii=False, separators=(",", ":"), default=str)
    draft_truncated = len(draft_text) > MAX_DRAFT_CHARS
    draft_text = draft_text[:MAX_DRAFT_CHARS]
    draft_record = {"phase": "initial", "raw": draft_text, "truncated": draft_truncated}
    root, structure_error = _strict_root(raw)
    if root is None:
        return None, [draft_record], [], [], [structure_error]
    assert root is not None
    warnings: list[str] = []
    errors: list[str] = [structure_error] if structure_error else []
    invalid_rows: list[str] = []
    actions: list[dict[str, Any]] = []
    rows = root["actions"]
    if len(rows) > MAX_ACTIONS:
        warnings.append(f"Only the first {MAX_ACTIONS} actions were reviewed; the rest were omitted.")
        rows = rows[:MAX_ACTIONS]
    for offset, row in enumerate(rows):
        action, error = _read_action_row(row, start_index + offset, proposal_id)
        if error:
            invalid_rows.append(f"Action {offset + 1}: {error}")
            continue
        assert action is not None
        validation_error = _validate_action(action, context, pinned_ids)
        if validation_error:
            invalid_rows.append(f"Action {offset + 1}: {validation_error}")
            continue
        actions.append(action)
    if invalid_rows:
        warnings.extend(invalid_rows[:32])
    missing_media: list[str] = []
    if len(root["missing_media"]) > MAX_MISSING_MEDIA:
        warnings.append(f"Only {MAX_MISSING_MEDIA} missing-media notes were retained.")
    for value in root["missing_media"][:MAX_MISSING_MEDIA]:
        if isinstance(value, str) and value.strip():
            missing_media.append(_redact_text(value.strip(), 240))
        else:
            warnings.append("A malformed missing-media note was omitted.")
    normalized = {
        "summary": _redact_text(root["summary"].strip(), 400),
        "actions": actions,
        "missing_media": missing_media,
    }
    return normalized, [draft_record], warnings, invalid_rows, errors


def _call_writer(writer: Callable[..., Any], ledger: Mapping[str, Any], request: Mapping[str, Any]) -> Any:
    return writer(ledger=ledger, request=request, schema=copy.deepcopy(ACTION_SCHEMA))


def create_editor_auto_edit_proposal(
    settings_dir: str | os.PathLike[str],
    project: Mapping[str, Any],
    *,
    mode: str,
    workspace: str,
    client_intent_id: str,
    client_signature: str | None = None,
    writer_identity: Mapping[str, Any] | None = None,
    brief: str = "",
    target_seconds: float | None = None,
    writer: Callable[..., Any] | None = None,
    pinned_item_ids: list[str] | tuple[str, ...] | set[str] = (),
) -> dict[str, Any]:
    """Create a reviewable proposal; never apply or submit it as an edit."""
    proposal_id = _safe_id(client_intent_id, "client intent id")
    if len(proposal_id) > 144:
        raise EditorAutoEditError("Client intent id is too long.")
    if mode not in _MODES:
        raise EditorAutoEditError("Unsupported auto-edit mode.")
    workspace = str(workspace or "").strip()
    if not workspace or len(workspace) > 512:
        raise EditorAutoEditError("A valid Editor workspace is required.")
    brief = str(brief or "")
    if len(brief) > MAX_BRIEF_CHARS:
        raise EditorAutoEditError(f"Auto-edit brief must be at most {MAX_BRIEF_CHARS} characters.")
    brief = _redact_text(brief, MAX_BRIEF_CHARS)
    if target_seconds is not None:
        target_seconds = _finite(target_seconds, "Target duration", minimum=0, maximum=MAX_TARGET_SECONDS)
    if client_signature is not None:
        client_signature = str(client_signature)
        if len(client_signature) > 256:
            raise EditorAutoEditError("Client signature is too long.")
    pins = {str(value) for value in pinned_item_ids if str(value)}
    if len(pins) > MAX_CLIPS:
        raise EditorAutoEditError(f"At most {MAX_CLIPS} pinned clip ids can be supplied.")
    ledger, context = _project_ledger(project, workspace, pins)
    director_plan: tuple[list[dict[str, Any]], list[str], str] | None = None
    if mode == "director":
        director_plan = _director_actions("director-preview", ledger, context)
        if len(director_plan[0]) > MAX_ACTIONS:
            raise EditorAutoEditError(
                f"Director reorder needs more than {MAX_ACTIONS} actions; split the shots into a smaller review plan."
            )
    project_id = ledger["project_id"]
    base_revision = ledger["revision"]
    request_fingerprint = hashlib.sha256(json.dumps({
        "mode": mode,
        "brief": brief,
        "target_seconds": target_seconds,
        "workspace": workspace,
        "project_id": project_id,
        "base_revision": base_revision,
        "client_signature": client_signature,
        "ledger_sha256": context["sha256"],
        "pinned_item_ids": sorted(pins),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    now = time.time()
    record = {
        "schema_version": 1,
        "proposal_id": proposal_id,
        "client_intent_id": proposal_id,
        "project_id": project_id,
        "workspace": workspace,
        "base_revision": base_revision,
        "client_signature": client_signature,
        "ledger_sha256": context["sha256"],
        "request_fingerprint": request_fingerprint,
        "mode": mode,
        "brief": brief,
        "target_seconds": target_seconds,
        "status": "running",
        "writer": {key: _redact_text((writer_identity or {}).get(key), 200) for key in ("provider", "model_id")},
        "summary": "Preparing a reviewable edit proposal.",
        "actions": [],
        "warnings": [],
        "missing_media": [],
        "drafts": [],
        "errors": [],
        "created_at": now,
        "updated_at": now,
    }
    path = _proposal_path(settings_dir, proposal_id)
    with _lock:
        if path.exists():
            existing = _read_record(path)
            if existing.get("request_fingerprint") != request_fingerprint:
                raise EditorAutoEditError("This client intent id is already bound to another proposal request.")
            return copy.deepcopy(existing)
        if mode != "director" and writer is None:
            raise EditorAutoEditError("No local proposal writer is available.")
        _atomic_write(path, record)

    if mode == "director":
        assert director_plan is not None
        preview_actions, warnings, summary = director_plan
        actions = [
            {**action, "id": f"{proposal_id}-act-{index:03d}"}
            for index, action in enumerate(preview_actions, start=1)
        ]
        record.update({"status": "ready" if actions else "needs_review", "summary": summary, "actions": actions,
                       "warnings": warnings, "updated_at": time.time()})
        return save_editor_auto_edit_proposal(settings_dir, record)

    request: dict[str, Any] = {
        "mode": mode,
        "brief": brief,
        "target_seconds": target_seconds,
        "project_id": project_id,
        "workspace": workspace,
        "base_revision": base_revision,
        "max_actions": MAX_ACTIONS,
        "max_missing_media": MAX_MISSING_MEDIA,
        "instruction": "Return a proposal only. Do not apply edits, call tools, use paths, or add fields outside the schema.",
    }
    try:
        raw = _call_writer(writer, ledger, request)
    except Exception as error:  # keep the exception detail bounded and path-free
        record.update({"status": "failed", "summary": "The proposal writer could not prepare a plan.",
                       "errors": [_redact_text(error, 400)], "updated_at": time.time()})
        return save_editor_auto_edit_proposal(settings_dir, record)

    normalized, drafts, warnings, invalid_rows, errors = _normalize_model_plan(
        raw, proposal_id=proposal_id, context=context, pinned_ids=pins, start_index=1,
    )
    record["drafts"] = drafts
    record["warnings"] = warnings
    record["errors"] = errors
    if normalized is None:
        record["errors"] = errors or warnings
        record["summary"] = "The proposal response did not match the required JSON schema."
    else:
        record["summary"] = normalized["summary"]
        record["actions"] = normalized["actions"]
        record["missing_media"] = normalized["missing_media"]

    # One focused repair is allowed for invalid structure/rows. The repair
    # request contains only bounded diagnostics and the previous redacted
    # draft; valid rows from the first pass stay intact.
    repair_needed = normalized is None or bool(invalid_rows) or bool(errors)
    if repair_needed:
        repair_request = dict(request)
        repair_request["repair"] = {
            "instruction": "Repair only the invalid/missing plan rows. Return the same strict JSON object and keep valid prior rows unchanged.",
            "errors": (errors or invalid_rows)[:32],
            "invalid_action_indexes": [int(match.group(1)) for value in invalid_rows
                                       if (match := re.match(r"Action (\d+):", value))],
            "previous_draft": drafts[0]["raw"] if drafts else "",
            "valid_actions": copy.deepcopy(record["actions"]),
        }
        try:
            repaired_raw = _call_writer(writer, ledger, repair_request)
            repaired, repair_drafts, repair_warnings, repair_invalid, repair_errors = _normalize_model_plan(
                repaired_raw, proposal_id=proposal_id, context=context, pinned_ids=pins,
                start_index=len(record["actions"]) + 1,
            )
            if repair_drafts:
                repair_drafts[0]["phase"] = "repair"
                record["drafts"] = (record["drafts"] + repair_drafts)[:2]
            if repaired is not None:
                seen = {_action_dedupe_key(action) for action in record["actions"]}
                for action in repaired["actions"]:
                    key = _action_dedupe_key(action)
                    if key not in seen and len(record["actions"]) < MAX_ACTIONS:
                        record["actions"].append(action)
                        seen.add(key)
                if not record["summary"] or record["summary"] == "The proposal response did not match the required JSON schema.":
                    record["summary"] = repaired["summary"]
                if errors:
                    record["warnings"] = (record["warnings"] + errors)[:64]
                    record["errors"] = repair_errors[:16]
                record["missing_media"] = list(dict.fromkeys(
                    [*record["missing_media"], *repaired["missing_media"]]
                ))[:MAX_MISSING_MEDIA]
                record["status"] = "ready"
            else:
                record["status"] = "ready" if record["actions"] else "needs_review"
            record["warnings"] = (record["warnings"] + repair_warnings + repair_invalid)[:64]
            record["errors"] = (record["errors"] + repair_errors)[:16]
        except Exception as error:
            record["warnings"].append(f"Focused repair was unavailable: {_redact_text(error, 240)}")
            record["status"] = "ready" if record["actions"] else "needs_review"
    else:
        record["status"] = "ready"
    if not record["actions"] and record["status"] == "ready":
        record["status"] = "needs_review"
        if not record["warnings"]:
            record["warnings"] = ["No applicable edits were produced; review the proposal before continuing."]
    record["warnings"] = list(dict.fromkeys(record["warnings"]))[:64]
    record["errors"] = list(dict.fromkeys(record["errors"]))[:16]
    for index, action in enumerate(record["actions"], start=1):
        action["id"] = f"{proposal_id}-act-{index:03d}"
    record["warnings"] = record["warnings"][:64]
    record["errors"] = record["errors"][:16]
    record["updated_at"] = time.time()
    return save_editor_auto_edit_proposal(settings_dir, record)


def get_editor_auto_edit_proposal(settings_dir: str | os.PathLike[str], proposal_id: str) -> dict[str, Any]:
    path = _proposal_path(settings_dir, proposal_id)
    try:
        return _read_record(path)
    except FileNotFoundError as error:
        raise EditorAutoEditError("Editor auto-edit proposal was not found.") from error


def list_editor_auto_edit_proposals(
    settings_dir: str | os.PathLike[str], *, project_id: str | None = None,
    workspace: str | None = None, limit: int = 100,
) -> list[dict[str, Any]]:
    root = _settings_root(settings_dir)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise EditorAutoEditError("Proposal list limit must be between 1 and 500.")
    result: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        if path.is_symlink() or path.resolve(strict=False).parent != root:
            continue
        try:
            item = _read_record(path)
        except (OSError, EditorAutoEditError):
            continue
        if project_id is not None and item.get("project_id") != project_id:
            continue
        if workspace is not None and item.get("workspace") != workspace:
            continue
        result.append(item)
    def updated_at(item: Mapping[str, Any]) -> float:
        try:
            value = float(item.get("updated_at") or 0)
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0
    result.sort(key=updated_at, reverse=True)
    return result[:limit]


def save_editor_auto_edit_proposal(settings_dir: str | os.PathLike[str], proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Save a server-side proposal lifecycle update without changing identity."""
    if not isinstance(proposal, Mapping):
        raise EditorAutoEditError("Proposal must be an object.")
    proposal_id = _safe_id(proposal.get("proposal_id"), "proposal id")
    path = _proposal_path(settings_dir, proposal_id)
    with _lock:
        try:
            existing = _read_record(path)
        except FileNotFoundError as error:
            raise EditorAutoEditError("Editor auto-edit proposal was not found.") from error
        immutable = ("client_intent_id", "project_id", "workspace", "base_revision", "client_signature",
                     "ledger_sha256", "request_fingerprint", "mode", "brief", "target_seconds", "created_at", "writer")
        if any(proposal.get(key) != existing.get(key) for key in immutable):
            raise EditorAutoEditError("Proposal identity and source snapshot are immutable.")
        status = str(proposal.get("status") or "")
        if status not in _STATUSES:
            raise EditorAutoEditError("Proposal status is invalid.")
        actions = proposal.get("actions", existing.get("actions", []))
        warnings = proposal.get("warnings", existing.get("warnings", []))
        missing_media = proposal.get("missing_media", existing.get("missing_media", []))
        drafts = proposal.get("drafts", existing.get("drafts", []))
        errors = proposal.get("errors", existing.get("errors", []))
        if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
            raise EditorAutoEditError("Proposal action count exceeds the limit.")
        if existing.get("status") != "running" and actions != existing.get("actions", []):
            raise EditorAutoEditError("Reviewable proposal actions are immutable.")
        allowed_action_keys = set(ACTION_SCHEMA["properties"]["actions"]["items"]["required"])
        safe_actions: list[dict[str, Any]] = []
        action_ids: set[str] = set()
        for action in actions:
            if not isinstance(action, Mapping) or set(action) != allowed_action_keys:
                raise EditorAutoEditError("Proposal contains a malformed action.")
            action_id = str(action.get("id") or "")
            if not _ID_RE.fullmatch(action_id) or not action_id.startswith(f"{proposal_id}-act-"):
                raise EditorAutoEditError("Proposal action id was not assigned by this server proposal.")
            if action_id in action_ids or action.get("type") not in _ACTION_TYPES:
                raise EditorAutoEditError("Proposal action ids and types must be unique and valid.")
            action_ids.add(action_id)
            safe_action = dict(action)
            for key in ("item_id", "asset_id", "track_id"):
                value = safe_action.get(key)
                if value is not None and (not isinstance(value, str) or len(value) > 160):
                    raise EditorAutoEditError(f"Proposal action {key} is invalid.")
            for key in ("start", "duration", "source_in", "volume", "seconds"):
                if safe_action.get(key) is not None:
                    safe_action[key] = _finite(safe_action[key], f"Proposal action {key}")
            reason = safe_action.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise EditorAutoEditError("Proposal action reason must be text or null.")
            safe_action["reason"] = _redact_text(reason, 300) if reason else None
            safe_actions.append(safe_action)
        transition_map = {
            "running": {"running", "ready", "needs_review", "failed"},
            "ready": {"ready", "accepted", "rejected", "superseded"},
            "needs_review": {"needs_review", "accepted", "rejected", "superseded"},
            "failed": {"failed"},
            "accepted": {"accepted"},
            "rejected": {"rejected"},
            "superseded": {"superseded"},
        }
        if status not in transition_map.get(str(existing.get("status")), set()):
            raise EditorAutoEditError("Proposal status transition is not allowed.")
        for name, values, cap in (("warnings", warnings, 64), ("missing_media", missing_media, MAX_MISSING_MEDIA),
                                  ("drafts", drafts, 2), ("errors", errors, 16)):
            if not isinstance(values, list) or len(values) > cap:
                raise EditorAutoEditError(f"Proposal {name} exceed the limit.")
        result = dict(existing)
        result.update({
            "status": status,
            "summary": _redact_text(proposal.get("summary", existing.get("summary", "")), 400),
            "actions": safe_actions,
            "warnings": [_redact_text(value, 400) for value in warnings],
            "missing_media": [_redact_text(value, 240) for value in missing_media],
            "drafts": [
                {"phase": _redact_text(item.get("phase") or "draft", 20),
                 "raw": _redact_text(item.get("raw") or "", MAX_DRAFT_CHARS),
                 "truncated": bool(item.get("truncated")) or len(str(item.get("raw") or "")) > MAX_DRAFT_CHARS}
                if isinstance(item, Mapping) else {"phase": "draft", "raw": _redact_text(item, MAX_DRAFT_CHARS)}
                for item in drafts
            ],
            "errors": [_redact_text(value, 400) for value in errors],
            "updated_at": float(proposal.get("updated_at") or time.time()),
        })
        _atomic_write(path, result)
        return copy.deepcopy(result)


__all__ = [
    "ACTION_SCHEMA",
    "EditorAutoEditError",
    "create_editor_auto_edit_proposal",
    "get_editor_auto_edit_proposal",
    "list_editor_auto_edit_proposals",
    "save_editor_auto_edit_proposal",
]
