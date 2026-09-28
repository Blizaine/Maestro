"""Local LoRA presentation metadata, independent of weight filenames and IDs."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import tempfile
import threading


_NAMES_PATH = Path(__file__).resolve().parents[1] / "settings" / "lora_display_names.json"
_NAMES_LOCK = threading.RLock()
_GENERIC_TITLES = {"readme", "model card", "usage", "introduction", "lora", "model", "description"}


def _lora_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def lora_date_fields(weight_path: str | Path, meta: dict | None = None) -> dict:
    """Release/download dates are independent of mutable guides and sidecars.

    Old hash scans incorrectly stamped their scan time as downloadedAt. For
    unmarked legacy records, the original weight file bounds that date so a
    metadata refresh cannot make an older adapter newly downloaded. New real
    downloads have explicit provenance and retain their recorded install date.
    """
    meta = meta if isinstance(meta, dict) else {}
    released = _lora_timestamp(meta.get("publishedAt"))
    downloaded = _lora_timestamp(meta.get("downloadedAt"))
    try:
        file_date = datetime.fromtimestamp(os.path.getmtime(weight_path), timezone.utc)
    except (OSError, ValueError, OverflowError):
        file_date = None
    if file_date and (downloaded is None or (
        meta.get("downloadedAtSource") != "download" and file_date < downloaded
    )):
        downloaded = file_date
    def iso(value):
        return value.isoformat(timespec="seconds").replace("+00:00", "Z") if value else None
    return {"released_at": iso(released), "downloaded_at": iso(downloaded)}


def _plain_title(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = html.unescape(re.sub(r"<[^>]*>", "", value))
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[*`#]", "", value)
    return " ".join(value.split()).strip()[:160]


def suggested_lora_name(filename: str, meta: dict | None = None, guide_text: str = "") -> str:
    """Prefer publisher titles, with old HF sidecars upgraded at read time."""
    meta = meta if isinstance(meta, dict) else {}
    name = _plain_title(meta.get("name"))
    description = meta.get("description") or ""
    if meta.get("source") == "huggingface" and isinstance(description, str):
        heading = re.search(r"^#\s+(.+?)\s*#*\s*$", description, re.MULTILINE)
        title = _plain_title(heading.group(1)) if heading else ""
        if title and title.casefold() not in _GENERIC_TITLES:
            return title
    if name and name.casefold() not in _GENERIC_TITLES:
        return name
    suggested = _plain_title(meta.get("suggestedName"))
    if suggested:
        return suggested
    heading = re.search(r"^#\s+(.+?)\s*#*\s*$", guide_text, re.MULTILINE)
    title = _plain_title(heading.group(1)) if heading else ""
    if title and title.casefold() not in _GENERIC_TITLES:
        return title
    return Path(filename).stem.replace("_", " ")


def lora_version_label(filename: str, meta: dict | None = None) -> str | None:
    """Keep release/variant details separate from the user's editable title.

    Older sidecars omit the publisher's version name. Explicit version suffixes
    in their unchanged filenames still distinguish V1, V4 and V4-ref2va offline.
    """
    meta = meta if isinstance(meta, dict) else {}
    version = _plain_title(meta.get("versionName") or meta.get("version_name"))
    stem = Path(filename.replace("\\", "/")).stem
    matches = list(re.finditer(
        r"(?:^|[\s_-])v(?:ersion)?[\s_-]?(\d+(?:\.\d+)*)(?=$|[\s_-])",
        stem, re.IGNORECASE,
    ))
    if matches:
        match = matches[-1]
        if not version:
            version = "V" + match.group(1)
        variant = _plain_title(re.sub(r"[_-]+", " ", stem[match.end():]).strip())
        # A publisher may already include the variant in its release title.
        def normalize(value):
            return re.sub(r"[^\w]+", " ", value.casefold()).strip()
        if variant and normalize(variant) not in normalize(version):
            version += " · " + variant
    return version[:160] or None


def _alias_key(filename: str, meta: dict | None, directory: str) -> str:
    meta = meta if isinstance(meta, dict) else {}
    try:
        model_id = int(meta["modelId"])
    except (KeyError, TypeError, ValueError):
        model_id = None
    if model_id is not None:
        # A CivitAI update may change the filename; the user name belongs to
        # the same model across versions, just like activation/weight state.
        return f"civitai:{model_id}"
    # Local/HF filenames are the existing app identity. Include the library
    # directory so two different models named adapter.safetensors stay separate.
    return "local:" + os.path.normcase(os.path.normpath(os.path.join(directory, filename))).replace("\\", "/")


def load_lora_display_names(path: Path | None = None) -> dict:
    try:
        data = json.loads((path or _NAMES_PATH).read_text(encoding="utf-8"))
        names = data.get("names", {})
        return names if isinstance(names, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def lora_name_fields(filename: str, meta: dict | None, directory: str,
                     guide_text: str = "", names: dict | None = None) -> dict:
    names = load_lora_display_names() if names is None else names
    custom = names.get(_alias_key(filename, meta, directory))
    custom = custom.strip() if isinstance(custom, str) else ""
    suggested = suggested_lora_name(filename, meta, guide_text)
    return {"display_name": custom or suggested,
            "display_name_override": custom or None, "suggested_name": suggested,
            "version_label": lora_version_label(filename, meta)}


def set_lora_display_name(filename: str, meta: dict | None, directory: str,
                          value: str | None, *, path: Path | None = None) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError("Display name must be text or null.")
    value = (value or "").strip()
    if len(value) > 120 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Use a display name of up to 120 characters on one line.")
    target = path or _NAMES_PATH
    with _NAMES_LOCK:
        names = load_lora_display_names(target)
        key = _alias_key(filename, meta, directory)
        if value:
            names[key] = value
        else:
            names.pop(key, None)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                             prefix=".lora-names-", suffix=".tmp", delete=False) as output:
                temp_name = output.name
                json.dump({"version": 1, "names": names}, output, ensure_ascii=False, indent=2)
            os.replace(temp_name, target)
        finally:
            if temp_name and os.path.isfile(temp_name):
                os.remove(temp_name)


def _creator_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", value, flags=re.I | re.S)
    value = re.sub(r"</?(?:p|div|br|li|ul|ol|h[1-6]|table|thead|tbody|tr|td|th)\b[^>]*>", "\n", value, flags=re.I)
    value = re.sub(r"</?(?:a|span|strong|em|b|i|img|video|source)\b[^>]*>", "", value, flags=re.I)
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    return html.unescape(value).strip()


def source_lora_guide(meta: dict, filename: str = "") -> str:
    """Build useful creator-sourced guidance without loading an LLM/GPU.

    This is a source excerpt, not AI verification. The prompt writer receives
    it as bounded, untrusted LoRA context, separate from the user's scene.
    """
    parts = []
    triggers = meta.get("trainedWords") or []
    if isinstance(triggers, list):
        triggers = [word.strip() for word in triggers if isinstance(word, str) and word.strip()]
        if triggers:
            parts.append("Creator-declared trigger words: " + ", ".join(triggers[:20]))
    description = _creator_text(meta.get("description"))
    if description:
        # Keep the introduction plus usage sections; exclude long training
        # logs, credits and comparison galleries when Markdown separates them.
        sections = re.split(r"(?m)^(?=#{1,6}\s)", description)
        selected = []
        for index, section in enumerate(sections):
            heading = section.split("\n", 1)[0].lstrip("# ").lower()
            if index == 0 or (index == 1 and not sections[0].strip()) or re.search(
                r"\b(prompt|usage|use|recommend|recommended|trigger|tips|avoid|limitations|settings)\b", heading
            ):
                selected.append(section.strip())
        excerpt = "\n\n".join(filter(None, selected))[:5000]
        if excerpt:
            parts.append("Creator's usage notes:\n" + excerpt)
    version = _creator_text(meta.get("versionDescription"))
    if version:
        parts.append("Version notes:\n" + version[:1000])
    examples = meta.get("examplePrompts") or []
    if isinstance(examples, list):
        for example in examples[:2]:
            if isinstance(example, str) and example.strip():
                parts.append("Creator example (adapt the subject to the user's scene):\n" + example.strip()[:1200])
    if not parts:
        return ""
    return "Source guide for " + suggested_lora_name(filename, meta) + ".\n\n" + "\n\n".join(parts)
