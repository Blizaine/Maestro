"""Bounded review of visible music-video action, excluding prompt boilerplate."""
from __future__ import annotations

import copy
import hashlib
import json
import re


def music_visual_body(prompt):
    text = str(prompt or "")
    match = re.search(r"\bdetailed_description\s*:\s*(.*?)(?=\boverall_soundscape\s*:|\bnon_diegetic_music\s*:|\Z)", text, re.I | re.S)
    return " ".join((match.group(1) if match else text).split())


def music_review_fingerprint(rows, arc):
    payload = json.dumps({"revision": 2, "rows": rows, "arc": arc}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def music_visual_reset_issue(prompt, source):
    """Reject literal long global-brief reinsertion in the native visual body."""
    def normalized(text):
        return " ".join(re.findall(r"\w+", str(text or "").casefold()))
    brief = normalized(source)
    body = normalized(music_visual_body(prompt))
    if len(brief.split()) >= 18 and brief in body:
        return "The visual body copies the complete project brief; describe only this clip's assigned beat."
    return None


def repeated_music_visuals(rows, arc):
    """Catch literal recycled visual bodies even when a small reviewer misses them."""
    seen = {}
    findings = {}
    for index, row in enumerate(rows):
        body = music_visual_body(row.get("video_prompt"))
        key = " ".join(re.findall(r"\w+", body.casefold()))
        if len(key) < 50:
            continue
        if key in seen and not arc[index].get("intentional_reprise"):
            findings[index] = (
                f"This visual body repeats clip {seen[key] + 1} verbatim. "
                "Develop this clip's assigned beat while retaining its physical boundary facts."
            )
        else:
            seen[key] = index
    return findings


def creative_review_schema():
    return {"type": "array", "maxItems": 6, "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "clip_index": {"type": "integer", "minimum": 0},
            "category": {"type": "string", "enum": ["repeated_action", "whole_brief_reset", "arc_mismatch"]},
            "evidence": {"type": "string", "minLength": 24, "maxLength": 500},
            "related_clip_index": {"type": "integer", "minimum": -1},
            "related_evidence": {"type": "string", "maxLength": 500},
            "repair": {"type": "string", "minLength": 12, "maxLength": 600},
        },
        "required": ["clip_index", "category", "evidence", "related_clip_index", "related_evidence", "repair"],
    }}


def grounded_review_findings(findings, rows, allowed_indices):
    """A reviewer may flag only supplied clips with actual visual evidence."""
    result = {}
    if not isinstance(findings, list):
        raise ValueError("Creative review must return a JSON array")
    for item in findings:
        if not isinstance(item, dict):
            raise ValueError("Creative review finding must be an object")
        index = item.get("clip_index")
        if type(index) is not int or index not in allowed_indices:
            raise ValueError("Creative review cited a clip outside its review window")
        if index in result:
            raise ValueError("Creative review repeated a clip index")
        body = music_visual_body(rows[index].get("video_prompt"))
        quote = " ".join(str(item.get("evidence") or "").split())
        repair = str(item.get("repair") or "").strip()
        if len(quote) < 24 or quote not in body or len(repair) < 12:
            raise ValueError(f"Creative review of clip {index + 1} needs an exact visual quote and specific repair")
        category = item.get("category")
        if category not in {"repeated_action", "whole_brief_reset", "arc_mismatch"}:
            raise ValueError("Creative review returned an unknown category")
        if category == "repeated_action":
            other = item.get("related_clip_index")
            other_quote = " ".join(str(item.get("related_evidence") or "").split())
            if (type(other) is not int or other not in allowed_indices or other == index
                    or len(other_quote) < 24
                    or other_quote not in music_visual_body(rows[other].get("video_prompt"))):
                raise ValueError("Repeated-action review needs evidence from two supplied clips")
        result[index] = f"{category}: {quote}\nRepair only this clip: {repair}"
    return result


def repair_music_rows(rows, findings, repair_row, validate_row, *, checkpoint=None):
    """Repair flagged rows only, keeping every unflagged row byte-equivalent."""
    current = copy.deepcopy(rows)
    for index, feedback in sorted(findings.items()):
        issue = ""
        for _ in range(2):
            replacement = repair_row(index, feedback + ("\nValidation: " + issue if issue else ""), current)
            if not isinstance(replacement, list) or len(replacement) != 1 or not isinstance(replacement[0], dict):
                issue = "Return exactly one complete replacement clip"
                continue
            candidate = copy.deepcopy(current)
            candidate[index] = replacement[0]
            issue = ""
            # Recheck all boundaries, including the unchanged following clip.
            # A focused repair cannot invalidate good downstream physical state.
            for position, row in enumerate(candidate):
                issue = validate_row(position, row, candidate[:position])
                if issue:
                    break
            if not issue:
                current = candidate
                if checkpoint:
                    checkpoint(current)
                break
        if issue:
            raise RuntimeError(f"Music-video clip {index + 1} creative repair needs revision: {issue}")
    return current
