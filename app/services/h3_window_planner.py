"""Structured MiniMax H3 sliding-window prompt planning.

The generic rolling-window scheduler can select a different prompt for every
pass, but historically H3 received the same complete timeline each time.  This
module plans the timeline once, then deterministically compiles one compact
Context-IR prompt per continuation window.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable


def compute_h3_window_boundaries(
    total_frames: int,
    window_frames: int,
    *,
    fps: float = 24.0,
    overlap_frames: int = 1,
    discard_frames: int = 0,
) -> list[dict[str, Any]]:
    """Return the exact committed-output span owned by every H3 pass."""

    total = max(1, int(total_frames))
    window = max(1, int(window_frames))
    fps_value = max(1.0, float(fps))
    overlap = max(0, min(window - 1, int(overlap_frames)))
    discard = max(0, min(window - overlap - 1, int(discard_frames)))
    stride = max(1, window - overlap - discard)

    if total <= window:
        count = 1
    else:
        left_after_first = total - window + discard
        count = 1 + int(math.ceil(left_after_first / float(stride)))

    boundaries: list[dict[str, Any]] = []
    committed_start = 0
    for index in range(count):
        committed_length = window if index == 0 else stride
        committed_end = min(total, committed_start + committed_length)
        boundaries.append(
            {
                "index": index + 1,
                "start_frame": committed_start,
                "end_frame": committed_end,
                "start_seconds": round(committed_start / fps_value, 3),
                "end_seconds": round(committed_end / fps_value, 3),
            }
        )
        committed_start = committed_end
    return boundaries


def h3_window_plan_signature(
    prompt: str,
    *,
    model_type: str,
    resolution: str,
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
    discard_frames: int,
    fps: float,
    has_start_image: bool,
    has_end_image: bool,
) -> str:
    """Fingerprint every input that can change a window plan."""

    payload = {
        "prompt": str(prompt or "").strip(),
        "model_type": str(model_type or ""),
        "resolution": str(resolution or ""),
        "total_frames": int(total_frames),
        "window_frames": int(window_frames),
        "overlap_frames": int(overlap_frames),
        "discard_frames": int(discard_frames),
        "fps": round(float(fps), 6),
        "has_start_image": bool(has_start_image),
        "has_end_image": bool(has_end_image),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _compact(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip(" \t\r\n-.,;:!?")
    if len(text) <= limit:
        return text
    prefix = text[: limit + 1]
    # Prefer a complete sentence, then a complete clause. The old hard word
    # cut produced fragments such as "The road continues toward the.." and
    # "His posture is..", which are actively harmful inside an H3 prompt.
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?](?=\s|$)", prefix)
        if match.end() >= max(36, int(limit * 0.35))
    ]
    if sentence_ends:
        shortened = prefix[: sentence_ends[-1]]
    else:
        clause_ends = [
            match.start()
            for match in re.finditer(r"[,;:](?=\s|$)", prefix)
            if match.start() >= max(36, int(limit * 0.45))
        ]
        if clause_ends:
            shortened = prefix[: clause_ends[-1]]
        else:
            shortened = prefix[:limit].rsplit(" ", 1)[0]
    return shortened.rstrip(" \t\r\n-.,;:!?")


def _dialogue_sentence(item: Any, speaker_ids: dict[str, str]) -> str:
    if not isinstance(item, dict):
        return ""
    text = " ".join(str(item.get("text") or "").split()).strip()
    if not text:
        return ""
    speaker = _compact(item.get("speaker") or "Speaker", 80)
    key = speaker.casefold()
    requested_id = str(item.get("speaker_id") or "").upper().strip("() ")
    if not re.fullmatch(r"S\d+", requested_id):
        requested_id = speaker_ids.get(key) or f"S{len(speaker_ids) + 1}"
    speaker_ids.setdefault(key, requested_id)
    stable_id = speaker_ids[key]
    language = _compact(item.get("language") or "English", 30)
    delivery = _compact(item.get("delivery") or "speaks naturally", 100)
    action = _compact(item.get("action") or "", 120)
    lead = f"{speaker} ({stable_id}) {delivery}"
    if action:
        lead += f" while {action}"
    return f"{lead}: <d>[{language}] {text}</d>. Immediately after the line, the speaker closes their mouth."


def compile_h3_window_prompts(
    plan: dict[str, Any],
    boundaries: Iterable[dict[str, Any]],
    *,
    has_start_image: bool = False,
    has_end_image: bool = False,
) -> list[dict[str, Any]]:
    """Compile a planner JSON object into complete, window-local H3 prompts."""

    spans = list(boundaries)
    windows = plan.get("windows") if isinstance(plan, dict) else None
    if not isinstance(windows, list) or len(windows) != len(spans):
        raise ValueError(
            f"H3 window planner returned {len(windows or [])} windows; expected {len(spans)}."
        )

    subjects = _compact(plan.get("subject_continuity"), 260)
    setting = _compact(plan.get("setting_continuity"), 190)
    visual = _compact(plan.get("visual_continuity"), 150)
    initial_state = _compact(plan.get("initial_state"), 170)
    ambient = _compact(plan.get("ambient_audio") or "Natural location ambience", 150)
    music = _compact(plan.get("music") or "N/A", 100)
    shared_visual = ". ".join(item for item in (subjects, setting, visual) if item)
    speaker_ids: dict[str, str] = {}
    compiled: list[dict[str, Any]] = []
    previous_closing = initial_state or "The requested scene is established in its opening composition"

    for position, (span, item) in enumerate(zip(spans, windows)):
        if not isinstance(item, dict):
            raise ValueError(f"H3 window {position + 1} is not an object.")
        action = _compact(item.get("action"), 430)
        closing = _compact(item.get("closing_state"), 180)
        effects = _compact(item.get("sound_effects") or "No one-time effect", 150)
        if not action:
            raise ValueError(f"H3 window {position + 1} has no action.")
        if not closing:
            closing = "The action holds in a concrete state ready to continue" if position + 1 < len(spans) else "The requested final beat settles naturally"

        duration = max(0.1, float(span["end_seconds"]) - float(span["start_seconds"]))
        continuity_instruction = (
            "This is the opening window of one continuous shot."
            if position == 0
            else "Continue directly from the supplied previous frame; do not restart, recap, or repeat earlier action."
        )
        outcome_instruction = (
            "The central outcome remains unresolved at this boundary."
            if position + 1 < len(spans)
            else "Complete the requested outcome only in this final window."
        )
        dialogue = " ".join(
            sentence
            for sentence in (
                _dialogue_sentence(dialogue_item, speaker_ids)
                for dialogue_item in (item.get("dialogue") or [])
            )
            if sentence
        )
        silence = (
            "Only the tagged lines are spoken once in order. Outside those lines, everyone is silent with mouths closed; no background voices, muttering, or gibberish."
            if dialogue
            else "Everyone remains silent with mouths closed throughout this window; no voices, muttering, or gibberish."
        )

        picture_instructions: list[str] = []
        if position > 0 or has_start_image:
            picture_instructions.append(
                "For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced."
            )
        if has_end_image and position + 1 == len(spans):
            picture_instructions.append(
                f"At {duration:.2f} seconds, <Picture 2> is the required final-frame destination."
            )

        visual_parts = [
            f"[Shot 1] {shared_visual}" if shared_visual else "[Shot 1] A continuous cinematic scene",
            continuity_instruction,
            f"At 0.00 seconds, {previous_closing}.",
            f"From 0.00 to {duration:.2f} seconds, {action}.",
        ]
        if dialogue:
            visual_parts.append(dialogue)
        visual_parts.extend(
            [
                silence,
                outcome_instruction,
                f"The window ends with {closing}.",
            ]
        )
        soundscape = ambient
        if position > 0:
            soundscape += "; the same ambience continues seamlessly without restarting"
        if effects and effects.casefold() not in {"n/a", "none", "no one-time effect"}:
            soundscape += f". Synchronized effects in this window: {effects}"
        music_value = music
        if music.casefold() != "n/a" and position > 0:
            music_value = f"{music}; continue seamlessly from the preceding window without restarting"

        prompt_parts = picture_instructions + [
            f"integrated_multimodal_description: {' '.join(visual_parts)}",
            f"overall_soundscape: {soundscape}.",
            f"non_diegetic_music: {music_value}",
        ]
        prompt = "\n\n".join(prompt_parts)
        compiled.append(
            {
                **span,
                "title": _compact(item.get("title") or f"Window {position + 1}", 80),
                "opening_state": previous_closing,
                "closing_state": closing,
                "prompt": prompt,
            }
        )
        previous_closing = closing
    return compiled


def _schema(window_count: int) -> dict[str, Any]:
    dialogue_item = {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "speaker_id": {"type": "string"},
            "language": {"type": "string"},
            "delivery": {"type": "string"},
            "action": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["speaker", "speaker_id", "language", "delivery", "action", "text"],
        "additionalProperties": False,
    }
    window_item = {
        "type": "object",
        "properties": {
            "window": {"type": "integer"},
            "title": {"type": "string"},
            "action": {"type": "string"},
            "dialogue": {"type": "array", "items": dialogue_item},
            "sound_effects": {"type": "string"},
            "closing_state": {"type": "string"},
        },
        "required": ["window", "title", "action", "dialogue", "sound_effects", "closing_state"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "subject_continuity": {"type": "string"},
            "setting_continuity": {"type": "string"},
            "visual_continuity": {"type": "string"},
            "initial_state": {"type": "string"},
            "ambient_audio": {"type": "string"},
            "music": {"type": "string"},
            "windows": {
                "type": "array",
                "minItems": window_count,
                "maxItems": window_count,
                "items": window_item,
            },
        },
        "required": [
            "subject_continuity",
            "setting_continuity",
            "visual_continuity",
            "initial_state",
            "ambient_audio",
            "music",
            "windows",
        ],
        "additionalProperties": False,
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    try:
        import json_repair

        value = json_repair.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


_UNREQUESTED_SPECTACLE_PATTERNS = (
    r"\bgolden\s+energy\b",
    r"\b(?:visible|glowing|luminous|colored|coloured)?\s*energy\s+"
    r"(?:wave|pulse|blast|beam|field|surge|aura)\b",
    r"\bforce\s+field\b",
    r"\btelekin(?:esis|etic)\b",
    r"\b(?:magic|magical)\s+(?:aura|blast|beam|wave|field)\b",
)


def _narrative_dialogue_expected(prompt: str, window_count: int) -> bool:
    """Return whether a long character interaction should not be all-mute."""

    source = " ".join(str(prompt or "").split())
    lowered = source.casefold()
    if int(window_count) < 2:
        return False
    if re.search(
        r"\b(?:silent|silently|no dialogue|without dialogue|nonverbal|"
        r"music video|montage|instrumental|landscape|establishing shot)\b",
        lowered,
    ):
        return False
    if re.search(r"\".+?\"", source):
        return True
    interaction = re.search(
        r"\b(?:talk|speak|say|ask|answer|discuss|argue|confront|attack|"
        r"threaten|rescue|save|protect|warn|interview|can't believe|"
        r"cannot believe|disbelie|astonish|surpris)\w*\b",
        lowered,
    )
    # Two distinct multi-token proper names are a conservative signal that
    # this is a character scene rather than a generic action montage.
    names = {
        match.group(0).casefold()
        for match in re.finditer(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
            source,
        )
    }
    return bool(interaction and len(names) >= 2)


def _plan_contract_violations(
    source_prompt: str,
    plan: dict[str, Any] | None,
    *,
    expect_dialogue: bool,
) -> list[str]:
    """Find high-impact errors that a small planner LLM may ignore."""

    if not isinstance(plan, dict):
        return ["invalid plan"]
    rendered = json.dumps(plan, ensure_ascii=False)
    lowered_source = str(source_prompt or "").casefold()
    lowered_plan = rendered.casefold()
    violations: list[str] = []

    invented = []
    for pattern in _UNREQUESTED_SPECTACLE_PATTERNS:
        generated_match = re.search(pattern, lowered_plan, flags=re.IGNORECASE)
        if (
            generated_match
            and not re.search(pattern, lowered_source, flags=re.IGNORECASE)
        ):
            invented.append(generated_match.group(0).strip())
    if invented:
        violations.append(
            "invented unrequested power/effect: "
            + ", ".join(sorted(set(invented)))
        )

    windows = plan.get("windows") or []
    timeline_text = json.dumps(
        [
            {
                "action": item.get("action"),
                "closing_state": item.get("closing_state"),
                "sound_effects": item.get("sound_effects"),
                "dialogue": item.get("dialogue"),
            }
            for item in windows
            if isinstance(item, dict)
        ],
        ensure_ascii=False,
    ).casefold()
    if re.search(
        r"(?:\bcut\s+at\s+\d|\bglobal\s+\d+(?:\.\d+)?\s*s|"
        r"\[shot\s+[2-9]\]|\bwindow\s+[1-9]\b)",
        timeline_text,
        flags=re.IGNORECASE,
    ):
        violations.append("used a window boundary as a global edit/timestamp")

    if expect_dialogue:
        has_dialogue = any(
            isinstance(item, dict)
            and any(
                isinstance(line, dict)
                and str(line.get("text") or "").strip()
                for line in (item.get("dialogue") or [])
            )
            for item in windows
        )
        if not has_dialogue:
            violations.append(
                "left a long named-character interaction entirely mute"
            )

    return violations


def _fallback_plan(prompt: str, window_count: int) -> dict[str, Any]:
    """A no-LLM fallback that never exposes the full plot to every window."""

    source = " ".join(str(prompt or "").split()).strip()
    source = re.sub(
        r"^(?:integrated_multimodal_description|overall_soundscape|non_diegetic_music):\s*",
        "",
        source,
        flags=re.IGNORECASE,
    )
    # Split explicit temporal transitions and common "setup + obligation"
    # concepts.  The latter matters for prompts such as "walks through town
    # and has to save a truck": leaving that whole sentence in pass one lets
    # H3 see (and finish) the rescue before a continuation begins.
    beat_separator = re.compile(
        r"(?<=[.!?])\s+|\s*;\s*|"
        r"\s+(?:and\s+then|then|after\s+that|next)\s+|"
        r"\s+and\s+(?:has|needs|must|tries|attempts)\s+to\s+",
        flags=re.IGNORECASE,
    )
    beats = [
        part.strip(" ,;:-")
        for part in beat_separator.split(source)
        if part.strip(" ,;:-")
    ]
    if not beats:
        beats = ["Establish and begin the requested action"]
    beat_buckets: list[list[str]] = [[] for _ in range(window_count)]
    if len(beats) == 1:
        beat_buckets[0].append(beats[0])
    else:
        for beat_index, beat in enumerate(beats):
            target = int(round(beat_index * (window_count - 1) / (len(beats) - 1)))
            beat_buckets[min(window_count - 1, max(0, target))].append(beat)
    windows = []
    for index in range(window_count):
        assigned = beat_buckets[index]
        if index == 0:
            opening_beat = assigned or beats[:1]
            action = f"Establish the scene and begin: {' '.join(opening_beat)}. Do not complete the central outcome"
        elif index + 1 == window_count:
            remaining = " ".join(assigned).strip()
            action = "Continue from the prior physical state and complete the requested central action and outcome"
            if remaining:
                action += f": {remaining}"
        else:
            middle = " ".join(assigned).strip()
            action = "Continue from the prior physical state and visibly advance toward the central outcome"
            if middle:
                action += f": {middle}"
            action += ". Keep the outcome unresolved"
        windows.append(
            {
                "window": index + 1,
                "title": f"Continuation {index + 1}",
                "action": action,
                "dialogue": [],
                "sound_effects": "Natural synchronized effects for the visible action",
                "closing_state": (
                    "The visible action pauses in a concrete position ready for the next continuation"
                    if index + 1 < window_count
                    else "The requested outcome settles into a clear final beat"
                ),
            }
        )
    return {
        "subject_continuity": "Keep every established subject's identity, appearance, wardrobe, and carried objects unchanged",
        "setting_continuity": "Keep the established location, geography, time of day, and background elements unchanged",
        "visual_continuity": "One continuous shot with consistent lighting, color, and camera language",
        "initial_state": "The requested scene is established and the subjects begin in natural positions",
        "ambient_audio": "Continuous natural ambience appropriate to the established location",
        "music": "N/A",
        "windows": windows,
    }


def plan_h3_sliding_windows(
    prompt: str,
    *,
    model_type: str,
    resolution: str,
    total_frames: int,
    window_frames: int,
    overlap_frames: int = 1,
    discard_frames: int = 0,
    fps: float = 24.0,
    has_start_image: bool = False,
    has_end_image: bool = False,
    image_paths: list[str] | None = None,
    nsfw: bool = False,
) -> dict[str, Any]:
    """Use Maestro's configured LLM to create and compile an H3 window plan."""

    boundaries = compute_h3_window_boundaries(
        total_frames,
        window_frames,
        fps=fps,
        overlap_frames=overlap_frames,
        discard_frames=discard_frames,
    )
    signature = h3_window_plan_signature(
        prompt,
        model_type=model_type,
        resolution=resolution,
        total_frames=total_frames,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        discard_frames=discard_frames,
        fps=fps,
        has_start_image=has_start_image,
        has_end_image=has_end_image,
    )
    if len(boundaries) <= 1:
        return {
            "source_prompt": str(prompt or ""),
            "signature": signature,
            "planned_by": "not_needed",
            "total_frames": int(total_frames),
            "window_frames": int(window_frames),
            "window_count": 1,
            "windows": [],
            "window_prompts": [],
        }

    from services import llm_service
    from services.guide_loader import load_guide

    guide = load_guide("enhance", "minimax_h3_sliding_windows")
    if nsfw:
        # This planner only divides an existing request over time. The general
        # mature-mode enhancer supplement is intentionally not injected here:
        # it is much longer than this guide and can make a clean prompt more
        # elaborate even though its rules are nominally self-gating. Preserve
        # requested mature material without censoring or creatively escalating
        # it, exactly as the ordinary source-fidelity rules require.
        guide += (
            "\n\nMATURE-MODE FIDELITY\n"
            "Preserve mature or intense material only when the user explicitly "
            "requested it. Do not censor it, add to it, or intensify it."
        )

    boundary_lines = "\n".join(
        f"- Window {span['index']}: owns global {span['start_seconds']:.3f}s to {span['end_seconds']:.3f}s "
        f"({span['start_frame']}..{span['end_frame']} committed frames); write its JSON action using "
        f"local time 0.000s to {(span['end_seconds'] - span['start_seconds']):.3f}s"
        for span in boundaries
    )
    media_context = []
    if has_start_image:
        media_context.append("<Picture 1> is the exact first frame and visual identity/scene anchor.")
    if has_end_image:
        picture_index = 2 if has_start_image else 1
        media_context.append(f"<Picture {picture_index}> is the required final frame destination.")
    user_prompt = (
        f"Total output: {int(total_frames)} frames at {float(fps):g} fps.\n"
        f"Exact continuation windows:\n{boundary_lines}\n\n"
        "Global times only assign story beats to windows. Every generated H3 pass has its own local "
        "timeline beginning at 0.000s; never place a global timestamp inside a JSON field.\n\n"
        + ("Reference roles:\n" + "\n".join(media_context) + "\n\n" if media_context else "")
        + f"User concept:\n{prompt}"
    )

    schema = _schema(len(boundaries))
    expect_dialogue = _narrative_dialogue_expected(prompt, len(boundaries))
    if expect_dialogue:
        user_prompt += (
            "\n\nThis is a long narrative interaction between named "
            "characters and the user did not request silence. Include a "
            "concise, natural, portrayal-appropriate exchange or vocal "
            "reaction in the dialogue arrays of the windows where it belongs."
        )
    planned_by = "llm"
    raw = ""
    try:
        raw = llm_service.generate(
            prompt=user_prompt,
            system_prompt=guide,
            max_new_tokens=max(1400, len(boundaries) * 520 + 500),
            temperature=0.25,
            top_p=0.85,
            image_paths=image_paths or None,
            enable_thinking=False,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            json_schema=schema,
        )
        plan = _parse_json_object(raw)
        violations = _plan_contract_violations(
            prompt,
            plan,
            expect_dialogue=expect_dialogue,
        )
        if (
            not plan
            or len(plan.get("windows") or []) != len(boundaries)
            or violations
        ):
            reason = (
                "; ".join(violations)
                if violations
                else "invalid JSON/window count"
            )
            print(
                f"[MiniMax H3] Window planner contract repair: "
                f"{reason}; retrying once."
            )
            raw = llm_service.generate(
                prompt=(
                    user_prompt
                    + f"\n\nRETRY: Return exactly {len(boundaries)} window objects. "
                    "Do not omit or add a window. Treat every boundary as a "
                    "continuation using local time, never as a cut or global "
                    "timestamp. Do not invent powers, energy effects, "
                    "wardrobe, or lore absent from the exact named portrayal. "
                    + (
                        "Include at least one concise, natural dialogue or "
                        "vocal-reaction turn in the appropriate window."
                        if expect_dialogue
                        else "Preserve the user's requested speech or silence exactly."
                    )
                ),
                system_prompt=guide,
                max_new_tokens=max(1400, len(boundaries) * 520 + 500),
                temperature=0.1,
                top_p=0.8,
                image_paths=image_paths or None,
                enable_thinking=False,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                json_schema=schema,
            )
            plan = _parse_json_object(raw)
            violations = _plan_contract_violations(
                prompt,
                plan,
                expect_dialogue=expect_dialogue,
            )
        if not plan or len(plan.get("windows") or []) != len(boundaries):
            raise ValueError("The local LLM did not return the required H3 window count.")
        if violations:
            raise ValueError(
                "The local LLM's repaired H3 window plan still violated "
                "source fidelity: " + "; ".join(violations)
            )
        compiled = compile_h3_window_prompts(
            plan,
            boundaries,
            has_start_image=has_start_image,
            has_end_image=has_end_image,
        )
    except Exception as error:
        print(f"[MiniMax H3] Window planner fallback: {error}")
        planned_by = "deterministic_fallback"
        plan = _fallback_plan(prompt, len(boundaries))
        compiled = compile_h3_window_prompts(
            plan,
            boundaries,
            has_start_image=has_start_image,
            has_end_image=has_end_image,
        )

    return {
        "source_prompt": str(prompt or ""),
        "signature": signature,
        "planned_by": planned_by,
        "total_frames": int(total_frames),
        "window_frames": int(window_frames),
        "window_count": len(compiled),
        "resolution": str(resolution or ""),
        "model_type": str(model_type or ""),
        "subject_continuity": plan.get("subject_continuity", ""),
        "setting_continuity": plan.get("setting_continuity", ""),
        "windows": compiled,
        "window_prompts": [item["prompt"] for item in compiled],
    }
