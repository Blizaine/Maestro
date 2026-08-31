"""Window-local prompt planning for Maestro's LTX long-form sequences.

LTX rolling generation has always accepted one prompt per sliding window, but
the old Studio UI exposed that behavior implicitly: extending Duration could
silently create more windows and a multiline prompt was split without an
explicit contract.  This module gives every LTX generation the same durable
contract as H3: compute the exact window geometry, validate manual prompts,
or expand one story idea into one chronological prompt per native pass.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence


def compute_ltx_window_count(
    total_frames: int,
    window_frames: int,
    *,
    overlap_frames: int = 0,
    discard_frames: int = 0,
) -> int:
    """Match WanGP's rolling-window count without importing its GPU runtime."""

    total = max(1, int(total_frames or 0))
    window = max(1, int(window_frames or 0))
    overlap = max(0, int(overlap_frames or 0))
    discard = max(0, int(discard_frames or 0))
    if total <= window:
        return 1
    stride = window - discard - overlap
    if stride <= 0:
        raise ValueError(
            "LTX Window Length must be greater than its overlap and discarded tail."
        )
    return 1 + math.ceil((total - window + discard) / stride)


def _collapse_prompt(value: str) -> str:
    return " ".join(str(value or "").strip().split())


_OPEN_ENDED_MOTION_RE = re.compile(
    r"\b(?:non[-\s]?stop|never[-\s]?ending|endless|perpetual|infinite|"
    r"forever|never\s+(?:slows?|stops?|stopping|ends?|ending)|"
    r"keeps?\s+(?:moving|falling|travelling|traveling|going))\b",
    re.IGNORECASE,
)

_VOCAL_PERFORMANCE_RE = re.compile(
    r"\b(?:lip[-\s]?sync(?:s|ing|ed)?|sing(?:s|ing|er|ers)?|"
    r"rap(?:s|ping|per|pers)?|vocalist(?:s)?|perform(?:s|ing)\s+"
    r"(?:the\s+)?(?:song|lyrics|verse|chorus))\b",
    re.IGNORECASE,
)


def requests_open_ended_ltx_motion(prompt: str) -> bool:
    """Whether the requested sequence must remain active beyond its last pass."""

    return bool(_OPEN_ENDED_MOTION_RE.search(_collapse_prompt(prompt)))


def extract_ltx_global_invariants(prompt: str) -> list[str]:
    """Extract explicit sequence-wide rules that every native pass must see.

    The LLM guide remains responsible for arbitrary creative continuity.  This
    small deterministic layer protects the high-impact rules users most often
    express globally (capture language, pace, persistent motion, era, spatial
    domain, one-take continuity, and unusual path geometry).  It deliberately
    avoids copying the complete story, which would make every pass perform all
    future beats again.
    """

    source = _collapse_prompt(prompt)
    lowered = source.lower()
    invariants: list[str] = []
    open_ended_motion = requests_open_ended_ltx_motion(source)

    def add(value: str) -> None:
        if value and value not in invariants:
            invariants.append(value)

    has_handheld = bool(re.search(r"\bhandheld\b", lowered))
    has_pov = bool(
        re.search(r"\b(?:pov|point[-\s]of[-\s]view|first[-\s]person)\b", lowered)
    )
    has_static = bool(re.search(r"\b(?:locked[-\s]?off|static camera)\b", lowered))
    has_aerial = bool(
        re.search(r"\b(?:aerial|drone)\s+(?:camera|shot|view)\b", lowered)
    )
    camera_mode_count = sum((has_handheld, has_pov, has_static, has_aerial))
    handheld_is_global = bool(
        re.search(
            r"\b(?:amateur\s+)?handheld\s+(?:recording|footage|video|camera|style)\b",
            lowered,
        )
    )
    if has_handheld and camera_mode_count == 1 and handheld_is_global:
        add(
            "an amateur handheld recording"
            if "amateur" in lowered
            else "handheld camera movement"
        )
    elif camera_mode_count <= 1 and re.search(r"\bfound[-\s]?footage\b", lowered):
        add("found-footage camera language")
    elif has_pov and camera_mode_count == 1:
        add("a first-person point of view")
    elif has_static and camera_mode_count == 1:
        add("a locked-off static camera")

    has_fast_pacing = bool(
        re.search(
            r"\b(?:very\s+fast|extremely\s+fast|high[-\s]?speed|rapid|frantic|"
            r"breakneck|supersonic)\b",
            lowered,
        )
    )
    has_slow_pacing = bool(
        re.search(
            r"\b(?:slow[-\s]?moving|very\s+slow|gentle pace|slow motion)\b",
            lowered,
        )
    )
    if has_fast_pacing and not has_slow_pacing and open_ended_motion:
        add("very fast movement and pacing")
    elif has_slow_pacing and not has_fast_pacing and open_ended_motion:
        add("deliberately slow movement and pacing")

    has_falling_motion = bool(
        re.search(r"\b(?:falling|plummeting|plunging|descending|descent)\b", lowered)
    )
    if has_falling_motion and (
        open_ended_motion
        or re.search(
            r"\b(?:continuous(?:ly)?\s+(?:falling|plummeting|descending)|"
            r"keeps?\s+falling)\b",
            lowered,
        )
    ):
        add("continuous falling motion")
    elif re.search(r"\b(?:keeps? moving|continuous movement|moving forward)\b", lowered):
        add("continuous forward motion")

    if open_ended_motion:
        add("nonstop motion that never slows, settles, stops, or resolves")

    decade = re.search(
        r"\b((?:19|20)\d{2}|\d{2})s[-\s]?(?:style|aesthetic|look)\b",
        lowered,
    )
    if not decade:
        decade = re.search(
            r"\b(?:all|every|each|different|throughout)\b.{0,48}?"
            r"\b((?:19|20)\d{2}|\d{2})s\b",
            lowered,
        )
    if decade:
        add(f"{decade.group(1)}s visual design")
    for pattern, description in (
        (r"\bphotoreal(?:istic|ism)?\b", "photorealistic live-action imagery"),
        (r"\blive[-\s]?action\b", "live-action imagery"),
        (r"\bstop[-\s]?motion\b", "stop-motion animation"),
        (r"\banime\b", "anime visual styling"),
        (r"\bfilm noir\b", "film-noir visual styling"),
        (r"\bblack[-\s]and[-\s]white\b", "black-and-white imagery"),
        (r"\bvhs\b", "VHS-era image texture"),
    ):
        if re.search(pattern, lowered):
            add(description)

    if re.search(
        r"\b(?:all|entirely|completely)\s+(?:remain\s+)?indoors?\b",
        lowered,
    ):
        add("every environment remaining indoors")
    elif re.search(
        r"\b(?:all|entirely|completely)\s+(?:remain\s+)?outdoors?\b",
        lowered,
    ):
        add("every environment remaining outdoors")

    if (
        re.search(r"\b(?:seamless|unbroken|continuous)\b", lowered)
        or re.search(r"\b(?:one[-\s]?take|oner|one[-\s]?er)\b", lowered)
    ) and re.search(
        r"\b(?:no\s+(?:clear\s+)?cuts?|without\s+cuts?|one[-\s]?take|oner|one[-\s]?er)\b",
        lowered,
    ):
        add("one seamless continuous take with no cuts or invisible resets")

    if "torus" in lowered or "toroid" in lowered:
        if "rising" in lowered and ("sweep" in lowered or "sweeping" in lowered):
            add("an ever-rising sweeping path inside a vast torus")
        else:
            add("a continuous path around the inside of a vast torus")

    if re.search(r"\b(?:no dialogue|without dialogue|silent film)\b", lowered):
        add("no spoken dialogue")
    if re.search(r"\b(?:no music|without music)\b", lowered):
        add("no non-diegetic music")

    # Each rolling pass receives only a little audiovisual history and no
    # previous prompt text. Repeat the source-audio performance contract in
    # every pass so later windows do not turn "singing" into unrelated mouth
    # motion while the correct soundtrack continues underneath them.
    if _VOCAL_PERFORMANCE_RE.search(source):
        add(
            "every visible singer or rapper lip-syncing each vocal syllable "
            "to the supplied source soundtrack with exact timing"
        )

    return invariants


def reinforce_ltx_window_invariants(
    prompts: Sequence[str],
    source_prompt: str,
) -> list[str]:
    """Prepend the same compact global contract to every independent pass."""

    cleaned = [_collapse_prompt(item) for item in prompts if _collapse_prompt(item)]
    invariants = extract_ltx_global_invariants(source_prompt)
    if not invariants:
        return cleaned
    if len(invariants) == 1:
        joined = invariants[0]
    else:
        joined = ", ".join(invariants[:-1]) + f", and {invariants[-1]}"
    prefix = f"Throughout this complete window, preserve {joined}."
    return [
        item if item.startswith(prefix) else _collapse_prompt(f"{prefix} {item}")
        for item in cleaned
    ]


def parse_ltx_window_prompts(
    value: str | Sequence[str] | None,
    *,
    expected_count: int,
) -> list[str]:
    """Return exactly one non-empty, single-line prompt per LTX window.

    The model guide asks the LLM for blank-line-separated paragraphs while
    Manual mode asks the user for one physical line per window.  Accept both
    forms, plus a list returned by a previous server-side plan.
    """

    count = max(1, int(expected_count or 1))
    if isinstance(value, (list, tuple)):
        prompts = [_collapse_prompt(item) for item in value if _collapse_prompt(item)]
    else:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        blocks = [
            _collapse_prompt(block)
            for block in re.split(r"\n\s*\n+", text)
            if _collapse_prompt(block)
        ]
        if len(blocks) == count:
            prompts = blocks
        else:
            prompts = [
                _collapse_prompt(line)
                for line in text.split("\n")
                if _collapse_prompt(line)
            ]

    if len(prompts) != count:
        raise ValueError(
            f"LTX multi-window sequence needs exactly {count} non-empty prompt "
            f"{'line' if count == 1 else 'lines'} (window 1 through window {count}); "
            f"found {len(prompts)}."
        )
    return prompts


def _split_story_sentences(prompt: str) -> list[str]:
    normalized = _collapse_prompt(prompt)
    if not normalized:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", normalized)
        if part.strip()
    ]


def deterministic_ltx_window_prompts(prompt: str, count: int) -> list[str]:
    """Usable no-LLM fallback that advances rather than restarting the story."""

    count = max(1, int(count or 1))
    source = _collapse_prompt(prompt)
    sentences = _split_story_sentences(source)
    prompts: list[str] = []
    for index in range(count):
        start = math.floor(index * len(sentences) / count) if sentences else 0
        end = math.floor((index + 1) * len(sentences) / count) if sentences else 0
        local = " ".join(sentences[start:end]).strip()
        if not local:
            local = source
        if index == 0:
            phase = "Establish the requested characters, setting, and action, then begin this beat."
        elif index == count - 1 and requests_open_ended_ltx_motion(source):
            phase = (
                "Continue directly from the preceding window without restarting; "
                "keep the requested motion visibly active through the final frame "
                "and beyond without slowing, settling, stopping, or resolving."
            )
        elif index == count - 1:
            phase = "Continue directly from the preceding window without restarting, then complete the requested final beat."
        else:
            phase = "Continue directly from the preceding window without restarting and advance only this middle beat."
        prompts.append(
            _collapse_prompt(
                f"{phase} {local} Preserve the same identities, wardrobe, location, "
                "lighting, screen direction, and visible continuity."
            )
        )
    return reinforce_ltx_window_invariants(prompts, source)


def plan_ltx_sliding_windows(
    prompt: str,
    *,
    model_type: str,
    duration_seconds: float,
    window_count: int,
    window_size_seconds: float,
    image_paths: Iterable[str] | None = None,
    nsfw: bool = False,
) -> dict:
    """Use Maestro's configured enhancer to create exact LTX window prompts.

    Planning deliberately uses the ordinary model-specific LTX guide so LTX
    0.9, 2.x, and 2.5 retain their established prompt vocabulary.  Geometry is
    supplied explicitly and malformed output falls back to a deterministic
    chronological plan instead of failing an expensive generation request.
    """

    count = max(1, int(window_count or 1))
    source = str(prompt or "").strip()
    planned_by = "llm"
    error = None
    try:
        from services import llm_service

        enhanced = llm_service.enhance_prompt(
            source,
            mode="video",
            max_new_tokens=max(512, count * 320),
            temperature=0.45,
            nsfw=bool(nsfw),
            model_type=str(model_type or ""),
            image_paths=list(image_paths or []) or None,
            duration_seconds=max(1, round(float(duration_seconds), 1)),
            window_count=count,
            window_size_seconds=max(1, round(float(window_size_seconds), 1)),
        )
        prompts = parse_ltx_window_prompts(enhanced, expected_count=count)
    except Exception as exc:  # LLM is optional; generation itself is not.
        error = str(exc)
        planned_by = "deterministic_fallback"
        prompts = deterministic_ltx_window_prompts(source, count)

    prompts = reinforce_ltx_window_invariants(prompts, source)

    return {
        "source_prompt": source,
        "window_count": count,
        "window_prompts": prompts,
        "planned_by": planned_by,
        "planning_error": error,
    }
