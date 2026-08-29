"""MiniMax H3 prompt measurement and quality-oriented compaction.

MiniMax H3 does not publish a 512-token prompt limit, and Maestro's conditioner
passes the complete token sequence to Qwen.  This module therefore must never
act as a generation gate.  It gives Studio, Director, and the window planners
one exact counter plus a conservative structure-aware compactor for overly
verbose AI-authored prompts.  If a prompt cannot be shortened without changing
protected dialogue or timing, the complete prompt is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re
from typing import Any, Iterable

from services.text_integrity import repair_text


H3_PROMPT_QUALITY_TARGET = 1024
# Compatibility name used by the Director and window planners.  This is a
# quality target for AI-authored prose, not a model or runtime limit.
H3_ENHANCED_TEXT_TOKEN_TARGET = H3_PROMPT_QUALITY_TARGET


class H3PromptBudgetError(ValueError):
    """Raised when required H3 prompt structure or dialogue is invalid."""


@dataclass(frozen=True)
class H3PromptBudgetResult:
    prompt: str
    token_count: int
    original_token_count: int
    compacted: bool


_DIALOGUE_RE = re.compile(r"<d>\s*\[[^\]\r\n]+\]\s+.*?</d>", re.IGNORECASE | re.DOTALL)
_FIELD_RE = {
    "visual": re.compile(r"\bintegrated_multimodal_description\s*:", re.IGNORECASE),
    "sound": re.compile(r"\boverall_soundscape\s*:", re.IGNORECASE),
    "music": re.compile(r"\bnon_diegetic_music\s*:", re.IGNORECASE),
}
_BOILERPLATE_PATTERNS = (
    re.compile(
        r"Immediately after the line, the speaker closes (?:their|his|her) mouth\.\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"Keep every requested subject(?:'s)? identity, appearance, wardrobe, and carried objects unchanged\.\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"Keep the requested location, geography, time of day, and background elements coherent\.\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"Keep lighting, color, screen direction, and established geography coherent\.\s*",
        re.IGNORECASE,
    ),
)
_TIMED_CLAUSE_RE = re.compile(
    r"\b(?:"
    r"from\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\s*(?:seconds?|s)"
    r"|(?:at|through|before|after)\s+\d+(?:\.\d+)?\s*(?:seconds?|s)"
    r")\b",
    re.IGNORECASE,
)


def _normalize(value: Any) -> str:
    return repair_text(str(value or "")).replace("\r\n", "\n").replace("\r", "\n").strip()


@lru_cache(maxsize=1)
def _h3_plain_tokenizer():
    """Load H3's tokenizer without importing Transformers or model weights."""

    tokenizer_path = (
        Path(__file__).resolve().parents[1]
        / "ckpts"
        / "minimax_h3"
        / "processor"
        / "tokenizer.json"
    )
    if not tokenizer_path.is_file():
        return None
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(tokenizer_path))
    except Exception:
        return None


def h3_prompt_token_count(value: Any) -> int:
    """Return H3's exact text-token count, or a safe pre-download estimate."""

    text = _normalize(value)
    tokenizer = _h3_plain_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
        except Exception:
            pass

    lexical = len(
        re.findall(
            r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*|[^\w\s]",
            text,
        )
    )
    # Qwen splits markup, names, punctuation, and uncommon words more often
    # than a whitespace lexer. Keep a conservative estimate before tokenizer
    # assets are downloaded so quality compaction behaves consistently.
    return int(math.ceil(lexical * 1.25)) + 8


def _parse_base_fields(prompt: str) -> tuple[str, str, str, str] | None:
    visual = _FIELD_RE["visual"].search(prompt)
    sound = _FIELD_RE["sound"].search(prompt)
    music = _FIELD_RE["music"].search(prompt)
    if not visual or not sound or not music:
        return None
    if not (visual.start() < sound.start() < music.start()):
        return None
    prefix = prompt[: visual.start()].strip()
    visual_body = prompt[visual.end() : sound.start()].strip()
    sound_body = prompt[sound.end() : music.start()].strip()
    music_body = prompt[music.end() :].strip()
    return prefix, visual_body, sound_body, music_body


def _protect_dialogue(text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def replace(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f" H3DIALOGUEBLOCK{len(blocks) - 1} "

    return _DIALOGUE_RE.sub(replace, text), blocks


def _restore_dialogue(text: str, blocks: Iterable[str]) -> str:
    result = text
    for index, block in enumerate(blocks):
        result = result.replace(f"H3DIALOGUEBLOCK{index}", block)
    return result


def _word_cap(value: str, limit: int, *, keep_tail: int = 0) -> str:
    words = re.sub(r"\s+", " ", value).strip().split()
    if len(words) <= limit:
        return " ".join(words)
    if limit <= 0:
        return ""
    tail = max(0, min(keep_tail, limit // 2, len(words)))
    head = max(1, limit - tail)
    selected = words[:head]
    if tail:
        selected.extend(words[-tail:])
    return " ".join(selected).rstrip(" ,;:-")


def _clean_boilerplate(text: str) -> str:
    cleaned = text
    for pattern in _BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(
        r"\bSlow motion occurs only when explicitly requested\.\s*"
        r"(?:Slow motion occurs only when explicitly requested\.\s*)+",
        "Slow motion occurs only when explicitly requested. ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_clauses(text: str) -> list[str]:
    protected, _blocks = _protect_dialogue(text)
    protected = re.sub(r"\s+(?=\[Shot\s+\d+\])", "\n", protected, flags=re.IGNORECASE)
    rough = re.split(r"\n+|(?<=[.!?])\s+", protected)
    clauses: list[str] = []
    for item in rough:
        item = item.strip()
        if not item:
            continue
        if len(item.split()) > 70:
            pieces = re.split(r"\s*;\s*|\s+Then\s+", item, flags=re.IGNORECASE)
            clauses.extend(piece.strip() for piece in pieces if piece.strip())
        else:
            clauses.append(item)
    return clauses


def _dedupe_clauses(clauses: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        key = re.sub(r"[^a-z0-9]+", " ", clause.casefold()).strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(clause)
    return result


def _clause_score(clause: str, index: int, total: int) -> int:
    lowered = clause.casefold()
    score = 0
    if index == 0:
        score += 120
    if index == total - 1:
        score += 70
    if "h3dialogueblock" in lowered:
        score += 300
    if re.search(r"\[shot\s+\d+\]", lowered):
        score += 110
    if _TIMED_CLAUSE_RE.search(lowered):
        score += 90
    if re.search(r"\b(?:final|ends?|closing|outcome|destination)\b", lowered):
        score += 60
    if re.search(r"\b(?:camera|shot|frame|tracking|dolly|pan|tilt|close-up|wide)\b", lowered):
        score += 35
    if re.search(r"\(s\d+\)", lowered):
        score += 35
    if re.search(r"\b(?:opens?|begins?|moves?|runs?|walks?|flies?|falls?|turns?|strikes?|punches?|speaks?)\b", lowered):
        score += 25
    if re.match(r"\s*then\b", lowered):
        # The deterministic window planner emits each ordered source event as
        # its own ``Then ...`` clause. Those are story instructions, not
        # decorative prose, so prefer them over continuity boilerplate.
        score += 140
    if "only the tagged" in lowered or "no background voices" in lowered:
        score -= 35
    return score


def _cap_clause(clause: str, *, level: int) -> str:
    contains_dialogue = "H3DIALOGUEBLOCK" in clause
    if contains_dialogue:
        # Preserve the literal tag and enough nearby words for speaker,
        # delivery, action, and timing. The dialogue itself is never capped.
        parts = re.split(r"(H3DIALOGUEBLOCK\d+)", clause)
        prose_limit = max(8, 28 - level * 5)
        prose_parts = max(1, sum(1 for part in parts if not part.startswith("H3DIALOGUEBLOCK")))
        per_part = max(4, prose_limit // prose_parts)
        return " ".join(
            part if part.startswith("H3DIALOGUEBLOCK") else _word_cap(part, per_part, keep_tail=2)
            for part in parts
            if part.strip()
        ).strip()
    limits = (46, 34, 25, 18, 13)
    return _word_cap(clause, limits[min(level, len(limits) - 1)], keep_tail=4)


def _compact_candidate(
    prefix: str,
    visual_body: str,
    sound_body: str,
    music_body: str,
    *,
    dialogue_blocks: list[str],
    target_tokens: int,
    level: int,
) -> str:
    protected_visual, _ = _protect_dialogue(visual_body)
    protected_visual = _clean_boilerplate(protected_visual)
    clauses = _dedupe_clauses(_split_clauses(protected_visual))
    if not clauses:
        clauses = ["[Shot 1] The requested scene and action unfold clearly."]

    capped = [_cap_clause(clause, level=level) for clause in clauses]
    required: set[int] = {0, len(capped) - 1}
    for index, clause in enumerate(capped):
        lowered = clause.casefold()
        if "h3dialogueblock" in lowered:
            required.add(index)
        if re.search(r"\[shot\s+\d+\]", lowered):
            required.add(index)
        if _TIMED_CLAUSE_RE.search(lowered):
            required.add(index)
        if re.match(r"\s*then\b", lowered):
            # Every planner-assigned event must survive. In particular, short
            # transition beats such as "Then they mount their brooms" can be
            # visually decisive even though they consume very few tokens.
            required.add(index)
        if re.match(r"\s*coverage\s+is\b", lowered) or "pacing is" in lowered:
            # Speed and coverage are global motion controls. Dropping them can
            # turn a fast continuation into the slow-motion behavior the
            # window planner exists to prevent.
            required.add(index)

    selected = set(required)
    order = sorted(
        (index for index in range(len(capped)) if index not in selected),
        key=lambda index: (-_clause_score(capped[index], index, len(capped)), index),
    )

    sound_limit = max(12, 46 - level * 8)
    music_limit = max(8, 26 - level * 4)
    compact_sound = _word_cap(sound_body, sound_limit, keep_tail=4) or "N/A"
    compact_music = _word_cap(music_body, music_limit, keep_tail=3) or "N/A"

    def build(indices: set[int]) -> str:
        body = " ".join(capped[index] for index in sorted(indices) if capped[index]).strip()
        body = _restore_dialogue(body, dialogue_blocks)
        parts = []
        if prefix:
            parts.append(prefix)
        parts.extend(
            [
                f"integrated_multimodal_description: {body}",
                f"overall_soundscape: {compact_sound}",
                f"non_diegetic_music: {compact_music}",
            ]
        )
        return "\n\n".join(parts)

    candidate = build(selected)
    for index in order:
        trial = build(selected | {index})
        if h3_prompt_token_count(trial) <= target_tokens:
            selected.add(index)
            candidate = trial
    return candidate


def fit_h3_base_prompt(
    prompt: Any,
    *,
    target_tokens: int = H3_ENHANCED_TEXT_TOKEN_TARGET,
) -> H3PromptBudgetResult:
    """Compact verbose structured H3 prose while preserving protected data.

    No string slicing is used.  Alignment instructions, all literal ``<d>``
    blocks, field order, timed clauses, shot boundaries, and the first/final
    visual states receive priority.  ``target_tokens`` is only a quality goal;
    if it cannot be reached safely, the best safe form (or the untouched input)
    is returned and generation remains allowed.
    """

    text = _normalize(prompt)
    original_count = h3_prompt_token_count(text)
    if original_count <= target_tokens:
        return H3PromptBudgetResult(text, original_count, original_count, False)

    parsed = _parse_base_fields(text)
    if parsed is None:
        return H3PromptBudgetResult(text, original_count, original_count, False)
    prefix, visual_body, sound_body, music_body = parsed
    dialogue_blocks = _DIALOGUE_RE.findall(visual_body)

    best = text
    best_count = original_count
    for level in range(5):
        candidate = _compact_candidate(
            prefix,
            visual_body,
            sound_body,
            music_body,
            dialogue_blocks=dialogue_blocks,
            target_tokens=target_tokens,
            level=level,
        )
        candidate_blocks = _DIALOGUE_RE.findall(candidate)
        if candidate_blocks != dialogue_blocks:
            # Never trade exact speech for a cosmetic prompt-length target.
            return H3PromptBudgetResult(text, original_count, original_count, False)
        candidate_count = h3_prompt_token_count(candidate)
        if candidate_count < best_count:
            best, best_count = candidate, candidate_count
        if candidate_count <= target_tokens:
            return H3PromptBudgetResult(
                candidate,
                candidate_count,
                original_count,
                True,
            )

    return H3PromptBudgetResult(
        best,
        best_count,
        original_count,
        best != text,
    )
