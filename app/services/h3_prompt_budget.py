"""MiniMax H3 prompt measurement and non-destructive cleanup.

MiniMax H3 does not publish a 512-token prompt limit, and Maestro's conditioner
passes the complete token sequence to Qwen.  This module therefore must never
act as a generation gate.  It gives Studio, Director, and the window planners
one exact counter plus cleanup of application boilerplate. Semantic condensation
belongs to the LLM; the compiler retains unique actions, sounds, and dialogue
even when the result exceeds the cosmetic token target.
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
    re.compile(
        r"Silent visual action, never spoken narration:\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"Visual direction only, never spoken narration:\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"No words are spoken or mouthed in this shot;\s*only explicitly "
        r"requested nonverbal reactions may be heard\.?\s*",
        re.IGNORECASE,
    ),
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


def _clean_boilerplate(text: str) -> str:
    cleaned = text
    for pattern in _BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(
        r"while in the established target scene,\s*only\s+(.+?)'s mouth "
        r"moves while every other visible mouth stays closed",
        r"while only \1's mouth moves and all other mouths stay closed",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bSlow motion occurs only when explicitly requested\.\s*"
        r"(?:Slow motion occurs only when explicitly requested\.\s*)+",
        "Slow motion occurs only when explicitly requested. ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def fit_h3_base_prompt(
    prompt: Any,
    *,
    target_tokens: int = H3_ENHANCED_TEXT_TOKEN_TARGET,
) -> H3PromptBudgetResult:
    """Remove redundant boilerplate without shortening unique story content.

    ``target_tokens`` controls when cleanup is useful, not what may survive.
    Complete action, sound and speech remain, regardless of the final count.
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

    # A token target is not permission to delete story content. The former
    # clause scorer/word caps kept timing markers but cut away the action
    # beneath them. Semantic condensation belongs to the LLM, before this
    # compiler. Here we remove only known application boilerplate and exact
    # repetitions of static environment description, never action or speech.
    protected, blocks = _protect_dialogue(visual_body)
    cleaned = _clean_boilerplate(protected)
    seen: set[str] = set()
    clauses = []
    for clause in re.split(r"(?<=[.!?])\s+", cleaned):
        static_description = bool(re.match(
            r"The\s+(?:(?:\w+\s+){0,3}(?:environment|landscape|background)\s+contains\b|"
            r"camera\s+preserves\b)", clause, re.IGNORECASE,
        ))
        if static_description and clause in seen:
            continue
        seen.add(clause)
        clauses.append(clause)
    visual = _restore_dialogue(" ".join(clauses), blocks)
    parts = [prefix] if prefix else []
    parts.extend([
        f"integrated_multimodal_description: {visual}",
        f"overall_soundscape: {sound_body}",
        f"non_diegetic_music: {music_body}",
    ])
    candidate = "\n\n".join(parts)
    if _DIALOGUE_RE.findall(candidate) != dialogue_blocks:
        return H3PromptBudgetResult(text, original_count, original_count, False)
    return H3PromptBudgetResult(candidate, h3_prompt_token_count(candidate), original_count, candidate != text)
