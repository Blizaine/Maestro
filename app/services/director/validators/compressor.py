"""
Prompt Compressor — removes redundancy and optimizes token usage.

Non-creative layer: preserves meaning while reducing prompt length.
Preserves: action clarity, camera direction, dialogue phrasing, ending beat.
Removes: repeated adjectives, filler phrases, redundant identity restatement.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from ..schema import ShotPlan
from ..policies import compress_prompt_text


@dataclass
class CompressionResult:
    original: str
    compressed: str
    original_words: int
    compressed_words: int
    chars_removed: int
    compression_ratio: float  # 0.0 = nothing removed, 1.0 = everything removed


def compress_prompt(
    prompt: str,
    mode: str,
    shot: Optional[ShotPlan] = None,
    aggressive: bool = False,
) -> CompressionResult:
    """Compress a rendered prompt while preserving essential content.

    Args:
        prompt: The rendered prompt text.
        mode: Render mode (t2v, i2v, a2v, retake, extend, image_gen).
        shot: Optional ShotPlan for context-aware compression.
        aggressive: If True, apply more aggressive compression rules.

    Returns:
        CompressionResult with original and compressed text plus metrics.
    """
    original = prompt
    original_words = len(original.split())
    result = prompt

    # Step 1: Basic compression (filler removal, duplicate adjectives)
    result, chars_removed = compress_prompt_text(result)

    # Step 2: Mode-specific compression
    if mode == "i2v":
        result = _compress_i2v(result)
    elif mode == "a2v":
        result = _compress_a2v(result, shot)
    elif mode == "image_gen":
        result = _compress_image(result)

    # Step 3: Generic cleanup
    result = _generic_cleanup(result)

    # Step 4: Aggressive compression (if enabled)
    if aggressive:
        result = _aggressive_compress(result)

    compressed_words = len(result.split())
    total_removed = len(original) - len(result)

    return CompressionResult(
        original=original,
        compressed=result,
        original_words=original_words,
        compressed_words=compressed_words,
        chars_removed=total_removed,
        compression_ratio=total_removed / len(original) if original else 0.0,
    )


def _compress_i2v(text: str) -> str:
    """I2V-specific: reduce redundant static scene description.

    The start image already shows the scene — remove descriptors that
    don't add new motion/change information.
    """
    # Remove redundant "in the scene" / "visible in frame" type phrases
    redundant = [
        r'\bin the (?:current |existing )?(?:scene|frame|shot)\b',
        r'\balready (?:visible|shown|present)\b',
        r'\bas seen (?:in|from) the (?:image|photo|frame)\b',
    ]
    for pattern in redundant:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def _compress_a2v(text: str, shot: Optional[ShotPlan] = None) -> str:
    """A2V-specific: reduce visual density during dialogue-heavy shots.

    Don't overload with visual events that could fight audio timing.
    """
    if shot and shot.audio_plan.lip_sync_critical:
        # Remove excessive environmental description during dialogue
        # Keep it lean — audio is the star
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 6:
            # Keep first 2 (setting), dialogue-related, and last (ending)
            essential = []
            for s in sentences:
                s_lower = s.lower()
                is_dialogue = '"' in s
                is_camera = any(w in s_lower for w in ['camera', 'shot', 'framing', 'dolly', 'tracking'])
                is_action = any(w in s_lower for w in ['says', 'speaks', 'whispers', 'gestures', 'nods'])
                if is_dialogue or is_camera or is_action or len(essential) < 2:
                    essential.append(s)
                elif s == sentences[-1]:  # keep ending
                    essential.append(s)
            text = " ".join(essential)
    return text


def _compress_image(text: str) -> str:
    """Image prompt compression — remove any lingering motion language."""
    # Remove motion-implying phrases that may have leaked through
    motion_phrases = [
        r'\b(?:begins? to|starts? to|about to|ready to)\s+\w+',
        r'\bin (?:the process|the act) of\b',
    ]
    for pattern in motion_phrases:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def _generic_cleanup(text: str) -> str:
    """Generic cleanup applicable to all modes."""
    # Collapse repeated commas/spaces
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.\s*\.', '.', text)
    text = re.sub(r'  +', ' ', text)

    # Remove trailing/leading whitespace and punctuation issues
    text = text.strip()
    if text and not text[-1] in '.!?"':
        text += "."

    return text


def _aggressive_compress(text: str) -> str:
    """More aggressive compression for token-constrained scenarios."""
    # Remove hedging language
    hedges = [
        r'\b(?:somewhat|slightly|a bit|rather|quite|fairly|kind of|sort of)\s+',
        r'\b(?:appears to be|seems to be|looks like)\s+',
    ]
    for pattern in hedges:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Simplify compound descriptors
    text = re.sub(r'\b(soft and gentle|warm and inviting|dark and moody)\b',
                  lambda m: m.group(0).split(' and ')[0], text, flags=re.IGNORECASE)

    return text
