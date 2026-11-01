"""Compatibility exports for the shared, CPU-only TTS writing guides."""
from services.tts_prompt_guidance import (
    TTS_MONOLOGUE_OR_DIALOGUE_PROMPT,
    TTS_MONOLOGUE_PROMPT,
    TTS_QWEN3_DIALOGUE_PROMPT,
)

__all__ = ["TTS_MONOLOGUE_PROMPT", "TTS_QWEN3_DIALOGUE_PROMPT", "TTS_MONOLOGUE_OR_DIALOGUE_PROMPT"]
