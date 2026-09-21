"""YuE2 input rules shared by Studio, Director and queued jobs."""

import math


def writer_duration_instruction(duration_seconds) -> str:
    try:
        seconds = float(duration_seconds)
    except (TypeError, ValueError):
        seconds = 120.0
    if not math.isfinite(seconds):
        seconds = 120.0
    seconds = min(600.0, max(1.0, seconds))
    return (
        f"The maximum recording length is {seconds:g} seconds. Write a song that "
        "can finish naturally within that limit, allowing for sung phrasing and "
        "instrumental space. For a short excerpt, use fewer sections and lines. "
        "This is a ceiling, not a guarantee of exact duration. Preserve supplied "
        "lyrics; do not invent additional verses just to fill the limit."
    )


def validate_song_inputs(inputs, lyrics):
    from models.TTS.yue2.instrumental import is_instrumental, instrumental_settings
    custom = inputs.get("custom_settings") or {}
    if not isinstance(custom, dict):
        return "YuE2 score settings must be an object."
    if 'instrumental' in custom and not isinstance(custom['instrumental'], bool):
        return "YuE2 instrumental mode must be on or off."
    instrumental = is_instrumental(inputs, lyrics)
    if instrumental:
        inputs.update(model_mode=0, audio_prompt_type='', audio_guide=None,
                      custom_settings=instrumental_settings(custom))
    if (not instrumental and not str(lyrics or "").strip()) or not str(inputs.get("alt_prompt") or "").strip():
        return "YuE2 requires lyrics and a music style."
    mode = inputs.get("model_mode")
    if mode is None:
        # The shared task defaults supply None for models without a mode.
        # Older Studio clients displayed a choice without submitting one.
        mode = inputs["model_mode"] = 2
    if isinstance(mode, bool) or not isinstance(mode, int) or mode not in (0, 1, 2):
        return "Choose melody and chords, melody only, or direct generation."
    scoring = "A" in str(inputs.get("audio_prompt_type") or "")
    if scoring and not inputs.get("audio_guide"):
        return "Upload a source song to extract its score."
    if scoring and mode == 2:
        return "Scoring a source song requires Melody and chords or Melody only."
    try:
        guidance = float(inputs.get("guidance_scale", 1))
        duration = float(inputs.get("duration_seconds", 120))
    except (TypeError, ValueError):
        return "YuE2 duration and guidance must be numbers."
    if not math.isfinite(guidance) or guidance < 1:
        return "YuE2 guidance must be at least 1 (1 disables CFG)."
    if not math.isfinite(duration) or not 1 <= duration <= 600:
        return "YuE2 maximum song duration must be between 1 and 600 seconds."
    custom = inputs.get("custom_settings") or {}
    if not isinstance(custom, dict):
        return "YuE2 score settings must be an object."
    from services.music_styles import generation_styles
    try:
        artists = [] if instrumental else generation_styles(custom)
    except ValueError as error:
        return str(error)
    if artists:
        if mode != 2:
            return "Music LoRAs use Direct generation. Clear the LoRAs to use score planning."
    if not scoring and "abc" in custom:
        if not isinstance(custom["abc"], str):
            return "ABC score must be text."
        if custom["abc"].strip() and mode == 2:
            return "An ABC score requires a composition planning mode."
    return None
