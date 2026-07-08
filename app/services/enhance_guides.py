"""
Smart prompt enhancement — loads per-model guide files for the enhance LLM.

Maps model architecture/type to the appropriate prompt guide, then uses the
local LLM to rewrite the user's prompt following that guide.
"""

import os
from typing import Optional

_GUIDES_DIR = os.path.join(os.path.dirname(__file__), "llm_guides", "enhance")

# Map architecture prefixes to guide files
# Values can be a string (single guide) or a tuple (with_images_guide, without_images_guide)
_ARCHITECTURE_MAP = {
    # LTX-2 video models
    "ltx2": "ltx2_video.md",
    "ltxv": "ltx2_video.md",

    # Wan video models
    "t2v": "wan_video.md",
    "i2v": "wan_video.md",
    "ti2v": "wan_video.md",
    "animate": "wan_video.md",
    "wanmove": "wan_video.md",
    "ovi": "wan_video.md",
    "lucy": "wan_video.md",
    "multitalk": "wan_video.md",
    "phantom": "wan_video.md",
    "fun_inp": "wan_video.md",
    "alpha": "wan_video.md",
    "fantasy": "wan_video.md",
    "chrono": "wan_video.md",
    "flf2v": "wan_video.md",

    # Qwen image models — edit guide when reference image exists, gen guide otherwise
    "qwen_image_edit": ("qwen_image_edit.md", "qwen_image_gen.md"),
    "qwen_image_layered": ("qwen_image_edit.md", "qwen_image_gen.md"),
    "qwen_image": "qwen_image_gen.md",

    # Flux image models — edit guide when reference image exists, gen guide otherwise
    "flux": ("flux_image_edit.md", "flux_image.md"),
    "z_image": ("flux_image_edit.md", "flux_image.md"),

    # Audio/TTS models
    "qwen3_tts": "audio_tts.md",
    "ace_step": "audio_tts.md",
    "chatterbox": "audio_tts.md",

    # Hunyuan
    "hunyuan": "wan_video.md",
    "heartmula": "wan_video.md",
}

# Fallback by generation mode
_MODE_FALLBACK = {
    "video": "generic_video.md",
    "image": "generic_image.md",
    "audio": "audio_tts.md",
    "avatar": "generic_video.md",
}


def _load_guide(filename: str) -> Optional[str]:
    """Load a guide file from the enhance guides directory."""
    filepath = os.path.join(_GUIDES_DIR, filename)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def get_enhance_guide(model_type: str, generation_mode: str, has_images: bool = False) -> str:
    """Get the appropriate enhancement guide for a model + mode combination.

    Resolution order (first non-empty wins):
      1. Per-model override: if the model_def carries an `enhance_guide`
         field naming a file under llm_guides/prompt_enhancer/, load
         that file. Lets specific community fine-tunes (e.g. 10Eros)
         ship their own prompt-enhancer rules without having to extend
         the architecture map below.
      2. Architecture-prefix mapping (_ARCHITECTURE_MAP).
      3. Generation-mode fallback (_MODE_FALLBACK).
      4. Ultimate inline fallback string.

    Args:
        model_type: The model_type string (e.g., "ltx2_22B_distilled")
        generation_mode: "video", "image", "audio", "avatar"
        has_images: Whether reference/start images are attached

    Returns:
        Guide text for the LLM system prompt.
    """
    # 1. Per-model overrides via model_def.
    # Lazy-import wgp + the shared loader because enhance_guides.py is
    # imported early during server startup, before wgp's full model
    # definitions are populated. Lazy keeps the import cheap.
    inline_delta = ""
    try:
        from wgp import get_model_def
        md = get_model_def(model_type)
        if md:
            # 1a. Full-guide FILE override — a complete, standalone enhancer
            # guide shipped by a community fine-tune (e.g. 10Eros, Sulphur).
            # REPLACES the architecture base entirely (existing behavior).
            override = md.get("enhance_guide")
            if override:
                # Reuse the cross-domain loader so caching + path
                # resolution stays consistent with the TTS enhancer
                # path (Scenema dialogue/speech rules).
                from services.guide_loader import load_guide
                guide = load_guide("prompt_enhancer", override)
                if guide:
                    print(f"[Enhance] Loaded per-model guide: prompt_enhancer/{override} for {model_type}")
                    return guide
                print(f"[Enhance] enhance_guide={override!r} declared on {model_type} but file is empty/missing")
            # 1b. Inline generated DELTA — auto-written for CivitAI checkpoint
            # imports into the gitignored finetune JSON (model.enhance_guide_text)
            # so a possibly-mature, machine-generated guide never enters the repo.
            # Unlike (1a) this does NOT replace the base; it is APPENDED below, so
            # the architecture base still owns cinematic structure and the model
            # layer only augments vocabulary / trigger words / style.
            inline = md.get("enhance_guide_text")
            if isinstance(inline, str) and inline.strip():
                inline_delta = inline.strip()
    except Exception as e:
        # Don't let a model_def lookup failure prevent the enhancer
        # from running — fall through to architecture mapping.
        print(f"[Enhance] Per-model guide lookup failed for {model_type}: {e}")

    # 2. Architecture-specific match (longest prefix first)
    base = None
    model_lower = model_type.lower()
    best_match = None
    best_len = 0
    for prefix, guide_entry in _ARCHITECTURE_MAP.items():
        if model_lower.startswith(prefix) and len(prefix) > best_len:
            best_match = guide_entry
            best_len = len(prefix)

    if best_match:
        # Resolve tuple: (with_images_guide, without_images_guide)
        if isinstance(best_match, tuple):
            guide_file = best_match[0] if has_images else best_match[1]
        else:
            guide_file = best_match
        guide = _load_guide(guide_file)
        if guide:
            print(f"[Enhance] Loaded guide: {guide_file} (has_images={has_images})")
            base = guide

    # 3. Fallback by generation mode
    if base is None:
        fallback_file = _MODE_FALLBACK.get(generation_mode, "generic_video.md")
        base = _load_guide(fallback_file)

    # 4. Ultimate inline fallback
    if base is None:
        base = (
            f"Rewrite the user's {generation_mode} prompt to be more detailed and specific. "
            f"Preserve their intent and dialogue. Add visual/cinematic detail. "
            f"Output ONLY the rewritten prompt."
        )

    # Layer the per-model generated delta on top of whatever base resolved.
    # The base owns structure; the delta augments it with this checkpoint's
    # trigger words / preferred prompt style. The shared video + mature
    # appendices are added by the caller (llm_service.enhance_prompt) after
    # this returns, so the final stack is: base -> model delta -> shared.
    if inline_delta:
        print(f"[Enhance] Appending inline per-model guide delta for {model_type} ({len(inline_delta)} chars)")
        base = (
            f"{base}\n\n"
            f"MODEL-SPECIFIC PROMPTING NOTES — the active checkpoint is a community "
            f"fine-tune with its own conventions. Apply these on top of the rules "
            f"above; they augment vocabulary and trigger words, they do not override "
            f"the structure or pacing rules:\n{inline_delta}"
        )

    return base
