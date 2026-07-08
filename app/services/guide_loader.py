"""Generic loader for LLM prompt guides under app/services/llm_guides/.

Reads .md files from app/services/llm_guides/<category>/<name>.md and
returns the content as a stripped string. Caches results after the
first load (guides don't change at runtime).

Two domain-specific loaders already exist for historical reasons and
remain the right tool for their narrow jobs:
  - services.director.guide_loader -> llm_guides/director/ (Director planners)
  - services.enhance_guides        -> llm_guides/enhance/  (architecture-mapped
                                                              prompt enhancer)

This module is the catch-all for guides that don't fit either bucket:
in particular, model-specific prompt-enhancer guides like Scenema's
single-speaker and multi-speaker rules, which the model handler
injects directly via `text_prompt_enhancer_instructions[1]` rather
than going through the architecture mapping.

Designed so the same `load_guide(category, name)` call works from
Studio Mode's prompt enhancer today and from Director Mode's
screenplay-system-prompt builder when that integration lands.
"""

import os
from functools import lru_cache

_GUIDES_ROOT = os.path.join(os.path.dirname(__file__), "llm_guides")


@lru_cache(maxsize=64)
def load_guide(category: str, name: str) -> str:
    """Load app/services/llm_guides/<category>/<name>.md as a string.

    Returns empty string if the file is missing or unreadable (caller
    is responsible for handling that case — usually falling back to
    a built-in default prompt). Cached after first load.

    Args:
        category: Subfolder under llm_guides/ (e.g. "prompt_enhancer").
        name: Guide filename without the .md suffix. The .md extension
            is appended automatically if not already present.

    Examples:
        load_guide("prompt_enhancer", "scenema_dialogue_rules")
        load_guide("prompt_enhancer", "scenema_speech_rules")
    """
    filename = name if name.endswith(".md") else f"{name}.md"
    filepath = os.path.join(_GUIDES_ROOT, category, filename)
    if not os.path.isfile(filepath):
        print(f"[GuideLoader] Guide not found: {category}/{filename}")
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[GuideLoader] Failed to load {category}/{filename}: {e}")
        return ""
