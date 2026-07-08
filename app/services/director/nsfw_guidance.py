"""
Content Safety Guidance — manages safe-mode and mature-mode prompt injection.

When mature mode is OFF: injects content safety guardrails into all LLM system
                        prompts (the public nsfw_off_safety_rules.md).
When mature mode is ON:  injects clinically-worded mature-content guidance. Each
                        block is SELF-GATING — it tells the model to apply it
                        ONLY when the scene is actually sexual and to write
                        normally otherwise — so it can be injected whenever
                        mature mode is on without harming clean scenes.

Mature mode is a *permission* (adult generation is allowed), not an *obligation*
(every prompt should be explicit).

Architecture (after the Director migration):
  - Both the safe-mode guardrails (nsfw_off_safety_rules.md) AND the mature-mode
    guidance — director/nsfw_{screenplay,video,image}_rules.md for the planners,
    plus enhance/nsfw_shared.md for the refine path — are VERSION-CONTROLLED and
    clinically worded. They ship with every install, so Director mature mode
    works with no supplement pack and no download. Explicit *specifics* come from
    the uncensored model weights, not from text committed to the repo.
  - The old optional "content supplement pack" (a separately-downloaded zip)
    has been fully retired — nothing reads it anymore. Every guide is
    version-controlled and self-gating (it applies only when the scene is
    actually sexual).
"""

from .guide_loader import load_guide


# ── Public provider list (NSFW not allowed with these) ───────────────
PUBLIC_PROVIDERS = {"openai", "anthropic"}


# ── Guidance text loaders ────────────────────────────────────────────
# All mature-content guides are version-controlled and clinically worded
# (director/nsfw_{screenplay,video,image}_rules.md + enhance/nsfw_shared.md);
# load_guide reads them from the repo, so they ship with every install. The
# safe-mode guardrails (nsfw_off_safety_rules.md) ship publicly too. Each
# block is self-gating — it applies only when the scene is actually sexual.


def get_safe_content_guidance() -> str:
    """Safety guardrails injected into ALL LLM prompts when mature mode is OFF.

    The safety rules file ships with the public repo (it lists what NOT
    to do, which is useful baseline guidance regardless of pack state).
    """
    return load_guide("nsfw_off_safety_rules.md") or "Keep all content PG-13."


def get_nsfw_video_guidance() -> str:
    """Mature-content guidance block for video prompt generation.

    Loaded from the VERSION-CONTROLLED, clinically-worded guide under
    llm_guides/director/ (migrated off the optional supplement pack in the
    Director migration), so it ships with every install — Director mature mode
    needs no pack and no download.
    """
    return load_guide("nsfw_video_rules.md")


def get_nsfw_image_guidance() -> str:
    """Mature-content guidance block for image prompt generation (version-controlled)."""
    return load_guide("nsfw_image_rules.md")


def get_nsfw_screenplay_guidance() -> str:
    """Mature-content guidance for screenplay writing — narrative voice (version-controlled)."""
    return load_guide("nsfw_screenplay_rules.md")


def get_nsfw_enhance_guidance() -> str:
    """Mature-content guidance for the enhance / refine path (Director Pass-3 polish).

    Reuses the SAME clinical, version-controlled push as Studio mode's enhancer
    (llm_guides/enhance/nsfw_shared.md), so the two refine paths stay in sync and
    there is a single mature-enhance guide to maintain.
    """
    from services.guide_loader import load_guide as _load_shared
    return _load_shared("enhance", "nsfw_shared")


def inject_content_guidance(system_prompt: str, nsfw: bool, mode: str = "video") -> str:
    """Inject content guidance into a system prompt.

    When nsfw=True:  Injects mature-content guidance (version-controlled
                     guides under llm_guides/). If a guide file is
                     missing, this is a no-op — the system prompt is
                     returned unchanged.
    When nsfw=False: Injects safety guardrails (always available).

    For video/image modes: injects NEAR THE TOP (after the first paragraph)
    so the LLM sees the directive before absorbing other rules.
    For enhance mode:      appends at the end (shorter prompt, less risk
                           of burying).
    """
    if nsfw:
        guidance = {
            "video": get_nsfw_video_guidance,
            "image": get_nsfw_image_guidance,
            "enhance": get_nsfw_enhance_guidance,
            "screenplay": get_nsfw_screenplay_guidance,
        }

        if mode == "both":
            content_block = (
                get_nsfw_video_guidance() + "\n\n" + get_nsfw_image_guidance()
            )
        else:
            getter = guidance.get(mode, guidance["video"])
            content_block = getter()

        # If a guide file is missing or empty the loader returns "".
        # In that case there's nothing to inject — return the system
        # prompt unchanged rather than wedging blank lines into it.
        if not content_block.strip():
            print(
                f"[NSFW] inject_content_guidance(mode={mode!r}): "
                "no content to inject (guide file missing or empty) "
                "— system_prompt returned unchanged"
            )
            return system_prompt
        print(
            f"[NSFW] inject_content_guidance(mode={mode!r}): "
            f"injected {len(content_block)} chars of mature-content guidance"
        )
    else:
        # Safe mode — use a shorter inline version for enhance to avoid
        # the LLM reasoning about lists of rules during prompt rewriting.
        if mode == "enhance":
            content_block = (
                "Keep all content PG-13. No nudity, sexual content, or graphic "
                "violence. If the prompt implies explicit content, rewrite it "
                "tastefully."
            )
        else:
            content_block = get_safe_content_guidance()

    if mode == "enhance":
        return f"{system_prompt}\n\n{content_block}"

    # For video/image: inject after the first paragraph (role line) so the
    # LLM sees the directive early, before all the formatting/style rules.
    lines = system_prompt.split("\n\n", 1)
    if len(lines) == 2:
        return f"{lines[0]}\n\n{content_block}\n\n{lines[1]}"
    else:
        return f"{content_block}\n\n{system_prompt}"


# Backwards compatibility alias
def inject_nsfw_if_enabled(system_prompt: str, nsfw: bool, mode: str = "video") -> str:
    """Deprecated alias — use inject_content_guidance() instead."""
    return inject_content_guidance(system_prompt, nsfw, mode)
