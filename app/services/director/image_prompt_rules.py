"""
Shared image prompt rules for all Director planners.

Image guides retain model-specific prompt dialects. A shared style contract is
appended for both reference edits and text-to-image prompts.
"""

from .guide_loader import load_guide
from ..image_prompt_style import IMAGE_STYLE_GUIDANCE


# Image model architecture → Pass 2 guide file
_IMAGE_PASS2_GUIDE_MAP = {
    # Flux family — Flux.2 Klein 9b is the certified path
    "flux": "flux_image_edit_pass2.md",
    "z_image": "flux_image_edit_pass2.md",
    # Qwen image-edit family
    "qwen_image_edit": "qwen_image_edit_reference.md",
    "qwen_image_layered": "qwen_image_edit_reference.md",
    "qwen_image": "qwen_image_edit_reference.md",
}

_REFERENCE_ROLE_GUIDANCE = (
    "REFERENCE ROLES AND LIGHTING: Use each reference only for the role the "
    "user assigns. An identity-only reference supplies identity, not the target "
    "medium or style. Treat an image as a style or lighting reference only when "
    "the user assigns that role or asks to match it. Follow explicit user "
    "requests for medium, restyling, and lighting over the source image. Do not "
    "append a universal source-lighting or color-temperature instruction."
)


def _route_image_pass2_guide(image_model: str) -> str:
    """Pick the Pass 2 image guide filename for the given model."""
    if not image_model:
        return "qwen_image_edit_reference.md"  # legacy fallback
    model_lower = image_model.lower()
    best_match: str | None = None
    best_len = 0
    for prefix, guide_file in _IMAGE_PASS2_GUIDE_MAP.items():
        if model_lower.startswith(prefix) and len(prefix) > best_len:
            best_match = guide_file
            best_len = len(prefix)
    return best_match or "qwen_image_edit_reference.md"


def get_image_prompt_rules(
    has_reference: bool,
    num_character_refs: int = 0,
    num_location_refs: int = 0,
    character_ref_labels: list[str] | None = None,
    location_ref_labels: list[str] | None = None,
    seamless: bool = True,
    image_model: str = "",
) -> str:
    """Return the image prompt instruction block for LLM system prompts.

    image_model selects between dialect-aware Pass 2 guides. The text-to-image
    path uses a shared opening-image guide because it has no base image to edit.
    """

    # Seamless mode hint — kept minimal so it doesn't overwhelm scene planning.
    seamless_block = ""
    if seamless:
        seamless_block = """
SEAMLESS MODE: Each scene's start image is also used as the end frame of the previous scene.
Keep consecutive scenes visually compatible — avoid jarring pose or location jumps between scenes.
"""

    # Additional refs section (appended when character/location refs provided).
    extra_refs_block = ""
    if num_character_refs > 0 or num_location_refs > 0:
        lines = []
        if num_character_refs > 0:
            lines.append(f"{num_character_refs} CHARACTER REFERENCE image(s):")
            for i in range(num_character_refs):
                label = (
                    character_ref_labels[i]
                    if character_ref_labels and i < len(character_ref_labels)
                    else ""
                ) or f"(unlabeled character {i + 1})"
                lines.append(f"  - Character ref {i + 1}: {label}")
        if num_location_refs > 0:
            lines.append(f"{num_location_refs} LOCATION REFERENCE image(s):")
            for i in range(num_location_refs):
                label = (
                    location_ref_labels[i]
                    if location_ref_labels and i < len(location_ref_labels)
                    else ""
                ) or f"(unlabeled location {i + 1})"
                lines.append(f"  - Location ref {i + 1}: {label}")
        extra_refs_block = f"""
MULTIPLE REFERENCE IMAGES:
The image model receives these additional references, alongside a main scene image when one is attached:
{chr(10).join(lines)}

Use the labels to map characters and locations to their assigned references.
Add "use images as reference" at the end only when it helps identity matching.
When featuring a specific character, mention their label description so the model
matches them to the correct reference image.
"""

    if has_reference:
        guide_file = _route_image_pass2_guide(image_model)
        guide = load_guide(guide_file)
        if guide:
            print(
                f"[ImagePromptRules] Loaded {guide_file} "
                f"for image_model={image_model or '(unset)'}"
            )
        else:
            guide = ""
        blocks = [seamless_block, extra_refs_block, guide]
    else:
        # Text-to-image has no base scene image. Optional character/location refs
        # remain available through the separate reference block above.
        base = load_guide("flux_image_generation.md")
        if not base:
            base = "Describe one static opening image with full scene details."
        blocks = [seamless_block, extra_refs_block, base]

    blocks.extend([IMAGE_STYLE_GUIDANCE, _REFERENCE_ROLE_GUIDANCE])
    return (chr(10) + chr(10)).join(block.strip() for block in blocks if block and block.strip())
