"""Style fidelity shared by guided and dedicated image prompt writers."""

IMAGE_STYLE_GUIDANCE = (
    "IMAGE MEDIUM AND STYLE: Preserve the user's requested medium and rendering style, "
    "including anime, illustration, painting, pixel art, photography or 3D when requested. "
    "Do not substitute realism, realistic skin, photographic texture, lens effects, "
    "cinematic grading or another unrequested aesthetic. Keep deliberate stylized "
    "proportions, linework, flat shading and minimalism. An explicit restyle takes "
    "priority over the base scene's medium; otherwise preserve that scene's style "
    "for a bounded edit. A reference supplied only for identity or objects does "
    "not prescribe the target medium; use it for style only when that role is requested. "
    "A still image may depict an action pose or intentional motion effects at one "
    "instant; do not ban action words or replace the pose with standing still. "
    "Keep these writing instructions out of the finished image prompt."
)
