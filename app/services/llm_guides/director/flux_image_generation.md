DIRECTOR IMAGE PROMPT (image_prompt) — the opening image for a planned scene.
Describe the visible opening state before the planned video action advances. This is a temporal constraint; it does not choose the image's medium or style.

FORMAT:
"create new scene, [environment]. [subject or character] [opening position or pose]. [requested style and any useful lighting details]."

RULES:
- Start with "create new scene, [detailed environment]."
- Describe the scene's requested opening composition, subjects, relationships, and visible details.
- Keep the image at the start of the planned scene. Do not depict a later action or outcome as already complete.
- A still image may depict a requested action pose or visual effect at one instant. Do not turn the prompt into a sequence; do not ban action words, motion lines, blur, or other effects when the brief requests them.
- Use only subjects and details from the user's concept and supplied references. Additional labeled references may identify characters, objects, locations, or style only according to the roles the user assigned. Do not invent missing references, clothing, or source-image constraints.
- Follow the user's requested medium, visual style, and lighting. Do not default to photography or photorealism, copy reference lighting, or impose lens specifications, grading, or other aesthetic choices.
- If no style or lighting is specified, keep the choice open or add only scene-appropriate detail that does not impose a medium.
- Do not use character names when the Director's downstream image model requires visual descriptions.
- Output one image prompt. Do not describe the future video action or add commentary.
