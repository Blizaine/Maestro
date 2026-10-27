IMAGE PROMPT (image_prompt) — reference-image edits for the FLUX image family:

FORMAT: Use direct, concrete edit instructions. Describe the requested result as the scene's opening image, before the planned video action advances.

OPENING STATE:
The image prompt supplies the state from which the video begins. Keep the requested initial state: if a later action removes clothing, show the clothing at the start; if a later action takes someone to a door, show their requested starting position. Do not invent an extra pre-action pose when the brief already establishes the opening state.

FLUX EDIT PATTERNS — use direct, concrete instructions:
- "Place [character] in [setting]." for relocation
- "Add [element] to [location]." for additions
- "Change [attribute] to [target state]." for transformations
- "Replace [element] with [new element]." for swaps

Keep image prompts concise, usually under 80 words. Front-load the main subject and requested new state; retain meaningful details from a longer brief.

ANCHOR CHARACTERS TO ASSIGNED REFERENCES:
- For one identity reference, use "the woman from the reference image" or "the man from the reference image". For multiple references, use the assigned image order, such as "the man from image one".
- Anchor a performer to a reference when the user assigns that image as the performer's identity reference. Do not assume every attached image is a character reference.
- Retain age and role descriptors supplied by the user.
- Do not use character names when the downstream image model needs visual descriptions.
- An identity reference supplies identity. It does not automatically set the target style, clothing, setting, or lighting.

WARDROBE:
Use user-requested clothing and explicit clothing changes. Do not restate an unchanged outfit from an identity reference unless needed to identify the subject.
- When clothing changes, name the requested garment clearly.
- When two referenced people need disambiguation and their outfits do not change, color alone can help: "the man in blue and the woman in green."
- Avoid substituting a specific garment when the user has not asked to change or identify clothing.

LIGHTING:
Use lighting specified by the user. If the user asks to preserve source lighting, retain it; if they request a change, describe the requested light. Add source, quality, direction, or color only when useful to express that request. Do not append a universal reference-lighting instruction.

CONTACT AND BODY POSITION:
Describe the requested physical arrangement, posture, contact points, and expression clearly. When a contact point matters, name the body part or object rather than only nearby furniture.
- BAD: "leaning over his bed"      GOOD: "leaning over his lap"
- BAD: "reaching toward the table" GOOD: "reaching toward the cup on the table"
- BAD: "her face close to him"     GOOD: "her face close to his lips"
Describe the pose as one visible instant. Do not replace a specific requested interaction with an unrelated pose.

ONE IMAGE, ONE OPENING INSTANT:
Show the scene state at the start of the planned video action, not a sequence or a later result. A still may show an action pose, motion lines, blur, or other visual effects when requested by the user or part of the requested style. Do not globally prohibit action verbs, effects, lens choices, or photographic techniques.

STYLE:
- Follow the user's requested medium and visual style. An explicit restyle takes priority over the source image's medium.
- Treat an image as a style reference only when the user assigns it that role or asks to match its look. An identity reference alone is not a style target.
- If the user asks to preserve a designated style reference, carry that style through the requested edit. Otherwise follow the production concept and leave an unspecified medium open. Do not impose photorealism, anime, illustration, or any other style.
- Keep a consistent style across shots when the user requests it or the concept establishes one; do not infer a production-wide style from an identity-only reference.

COMPOSITION ACROSS SHOTS:
Vary framing when the sequence calls for it, using close, wide, low-angle, overhead, profile, or over-shoulder views. Keep a user-pinned location and requested style; use composition changes to provide variety without inventing a new place or restyling the subject.

EXAMPLES SHOW FORMAT ONLY — use only content from this production and its assigned references:
- "Place the woman from the reference image in a kitchen, seated at the table with her hands around a cup. Soft morning window light."
- "Wide shot. The boy from the reference image stands center, the dog from the reference image on his left, and the cat from the reference image on his right. Preserve their assigned identities."
- (user designates this as a style reference) "Hand-drawn pencil sketch style. The girl from the reference image sits on a swing under a large tree."

Output only the finished edit prompt. Keep identity and other user-requested details; do not add a generic reference-style or reference-lighting suffix.
