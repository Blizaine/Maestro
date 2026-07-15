IMAGE PROMPT (image_prompt) — edit instructions for FLUX.2 Klein:

FORMAT: Direct edit instructions. Describe the FIRST FRAME of the scene as a transformation of the reference image.

CRITICAL — THE IMAGE PROMPT IS THE OPENING FRAME OF THE SCENE:
The image prompt creates the FIRST FRAME that the video animates FROM.
It must show the STARTING STATE — before any action happens in the video prompt.
- If the scene involves a woman removing her top → the image shows her WEARING the top.
- If a man walks to a door → the image shows him AWAY from the door, where he starts.
- If characters sit down → the image shows them STANDING, about to sit.
- The VIDEO PROMPT handles the action. The IMAGE PROMPT sets up the moment BEFORE.

FLUX.2 KLEIN EDIT PATTERNS — use direct, concrete instructions:
- "Place [character] in [setting]." for relocation
- "Add [element] to [location]." for additions
- "Change [attribute] to [target state]." for transformations
- "Replace [element] with [new element]." for swaps

Keep image prompts under 80 words. Front-load the main subject and the
new state. Klein does not auto-enhance prompts — every word must contribute
visible information.

ANCHOR CHARACTERS TO REFERENCE IMAGES:
- Single reference: "the woman from the reference image", "the man from the reference image"
- Multiple references: "the man from image one", "the woman from image two"
- Always preserve the age/role descriptor: teen boy → "teen boy", elderly woman → "elderly woman"
- NEVER use character names — describe by appearance only.
- The MAIN PERFORMER / protagonist IS the character shown in the reference
  image. ALWAYS anchor them: "the mouse from the reference image", "the
  singer from the reference image". Describing them loosely ("a mouse in
  black") makes the image model invent a NEW character design — and hand
  the reference's look to the background characters instead.

CLOTHING — NEVER NAME GARMENTS, EVER:
The reference image already shows what each character wears. Naming a garment
("blue shirt", "khaki pants") tells Klein to generate that specific garment,
which substitutes whatever the reference actually shows.

ABSOLUTE BANS — never use ANY of these in an image_prompt:
  shirt, t-shirt, polo, blouse, sweater, sweatshirt, hoodie, jacket, coat,
  blazer, vest, cardigan, dress, gown, skirt, pants, slacks, trousers,
  jeans, shorts, leggings, tights, stockings, lab coat, scrubs, uniform,
  robe, bodice, corset, bikini, bra, lingerie, suit, tie, tuxedo.

DISAMBIGUATE WITH COLOR ALONE (no garment word):
- BAD:  "the man in the blue shirt and the woman in the green sweater"
- GOOD: "the man in blue and the woman in green"
The reference image already shows what each character is wearing.

The ONLY time naming a garment is acceptable is when clothing explicitly
CHANGES in the scene: "now in workout clothes after changing", "wearing a new red dress she just put on".

LIGHTING — Klein responds strongly to specific lighting cues. Always describe:
- source: window light, sunlight, neon, studio, candlelight
- quality: soft, harsh, diffused, direct, overcast
- direction: side-lit, backlit, overhead
- color: warm/cool, golden, blue, amber

End every image_prompt with: "Use lighting and color temp from reference image."
This preserves visual continuity across scenes.

ANCHOR CONTACT POINTS TO ANATOMY, NOT SCENERY:
When characters lean toward, reach for, or come close to another, name the
specific target BODY PART — not the furniture or general area.
- BAD: "leaning over his bed"      GOOD: "leaning over his lap"
- BAD: "reaching toward the table" GOOD: "reaching toward the cup on the table"
- BAD: "her face close to him"     GOOD: "her face close to his lips"

BODY POSITION — describe PHYSICAL MECHANICS, not emotions:
- Klein cannot interpret "lost in the moment" or "locked together emotionally" — describe physical arrangement.
  BAD:  "they greet each other warmly"
  GOOD: "she steps forward with arms raised, his hands extending to meet hers, faces inches apart"
- Specify POSTURE (standing, seated, kneeling), LIMB POSITIONS, FACIAL STATE, SPATIAL RELATIONSHIPS.

IMAGE PROMPTS DESCRIBE A FROZEN FRAME:
- Describe WHERE each person IS, not what they are DOING over time.
- POSITIONS not movements. The image is a single frozen moment.
- NO motion verbs: no walking, running, reaching, turning, dancing, gesturing.
- NO motion-photography effects: no motion blur, speed lines, long exposure,
  camera shake. The still frame is SHARP — motion belongs to the video.
- Describe EXPRESSIONS as physical states: "mouth open, brow furrowed" not "looking angry".

STYLE CONSISTENCY:
- Match the visual MEDIUM and ART STYLE of the reference image, whatever it is.
  - Photorealistic reference → every prompt stays photorealistic. Do NOT introduce
    cartoon, anime, or illustration styles.
  - Stylized reference (hand-drawn, pencil sketch, watercolor, anime, cartoon, oil
    painting, pixel art, etc.) → NAME that medium explicitly in EVERY image prompt
    (e.g. "hand-drawn pencil sketch style" or "watercolor illustration") and do NOT
    introduce photorealistic rendering. Without naming the medium, the image model
    defaults to photorealism and destroys the style.
- Never switch styles between shots; the reference image's medium is the medium of
  the whole production.

CRITICAL — EVERY IMAGE PROMPT MUST BE VISUALLY UNIQUE:
- VARY composition: close-up, wide shot, low angle, overhead, profile, over-shoulder.
- VARY environment WITHIN the scene's established location(s): "same environment" for
  most shots; move to a NEW location only when the scene description itself calls for
  it. If the user's description pins the scene to one location, EVERY shot stays there —
  vary the angle, framing, and distance, never the place.
- VARY who is featured: some shots focus on one character, others show the group.
- NEVER write the same framing/pose/environment for consecutive shots.

EXAMPLES SHOW FORMAT ONLY — never copy their subjects, animals, objects, or settings
into your prompts. Everything in your prompts must come from THIS production's scene
description and reference images.

GOOD EXAMPLES:
- "Place the woman from the reference image in a sunlit kitchen, seated at the table, hands wrapped around a coffee cup. Soft morning light through the window. Use lighting and color temp from reference image. Preserve character identity, attire, and body attributes from the reference image."
- "Wide shot. The boy from the reference image stands center, the golden retriever from the reference image on his left, the tabby cat from the reference image on his right. Brick alleyway, overcast daylight. Use lighting and color temp from reference image. Preserve character identity, attire, and body attributes from the reference image."
- (stylized reference) "Hand-drawn pencil sketch style, matching the reference image's art style. The girl from the reference image sits on a swing under a large tree, midday light. Preserve character identity, attire, body attributes, and the art style of the reference image."

BAD EXAMPLES:
- "create new scene, same environment. Blonde man playing guitar." — no reference anchoring, no lighting.
- "the woman in the red dress with curly hair" — names a garment.
- "cartoon style illustration of..." (for a PHOTOREALISTIC reference) — introduces a style not in the reference.
- "photorealistic, 8k detailed photo of..." (for a HAND-DRAWN reference) — same mistake in the other direction.

ALWAYS end every image_prompt with: "Preserve character identity, attire, body attributes, and the art style of the reference image."
